---
name: ml-training-window-assessor
description: |
  Assess whether a training window can be extended by adding a new data source (e.g., CRM-only
  features to bypass behavioral data limitations). Use when: (1) the user asks "can we train on
  more data?", "how far back can we go?", "can we extend the training window?", or "our model
  only has N months of data", (2) a multi-output model has different effective training windows
  per output due to dynamic lookforward, (3) you need to determine whether labels or features
  are the binding constraint for historical data availability, (4) the user mentions seasonal
  patterns with insufficient training cycles, (5) the user says "extend the training window",
  "not enough historical data", "how many complete seasonal cycles?", "which data source is
  the bottleneck for training history?", (6) someone asks about training on pre-launch data
  by dropping behavioral features, (7) the user mentions lookforward windows eating into
  effective training months, (8) the user wants to include older records or historical data
  by adding a new data source, (9) "barely any training data" or one output has much less
  data than others, (10) "should we add older data from before the tracking pixel was
  installed?", "include data from before [system] was set up".
  Do NOT use for: questions about individual column evaluation or "should we add feature X?"
  — use ml-feature-evaluator instead for that.
  Covers: per-output label validity computation, lookforward bridging, feature vs label binding
  constraints, drift-aware validation of extension periods, purged temporal cross-validation
  with embargo, and Option B (companion model) vs Option C (extended training with feature flag)
  architecture decisions.
author: Claude Code
version: 2.1.0
date: 2026-03-24
---

# ML Training Window Assessor

Evaluate whether and how far a model's training window can be extended by incorporating
a new data source or removing a data dependency bottleneck.

## Problem

Multi-output temporal models (e.g., per-term enrollment prediction) often have training
windows that are shorter than they appear. A nominal date range of "10 months" may actually
be 3 months for one output and 9 months for another, due to dynamic lookforward windows
and label validity constraints. This leads to:

- Models that have never seen a complete seasonal cycle for some outputs
- Overconfident claims about training data sufficiency
- Missed opportunities to extend the window by swapping data sources

## Context / Trigger Conditions

Use this skill when:
- The user says the model "only has X months of training data"
- Training data is bottlenecked by one data source (e.g., behavioral events start recently)
- A multi-output model has per-output lookforward windows of different lengths
- The user asks about using CRM/static features to train on older data
- Seasonal patterns exist but training doesn't cover full cycles

## Solution

### Step 1: Compute Per-Output Valid Training Months

The effective training window is NOT `end_date - start_date`. For each output:

```
valid_end_date[output] = max target_date WHERE target_date + lookforward[output] <= data_available_through
valid_months[output] = valid_end_date[output] - start_date
```

Where `lookforward[output]` is the maximum days needed for the label to materialize for
that output. In dynamic lookforward systems, this varies by output and by target_date.

**Key insight:** Outputs with earlier "next cycle starts" (e.g., Summer starting in May)
run out of valid labels sooner than outputs with later starts (e.g., Spring starting in January).

Always report per-output figures, never a single aggregate.

### Step 2: Identify the Binding Constraint

Check BOTH sides:

1. **Feature availability:** When does each data source have meaningful volume?
   - Query: `SELECT DATE_TRUNC(created_date, MONTH), COUNT(*) GROUP BY 1 ORDER BY 1`
   - Look for the month where volume becomes "meaningful" (hundreds+ per month)

2. **Label availability:** When do outcome labels begin?
   - Query: `SELECT DATE_TRUNC(label_date, MONTH), COUNT(*) GROUP BY 1 ORDER BY 1`
   - Labels may be sparse initially — check if volume is enough to learn from

The earliest valid target_date = `max(feature_start, label_start)`, adjusted for
lookback windows (features) and lookforward windows (labels).

**Either side can be the bottleneck:**
- Current model may be feature-bound (e.g., behavioral events start recently)
- Extended model may be label-bound (e.g., CRM features go back years but labels don't)

### Step 3: Check if Lookforward Bridges the Gap

A target_date doesn't need labels ON that exact date — it needs labels within its
lookforward window.

Example: `target_date = 2024-01-01` with 9-month lookforward:
- Label window = Jan 1 to Sep 28, 2024
- If labels start March 2024, this target_date is VALID (lookforward reaches past March)

This means the earliest valid target_date can be BEFORE the first label date, as long
as the lookforward window reaches into the period where labels exist.

### Step 4: Quantify the Extension Benefit

For each output, compute:

| Metric | Current | Extended | Gain |
|--------|---------|----------|------|
| Valid training months | X | Y | Y-X |
| Complete seasonal cycles | N | M | M-N |
| Label volume in extension period | - | K | new data |

The most compelling argument is usually **complete seasonal cycles gained** — going from
0 to 1 complete Summer cycle is more valuable than going from 5 to 6 months of Spring data.

### Step 5: Drift-Aware Feature Validation in Extension Period

Run three checks before accepting extension data. See `references/drift-validation.md`
for code, SQL queries, and implementation details.

**5a: PSI per feature.** Compute Population Stability Index between extension and current
periods. Thresholds: <0.10 safe, 0.10-0.25 caution, >0.25 investigate. If >20% of features
exceed 0.25, the bang-bang heuristic (Step 6) applies.

**5b: Label distribution stability.** Check positive rate per quarter per output. Shift
>5pp warrants investigation. If correlated with seasonal patterns → expected. If not → halt
extension for that output.

**5c: Feature-outcome relationship stability.** Compute per-period univariate AUC for top
features. If any feature's AUC flips direction between periods, that's concept drift at
the feature level.

### Step 5.5: Temporal Stability Test via Purged Walk-Forward CV

Standard k-fold CV on temporal data leaks information. Use purged walk-forward CV with
embargo = max(lookforward) across all outputs. See `references/drift-validation.md` for
implementation with `timeseriescv` or manual purge logic.

Run twice: (1) current window baseline, (2) extended window. If AUC degrades >0.01 on
recent test folds, the extension is hurting — apply bang-bang heuristic.

### Step 6: Choose Architecture (Drift-Informed)

**Option A: Extended Training Window (single model)**
- Change `start_date` to the earlier date
- Add `has_[data_source]` binary flag for the data source that's missing in older data
- Tree models (XGBoost, LightGBM) handle missing values natively
- Pro: Simpler. Con: Risk of model learning "no behavioral data" as temporal proxy.

**Option B: Companion Model**
- Train a separate model on the extended window using only universally-available features
- Feed its output as a feature into the main model
- Pro: Clean separation. Con: Two models to maintain.

**Decision heuristic (drift-informed):**

```
IF Step 5a shows <20% of features with PSI > 0.25
   AND Step 5c shows no feature-outcome relationship flips
   AND Step 5.5 shows no AUC degradation on recent test folds:
   → Option A (extend with full data)

ELIF Step 5a shows >50% of features with PSI > 0.25
   OR Step 5c shows multiple relationship flips:
   → BANG-BANG: Do NOT partially extend.
     Either use Option B (companion model on stable features only)
     or do not extend at all.

ELSE (moderate drift, 20-50% features shifted):
   → Option A with feature selection:
     Exclude features with PSI > 0.25 from extension period rows.
     Keep them as NaN (XGBoost handles natively).
```

**Bang-bang insight:** Under significant concept drift, the optimal policy is binary —
use all data from a period or none. Partial windows add noise without signal.
(arXiv:2512.12816)

## Implementation Pitfalls

When implementing Option A, avoid these traps. See `references/pitfalls.md` for full
details, code examples, and references.

1. **`.fillna(0)` destroys XGBoost's NaN routing.** Gate `.fillna(0)` on the data source's
   availability date — only fill 0 for post-tracking rows. Pre-tracking rows must stay NaN
   so XGBoost's sparsity-aware algorithm can learn separate routing per population.

2. **Sentinel values need gating too.** Use `999` (not `-1`) for "never occurred" sentinels.
   Pre-tracking rows should be NaN (not sentinel) — sentinel means "tracking existed but
   event never occurred," which is a different signal than "tracking didn't exist."

3. **Preprocessing parity across pipeline files.** The gated fillna logic must be identical
   in training, serving, dashboard, and export scripts. Extract into a shared function.
   Serving script may not need changes if all served data is post-tracking.

4. **Standard CV on temporal data leaks.** Always use purged CV with embargo matching
   the lookforward window (Step 5.5). Compare purged vs standard CV AUC — if gap > 0.02,
   you have leakage.

## Post-Extension Monitoring

After deploying, monitor for drift. See `references/drift-validation.md` for code.

- **Batch:** Monthly PSI checks. If >20% of features exceed 0.25 → retrain. If positive
  rate shifts >5pp → investigate.
- **Streaming:** Use River's ADWIN for real-time drift detection on prediction confidence.
- **Temporal safety:** Every feature must be computable from data available at
  `target_date - 1 day`. Temporian enforces this by default.

## Verification

After computing per-output training windows:

1. Cross-check against the actual SQL/config:
   - What is `start_date`, `end_date`, `data_available_through`?
   - How is lookforward computed? (static vs dynamic)
   - How is `label_valid` defined?

2. Validate label volume in the extension period:
   - Are there enough positive labels per month to learn from?
   - Is the positive rate stable or does it shift? (Step 5b)

3. Run drift checks on the extension period:
   - PSI for all features (Step 5a)
   - Feature-outcome relationship stability (Step 5c)

4. Test the extended model with proper temporal CV:
   - Use purged walk-forward CV with embargo (Step 5.5)
   - Compare per-output AUC: current window vs extended window
   - Check calibration across both time periods
   - Verify no temporal leakage (compare purged vs standard CV AUC)

5. Validate feature importance across periods:
   - Compare feature importance rankings between pre-extension and post-extension subsets
   - CRM/static features should dominate in pre-extension period (where behavioral = NaN)
   - Behavioral features should add value in post-extension period
   - If behavioral features rank high in the pre-extension period, something is wrong
     (they should be NaN there)
   - Use partial dependence plots for top CRM features — should be similar across periods

## Example

**Scenario:** Enrollment propensity model with 3 term outputs (Fall, Spring/Winter, Summer).
Behavioral features start Aug 2024. Dynamic lookforward per term.

**Step 1 result:**
| Term | Lookforward | Valid Through | Training Months |
|------|-------------|---------------|-----------------|
| Summer | ~120 days | Jun 1, 2025 | 3.3 months |
| Fall | ~180 days | Jul 27, 2025 | 5.2 months |
| Spring | ~270 days | Nov 30, 2025 | 9.4 months |

The "~10 months" claim was only true for Spring. Summer had just 3.3 months.

**Step 2 result:** CRM features go back to 2023, but enrollment labels start March 2024.
Labels are the binding constraint for the extension (not features).

**Step 3 result:** target_date=2024-01-01 with 9-month lookforward reaches Sep 2024 —
well past when labels begin. Valid.

**Step 4 result:** Extension adds +14 months per term. Summer goes from 3.3 to 17.3 months
(5.2x), gaining its first complete enrollment cycle.

**Step 5 result:** PSI computed for 45 CRM features between extension and current period.
3 features (6.7%) show PSI > 0.25 (application_stage codes changed). Remaining 93% stable.
Feature-outcome AUC consistent across periods. → Safe to extend.

**Step 5.5 result:** Purged walk-forward CV (embargo=270 days, 5 splits):
| Term | Current AUC | Extended AUC | Delta |
|------|-------------|--------------|-------|
| Summer | 0.71 | 0.76 | +0.05 |
| Fall | 0.74 | 0.75 | +0.01 |
| Spring | 0.78 | 0.78 | +0.00 |

Extension helps Summer (more seasonal data), neutral for others. No degradation. → Proceed.

**Step 6 result:** Low drift (<20% features affected) + no AUC degradation → Option A
(single model, extended window, NaN for missing behavioral features).

## Input / Output Contract

**Input requires:** The user provides (or the skill asks for) the following context:
- Model type (single-output or multi-output) and output names
- Current training window dates (`start_date`, `end_date`, `data_available_through`)
- Lookforward window per output (static or dynamic)
- Data sources and their availability dates (e.g., "CRM since 2022, behavioral since Aug 2024")
- Label definition and when labels begin

**Output produces:** A structured assessment containing:
- Per-output valid training months table (current vs extended)
- Binding constraint identification (features vs labels)
- Drift analysis results (PSI, label stability, feature-outcome AUC)
- Architecture recommendation (Option A / Option B / do not extend)
- Purged walk-forward CV comparison if data is available

## Composability

**Handoff points:**
- Then use `ml-feature-evaluator` to evaluate individual features available in the extension period
- If the user asks "should we add feature X?" suggest using `ml-feature-evaluator` instead
- After this assessment, the next step is implementation (modify training pipeline `start_date` and preprocessing)

**Error handling:** If the assessment fails due to insufficient information (e.g., unknown lookforward windows or missing data source dates), the skill asks clarifying questions before proceeding. If PSI computation fails due to sparse data, it gracefully degrades to qualitative drift assessment.

**Safety:** This skill is safe to re-run multiple times. Running the assessment again with updated dates produces updated results with no side-effects. The analysis is read-only and does not modify any training pipeline or data.

**Dependencies:** Depends on Python with `numpy`, `pandas`, `sklearn` for drift analysis code snippets. The `timeseriescv` package is recommended for purged CV but manual implementation is provided as a fallback. Alternatively, `mlfinlab` provides equivalent CPCV functionality.

**Scope:** This skill is scoped to training window extension decisions only. It is out of scope for feature engineering, model selection, hyperparameter tuning, deployment, or serving pipeline changes.

**Compatibility:** Requires XGBoost >= 1.0, LightGBM >= 3.0, or any scikit-learn compatible model. PSI thresholds (0.10, 0.25) follow industry standard conventions. Requires Python >= 3.8.

## Notes

- This skill complements `ml-feature-evaluator` — that skill asks "should we add feature X?",
  this one asks "can we extend the training window?"
- The per-output computation is critical for multi-output models. Single-output models
  can use a simpler calculation but should still check label validity.
- `data_available_through` is a moving target — as more time passes, the valid window
  for each output extends. Recompute periodically.
- See also: `ml-feature-evaluator` for the 6-query diagnostic pattern when evaluating
  the features available in the extension period.

## References

See `references/drift-validation.md` and `references/pitfalls.md` for full reference
tables with links and star counts.
