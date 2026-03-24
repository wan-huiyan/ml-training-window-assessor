---
name: ml-training-window-assessor
description: |
  Assess whether a training window can be extended by adding a new data source (e.g., CRM-only
  features to bypass behavioral data limitations). Use when: (1) the user asks "can we train on
  more data?", "how far back can we go?", or "our model only has N months of data", (2) a
  multi-output model has different effective training windows per output due to dynamic lookforward,
  (3) you need to determine whether labels or features are the binding constraint for historical
  data availability, (4) the user mentions seasonal patterns with insufficient training cycles,
  (5) the user says "extend the training window", "not enough historical data", "how many
  complete seasonal cycles?", "which data source is the bottleneck for training history?",
  (6) someone asks about training on pre-launch data by dropping behavioral features,
  (7) the user mentions lookforward windows eating into effective training months.
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

Before accepting extension data, verify that the extension period's data distribution
is compatible with the current period. This goes beyond simple coverage checks — you need
statistical evidence that older data won't degrade the model.

#### 5a: Population Stability Index (PSI)

For each feature, compute PSI between the extension period and the current period:

```python
def compute_psi(expected, actual, bins=10):
    """PSI between two distributions. Lower = more stable."""
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 0.001, None)
    actual_pct = np.clip(actual_pct, 0.001, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

**PSI thresholds (industry standard):**
| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | Insignificant shift | Safe to extend |
| 0.10 - 0.25 | Moderate shift | Extend with caution, monitor |
| > 0.25 | Significant shift | Investigate before extending — this feature may need exclusion or transformation |

Compute PSI for every feature. If >20% of features have PSI > 0.25, the extension period
is structurally different and the "bang-bang" heuristic (Step 6) applies.

#### 5b: Label Distribution Stability

Check that the outcome rate (positive class proportion) is stable:

```sql
SELECT
  DATE_TRUNC(target_date, QUARTER) AS quarter,
  output_name,
  COUNT(*) AS n_samples,
  AVG(CAST(label AS FLOAT64)) AS positive_rate
FROM training_data
WHERE label_valid = TRUE
GROUP BY 1, 2
ORDER BY 1, 2
```

A positive rate shift > 5 percentage points between periods warrants investigation.
It may reflect genuine seasonal patterns (expected) or data quality issues (dangerous).

#### 5c: Feature-Outcome Relationship Stability

The most critical check: does the relationship between features and outcomes hold
across periods? Compute per-period AUC for the top features:

```python
from sklearn.metrics import roc_auc_score

for period in ['extension', 'current']:
    period_data = df[df['period'] == period]
    for feature in top_features:
        valid = period_data[[feature, 'label']].dropna()
        if len(valid) > 50:
            auc = roc_auc_score(valid['label'], valid[feature])
            print(f"{period} | {feature} | AUC={auc:.3f}")
```

If a feature's univariate AUC flips direction (e.g., 0.62 in current → 0.45 in extension),
that feature's relationship with the outcome has fundamentally changed. This is concept
drift at the feature level.

**References:**
- [Frouros](https://github.com/IFCA-Advanced-Computing/frouros) — 28 drift detection
  algorithms including PSI, KS test, Chi-squared, for both batch and streaming contexts
- [River ADWIN](https://github.com/online-ml/river) — Adaptive windowing for streaming
  drift detection; useful for monitoring post-deployment

### Step 5.5: Temporal Stability Test via Purged Walk-Forward CV

Standard k-fold CV on temporal data causes information leakage. When evaluating an
extended training window, you MUST use temporal cross-validation with purging and embargo.

#### Why Purging and Embargo Matter

Each training sample has two timestamps:
- **pred_time**: when features are observed (the target_date / prediction date)
- **eval_time**: when the outcome becomes known (target_date + lookforward)

Without purging, training samples whose eval_time overlaps with the test fold's pred_time
will leak future information into the model.

#### Purge Condition

Remove any training sample where:
```
train_eval_time >= test_pred_time_start
```

In practice: if your test fold predicts enrollment for Jan 2025, purge training samples
whose lookforward window extends past Jan 2025.

#### Embargo Period

After purging, add a temporal buffer (embargo) between the test fold and subsequent
training data:

```
embargo_threshold = max(test_eval_times) + embargo_duration
```

Exclude training samples with `pred_time <= embargo_threshold`. This prevents temporal
autocorrelation from leaking through adjacent time periods.

**Embargo duration heuristic:** Set embargo = max(lookforward) across all outputs.
For enrollment models with 9-month lookforward, embargo should be ~270 days.

#### Implementation

```python
from timeseriescv.cross_validation import PurgedWalkForwardCV

cv = PurgedWalkForwardCV(
    n_splits=5,
    n_test_splits=1
)

# pred_times = target_date for each sample
# eval_times = target_date + lookforward for each sample
for train_idx, test_idx in cv.split(X, pred_times=pred_times, eval_times=eval_times):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

Or implement manually:

```python
def purged_temporal_split(df, test_start, test_end, lookforward_days, embargo_days=0):
    """Split with purging and embargo for temporal models."""
    test = df[(df['target_date'] >= test_start) & (df['target_date'] < test_end)]

    # Purge: remove training samples whose labels overlap with test predictions
    test_pred_start = test['target_date'].min()
    purge_mask = df['eval_time'] < test_pred_start

    # Embargo: add buffer after test fold
    test_eval_end = test['eval_time'].max()
    embargo_end = test_eval_end + pd.Timedelta(days=embargo_days)

    train_before = df[purge_mask & (df['target_date'] < test_start)]
    train_after = df[df['target_date'] > embargo_end]  # if using future folds
    train = pd.concat([train_before, train_after])

    return train, test
```

#### Comparing Current vs Extended Window

Run purged walk-forward CV twice:
1. **Current window only** — establish baseline per-output AUC
2. **Extended window** — measure impact of adding older data

If extended window shows AUC degradation > 0.01 for any output on recent test folds
(where all features are available), the extension is hurting. Apply the bang-bang
heuristic from Step 6.

**References:**
- [timeseriescv](https://github.com/sam31415/timeseriescv) — Scikit-learn style purged
  walk-forward and combinatorial purged k-fold CV (284 stars, actively maintained)
- [MLFinLab](https://github.com/hudson-and-thames/mlfinlab) — De Prado's Combinatorial
  Purged Cross-Validation (CPCV) from "Advances in Financial Machine Learning" (4,613 stars)
- De Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 7:
  Cross-Validation in Finance

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

**The bang-bang insight** (from optimal control theory, applied to ML training under
concept drift): When concept drift is significant, the optimal policy is binary — either
use all the data from a period or none of it. Partial windows from drifted periods add
noise without enough signal. This is proven optimal when concept durations follow
Decreasing Mean Residual Life (DMRL) distributions, which is common in practice (the
longer a stable regime has lasted, the more likely it is to end soon).

**Reference:** "Optimal Resource Allocation for ML Model Training and Deployment under
Concept Drift" (arXiv:2512.12816, Dec 2025)

## Critical Pitfall: `.fillna(0)` Destroys XGBoost's Missing Value Handling

When implementing Option A (extended training window), XGBoost's sparsity-aware algorithm
(Algorithm 2, Chen & Guestrin 2016) learns an optimal "default direction" for NaN values
at every tree node. This is how it naturally handles rows where a data source didn't exist —
NaN rows get routed down CRM-feature branches, non-NaN rows use behavioral+CRM branches.

**The trap:** Many ML pipelines impute missing values to 0 before training (e.g.,
`.fillna(0)` for event counts). This destroys the missingness signal:
- "No tracking existed" (structural NaN) → 0
- "Tracked but zero events" (genuine zero) → 0

XGBoost can no longer distinguish the two populations. The sparsity-aware algorithm
has nothing to work with — there are no NaN values left.

**The fix:** Gate `.fillna(0)` on the data source's availability date:

```python
has_tracking = df['target_date'] >= pd.Timestamp('2025-02-23')  # Bloomreach start
behavioral_cols = [c for c in df.columns if c.startswith(('session_start_', 'page_view_', ...))]

for col in behavioral_cols:
    df.loc[has_tracking, col] = df.loc[has_tracking, col].fillna(0)
    # Pre-tracking rows: leave as NaN → XGBoost handles natively

# Sentinel features also need gating:
df.loc[has_tracking, 'days_since_deposit'] = df.loc[has_tracking, 'days_since_deposit'].fillna(999)
# Pre-tracking rows: leave as NaN (not sentinel)
```

**Serving script impact:** If all served data comes from the active-tracking period
(current date is always post-launch), the existing `.fillna(0)` in the serving script
remains correct. No serving change needed.

**Cold-start benefit:** At serving time, new users who haven't interacted with the
platform yet also have NaN behavioral features (from the SQL query returning NULL).
The model's learned NaN routing — "when behavioral data is missing, rely on CRM
features" — directly applies to cold-start users. This is a **free benefit** of
Option A: the model implicitly learns a cold-start strategy from the pre-tracking
training data.

**How XGBoost actually handles the missing block (per Chen & Guestrin 2016):**

XGBoost does NOT build separate "CRM-only trees" and "CRM+behavioral trees." What
happens is more granular: when a tree considers splitting on a behavioral feature,
all NaN rows get routed to one side via the learned default direction. The NaN-side
child node then only has CRM features available for further splitting (since all
behavioral features are simultaneously NaN for pre-tracking rows). The non-NaN side
can split on both CRM and behavioral features. So within a single tree, you get
**branches** that are effectively CRM-only and branches that use all features — it's
separate subtrees within the same tree, not separate trees.

Key mechanics:
- The default direction is learned **per-node, per-feature** — context-dependent,
  not a global decision. The same feature's NaN can go left at the root and right
  at a deeper node.
- Only non-missing observations are visited when computing split gains (efficient).
- **Edge case:** If no missing values exist in training data for a feature, missing
  values at inference time default to the right branch. This matters if your training
  data has no NaN for a feature but serving data does.

**Why NaN is better than a `has_behavioral_data` flag:**
- NaN lets XGBoost learn per-feature, per-node routing (more expressive)
- A binary flag forces a single global split — it's largely redundant with NaN
  routing, since XGBoost already implicitly learns a `has_behavioral_data` split
  when it routes all-NaN rows to one side
- NaN is the native interface for XGBoost's sparsity-aware algorithm
- **However:** if you must impute to 0 (e.g., pipeline constraints), then a
  `has_behavioral_data` flag becomes **necessary** — it's the only way the model
  can distinguish "no tracking existed" from "tracked, zero events"

**Feature importance diagnostic:** When training on mixed pre/post-tracking data,
behavioral features will appear **less important than they truly are**, because
pre-tracking rows (where they're NaN/0) dilute their discriminative power. Compare
feature importance on the post-tracking subset alone vs the full dataset — if
behavioral features rank much higher on the subset, the dilution effect is real
but the model is still learning the correct routing.

**Temporal proxy risk (MNAR):** When missingness is perfectly correlated with time period
(all pre-launch rows are NaN), the learned default direction encodes the old cohort's
patterns. Mitigate by: (1) verifying feature-outcome relationships are stable across
periods (Step 5c), (2) using purged temporal CV (Step 5.5), (3) monitoring per-period
calibration.

**References:**
- [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)](https://arxiv.org/pdf/1603.02754) — Algorithm 2: sparsity-aware split finding
- [XGBoost FAQ: Missing Values](https://xgboost.readthedocs.io/en/stable/faq.html)

## Critical Pitfall: Sentinel Values Interact with Missing Value Handling

When extending the training window, "days since X" features often use sentinel values
(e.g., 999 for "event never occurred"). The sentinel choice interacts with NaN handling:

**The trap:** A sentinel of `-1` for "never visited" groups semantically wrong with
XGBoost's threshold splits. A split at `> -0.5` puts "never visited" (-1) alongside
"visited today" (0) — the opposite of what you want.

**The fix:** Use `999` (not `-1`) for "never occurred" sentinels. This groups "never"
alongside "stale" (e.g., >180 days), which is semantically correct — someone who never
visited is more similar to someone who visited 6 months ago than someone who visited
today.

**Interaction with NaN gating:** For the extension period, sentinel features also need
gating. Pre-tracking rows should be NaN (not sentinel), because the sentinel means
"tracking existed but event never occurred" — a different signal than "tracking didn't
exist." Only fill sentinels for post-tracking rows.

## Critical Pitfall: Preprocessing Parity Across Pipeline Files

The gated `.fillna()` logic must be **identically synchronized** across every file that
preprocesses features. In a typical ML pipeline, this includes:

1. **Training script** (Cloud Run / batch job)
2. **Serving script** (Cloud Run / API endpoint)
3. **Dashboard / analysis script** (Streamlit / Jupyter)
4. **Export / SHAP analysis script**

If the training script gates `.fillna(0)` on target_date but the serving script doesn't
(or vice versa), you get **silent train-serve skew** — the model produces wrong
predictions with no error.

**Practical safeguard:** Extract the gating logic into a shared preprocessing function
imported by all scripts:

```python
def gate_fillna_behavioral(df, tracking_start='2025-02-23'):
    """Fill behavioral NaN with 0 only for rows where tracking existed."""
    has_tracking = df['target_date'] >= pd.Timestamp(tracking_start)
    behavioral_cols = [c for c in df.columns if any(
        c.startswith(p) for p in BEHAVIORAL_PREFIXES
    )]
    for col in behavioral_cols:
        df.loc[has_tracking, col] = df.loc[has_tracking, col].fillna(0)
    return df
```

Mark any legacy `.replace(-1, 999)` patches with comments referencing the
migration ticket, so they can be removed after the training data pipeline is rebuilt.

## Critical Pitfall: Standard CV on Temporal Data Causes Silent Leakage

Standard k-fold or even `TimeSeriesSplit` without purging allows information leakage
when samples have overlapping prediction-evaluation windows (which is the norm for models
with lookforward periods).

**The problem:** A training sample with `target_date = 2024-06-01` and `lookforward = 180 days`
has its outcome determined by events through Dec 2024. If a test sample has
`target_date = 2024-09-01`, standard temporal splits would include the June training sample —
but its label was determined using information from the test sample's prediction period.

**The fix:** Always use purged CV with embargo matching the lookforward window (Step 5.5).
Without this, evaluation of the extended window is unreliable — you may conclude the
extension helps when it's actually just leaking future information.

**How to detect if your current evaluation is leaking:**
1. Compare purged CV AUC vs standard CV AUC
2. If standard CV shows significantly higher AUC (>0.02 gap), you have leakage
3. The longer the lookforward window, the worse the leakage

**References:**
- De Prado, M.L. (2018). Chapter 7: "The dangers of ordinary cross-validation are
  compounded when the investment strategy involves dynamic position sizing"
- [timeseriescv](https://github.com/sam31415/timeseriescv) — implements the purge/embargo
  logic correctly

## Post-Extension Monitoring

After deploying a model trained on the extended window, monitor for concept drift
in production using these approaches:

### Batch Monitoring (recommended for periodic retraining)

Run PSI checks monthly between the training distribution and recent inference data:
- If PSI > 0.25 for >20% of features → trigger retraining
- If positive rate shifts > 5pp from training → investigate

### Streaming Monitoring (for real-time systems)

Use ADWIN (Adaptive Windowing) from the River library to detect drift in prediction
confidence or error rate:

```python
from river.drift import ADWIN

adwin = ADWIN(delta=0.002)  # Lower delta = more sensitive

for prediction_error in error_stream:
    in_drift, _ = adwin.update(prediction_error)
    if in_drift:
        print("Drift detected — consider retraining or window adjustment")
```

ADWIN maintains an adaptive-length window and detects when the mean of recent values
diverges statistically from historical values. It automatically shrinks the window
when drift is detected, making it suitable for detecting when the training window's
assumptions have expired.

**Reference:** [River](https://github.com/online-ml/river) — 5,700+ stars, actively maintained

### Temporal Safety in Feature Engineering

When extending the training window, ensure feature engineering pipelines don't introduce
future leakage. Google's Temporian library enforces this by default — all temporal
operators are causal (cannot depend on future data) unless explicitly overridden with
`tp.leak()`. Even if not using Temporian, adopt this principle:

**Rule:** Every feature computation must be expressible as a function of data available
at `target_date - 1 day` (or the appropriate lookback boundary). If extending the window
introduces features computed from data that wasn't available at prediction time, you
have leakage.

**Reference:** [Temporian](https://github.com/google/temporian) — Google's temporal
data library with built-in leakage prevention (710 stars)

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

## References Summary

| Source | What it contributes | Stars |
|--------|-------------------|-------|
| [timeseriescv](https://github.com/sam31415/timeseriescv) | Purged walk-forward CV + combinatorial purged k-fold with embargo | 284 |
| [River](https://github.com/online-ml/river) | ADWIN adaptive windowing for streaming drift detection | 5,752 |
| [Temporian](https://github.com/google/temporian) | Temporal safety / leakage prevention in feature engineering | 710 |
| [MLFinLab](https://github.com/hudson-and-thames/mlfinlab) | De Prado's CPCV, triple barrier labeling (analogous to dynamic lookforward) | 4,613 |
| [Frouros](https://github.com/IFCA-Advanced-Computing/frouros) | 28 drift detection algorithms (PSI, KS, Chi-squared) | 252 |
| [arXiv:2512.12816](https://arxiv.org/abs/2512.12816) | Bang-bang optimality for training window under concept drift | — |
| Chen & Guestrin (2016) | XGBoost sparsity-aware split finding (Algorithm 2) | — |
| De Prado (2018) | Purged CV theory, temporal leakage in finance ML | — |
