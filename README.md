# ML Training Window Assessor

A [Claude Code](https://claude.com/claude-code) skill that provides a structured, drift-aware methodology for evaluating whether and how far an ML model's training window can be extended.

![Steps 1-4: Training window assessment](docs/demo-diagnostic-1.png)
*Per-output training window computation, binding constraint identification, and +31 month extension benefit for a SaaS churn model.*

<details>
<summary>See drift validation, XGBoost routing, and architecture decision</summary>

![Step 5: PSI drift validation, label stability, feature-outcome AUC](docs/demo-diagnostic-2.png)

*PSI-based drift validation with color-coded bars, label distribution stability, and feature-outcome AUC stability across years.*

![Step 6: Decision heuristic, XGBoost sparsity-aware routing, .fillna(0) pitfall, verdict](docs/demo-diagnostic-3.png)

*Drift-informed architecture decision, XGBoost NaN routing diagram, critical .fillna(0) pitfall, and final EXTEND verdict.*
</details>

## Core Problem It Solves

Multi-output temporal models often have training windows that are shorter than they appear. A nominal "10 months of data" may actually be 3 months for one output and 9 months for another, due to dynamic lookforward windows and label validity constraints. This leads to:

- Models that have never seen a complete seasonal cycle for some outputs
- Overconfident claims about training data sufficiency
- Missed opportunities to extend the window by swapping data sources
- Silent temporal leakage in evaluation when using standard cross-validation

## How It Works

The skill runs a **7-step diagnostic**:

**Steps 1-3: Structural Assessment**
- Compute per-output valid training months (not just `end_date - start_date`)
- Identify whether features or labels are the binding constraint
- Check if lookforward windows bridge the gap to earlier data

**Step 4: Quantify Extension Benefit**
- Per-output gains in months, seasonal cycles, and label volume
- Prioritize by complete seasonal cycles gained (0 to 1 > 5 to 6)

**Step 5: Drift-Aware Feature Validation** *(v2.0 — new)*
- **PSI (Population Stability Index)** for every feature between periods (<0.10 safe, 0.10-0.25 caution, >0.25 investigate)
- **Label distribution stability** — positive rate shift detection across periods
- **Feature-outcome relationship stability** — per-feature univariate AUC to detect concept drift at the feature level

**Step 5.5: Purged Walk-Forward CV with Embargo** *(v2.0 — new)*
- Temporal cross-validation that respects `pred_times` / `eval_times` overlap
- Purge condition: `train_eval_time < test_pred_time` prevents information leakage
- Embargo buffer matching the lookforward window prevents temporal autocorrelation
- Compare purged CV results between current and extended windows

**Step 6: Drift-Informed Architecture Decision** *(v2.0 — upgraded)*
- Three-tier decision tree based on drift severity
- **Bang-bang optimality**: when drift is significant, use all extension data or none — partial windows are suboptimal under DMRL distributions
- Option A (single model + NaN) vs Option B (companion model) selection

## Key Pitfalls Addressed

### `.fillna(0)` Destroys XGBoost's Missing Value Handling
When extending windows, older rows lack certain data sources. XGBoost's sparsity-aware algorithm (Algorithm 2, Chen & Guestrin 2016) learns optimal routing for NaN values — but `.fillna(0)` destroys this by conflating "no tracking existed" with "tracked but zero events." The skill provides the gated `.fillna()` fix.

### Standard CV on Temporal Data Causes Silent Leakage *(v2.0 — new)*
Standard `TimeSeriesSplit` without purging allows leakage when samples have overlapping prediction-evaluation windows. A training sample with 180-day lookforward can leak information into test samples predicted during that window. The skill provides purged CV with embargo as the correct evaluation method.

## Research & Open-Source Foundations

| Source | Contribution |
|--------|-------------|
| [timeseriescv](https://github.com/sam31415/timeseriescv) | Purged walk-forward CV + combinatorial purged k-fold with embargo |
| [River](https://github.com/online-ml/river) | ADWIN adaptive windowing for streaming drift detection |
| [Temporian](https://github.com/google/temporian) | Temporal safety / leakage prevention in feature engineering |
| [MLFinLab](https://github.com/hudson-and-thames/mlfinlab) | De Prado's Combinatorial Purged CV (CPCV) |
| [Frouros](https://github.com/IFCA-Advanced-Computing/frouros) | 28 drift detection algorithms (PSI, KS, Chi-squared) |
| [arXiv:2512.12816](https://arxiv.org/abs/2512.12816) | Bang-bang optimality for training window under concept drift |
| Chen & Guestrin (2016) | XGBoost sparsity-aware split finding |
| De Prado (2018) | Purged CV theory from *Advances in Financial Machine Learning* |

## Installation

### Via Claude Code CLI
```bash
claude install-skill github:wan-huiyan/ml-training-window-assessor
```

### Via Git Clone
```bash
git clone https://github.com/wan-huiyan/ml-training-window-assessor.git
cp -r ml-training-window-assessor/skills/ml-training-window-assessor ~/.claude/skills/
```

### Manual
Copy `SKILL.md` into `~/.claude/skills/ml-training-window-assessor/SKILL.md`

## Trigger Conditions

The skill activates when:
- "Can we train on more data?" / "How far back can we go?"
- "Our model only has X months of training data"
- Training data is bottlenecked by one data source
- A multi-output model has per-output lookforward windows of different lengths
- Seasonal patterns exist but training doesn't cover full cycles

## Related Skills

- **[`ml-feature-evaluator`](https://github.com/wan-huiyan/ml-feature-evaluator)**: When the question is "should we add feature X?" rather than "can we extend the training window?" — covers the 10-step diagnostic for go/no-go feature decisions.

## License

MIT
