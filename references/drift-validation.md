# Drift Validation Code & Implementation

## Population Stability Index (PSI)

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
| > 0.25 | Significant shift | Investigate before extending |

## Label Distribution Stability Query

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
If correlated with a known seasonal pattern, it may be expected. If not, halt extension
for that output until the cause is identified.

## Feature-Outcome Relationship Stability

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

If a feature's univariate AUC flips direction (e.g., 0.62 → 0.45), that feature's
relationship with the outcome has fundamentally changed — concept drift at the feature level.

## Purged Walk-Forward CV

### Why Purging and Embargo Matter

Each training sample has two timestamps:
- **pred_time**: when features are observed (the target_date / prediction date)
- **eval_time**: when the outcome becomes known (target_date + lookforward)

Without purging, training samples whose eval_time overlaps with the test fold's pred_time
leak future information into the model.

### Purge Condition

Remove any training sample where:
```
train_eval_time >= test_pred_time_start
```

### Embargo Period

After purging, add a temporal buffer (embargo = max lookforward across all outputs):
```
embargo_threshold = max(test_eval_times) + embargo_duration
```

### Implementation

```python
from timeseriescv.cross_validation import PurgedWalkForwardCV

cv = PurgedWalkForwardCV(n_splits=5, n_test_splits=1)

for train_idx, test_idx in cv.split(X, pred_times=pred_times, eval_times=eval_times):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

Or manually:

```python
def purged_temporal_split(df, test_start, test_end, lookforward_days, embargo_days=0):
    """Split with purging and embargo for temporal models."""
    test = df[(df['target_date'] >= test_start) & (df['target_date'] < test_end)]

    test_pred_start = test['target_date'].min()
    purge_mask = df['eval_time'] < test_pred_start

    test_eval_end = test['eval_time'].max()
    embargo_end = test_eval_end + pd.Timedelta(days=embargo_days)

    train_before = df[purge_mask & (df['target_date'] < test_start)]
    train_after = df[df['target_date'] > embargo_end]
    train = pd.concat([train_before, train_after])

    return train, test
```

### Comparing Current vs Extended Window

Run purged walk-forward CV twice:
1. **Current window only** — baseline per-output AUC
2. **Extended window** — measure impact of older data

If extended window shows AUC degradation > 0.01 on recent test folds, apply bang-bang heuristic.

## Post-Extension Monitoring

### Batch Monitoring (recommended for periodic retraining)

Run PSI checks monthly between the training distribution and recent inference data:
- If PSI > 0.25 for >20% of features → trigger retraining
- If positive rate shifts > 5pp from training → investigate

### Streaming Monitoring (for real-time systems)

```python
from river.drift import ADWIN

adwin = ADWIN(delta=0.002)  # Lower delta = more sensitive

for prediction_error in error_stream:
    in_drift, _ = adwin.update(prediction_error)
    if in_drift:
        print("Drift detected — consider retraining or window adjustment")
```

### Temporal Safety in Feature Engineering

Every feature computation must be expressible as a function of data available
at `target_date - 1 day`. If extending the window introduces features computed from
data that wasn't available at prediction time, you have leakage.

## References

| Source | What it contributes | Stars |
|--------|-------------------|-------|
| [timeseriescv](https://github.com/sam31415/timeseriescv) | Purged walk-forward CV + combinatorial purged k-fold with embargo | 284 |
| [River](https://github.com/online-ml/river) | ADWIN adaptive windowing for streaming drift detection | 5,752 |
| [Temporian](https://github.com/google/temporian) | Temporal safety / leakage prevention in feature engineering | 710 |
| [MLFinLab](https://github.com/hudson-and-thames/mlfinlab) | De Prado's CPCV, triple barrier labeling | 4,613 |
| [Frouros](https://github.com/IFCA-Advanced-Computing/frouros) | 28 drift detection algorithms (PSI, KS, Chi-squared) | 252 |
| [arXiv:2512.12816](https://arxiv.org/abs/2512.12816) | Bang-bang optimality for training window under concept drift | — |
| Chen & Guestrin (2016) | XGBoost sparsity-aware split finding (Algorithm 2) | — |
| De Prado (2018) | Purged CV theory, temporal leakage in finance ML | — |
