# Implementation Pitfalls for Training Window Extension

## `.fillna(0)` Destroys XGBoost's Missing Value Handling

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

## Sentinel Values Interact with Missing Value Handling

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

## Preprocessing Parity Across Pipeline Files

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

## Standard CV on Temporal Data Causes Silent Leakage

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
