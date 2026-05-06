# Feature Pruning Reports

Phase 3 Task 3.3 adds report-only feature pruning utilities in
`src/evaluation/feature_pruning.py`.

## Purpose

Feature pruning reports consume feature attribution importance scores and mark
low-importance features as removal candidates. The utility does not edit
configuration files, feature lists, checkpoints, or preprocessing artifacts.

## Usage

```python
from src.evaluation.feature_attribution import compute_feature_attribution_report
from src.evaluation.feature_pruning import create_feature_pruning_report

attribution_report = compute_feature_attribution_report(
    model=model,
    data=validation_loader,
    feature_names=feature_cols,
    method="integrated_gradients",
    max_batches=1,
    max_samples=128,
)

pruning_report = create_feature_pruning_report(
    feature_importance=attribution_report["feature_importance"],
    bottom_percent=0.3,
    data_split="validation",
)
```

The report includes:

- `feature_ranking`: ranked importance table with pruning flags and reasons
- `pruning_candidates`: exact candidate feature names
- `retained_features`: features not marked for removal
- `config_patch_suggestion`: manual patch guidance only
- `metadata`: thresholds, split provenance, warnings, and interpretation notes

## Leakage Controls

Generate pruning reports from validation batches, walk-forward folds, or
out-of-fold analyses. The utility rejects `test`, `holdout`, and `production`
split names by default because using final evaluation data for pruning decisions
can leak model-selection information.

Set `allow_test_data=True` only for diagnostic investigation. Do not use such a
report to choose features for a final model.

## Interpretation Guidance

Low attribution importance is diagnostic, not causal proof that a feature is
useless. Financial feature importance can vary by time window and market regime.
Compare candidates across validation windows, walk-forward folds, and regimes,
then retrain and validate out of sample before removing features.
