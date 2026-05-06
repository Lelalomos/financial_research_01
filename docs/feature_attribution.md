# Feature Attribution Utilities

Phase 3 Task 3.2 adds optional Captum-based feature attribution utilities in
`src/evaluation/feature_attribution.py`.

## Optional Dependency

Captum is not required for normal training, prediction, or testing. Attribution
helpers import Captum only when attribution is requested. If Captum is missing,
the utility raises a clear `AttributionDependencyError`.

Install Captum only when you want to run attribution:

```bash
pip install captum
```

## Supported Methods

- `integrated_gradients`
- `feature_ablation`

The helpers attribute only the numeric `features` tensor. Categorical inputs
such as `stock_id`, `group_id`, `day`, `month`, and `dividend_flag` are passed
as fixed model inputs.

## Usage

```python
from src.evaluation.feature_attribution import compute_feature_attribution_report

report = compute_feature_attribution_report(
    model=model,
    data=validation_loader,
    feature_names=feature_cols,
    method="integrated_gradients",
    baseline="zero",
    max_batches=1,
    max_samples=128,
)

raw = report["raw_attributions"]
importance = report["feature_importance"]
metadata = report["metadata"]
```

`raw_attributions` has shape:

```text
(sample_count, sequence_length, num_features)
```

`feature_importance` ranks features by mean absolute attribution by default.

## Interpretation Guidance

Attribution reports are diagnostic, not causal proof. Financial features can be
noisy and unstable, so compare attribution reports across validation windows,
walk-forward folds, and market regimes before acting on them.

Reports include metadata such as method, baseline, sample count, sequence
length, and whether absolute attribution was used. Task 3.2 does not remove
features or mutate config files.

Task 3.3 adds report-only pruning recommendations in
`src/evaluation/feature_pruning.py`. Use those reports as manual review inputs;
do not remove features until candidates are stable across validation windows,
walk-forward folds, and market regimes.
