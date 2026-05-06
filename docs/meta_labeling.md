# Meta-Labeling Utilities

Meta-labeling is implemented as an offline analysis tool in
`src/evaluation/meta_labeling.py`.

## Purpose

The primary model predicts return or direction. Meta-labeling converts those
predictions into a binary target:

- `1`: predicted direction matches realized target direction
- `0`: predicted direction does not match realized target direction

This dataset can later be used to train a second-stage trade-filtering model,
but Task 2.3 does not train that production model.

## Leakage Controls

Production-quality meta-labels should come from out-of-sample predictions. Use
walk-forward, out-of-fold, or purged/embargoed prediction workflows so each row
is predicted by a model that did not train on that same row.

The utility records prediction provenance through `prediction_source`:

- `out_of_sample`
- `out_of_fold`
- `walk_forward`
- `purged_cv`
- `in_sample`

Using `prediction_source="in_sample"` emits a warning. Set
`require_out_of_sample=True` to reject in-sample labels.

When supplied, prediction and target `date` / `tic` arrays are checked row by
row. Mismatches fail early to prevent labels from being assigned to the wrong
sample.

## Example

```python
from src.evaluation.meta_labeling import create_meta_labels

meta_df = create_meta_labels(
    predictions=predictions,
    targets=targets,
    prediction_dates=prediction_dates,
    target_dates=target_dates,
    prediction_tickers=prediction_tickers,
    target_tickers=target_tickers,
    features=feature_df,
    confidence_threshold=0.1,
    prediction_source="walk_forward",
    require_out_of_sample=True,
)
```

`confidence_threshold` filters rows by `abs(prediction)`. Zero has direction
`0`, so it only matches another zero.
