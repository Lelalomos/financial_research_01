"""
Meta-labeling utilities for offline analysis.

Meta-labels should be generated from out-of-sample predictions, ideally from
walk-forward or purged out-of-fold evaluation. In-sample labels are allowed for
experimentation, but they are explicitly marked because they can leak training
fit quality into a second-stage model.
"""

from typing import Iterable, Optional
import warnings

import numpy as np
import pandas as pd


VALID_PREDICTION_SOURCES = {"out_of_sample", "out_of_fold", "walk_forward", "purged_cv", "in_sample"}


def _as_1d_array(values: Iterable, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim > 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _optional_1d_array(values: Optional[Iterable], name: str) -> Optional[np.ndarray]:
    if values is None:
        return None
    return _as_1d_array(values, name)


def _validate_lengths(reference_length: int, **arrays: Optional[np.ndarray]) -> None:
    for name, array in arrays.items():
        if array is not None and len(array) != reference_length:
            raise ValueError(f"{name} length {len(array)} does not match predictions length {reference_length}")


def validate_prediction_target_alignment(
    prediction_dates: Optional[Iterable] = None,
    target_dates: Optional[Iterable] = None,
    prediction_tickers: Optional[Iterable] = None,
    target_tickers: Optional[Iterable] = None,
) -> None:
    """
    Validate that prediction and target identifiers are aligned row-by-row.

    Args:
        prediction_dates: Dates attached to predictions.
        target_dates: Dates attached to targets.
        prediction_tickers: Tickers attached to predictions.
        target_tickers: Tickers attached to targets.
    """
    pred_dates = _optional_1d_array(prediction_dates, "prediction_dates")
    targ_dates = _optional_1d_array(target_dates, "target_dates")
    pred_tickers = _optional_1d_array(prediction_tickers, "prediction_tickers")
    targ_tickers = _optional_1d_array(target_tickers, "target_tickers")

    if (pred_dates is None) != (targ_dates is None):
        raise ValueError("prediction_dates and target_dates must be provided together")
    if (pred_tickers is None) != (targ_tickers is None):
        raise ValueError("prediction_tickers and target_tickers must be provided together")

    if pred_dates is not None:
        _validate_lengths(
            len(pred_dates),
            target_dates=targ_dates,
            prediction_tickers=pred_tickers,
            target_tickers=targ_tickers,
        )
        if not np.array_equal(pd.to_datetime(pred_dates).to_numpy(), pd.to_datetime(targ_dates).to_numpy()):
            raise ValueError("prediction_dates and target_dates are not aligned")

    if pred_tickers is not None:
        _validate_lengths(
            len(pred_tickers),
            target_tickers=targ_tickers,
            prediction_dates=pred_dates,
            target_dates=targ_dates,
        )
        if not np.array_equal(pred_tickers.astype(str), targ_tickers.astype(str)):
            raise ValueError("prediction_tickers and target_tickers are not aligned")


def create_meta_labels(
    predictions: Iterable[float],
    targets: Iterable[float],
    prediction_dates: Optional[Iterable] = None,
    target_dates: Optional[Iterable] = None,
    prediction_tickers: Optional[Iterable] = None,
    target_tickers: Optional[Iterable] = None,
    features: Optional[pd.DataFrame] = None,
    confidence_threshold: Optional[float] = None,
    prediction_source: str = "out_of_sample",
    require_out_of_sample: bool = False,
) -> pd.DataFrame:
    """
    Create binary labels for whether primary predictions got direction correct.

    Direction is calculated with ``np.sign``. Zero has direction 0 and only
    matches another zero. Confidence is ``abs(prediction)``.

    Args:
        predictions: Primary model predictions.
        targets: Realized targets aligned to predictions.
        prediction_dates: Optional prediction date identifiers.
        target_dates: Optional target date identifiers used for alignment checks.
        prediction_tickers: Optional prediction ticker identifiers.
        target_tickers: Optional target ticker identifiers used for alignment checks.
        features: Optional tabular features for a second-stage model.
        confidence_threshold: Keep only rows with abs(prediction) >= threshold.
        prediction_source: Provenance of predictions. Prefer out-of-sample,
            out-of-fold, walk-forward, or purged_cv for production analysis.
        require_out_of_sample: If True, reject in_sample predictions.

    Returns:
        DataFrame with prediction, target, directions, confidence, meta_label,
        and optional identifiers/features.
    """
    if prediction_source not in VALID_PREDICTION_SOURCES:
        raise ValueError(
            f"prediction_source must be one of {sorted(VALID_PREDICTION_SOURCES)}"
        )
    if require_out_of_sample and prediction_source == "in_sample":
        raise ValueError("in_sample predictions are not allowed when require_out_of_sample=True")
    if prediction_source == "in_sample":
        warnings.warn(
            "In-sample meta-labels can leak primary-model fit quality. Use walk-forward, "
            "purged_cv, or out_of_fold predictions for production-quality meta-labeling.",
            UserWarning,
            stacklevel=2,
        )
    if confidence_threshold is not None and confidence_threshold < 0:
        raise ValueError("confidence_threshold must be non-negative")

    predictions_array = _as_1d_array(predictions, "predictions").astype(float)
    targets_array = _as_1d_array(targets, "targets").astype(float)
    _validate_lengths(len(predictions_array), targets=targets_array)

    pred_dates = _optional_1d_array(prediction_dates, "prediction_dates")
    targ_dates = _optional_1d_array(target_dates, "target_dates")
    pred_tickers = _optional_1d_array(prediction_tickers, "prediction_tickers")
    targ_tickers = _optional_1d_array(target_tickers, "target_tickers")
    _validate_lengths(
        len(predictions_array),
        prediction_dates=pred_dates,
        target_dates=targ_dates,
        prediction_tickers=pred_tickers,
        target_tickers=targ_tickers,
    )

    validate_prediction_target_alignment(
        prediction_dates=pred_dates,
        target_dates=targ_dates,
        prediction_tickers=pred_tickers,
        target_tickers=targ_tickers,
    )

    pred_direction = np.sign(predictions_array).astype(np.int8)
    target_direction = np.sign(targets_array).astype(np.int8)
    confidence = np.abs(predictions_array)

    result = pd.DataFrame({
        "prediction": predictions_array,
        "target": targets_array,
        "prediction_direction": pred_direction,
        "target_direction": target_direction,
        "confidence": confidence,
        "meta_label": (pred_direction == target_direction).astype(np.int8),
        "prediction_source": prediction_source,
        "is_out_of_sample": prediction_source != "in_sample",
    })

    if pred_dates is not None:
        result.insert(0, "date", pd.to_datetime(pred_dates))
    if pred_tickers is not None:
        insert_at = 1 if "date" in result.columns else 0
        result.insert(insert_at, "tic", pred_tickers.astype(str))

    if features is not None:
        if len(features) != len(result):
            raise ValueError(f"features length {len(features)} does not match predictions length {len(result)}")
        reserved = set(result.columns)
        conflicts = reserved.intersection(features.columns)
        if conflicts:
            raise ValueError(f"features contain reserved meta-label columns: {sorted(conflicts)}")
        result = pd.concat([result.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    if confidence_threshold is not None:
        result = result[result["confidence"] >= confidence_threshold].reset_index(drop=True)

    return result


def prepare_meta_label_dataset(*args, **kwargs) -> pd.DataFrame:
    """Alias for create_meta_labels for callers building second-stage datasets."""
    return create_meta_labels(*args, **kwargs)
