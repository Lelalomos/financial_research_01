"""
Unit tests for meta-labeling utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.meta_labeling import (
    create_meta_labels,
    prepare_meta_label_dataset,
    validate_prediction_target_alignment,
)


def test_create_meta_labels_direction_correctness():
    labels = create_meta_labels(
        predictions=[0.2, -0.1, 0.0, 0.3],
        targets=[0.5, 0.2, 0.0, -0.1],
        prediction_source="out_of_sample",
    )

    assert labels["meta_label"].tolist() == [1, 0, 1, 0]
    assert labels["prediction_direction"].tolist() == [1, -1, 0, 1]
    assert labels["target_direction"].tolist() == [1, 1, 0, -1]
    assert labels["is_out_of_sample"].all()


def test_create_meta_labels_rejects_misaligned_dates():
    with pytest.raises(ValueError, match="dates are not aligned"):
        create_meta_labels(
            predictions=[0.1, 0.2],
            targets=[0.1, 0.2],
            prediction_dates=["2024-01-01", "2024-01-02"],
            target_dates=["2024-01-01", "2024-01-03"],
        )


def test_create_meta_labels_rejects_misaligned_tickers():
    with pytest.raises(ValueError, match="tickers are not aligned"):
        create_meta_labels(
            predictions=[0.1, 0.2],
            targets=[0.1, 0.2],
            prediction_tickers=["AAPL", "MSFT"],
            target_tickers=["AAPL", "GOOGL"],
        )


def test_create_meta_labels_confidence_threshold_filtering():
    labels = create_meta_labels(
        predictions=[0.01, -0.2, 0.5],
        targets=[0.1, -0.1, -0.1],
        confidence_threshold=0.2,
        prediction_source="walk_forward",
    )

    assert len(labels) == 2
    np.testing.assert_allclose(labels["prediction"].to_numpy(), np.array([-0.2, 0.5]))
    assert labels["prediction_source"].unique().tolist() == ["walk_forward"]


def test_create_meta_labels_adds_identifiers_and_features():
    features = pd.DataFrame({
        "rsi_14": [0.4, 0.6],
        "vix": [15.0, 20.0],
    })

    labels = prepare_meta_label_dataset(
        predictions=[0.1, -0.1],
        targets=[0.2, -0.3],
        prediction_dates=["2024-01-01", "2024-01-02"],
        target_dates=["2024-01-01", "2024-01-02"],
        prediction_tickers=["AAPL", "MSFT"],
        target_tickers=["AAPL", "MSFT"],
        features=features,
        prediction_source="purged_cv",
    )

    assert labels.columns[:2].tolist() == ["date", "tic"]
    assert labels["tic"].tolist() == ["AAPL", "MSFT"]
    assert labels["rsi_14"].tolist() == [0.4, 0.6]
    assert labels["meta_label"].tolist() == [1, 1]


def test_create_meta_labels_warns_for_in_sample_predictions():
    with pytest.warns(UserWarning, match="In-sample meta-labels"):
        labels = create_meta_labels(
            predictions=[0.1],
            targets=[0.2],
            prediction_source="in_sample",
        )

    assert labels["is_out_of_sample"].tolist() == [False]


def test_create_meta_labels_rejects_in_sample_when_required():
    with pytest.raises(ValueError, match="in_sample predictions are not allowed"):
        create_meta_labels(
            predictions=[0.1],
            targets=[0.2],
            prediction_source="in_sample",
            require_out_of_sample=True,
        )


def test_validate_prediction_target_alignment_requires_pairs():
    with pytest.raises(ValueError, match="provided together"):
        validate_prediction_target_alignment(prediction_dates=["2024-01-01"])


def test_create_meta_labels_rejects_reserved_feature_columns():
    features = pd.DataFrame({"prediction": [1.0]})

    with pytest.raises(ValueError, match="reserved"):
        create_meta_labels(
            predictions=[0.1],
            targets=[0.2],
            features=features,
        )
