"""
Unit tests for ensemble prediction utilities.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.prediction.ensemble import EnsembleCompatibilityError, EnsemblePredictor


class _FakePreparator:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols


class _FakePredictor:
    def __init__(
        self,
        model_path,
        prediction,
        feature_cols=None,
        num_features=2,
        normalize_target=True,
        target_threshold=10.0,
    ):
        self.model_path = Path(model_path)
        self.prediction = np.asarray(prediction, dtype=np.float32)
        self.device = "cpu"
        self.model = object()
        self.preparator = _FakePreparator(feature_cols or ["open", "close"])
        self.model_metadata = {
            "metadata": {"model_type": "fake"},
            "num_features": num_features,
            "num_stocks": 3,
            "num_groups": 2,
            "feature_cols": feature_cols or ["open", "close"],
            "target_normalization": {
                "NORMALIZE_TARGET": normalize_target,
                "TARGET_THRESHOLD": target_threshold,
            },
            "normalize_target": None,
            "target_threshold": None,
        }

    def predict(self, data, return_raw=False):
        return self.prediction

    def get_model_info(self):
        return {
            "model_path": str(self.model_path),
            "model_type": "fake",
            "num_features": self.model_metadata["num_features"],
            "num_stocks": self.model_metadata["num_stocks"],
            "num_groups": self.model_metadata["num_groups"],
            "device": "cpu",
            "training_epochs": None,
            "best_val_loss": None,
            "feature_cols": self.model_metadata["feature_cols"],
        }


def _fake_predictor_factory(**overrides):
    def factory(model_path, **kwargs):
        if str(model_path).endswith("one.pt"):
            return _FakePredictor(model_path, [[1.0], [3.0]], **overrides)
        return _FakePredictor(model_path, [[5.0], [7.0]], **overrides)

    return factory


def test_ensemble_predictor_weighted_average():
    with patch("src.prediction.ensemble.Predictor", side_effect=_fake_predictor_factory()):
        predictor = EnsemblePredictor(
            model_paths=["one.pt", "two.pt"],
            weights=[0.25, 0.75],
            device="cpu",
        )

    predictions = predictor.predict({"features": np.zeros((2, 1, 2), dtype=np.float32)})

    np.testing.assert_allclose(predictions, np.array([[4.0], [6.0]], dtype=np.float32))


def test_ensemble_rejects_mismatched_feature_columns():
    def factory(model_path, **kwargs):
        if str(model_path).endswith("one.pt"):
            return _FakePredictor(model_path, [[1.0]], feature_cols=["open", "close"])
        return _FakePredictor(model_path, [[2.0]], feature_cols=["close", "open"])

    with patch("src.prediction.ensemble.Predictor", side_effect=factory):
        with pytest.raises(EnsembleCompatibilityError, match="feature_cols"):
            EnsemblePredictor(model_paths=["one.pt", "two.pt"], device="cpu")


def test_ensemble_rejects_mismatched_num_features():
    def factory(model_path, **kwargs):
        if str(model_path).endswith("one.pt"):
            return _FakePredictor(model_path, [[1.0]], num_features=2)
        return _FakePredictor(model_path, [[2.0]], num_features=3)

    with patch("src.prediction.ensemble.Predictor", side_effect=factory):
        with pytest.raises(EnsembleCompatibilityError, match="num_features"):
            EnsemblePredictor(model_paths=["one.pt", "two.pt"], device="cpu")


def test_ensemble_rejects_mismatched_target_normalization():
    def factory(model_path, **kwargs):
        if str(model_path).endswith("one.pt"):
            return _FakePredictor(model_path, [[1.0]], target_threshold=10.0)
        return _FakePredictor(model_path, [[2.0]], target_threshold=20.0)

    with patch("src.prediction.ensemble.Predictor", side_effect=factory):
        with pytest.raises(EnsembleCompatibilityError, match="target normalization"):
            EnsemblePredictor(model_paths=["one.pt", "two.pt"], device="cpu")


def test_ensemble_rejects_prediction_shape_mismatch():
    def factory(model_path, **kwargs):
        if str(model_path).endswith("one.pt"):
            return _FakePredictor(model_path, [[1.0], [2.0]])
        return _FakePredictor(model_path, [[3.0]])

    with patch("src.prediction.ensemble.Predictor", side_effect=factory):
        predictor = EnsemblePredictor(model_paths=["one.pt", "two.pt"], device="cpu")

    with pytest.raises(EnsembleCompatibilityError, match="different shapes"):
        predictor.predict({"features": np.zeros((2, 1, 2), dtype=np.float32)})
