"""
Unit tests for optional feature attribution utilities.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation import feature_attribution
from src.evaluation.feature_attribution import (
    AttributionDependencyError,
    aggregate_feature_attributions,
    attribute_batch,
    compute_feature_attribution_report,
)


class TinyFinancialModel(torch.nn.Module):
    def __init__(self, num_features=3):
        super().__init__()
        self.fc = torch.nn.Linear(num_features, 1)

    def forward(self, features, stock_id, group_id, day, month, dividend_flag):
        return self.fc(features.mean(dim=1))


def _batch(batch_size=4, seq_len=5, num_features=3):
    return {
        "features": torch.randn(batch_size, seq_len, num_features),
        "stock_id": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "group_id": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "day": torch.ones(batch_size, seq_len, dtype=torch.long),
        "month": torch.ones(batch_size, seq_len, dtype=torch.long),
        "dividend_flag": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


def test_missing_captum_dependency_has_clear_error(monkeypatch):
    def raise_missing(method):
        raise AttributionDependencyError("Captum is required")

    monkeypatch.setattr(feature_attribution, "_require_captum", raise_missing)

    with pytest.raises(AttributionDependencyError, match="Captum is required"):
        attribute_batch(TinyFinancialModel(), _batch())


def test_aggregate_feature_attributions_schema_and_sorting():
    attributions = np.array([
        [[1.0, -3.0, 0.5], [2.0, -1.0, 0.5]],
        [[1.0, -2.0, 0.5], [0.0, -4.0, 0.5]],
    ])

    report = aggregate_feature_attributions(
        attributions,
        feature_names=["open", "close", "volume"],
    )

    assert report.columns.tolist() == [
        "feature",
        "importance",
        "mean_attribution",
        "mean_abs_attribution",
    ]
    assert report.iloc[0]["feature"] == "close"
    assert report["importance"].is_monotonic_decreasing


def test_aggregate_feature_attributions_rejects_bad_feature_names():
    with pytest.raises(ValueError, match="feature_names length"):
        aggregate_feature_attributions(np.ones((2, 3, 4)), feature_names=["only_one"])


def test_compute_feature_attribution_report_uses_sampling_and_metadata(monkeypatch):
    def fake_attribute(model, batch, method, baseline, target, n_steps, perturbations_per_eval, device):
        features = batch["features"]
        values = torch.zeros_like(features)
        values[:, :, 0] = 2.0
        values[:, :, 1] = -1.0
        values[:, :, 2] = 0.5
        return values.detach().cpu().numpy()

    monkeypatch.setattr(feature_attribution, "attribute_batch", fake_attribute)

    batches = [_batch(batch_size=3), _batch(batch_size=3)]
    report = compute_feature_attribution_report(
        model=TinyFinancialModel(),
        data=batches,
        feature_names=["open", "close", "volume"],
        method="integrated_gradients",
        baseline="zero",
        max_batches=2,
        max_samples=4,
        n_steps=8,
    )

    assert report["raw_attributions"].shape == (4, 5, 3)
    assert report["feature_importance"].iloc[0]["feature"] == "open"
    assert report["metadata"]["sample_count"] == 4
    assert report["metadata"]["method"] == "integrated_gradients"
    assert report["metadata"]["n_steps"] == 8
    assert "diagnostic" in report["metadata"]["interpretation_note"]


def test_compute_feature_attribution_report_accepts_single_batch(monkeypatch):
    def fake_attribute(model, batch, method, baseline, target, n_steps, perturbations_per_eval, device):
        return np.ones_like(batch["features"].detach().cpu().numpy())

    monkeypatch.setattr(feature_attribution, "attribute_batch", fake_attribute)

    report = compute_feature_attribution_report(
        model=TinyFinancialModel(),
        data=_batch(batch_size=2),
        feature_names=["a", "b", "c"],
        method="feature_ablation",
        perturbations_per_eval=2,
    )

    assert isinstance(report["feature_importance"], pd.DataFrame)
    assert report["metadata"]["perturbations_per_eval"] == 2


def test_compute_feature_attribution_report_validates_limits():
    with pytest.raises(ValueError, match="max_batches"):
        compute_feature_attribution_report(TinyFinancialModel(), [_batch()], max_batches=0)

    with pytest.raises(ValueError, match="max_samples"):
        compute_feature_attribution_report(TinyFinancialModel(), [_batch()], max_samples=0)
