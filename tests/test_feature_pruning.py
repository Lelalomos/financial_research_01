"""
Unit tests for report-only feature pruning utilities.
"""

import pandas as pd
import pytest

from src.evaluation.feature_pruning import create_feature_pruning_report


def _importance_report():
    return pd.DataFrame({
        "feature": ["close", "volume", "rsi", "vix", "ema_50"],
        "importance": [0.9, 0.4, 0.1, 0.05, 0.01],
        "mean_attribution": [0.9, 0.4, 0.1, 0.05, 0.01],
    })


def test_feature_pruning_report_selects_bottom_percentile_candidates():
    report = create_feature_pruning_report(
        _importance_report(),
        bottom_percent=0.4,
        data_split="validation",
    )

    assert report["pruning_candidates"] == ["vix", "ema_50"]
    assert report["retained_features"] == ["close", "volume", "rsi"]
    assert report["metadata"]["candidate_count"] == 2
    assert report["metadata"]["data_split"] == "validation"
    assert report["feature_ranking"]["importance"].is_monotonic_decreasing
    assert report["feature_ranking"].loc[3, "prune_reason"] == "bottom_percentile"


def test_feature_pruning_report_can_add_min_importance_threshold():
    report = create_feature_pruning_report(
        _importance_report(),
        bottom_percent=0.0,
        min_importance=0.1,
        data_split="walk_forward_fold_1",
    )

    assert report["pruning_candidates"] == ["rsi", "vix", "ema_50"]
    assert report["metadata"]["bottom_count"] == 0
    assert "importance_below_or_equal_0.1" in report["feature_ranking"].loc[2, "prune_reason"]


def test_feature_pruning_report_rejects_test_data_by_default():
    with pytest.raises(ValueError, match="test/holdout"):
        create_feature_pruning_report(_importance_report(), data_split="test")


def test_feature_pruning_report_allows_explicit_test_data_with_warning():
    report = create_feature_pruning_report(
        _importance_report(),
        data_split="holdout",
        allow_test_data=True,
    )

    assert report["metadata"]["allow_test_data"] is True
    assert any("test/holdout data" in warning for warning in report["metadata"]["warnings"])


def test_feature_pruning_report_validates_schema_and_parameters():
    with pytest.raises(ValueError, match="required columns"):
        create_feature_pruning_report(pd.DataFrame({"feature": ["a"]}))

    with pytest.raises(ValueError, match="bottom_percent"):
        create_feature_pruning_report(_importance_report(), bottom_percent=1.0)

    with pytest.raises(ValueError, match="duplicate"):
        create_feature_pruning_report(pd.DataFrame({
            "feature": ["a", "a"],
            "importance": [0.1, 0.2],
        }))


def test_feature_pruning_report_does_not_mutate_inputs():
    feature_importance = _importance_report()
    original = feature_importance.copy(deep=True)

    report = create_feature_pruning_report(feature_importance, bottom_percent=0.4)

    pd.testing.assert_frame_equal(feature_importance, original)
    assert report["config_patch_suggestion"]["remove_features"] == ["vix", "ema_50"]
    assert "does not edit config files" in report["config_patch_suggestion"]["note"]
