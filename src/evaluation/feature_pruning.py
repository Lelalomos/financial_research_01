"""
Feature pruning report utilities.

These helpers are intentionally report-only. They identify low-importance
feature candidates from attribution reports but never mutate config files or
feature lists.
"""

from math import ceil
from typing import Dict, List, Optional

import pandas as pd


REQUIRED_IMPORTANCE_COLUMNS = {"feature", "importance"}
TEST_SPLIT_NAMES = {"test", "holdout", "production"}


def _validate_feature_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_importance, pd.DataFrame):
        raise TypeError("feature_importance must be a pandas DataFrame")

    missing_columns = REQUIRED_IMPORTANCE_COLUMNS - set(feature_importance.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"feature_importance is missing required columns: {missing}")

    if feature_importance["feature"].isna().any():
        raise ValueError("feature_importance contains missing feature names")
    if feature_importance["feature"].duplicated().any():
        raise ValueError("feature_importance contains duplicate feature names")
    if feature_importance["importance"].isna().any():
        raise ValueError("feature_importance contains missing importance values")

    validated = feature_importance.copy(deep=True)
    validated["importance"] = pd.to_numeric(validated["importance"], errors="raise")
    return validated


def _validate_split(data_split: Optional[str], allow_test_data: bool) -> str:
    split = "unknown" if data_split is None else str(data_split).strip().lower()
    if split in TEST_SPLIT_NAMES and not allow_test_data:
        raise ValueError(
            "Feature pruning reports should not be generated from test/holdout "
            "data unless allow_test_data=True. Use validation or walk-forward "
            "folds to avoid evaluation leakage."
        )
    return split


def create_feature_pruning_report(
    feature_importance: pd.DataFrame,
    bottom_percent: float = 0.3,
    min_importance: Optional[float] = None,
    data_split: Optional[str] = "validation",
    allow_test_data: bool = False,
) -> Dict[str, object]:
    """
    Create a report-only feature pruning recommendation.

    Args:
        feature_importance: DataFrame with at least `feature` and `importance`
            columns, such as the output from `aggregate_feature_attributions()`.
        bottom_percent: Fraction of lowest-ranked features to mark as pruning
            candidates. Set to 0 to disable percentile selection.
        min_importance: Optional absolute threshold. Features with importance
            less than or equal to this value are also marked as candidates.
        data_split: Name of the data split used to produce importance scores.
            Test/holdout splits are rejected by default to reduce leakage risk.
        allow_test_data: Explicit opt-in for test/holdout reports.

    Returns:
        Dict containing ranked features, candidate lists, a config patch
        suggestion, and metadata. No inputs or project files are mutated.
    """
    if bottom_percent < 0 or bottom_percent >= 1:
        raise ValueError("bottom_percent must be in the range [0, 1)")
    if min_importance is not None and min_importance < 0:
        raise ValueError("min_importance must be non-negative when provided")

    split = _validate_split(data_split, allow_test_data)
    ranking = _validate_feature_importance(feature_importance)
    ranking = ranking.sort_values("importance", ascending=False).reset_index(drop=True)

    total_features = len(ranking)
    bottom_count = ceil(total_features * bottom_percent) if bottom_percent > 0 else 0
    bottom_features = set()
    if bottom_count > 0:
        bottom_features = set(ranking.tail(bottom_count)["feature"].tolist())

    threshold_features = set()
    if min_importance is not None:
        threshold_features = set(
            ranking.loc[ranking["importance"] <= min_importance, "feature"].tolist()
        )

    candidate_features = bottom_features | threshold_features
    ranking["rank"] = ranking.index + 1
    ranking["prune_candidate"] = ranking["feature"].isin(candidate_features)
    ranking["prune_reason"] = ranking["feature"].map(
        lambda feature: _candidate_reason(
            feature=feature,
            bottom_features=bottom_features,
            threshold_features=threshold_features,
            min_importance=min_importance,
        )
    )

    pruning_candidates = ranking.loc[ranking["prune_candidate"], "feature"].tolist()
    retained_features = ranking.loc[~ranking["prune_candidate"], "feature"].tolist()
    warnings = [
        "Report is advisory and must be validated by retraining before removing features.",
        "Compare pruning candidates across validation windows, walk-forward folds, and regimes.",
    ]
    if split == "unknown":
        warnings.append("Data split was not provided; confirm this report was not generated from test data.")
    if allow_test_data and split in TEST_SPLIT_NAMES:
        warnings.append("Report was generated from test/holdout data; do not use it for model selection.")

    return {
        "feature_ranking": ranking,
        "pruning_candidates": pruning_candidates,
        "retained_features": retained_features,
        "config_patch_suggestion": {
            "remove_features": pruning_candidates,
            "retain_features": retained_features,
            "note": (
                "Apply manually only after retraining and out-of-sample validation. "
                "This utility does not edit config files."
            ),
        },
        "metadata": {
            "total_features": total_features,
            "candidate_count": len(pruning_candidates),
            "bottom_percent": bottom_percent,
            "bottom_count": bottom_count,
            "min_importance": min_importance,
            "data_split": split,
            "allow_test_data": allow_test_data,
            "warnings": warnings,
            "interpretation_note": (
                "Low attribution importance is diagnostic, not causal proof that "
                "a feature is useless."
            ),
        },
    }


def _candidate_reason(
    feature: str,
    bottom_features: set,
    threshold_features: set,
    min_importance: Optional[float],
) -> str:
    reasons: List[str] = []
    if feature in bottom_features:
        reasons.append("bottom_percentile")
    if feature in threshold_features:
        reasons.append(f"importance_below_or_equal_{min_importance}")
    return ",".join(reasons)
