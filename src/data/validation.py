"""
Dataset validation for Multi-Model Financial Forecasting.

This module provides validation functions to verify that datasets contain
required columns and warns about missing optional columns.
"""

import pandas as pd
from typing import Dict, List, Optional, Set
from pathlib import Path

from src.config import load_config
from src.utils.logger import get_logger


class DatasetValidator:
    """
    Validator for dataset columns.

    Checks that required columns are present and warns about missing optional columns.
    """

    def __init__(self, config=None):
        """
        Initialize dataset validator.

        Args:
            config: Configuration object (defaults to load_config('main') if None)
        """
        if config is None:
            config = load_config('main')
        self.config = config
        self.logger = get_logger("dataset_validation", log_dir="logs")

        # Flatten required columns from categories
        self.required_columns = set()
        if hasattr(config.data, 'validation') and hasattr(config.data.validation, 'REQUIRED_COLUMNS'):
            for category, columns in config.data.validation.REQUIRED_COLUMNS._data.items():
                self.required_columns.update(columns)

        # Flatten optional columns from categories
        self.optional_columns = {}
        if hasattr(config.data, 'validation') and hasattr(config.data.validation, 'OPTIONAL_COLUMNS'):
            for category, columns in config.data.validation.OPTIONAL_COLUMNS._data.items():
                self.optional_columns[category] = set(columns)

    def validate_columns(
        self,
        df: pd.DataFrame,
        df_name: str = "dataset",
        raise_on_missing: bool = True,
        warn_on_missing_optional: bool = None
    ) -> Dict[str, any]:
        """
        Validate that DataFrame contains required columns.

        Args:
            df: DataFrame to validate
            df_name: Name of the dataset (for logging)
            raise_on_missing: Raise exception if required columns missing
            warn_on_missing_optional: Warn about missing optional columns
                                     (default from config if None)

        Returns:
            Dictionary with validation results:
            - valid: bool - True if all required columns present
            - missing_required: list - Missing required column names
            - missing_optional: dict - Missing optional columns by category
            - present_optional: dict - Present optional columns by category
        """
        if warn_on_missing_optional is None:
            warn_on_missing_optional = getattr(
                self.config.data.validation,
                'WARN_ON_MISSING_OPTIONAL',
                True
            )

        self.logger.info(f"Validating columns for {df_name}...")

        result = {
            'valid': True,
            'missing_required': [],
            'missing_optional': {},
            'present_optional': {}
        }

        # Check required columns
        for col in self.required_columns:
            if col not in df.columns:
                result['missing_required'].append(col)
                result['valid'] = False

        # Report on required columns
        if result['valid']:
            self.logger.info(f"  All {len(self.required_columns)} required columns present")
        else:
            msg = f"  Missing {len(result['missing_required'])} required columns: {result['missing_required']}"
            if raise_on_missing:
                self.logger.error(msg)
                raise ValueError(f"{df_name} is missing required columns: {result['missing_required']}")
            else:
                self.logger.warning(msg)

        # Check optional columns
        for category, columns in self.optional_columns.items():
            missing = []
            present = []

            for col in columns:
                if col in df.columns:
                    present.append(col)
                else:
                    missing.append(col)

            if present:
                result['present_optional'][category] = present
            if missing:
                result['missing_optional'][category] = missing

        # Report on optional columns
        if result['present_optional']:
            self.logger.info("  Optional columns present:")
            for category, cols in result['present_optional'].items():
                self.logger.info(f"    {category}: {', '.join(cols)}")

        if warn_on_missing_optional and result['missing_optional']:
            self.logger.warning("  Optional columns missing:")
            for category, cols in result['missing_optional'].items():
                self.logger.warning(f"    {category}: {', '.join(cols)}")

            # Special warning for financial metrics
            financial_metrics = result['missing_optional'].get('financial_metrics', [])
            if financial_metrics:
                self.logger.warning(f"  Financial metrics (EPS, PE, ROE, etc.) are missing!")
                self.logger.warning(f"  This may be because:")
                self.logger.warning(f"    1. The financial_metrics feature flag is disabled")
                self.logger.warning(f"    2. JSON files in {self.config.data.financial_metrics.FINANCIAL_METRICS_SOURCE} are missing")
                self.logger.warning(f"    3. The JSON files don't contain the required data")

        return result

    def get_all_expected_columns(self) -> Set[str]:
        """
        Get all expected columns (required + optional).

        Returns:
            Set of all expected column names
        """
        all_columns = self.required_columns.copy()
        for columns in self.optional_columns.values():
            all_columns.update(columns)
        return all_columns

    def get_required_columns(self) -> Set[str]:
        """
        Get required columns.

        Returns:
            Set of required column names
        """
        return self.required_columns.copy()

    def get_optional_columns(self) -> Dict[str, Set[str]]:
        """
        Get optional columns by category.

        Returns:
            Dictionary mapping category to set of column names
        """
        return {k: v.copy() for k, v in self.optional_columns.items()}


def validate_dataset(
    df: pd.DataFrame,
    df_name: str = "dataset",
    config=None,
    raise_on_missing: bool = True
) -> Dict[str, any]:
    """
    Convenience function to validate a dataset.

    Args:
        df: DataFrame to validate
        df_name: Name of the dataset (for logging)
        config: Configuration object (defaults to load_config('main') if None)
        raise_on_missing: Raise exception if required columns missing

    Returns:
        Dictionary with validation results

    Example:
        >>> result = validate_dataset(df, "training data")
        >>> if not result['valid']:
        ...     print(f"Missing columns: {result['missing_required']}")
    """
    validator = DatasetValidator(config)
    return validator.validate_columns(df, df_name, raise_on_missing=raise_on_missing)


def check_feature_consistency(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> bool:
    """
    Check that all splits have the same feature columns.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: Expected feature columns

    Returns:
        True if all splits have consistent features

    Raises:
        ValueError: If features are inconsistent
    """
    logger = get_logger("dataset_validation", log_dir="logs")

    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)
    test_cols = set(test_df.columns)

    # Check all have the same columns
    if train_cols != val_cols or train_cols != test_cols:
        missing_in_val = train_cols - val_cols
        missing_in_test = train_cols - test_cols
        extra_in_val = val_cols - train_cols
        extra_in_test = test_cols - train_cols

        error_msg = "Feature columns are inconsistent across splits:\n"
        if missing_in_val:
            error_msg += f"  Missing in val: {missing_in_val}\n"
        if missing_in_test:
            error_msg += f"  Missing in test: {missing_in_test}\n"
        if extra_in_val:
            error_msg += f"  Extra in val: {extra_in_val}\n"
        if extra_in_test:
            error_msg += f"  Extra in test: {extra_in_test}\n"

        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check all expected feature columns are present
    missing_features = set(feature_cols) - train_cols
    if missing_features:
        error_msg = f"Expected feature columns missing from data: {missing_features}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Feature columns are consistent across all splits")
    return True
