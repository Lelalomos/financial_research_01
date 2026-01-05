"""
Unit tests for dataset column validation and NaN handling.
"""

import sys
import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.validation import (
    DatasetValidator,
    validate_dataset,
    check_feature_consistency
)
from src.utils.validation import (
    check_tensor_for_nan_inf,
    sanitize_tensor,
    check_batch_for_invalid,
    sanitize_batch,
    check_model_parameters
)


class TestDatasetValidator:
    """Test DatasetValidator class."""

    def test_validator_initialization(self):
        """Test validator initializes with config."""
        config = load_config('main')
        validator = DatasetValidator(config)

        # Should have required columns
        assert 'date' in validator.required_columns
        assert 'tic' in validator.required_columns
        assert 'open' in validator.required_columns
        assert 'close' in validator.required_columns

        # Should have optional columns grouped by category
        assert 'financial_metrics' in validator.optional_columns
        assert 'pe_ratio' in validator.optional_columns['financial_metrics']
        assert 'eps' in validator.optional_columns['financial_metrics']
        assert 'roe' in validator.optional_columns['financial_metrics']

    def test_validate_all_required_columns_present(self):
        """Test validation passes when all required columns present."""
        config = load_config('main')
        validator = DatasetValidator(config)

        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10,
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000000, 10000000, 10),
            'target': np.random.rand(10)
        })

        result = validator.validate_columns(df, "test_data")

        assert result['valid'] is True
        assert len(result['missing_required']) == 0

    def test_validate_missing_required_columns(self):
        """Test validation fails when required columns missing."""
        config = load_config('main')
        validator = DatasetValidator(config)

        # Missing 'volume' and 'target'
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10,
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100
        })

        # Should raise by default
        with pytest.raises(ValueError, match="missing required columns"):
            validator.validate_columns(df, "test_data")

        # Should not raise if raise_on_missing=False
        result = validator.validate_columns(df, "test_data", raise_on_missing=False)
        assert result['valid'] is False
        assert 'volume' in result['missing_required']
        assert 'target' in result['missing_required']

    def test_validate_with_financial_metrics(self):
        """Test validation with financial metrics columns."""
        config = load_config('main')
        validator = DatasetValidator(config)

        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10,
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000000, 10000000, 10),
            'target': np.random.rand(10),
            'pe_ratio': np.random.rand(10) * 50,
            'eps': np.random.rand(10) * 10,
            'roe': np.random.rand(10) * 0.5
        })

        result = validator.validate_columns(df, "test_data")

        assert result['valid'] is True
        assert 'financial_metrics' in result['present_optional']
        assert 'pe_ratio' in result['present_optional']['financial_metrics']
        assert 'eps' in result['present_optional']['financial_metrics']
        assert 'roe' in result['present_optional']['financial_metrics']

    def test_validate_missing_financial_metrics(self):
        """Test validation detects missing financial metrics."""
        config = load_config('main')
        validator = DatasetValidator(config)

        # Basic required columns only
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10,
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000000, 10000000, 10),
            'target': np.random.rand(10)
        })

        result = validator.validate_columns(df, "test_data", warn_on_missing_optional=True)

        # Should still be valid (financial metrics are optional)
        assert result['valid'] is True
        # But should report missing financial metrics
        assert 'financial_metrics' in result['missing_optional']
        assert len(result['missing_optional']['financial_metrics']) > 0

    def test_get_all_expected_columns(self):
        """Test getting all expected columns."""
        config = load_config('main')
        validator = DatasetValidator(config)

        all_cols = validator.get_all_expected_columns()

        # Should include required and optional
        assert 'date' in all_cols
        assert 'close' in all_cols
        assert 'pe_ratio' in all_cols
        assert 'eps' in all_cols

    def test_get_required_columns(self):
        """Test getting required columns."""
        config = load_config('main')
        validator = DatasetValidator(config)

        required = validator.get_required_columns()

        assert 'date' in required
        assert 'tic' in required
        assert 'close' in required
        assert 'target' in required


class TestValidateDataset:
    """Test validate_dataset convenience function."""

    def test_validate_dataset_convenience(self):
        """Test validate_dataset function works correctly."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10,
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000000, 10000000, 10),
            'target': np.random.rand(10)
        })

        result = validate_dataset(df, "test_df")

        assert result['valid'] is True

    def test_validate_dataset_missing_columns(self):
        """Test validate_dataset raises on missing columns."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10
            # Missing all other required columns
        })

        with pytest.raises(ValueError):
            validate_dataset(df, "test_df")

    def test_validate_dataset_no_raise(self):
        """Test validate_dataset with raise_on_missing=False."""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10
        })

        result = validate_dataset(df, "test_df", raise_on_missing=False)

        assert result['valid'] is False
        assert len(result['missing_required']) > 0


class TestCheckFeatureConsistency:
    """Test check_feature_consistency function."""

    def test_consistent_features(self):
        """Test check passes when features are consistent."""
        feature_cols = ['open', 'high', 'low', 'close', 'volume']

        train_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10,
            **{col: np.random.rand(10) for col in feature_cols}
        })

        val_df = pd.DataFrame({
            'date': pd.date_range('2020-01-11', periods=10),
            'tic': ['AAPL'] * 10,
            **{col: np.random.rand(10) for col in feature_cols}
        })

        test_df = pd.DataFrame({
            'date': pd.date_range('2020-01-21', periods=10),
            'tic': ['AAPL'] * 10,
            **{col: np.random.rand(10) for col in feature_cols}
        })

        # Should not raise
        result = check_feature_consistency(train_df, val_df, test_df, feature_cols)
        assert result is True

    def test_inconsistent_features(self):
        """Test check fails when features are inconsistent."""
        feature_cols = ['open', 'high', 'low', 'close', 'volume']

        train_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'tic': ['AAPL'] * 10,
            **{col: np.random.rand(10) for col in feature_cols}
        })

        # Missing 'volume' in val
        val_df = pd.DataFrame({
            'date': pd.date_range('2020-01-11', periods=10),
            'tic': ['AAPL'] * 10,
            **{col: np.random.rand(10) for col in ['open', 'high', 'low', 'close']}
        })

        test_df = pd.DataFrame({
            'date': pd.date_range('2020-01-21', periods=10),
            'tic': ['AAPL'] * 10,
            **{col: np.random.rand(10) for col in feature_cols}
        })

        with pytest.raises(ValueError, match="Feature columns are inconsistent"):
            check_feature_consistency(train_df, val_df, test_df, feature_cols)

    def test_missing_expected_features(self):
        """Test check fails when expected features are missing."""
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ema_50']

        # All splits missing 'ema_50'
        all_splits = [
            pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=10),
                'tic': ['AAPL'] * 10,
                **{col: np.random.rand(10) for col in ['open', 'high', 'low', 'close', 'volume']}
            })
            for _ in range(3)
        ]

        with pytest.raises(ValueError, match="Expected feature columns missing"):
            check_feature_consistency(all_splits[0], all_splits[1], all_splits[2], feature_cols)


class TestNaNHandling:
    """Test NaN/Inf detection and handling utilities."""

    def test_check_tensor_for_nan(self):
        """Test NaN detection in tensors."""
        # Clean tensor
        tensor = torch.randn(10, 10)
        has_issues, msg = check_tensor_for_nan_inf(tensor, "test")
        assert has_issues is False
        assert msg == ""

        # Tensor with NaN
        tensor[0, 0] = float('nan')
        has_issues, msg = check_tensor_for_nan_inf(tensor, "test")
        assert has_issues is True
        assert "NaN" in msg

    def test_check_tensor_for_inf(self):
        """Test Inf detection in tensors."""
        # Tensor with positive Inf
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float('inf')
        has_issues, msg = check_tensor_for_nan_inf(tensor, "test")
        assert has_issues is True
        assert "Inf" in msg

        # Tensor with negative Inf
        tensor = torch.randn(10, 10)
        tensor[1, 1] = float('-inf')
        has_issues, msg = check_tensor_for_nan_inf(tensor, "test")
        assert has_issues is True
        assert "Inf" in msg

    def test_sanitize_tensor(self):
        """Test tensor sanitization."""
        # Tensor with NaN and Inf
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float('nan')
        tensor[1, 1] = float('inf')
        tensor[2, 2] = float('-inf')

        sanitized = sanitize_tensor(tensor, "test", replace_value=0.0)

        # Check no NaN or Inf remains
        assert not torch.isnan(sanitized).any()
        assert not torch.isinf(sanitized).any()
        assert sanitized[0, 0].item() == 0.0
        assert sanitized[1, 1].item() == 0.0
        assert sanitized[2, 2].item() == 0.0

    def test_sanitize_tensor_custom_value(self):
        """Test tensor sanitization with custom replacement value."""
        tensor = torch.randn(10, 10)
        tensor[0, 0] = float('nan')

        sanitized = sanitize_tensor(tensor, "test", replace_value=1.0)

        assert not torch.isnan(sanitized).any()
        assert sanitized[0, 0].item() == 1.0

    def test_check_batch_for_invalid(self):
        """Test batch validation."""
        # Clean batch
        batch = {
            'features': torch.randn(4, 30, 20),
            'stock_id': torch.randint(0, 10, (4, 30)),
            'target': torch.randn(4, 1)
        }

        has_issues, msg = check_batch_for_invalid(batch)
        assert has_issues is False
        assert msg == ""

        # Batch with NaN
        batch['features'][0, 0, 0] = float('nan')
        has_issues, msg = check_batch_for_invalid(batch)
        assert has_issues is True
        assert "features" in msg
        assert "NaN" in msg

    def test_sanitize_batch(self):
        """Test batch sanitization."""
        batch = {
            'features': torch.randn(4, 30, 20),
            'target': torch.randn(4, 1)
        }

        # Add NaN/Inf
        batch['features'][0, 0, 0] = float('nan')
        batch['target'][1, 0] = float('inf')

        sanitized = sanitize_batch(batch, replace_value=0.0)

        # Check sanitized
        assert not torch.isnan(sanitized['features']).any()
        assert not torch.isinf(sanitized['target']).any()
        assert sanitized['features'][0, 0, 0].item() == 0.0
        assert sanitized['target'][1, 0].item() == 0.0

    def test_check_model_parameters(self):
        """Test model parameter checking."""
        from src.models.crnn_attention import CRNNAttentionModel

        config = load_config('model')

        # Ensure nan_handling config exists
        if not hasattr(config.model, 'nan_handling'):
            from types import SimpleNamespace
            config.model.nan_handling = SimpleNamespace(
                CHECK_INPUTS=True,
                SANITIZE_INPUTS=True,
                CHECK_GRADIENTS=True,
                STOP_ON_NAN=True,
                LOG_NAN_DETAILS=True,
                MAX_GRAD_VALUE=100.0,
                REPLACE_VALUE=0.0
            )

        model = CRNNAttentionModel(
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=config
        )

        # Healthy model should pass
        is_valid, issues = check_model_parameters(model)
        assert is_valid is True
        assert len(issues) == 0

    def test_check_tensor_with_mixed_invalid(self):
        """Test detection of mixed NaN and Inf values."""
        tensor = torch.randn(100, 100)

        # Add multiple invalid values
        tensor[0, 0] = float('nan')
        tensor[1, 1] = float('inf')
        tensor[2, 2] = float('-inf')
        tensor[3, 3] = float('nan')

        has_issues, msg = check_tensor_for_nan_inf(tensor, "mixed_test")

        # Should detect issues
        assert has_issues is True
        assert "mixed_test" in msg
        # Should count NaN values (2 in our case)
        assert "NaN" in msg

    def test_sanitize_preserves_valid_values(self):
        """Test that sanitization preserves valid values."""
        tensor = torch.randn(10, 10)
        original_valid_value = tensor[5, 5].item()

        # Add some invalid values
        tensor[0, 0] = float('nan')
        tensor[1, 1] = float('inf')

        sanitized = sanitize_tensor(tensor, "test", replace_value=-999.0)

        # Check that valid value was preserved
        assert sanitized[5, 5].item() == original_valid_value
        # Check that invalid values were replaced
        assert sanitized[0, 0].item() == -999.0
        assert sanitized[1, 1].item() == -999.0
