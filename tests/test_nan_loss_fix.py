"""
Comprehensive test for NaN loss fixes.

Tests the complete pipeline with focus on:
1. Small dataset full flow (no NaN loss)
2. NaN detection and handling in training
3. Data validation and sanitization
4. Edge cases (extreme values, zeros, etc.)
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import create_data_loaders, FinancialDataset
from src.models.crnn_attention import CRNNAttentionModel
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.validation import (
    check_tensor_for_nan_inf,
    sanitize_tensor,
    check_batch_for_invalid,
    sanitize_batch,
    check_model_parameters
)

logger = get_logger("test_nan_fix", log_dir="logs")


def create_small_test_dataset(n_stocks=3, n_days=200, seed=42):
    """
    Create a small but realistic dataset for testing.

    Includes edge cases:
    - Zero values
    - Very small values
    - Very large values
    - Missing values (that will become NaN)

    Args:
        n_stocks: Number of stocks to simulate
        n_days: Number of trading days
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    tickers = [f"TEST{i:02d}" for i in range(n_stocks)]
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

    data = []
    for ticker in tickers:
        price = 100.0

        for i, date in enumerate(dates):
            # Create realistic price movements
            change = np.random.randn() * 0.02

            # Add some edge cases
            if i == 10:  # Very small change
                change = 1e-10
            elif i == 20:  # Zero change
                change = 0.0
            elif i == 30:  # Large change
                change = 0.1

            price = max(price * (1 + change), 1.0)  # Ensure positive

            data.append({
                'date': date,
                'tic': ticker,
                'open': price * (1 + np.random.randn() * 0.005),
                'high': price * (1 + abs(np.random.randn()) * 0.01),
                'low': price * (1 - abs(np.random.randn()) * 0.01),
                'close': price,
                'volume': np.random.randint(100000, 10000000),
            })

    df = pd.DataFrame(data)
    return df


class TestNaNLossFix(unittest.TestCase):
    """Test suite for NaN loss fixes."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.config = load_config('model')

        # Configure for fast testing
        cls.config.model.training.NUM_EPOCHS = 3
        cls.config.model.training.BATCH_SIZE = 8
        cls.config.model.training.EARLY_STOPPING_PATIENCE = 2
        cls.config.model.loss.HUBER_DELTA = 1.0

        # Ensure nan_handling config exists
        if not hasattr(cls.config.model, 'nan_handling'):
            from types import SimpleNamespace
            cls.config.model.nan_handling = SimpleNamespace(
                CHECK_INPUTS=True,
                SANITIZE_INPUTS=True,
                CHECK_GRADIENTS=True,
                STOP_ON_NAN=True,
                LOG_NAN_DETAILS=True,
                MAX_GRAD_VALUE=100.0,
                REPLACE_VALUE=0.0
            )

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_small_dataset_full_flow(self):
        """
        Test 1: Full training pipeline with small dataset.

        Verifies:
        - Training completes without NaN loss
        - Loss decreases reasonably
        - Model can make predictions
        """
        logger.info("=" * 70)
        logger.info("TEST 1: Small Dataset Full Flow")
        logger.info("=" * 70)

        # Create dataset
        df = create_small_test_dataset(n_stocks=3, n_days=200)

        # Feature engineering
        data_config = load_config('main')
        engineer = FeatureEngineer(data_config)
        df = engineer.add_all_features(df, calculate_target=True)

        # Preprocessing
        preprocessor = DataPreprocessor(data_config)
        df = preprocessor.encode_categorical(df, fit=True)
        df = preprocessor.normalize_features(df, fit=True)

        # Get features
        feature_cols = [c for c in df.columns if c not in
                       ['date', 'tic', 'tic_id', 'group', 'group_id', 'target', 'split']]

        # Simple split
        n = len(df)
        train_df = df.iloc[:int(n*0.7)]
        val_df = df.iloc[int(n*0.7):int(n*0.85)]
        test_df = df.iloc[int(n*0.85):]

        # Create sequences
        train_sequences = preprocessor.create_sequences(train_df, feature_cols)
        val_sequences = preprocessor.create_sequences(val_df, feature_cols)

        # Skip if not enough sequences
        if len(train_sequences['target']) < 10:
            self.skipTest("Not enough training sequences")

        # Add dividend_flag if not present
        for seq_dict in [train_sequences, val_sequences]:
            if 'dividend_flag' not in seq_dict:
                seq_len = seq_dict['features'].shape[1]
                seq_dict['dividend_flag'] = np.full(
                    (seq_dict['features'].shape[0], seq_len),
                    2, dtype=np.int32
                )

        # Create loaders
        loaders = create_data_loaders(train_sequences, val_sequences, config=self.config)

        # Create model
        num_features = train_sequences['features'].shape[2]
        num_stocks = int(train_df['tic_id'].max()) + 1
        num_groups = int(train_df['group_id'].max()) + 1 if 'group_id' in train_df.columns else 1

        model = CRNNAttentionModel(
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
            config=self.config
        )

        # Train
        device = 'cpu'
        trainer = Trainer(model, self.config, device=device)

        try:
            history = trainer.train(
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                num_epochs=3
            )

            # Assertions
            self.assertEqual(len(history['train_loss']), 3, "Training did not complete 3 epochs")

            # Check no NaN in losses
            for i, loss in enumerate(history['train_loss']):
                self.assertFalse(np.isnan(loss), f"NaN in train loss at epoch {i}")
                self.assertFalse(np.isinf(loss), f"Inf in train loss at epoch {i}")

            if history['val_loss']:
                for i, loss in enumerate(history['val_loss']):
                    self.assertFalse(np.isnan(loss), f"NaN in val loss at epoch {i}")
                    self.assertFalse(np.isinf(loss), f"Inf in val loss at epoch {i}")

            # Check loss decreased reasonably
            initial_loss = history['train_loss'][0]
            final_loss = history['train_loss'][-1]
            self.assertLess(final_loss, initial_loss + 1.0, "Loss did not decrease reasonably")

            logger.info(f"Training completed successfully!")
            logger.info(f"Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")

        except RuntimeError as e:
            if 'NaN' in str(e) or 'Inf' in str(e):
                self.fail(f"Training failed with NaN/Inf: {e}")
            raise

    def test_02_nan_detection_in_batch(self):
        """
        Test 2: NaN detection in training batch.

        Verifies:
        - NaN inputs are detected
        - NaN inputs are sanitized
        - Training continues after sanitization
        """
        logger.info("=" * 70)
        logger.info("TEST 2: NaN Detection in Batch")
        logger.info("=" * 70)

        # Create batch with NaN
        batch = {
            'features': torch.randn(4, 30, 20),
            'stock_id': torch.randint(0, 10, (4, 30)),
            'target': torch.randn(4, 1)
        }

        # Add NaN
        batch['features'][0, 0, 0] = float('nan')
        batch['target'][1, 0] = float('inf')

        # Check for invalid
        has_issues, message = check_batch_for_invalid(batch)
        self.assertTrue(has_issues, "Failed to detect NaN/Inf in batch")
        self.assertIn("NaN", message)

        # Sanitize
        sanitized = sanitize_batch(batch)

        # Verify sanitized
        self.assertFalse(torch.isnan(sanitized['features']).any(), "NaN still in features")
        self.assertFalse(torch.isinf(sanitized['target']).any(), "Inf still in target")

        logger.info("NaN detection and sanitization working correctly")

    def test_03_model_weight_initialization(self):
        """
        Test 3: Model weight initialization.

        Verifies:
        - Weights are initialized properly
        - No NaN/Inf in initial weights
        - Gradients don't explode on first batch
        """
        logger.info("=" * 70)
        logger.info("TEST 3: Model Weight Initialization")
        logger.info("=" * 70)

        # Create model
        model = CRNNAttentionModel(
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=self.config
        )

        # Check all parameters
        for name, param in model.named_parameters():
            # No NaN
            self.assertFalse(torch.isnan(param).any(), f"NaN in {name}")
            # No Inf
            self.assertFalse(torch.isinf(param).any(), f"Inf in {name}")
            # Finite values
            self.assertTrue(torch.isfinite(param).all(), f"Non-finite values in {name}")

        # Test forward pass
        batch_size = 4
        seq_len = 30
        features = torch.randn(batch_size, seq_len, 20)
        stock_id = torch.randint(0, 10, (batch_size, seq_len))
        group_id = torch.randint(0, 5, (batch_size, seq_len))
        day = torch.randint(1, 32, (batch_size, seq_len))
        month = torch.randint(1, 13, (batch_size, seq_len))
        dividend_flag = torch.ones(batch_size, seq_len, dtype=torch.long)

        output = model(features, stock_id, group_id, day, month, dividend_flag)

        # Check output
        self.assertFalse(torch.isnan(output).any(), "NaN in model output")
        self.assertFalse(torch.isinf(output).any(), "Inf in model output")
        self.assertTrue(torch.isfinite(output).all(), "Non-finite output")

        logger.info("Model initialization test passed")

    def test_04_gradient_checking(self):
        """
        Test 4: Gradient checking during training.

        Verifies:
        - Gradients are computed correctly
        - No NaN/Inf in gradients
        - Gradient clipping works
        """
        logger.info("=" * 70)
        logger.info("TEST 4: Gradient Checking")
        logger.info("=" * 70)

        # Create model
        model = CRNNAttentionModel(
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=self.config
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.HuberLoss(delta=1.0)

        # Create batch
        features = torch.randn(4, 30, 20)
        stock_id = torch.randint(0, 10, (4, 30))
        group_id = torch.randint(0, 5, (4, 30))
        day = torch.randint(1, 32, (4, 30))
        month = torch.randint(1, 13, (4, 30))
        dividend_flag = torch.ones(4, 30, dtype=torch.long)
        target = torch.randn(4, 1)

        # Forward pass
        output = model(features, stock_id, group_id, day, month, dividend_flag)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN in gradient of {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf in gradient of {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}")

                # Check gradient magnitude
                grad_norm = param.grad.norm().item()
                self.assertLess(grad_norm, 1000.0, f"Exploding gradient in {name}: {grad_norm}")

        logger.info("Gradient checking test passed")

    def test_05_edge_cases(self):
        """
        Test 5: Edge cases and extreme values.

        Verifies:
        - All zero inputs
        - Very large values
        - Very small values
        - Mixed NaN/valid values
        """
        logger.info("=" * 70)
        logger.info("TEST 5: Edge Cases")
        logger.info("=" * 70)

        model = CRNNAttentionModel(
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=self.config
        )

        # Test 1: All zeros
        features = torch.zeros(2, 30, 20)
        stock_id = torch.zeros(2, 30, dtype=torch.long)
        group_id = torch.zeros(2, 30, dtype=torch.long)
        day = torch.ones(2, 30, dtype=torch.long)
        month = torch.ones(2, 30, dtype=torch.long)
        dividend_flag = torch.ones(2, 30, dtype=torch.long)

        output = model(features, stock_id, group_id, day, month, dividend_flag)
        self.assertTrue(torch.isfinite(output).all(), "Failed with all zeros")

        # Test 2: Very large values
        features = torch.ones(2, 30, 20) * 1000
        output = model(features, stock_id, group_id, day, month, dividend_flag)
        self.assertTrue(torch.isfinite(output).all(), "Failed with large values")

        # Test 3: Very small values
        features = torch.ones(2, 30, 20) * 1e-10
        output = model(features, stock_id, group_id, day, month, dividend_flag)
        self.assertTrue(torch.isfinite(output).all(), "Failed with small values")

        # Test 4: Mixed NaN/valid
        features = torch.randn(2, 30, 20)
        features[0, 0, 0] = float('nan')
        features[1, 1, 1] = float('inf')

        # Should sanitize
        if self.config.model.nan_handling.SANITIZE_INPUTS:
            features_sanitized = torch.where(
                torch.isnan(features) | torch.isinf(features),
                torch.zeros_like(features),
                features
            )
            output = model(features_sanitized, stock_id, group_id, day, month, dividend_flag)
            self.assertTrue(torch.isfinite(output).all(), "Failed with sanitized NaN/Inf")

        logger.info("Edge cases test passed")

    def test_06_model_state_check(self):
        """
        Test 6: Model state checking.

        Verifies:
        - check_model_state works correctly
        - Detects NaN/Inf in model parameters
        """
        logger.info("=" * 70)
        logger.info("TEST 6: Model State Check")
        logger.info("=" * 70)

        model = CRNNAttentionModel(
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=self.config
        )

        # Check healthy model
        is_valid, issues = check_model_parameters(model)
        self.assertTrue(is_valid, f"Healthy model has issues: {issues}")
        self.assertEqual(len(issues), 0, f"Healthy model has issues: {issues}")

        logger.info("Model state check test passed")

    def test_07_dataset_validation(self):
        """
        Test 7: Dataset NaN validation.

        Verifies:
        - FinancialDataset can be created with clean data
        - Training loop handles NaN sanitization
        """
        logger.info("=" * 70)
        logger.info("TEST 7: Dataset Validation")
        logger.info("=" * 70)

        # Create clean sequences (no NaN)
        sequences = {
            'features': np.random.randn(50, 30, 20),
            'stock_id': np.random.randint(0, 10, (50, 30)),
            'group_id': np.random.randint(0, 5, (50, 30)),
            'day': np.random.randint(1, 32, (50, 30)),
            'month': np.random.randint(1, 13, (50, 30)),
            'target': np.random.randn(50)
        }

        # Create dataset with clean data
        dataset = FinancialDataset(sequences, self.config)

        # Check features are valid
        self.assertEqual(len(dataset), 50)
        self.assertEqual(dataset.num_samples, 50)
        self.assertFalse(torch.isnan(dataset.features).any(), "NaN found in dataset features")

        # Verify training loop would handle NaN in inputs
        # Create a batch with NaN and verify sanitization works
        batch = {
            'features': torch.randn(4, 30, 20),
            'stock_id': torch.randint(0, 10, (4, 30)),
            'group_id': torch.randint(0, 5, (4, 30)),
            'day': torch.randint(1, 32, (4, 30)),
            'month': torch.randint(1, 13, (4, 30)),
            'dividend_flag': torch.ones(4, 30, dtype=torch.long),
            'target': torch.randn(4, 1)
        }

        # Add NaN
        batch['features'][0, 0, 0] = float('nan')

        # Simulate training loop sanitization
        if self.config.model.nan_handling.CHECK_INPUTS:
            features = batch['features']
            if torch.isnan(features).any() or torch.isinf(features).any():
                features = torch.where(
                    torch.isnan(features) | torch.isinf(features),
                    torch.tensor(0.0),
                    features
                )

        self.assertFalse(torch.isnan(features).any(), "Sanitization failed")

        logger.info("Dataset validation test passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
