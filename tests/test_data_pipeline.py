"""
Unit tests for data pipeline components.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.downloader import DataDownloader
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import FinancialDataset


class TestDataPipeline(unittest.TestCase):
    """Test data pipeline components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = load_config('main')
        self.model_config = load_config('model')

        # Create sample data
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """Create sample stock data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')

        data = []
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            for i, date in enumerate(dates):
                # Random walk
                price = 100 + np.cumsum(np.random.randn(len(dates)))[i] * 0.5
                data.append({
                    'date': date,
                    'tic': ticker,
                    'open': price * (1 + np.random.randn() * 0.01),
                    'high': price * (1 + abs(np.random.randn()) * 0.01),
                    'low': price * (1 - abs(np.random.randn()) * 0.01),
                    'close': price,
                    'volume': np.random.randint(1000000, 10000000)
                })

        return pd.DataFrame(data)

    def test_feature_engineering(self):
        """Test feature engineering."""
        engineer = FeatureEngineer(self.config)

        # Test adding time features
        result = engineer.add_time_features(self.sample_data)
        self.assertIn('day', result.columns)
        self.assertIn('month', result.columns)

        # Test technical indicators
        result = engineer.add_technical_indicators(result)
        self.assertIn('ema_50', result.columns)
        self.assertIn('rsi_14', result.columns)

        # Test target calculation
        result = engineer.calculate_target(result)
        self.assertIn('target', result.columns)
        self.assertEqual(result['target'].isna().sum(), 0)  # No NaN after dropping

    def test_fibonacci_features(self):
        """Test Fibonacci retracement feature calculation."""
        feature_flags = self.config.data.features.FEATURE_FLAGS
        original_fibonacci_flag = feature_flags.get('fibonacci_features', False)
        feature_flags._data['fibonacci_features'] = True

        try:
            engineer = FeatureEngineer(self.config)

            # Test adding Fibonacci features
            result = engineer.add_fibonacci_features(self.sample_data)
        finally:
            feature_flags._data['fibonacci_features'] = original_fibonacci_flag

        # Verify all Fibonacci columns exist
        fibonacci_cols = ['swing_high', 'swing_low', 'fib_range', 'fib_38', 'fib_50',
                          'fib_61', 'dist_fib_38', 'dist_fib_50', 'dist_fib_61', 'break_fib_61']

        for col in fibonacci_cols:
            self.assertIn(col, result.columns, f"Column {col} should exist")

        # Verify swing high >= high (or NaN)
        self.assertTrue((result['swing_high'] >= result['high']).all() or
                        result['swing_high'].isna().any())

        # Verify swing low <= low (or NaN)
        self.assertTrue((result['swing_low'] <= result['low']).all() or
                        result['swing_low'].isna().any())

        # Verify Fibonacci levels are between swing high and low
        valid_mask = ~result['fib_range'].isna() & (result['fib_range'] > 0)
        if valid_mask.any():
            for fib_col in ['fib_38', 'fib_50', 'fib_61']:
                self.assertTrue(
                    (result.loc[valid_mask, fib_col] >= result.loc[valid_mask, 'swing_low']).all(),
                    f"{fib_col} should be >= swing_low"
                )
                self.assertTrue(
                    (result.loc[valid_mask, fib_col] <= result.loc[valid_mask, 'swing_high']).all(),
                    f"{fib_col} should be <= swing_high"
                )

        # Verify break indicator is 0 or 1
        self.assertTrue(
            result['break_fib_61'].isin([0, 1]).all() or result['break_fib_61'].isna().any()
        )

    def test_preprocessing(self):
        """Test preprocessing."""
        engineer = FeatureEngineer(self.config)

        # Add features
        df = engineer.add_all_features(self.sample_data, calculate_target=True)

        # Test preprocessing
        preprocessor = DataPreprocessor(self.config)

        # Test encoding
        df = preprocessor.encode_categorical(df, fit=True)
        self.assertIn('tic_id', df.columns)

        # Test normalization
        feature_cols = ['close', 'volume', 'ema_50', 'rsi_14']
        df = preprocessor.normalize_features(df, fit=True, feature_cols=feature_cols)
        self.assertIn('close', df.columns)  # Column should exist

    def test_dataset(self):
        """Test dataset creation."""
        # Create sample sequences
        sequences = {
            'features': np.random.randn(100, 30, 10),
            'stock_id': np.random.randint(0, 10, (100, 30)),
            'group_id': np.random.randint(0, 5, (100, 30)),
            'day': np.random.randint(1, 32, (100, 30)),
            'month': np.random.randint(1, 13, (100, 30)),
            'target': np.random.randn(100)
        }

        dataset = FinancialDataset(sequences, self.model_config)

        self.assertEqual(len(dataset), 100)

        # Test __getitem__
        sample = dataset[0]
        self.assertIn('features', sample)
        self.assertIn('target', sample)
        self.assertEqual(sample['features'].shape, (30, 10))

        # Test embedding sizes
        emb_sizes = dataset.get_embedding_sizes()
        self.assertIn('num_stocks', emb_sizes)
        self.assertIn('num_groups', emb_sizes)

    def test_local_index_loading(self):
        """Test loading stocks from local index file."""
        downloader = DataDownloader(self.config)

        # Check if index file exists before testing
        if not downloader.index_path.exists():
            self.skipTest(f"Index file not found: {downloader.index_path}")

        # Test listing available indices
        indices = downloader.list_available_indices()
        self.assertIsInstance(indices, list)
        self.assertGreater(len(indices), 0)

        # Test getting tickers from index
        tickers = downloader.get_sp500_tickers()
        self.assertIsInstance(tickers, list)
        self.assertGreater(len(tickers), 0)

        # Test listing stocks with details
        stocks = downloader.list_index_stocks()
        self.assertIsInstance(stocks, list)
        if stocks:
            self.assertIn('Code', stocks[0])
            self.assertIn('Name', stocks[0])
            self.assertIn('Sector', stocks[0])

    def test_stock_filtering(self):
        """Test filtering stocks by criteria."""
        downloader = DataDownloader(self.config)

        # Check if index file exists before testing
        if not downloader.index_path.exists():
            self.skipTest(f"Index file not found: {downloader.index_path}")

        # Test filtering by sector
        tech_stocks = downloader.filter_stocks_by_criteria(sectors=['Technology'])
        self.assertIsInstance(tech_stocks, list)

        # Test filtering by specific stocks
        specific = downloader.filter_stocks_by_criteria(
            include_stocks={'AAPL', 'MSFT', 'GOOGL'}
        )
        self.assertEqual(len(specific), 3)
        self.assertIn('AAPL', specific)
        self.assertIn('MSFT', specific)

        # Test excluding stocks
        filtered = downloader.filter_stocks_by_criteria(
            include_stocks={'AAPL', 'MSFT', 'GOOGL', 'TSLA'},
            exclude_stocks={'TSLA'}
        )
        self.assertEqual(len(filtered), 3)
        self.assertNotIn('TSLA', filtered)

    def test_export_normalized_data(self):
        """Test exporting normalized data to parquet."""
        engineer = FeatureEngineer(self.config)

        # Add features
        df = engineer.add_all_features(self.sample_data, calculate_target=True)

        # Create temporary file paths
        with tempfile.TemporaryDirectory() as tmpdir:
            export_pre_path = Path(tmpdir) / 'pre_normalized.parquet'
            export_normalized_path = Path(tmpdir) / 'normalized.parquet'

            # Test preprocessing with export
            preprocessor = DataPreprocessor(self.config)
            processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(
                df,
                fit=True,
                export_pre_normalize=str(export_pre_path),
                export_normalized=str(export_normalized_path)
            )

            # Verify files were created
            self.assertTrue(export_pre_path.exists(), "Pre-normalized file should exist")
            self.assertTrue(export_normalized_path.exists(), "Normalized file should exist")

            # Load and verify pre-normalized data
            pre_normalized_df = pd.read_parquet(export_pre_path)
            self.assertIn('close', pre_normalized_df.columns)

            # Load and verify normalized data
            normalized_df = pd.read_parquet(export_normalized_path)
            self.assertIn('close', normalized_df.columns)

            # Verify normalized values are different from pre-normalized
            # (normalized values should be transformed)
            pre_close_mean = pre_normalized_df['close'].mean()
            norm_close_mean = normalized_df['close'].mean()
            self.assertNotAlmostEqual(
                pre_close_mean, norm_close_mean, places=2,
                msg="Normalized values should differ from pre-normalized values"
            )

            # Verify NO string columns in exported normalized data
            non_numeric_cols = normalized_df.select_dtypes(exclude=['number']).columns
            self.assertEqual(
                len(non_numeric_cols), 0,
                f"Normalized export should have no string columns, found: {list(non_numeric_cols)}"
            )

            # Verify string columns were dropped
            self.assertNotIn('date', normalized_df.columns, "date column should be dropped")
            self.assertNotIn('tic', normalized_df.columns, "tic column should be dropped")
            self.assertNotIn('group', normalized_df.columns, "group column should be dropped")
            self.assertNotIn('split', normalized_df.columns, "split column should be dropped")

            # Verify integer encoded columns still exist
            self.assertIn('tic_id', normalized_df.columns, "tic_id column should exist")
            self.assertIn('group_id', normalized_df.columns, "group_id column should exist")

            # Verify target is NOT normalized (should be real values, not transformed)
            # Target should be same in both pre-normalized and normalized data
            pre_target_values = pre_normalized_df['target'].values
            norm_target_values = normalized_df['target'].values
            np.testing.assert_array_equal(
                pre_target_values, norm_target_values,
                err_msg="Target column should NOT be normalized - values should be identical"
            )

            # Verify NO NaN values in exported normalized data
            nan_count = normalized_df.isna().sum().sum()
            self.assertEqual(nan_count, 0, f"Normalized export should have no NaN values, found: {nan_count}")

    def test_nan_filling(self):
        """Test that NaN values are filled with 0."""
        engineer = FeatureEngineer(self.config)

        # Add features
        df = engineer.add_all_features(self.sample_data, calculate_target=True)

        # Store original values before introducing NaN
        original_close = df.loc[0, 'close']
        original_volume = df.loc[1, 'volume']
        original_target = df.loc[2, 'target']

        # Introduce some NaN values for testing
        df.loc[0, 'close'] = np.nan
        df.loc[1, 'volume'] = np.nan
        df.loc[2, 'target'] = np.nan

        # Verify NaN was introduced
        self.assertTrue(np.isnan(df.loc[0, 'close']), "close should be NaN")
        self.assertTrue(np.isnan(df.loc[1, 'volume']), "volume should be NaN")
        self.assertTrue(np.isnan(df.loc[2, 'target']), "target should be NaN")

        # Process data
        preprocessor = DataPreprocessor(self.config)
        processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(
            df,
            fit=True
        )

        # Verify no NaN values remain in numeric columns
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        nan_count = processed_df[numeric_cols].isna().sum().sum()
        self.assertEqual(nan_count, 0, f"All NaN values should be filled, found: {nan_count} NaN values")

        # Target should be filled with 0 (not normalized)
        self.assertEqual(processed_df.loc[2, 'target'], 0.0, "NaN target should be filled with 0")

        # Features are filled and then normalized (so they won't be exactly 0 after normalization)
        # But they should not be NaN
        self.assertFalse(np.isnan(processed_df.loc[0, 'close']), "close should not be NaN after processing")
        self.assertFalse(np.isnan(processed_df.loc[1, 'volume']), "volume should not be NaN after processing")


if __name__ == '__main__':
    unittest.main()
