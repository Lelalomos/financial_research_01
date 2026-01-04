"""
Unit tests for prediction functionality.
"""

import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.config import load_config
from src.data.prediction_prep import PredictionPreparator, create_prediction_preparator
from src.prediction.predictor import Predictor, create_predictor
from src.models.lstm3_attn_model import create_model as create_lstm3_attn
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer


class TestPredictionPreparator(unittest.TestCase):
    """Test prediction data preparation."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_config = load_config('main')
        self.model_config = load_config('model')
        self.preparator = PredictionPreparator(
            self.data_config,
            self.model_config
        )

        # Create sample data
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self, n_samples=300):
        """Create sample stock data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')

        data = []
        for ticker in ['AAPL', 'MSFT']:
            for i, date in enumerate(dates):
                price = 150 + np.cumsum(np.random.randn(len(dates)))[i] * 0.5
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

    def test_init(self):
        """Test initialization."""
        self.assertIsNotNone(self.preparator.data_config)
        self.assertIsNotNone(self.preparator.model_config)
        self.assertIsNotNone(self.preparator.feature_engineer)

    def test_prepare_single_row(self):
        """Test single row preparation."""
        data = {
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.5,
            'volume': 50000000
        }

        result = self.preparator.prepare_single_row(
            data,
            stock_ticker='AAPL',
            date='2024-01-15',
            group='Technology'
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertIn('tic', result.columns)
        self.assertIn('date', result.columns)
        self.assertIn('group', result.columns)
        self.assertEqual(result['tic'].iloc[0], 'AAPL')

    def test_prepare_batch(self):
        """Test batch preparation."""
        result = self.preparator.prepare_batch(self.sample_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertIn('tic_id', result.columns)
        self.assertIn('group_id', result.columns)
        self.assertIn('day', result.columns)
        self.assertIn('month', result.columns)

    def test_encode_categoricals(self):
        """Test categorical encoding."""
        df = self.sample_data.copy()
        df['group'] = 'Technology'

        result = self.preparator._encode_categoricals(df)

        self.assertIn('tic_id', result.columns)
        self.assertIn('group_id', result.columns)
        self.assertTrue(result['tic_id'].dtype in [np.int64, np.int32])

    def test_set_feature_columns(self):
        """Test setting feature columns."""
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        self.preparator.set_feature_columns(feature_cols)

        self.assertEqual(self.preparator.feature_cols, feature_cols)
        self.assertEqual(self.preparator.num_features, len(feature_cols))

    def test_create_sequences(self):
        """Test sequence creation."""
        # Create more sample data for proper sequence creation (single ticker, more dates)
        # Need at least 250 periods because EMA_200 needs 200 data points + sequence length
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=300, freq='D')

        data = []
        for ticker in ['AAPL']:
            for i, date in enumerate(dates):
                price = 150 + np.cumsum(np.random.randn(len(dates)))[i] * 0.5
                data.append({
                    'date': date,
                    'tic': ticker,
                    'open': price * (1 + np.random.randn() * 0.01),
                    'high': price * (1 + abs(np.random.randn()) * 0.01),
                    'low': price * (1 - abs(np.random.randn()) * 0.01),
                    'close': price,
                    'volume': np.random.randint(1000000, 10000000)
                })

        df = self.preparator.prepare_batch(pd.DataFrame(data))
        self.preparator.set_feature_columns(
            [c for c in df.columns if c not in ['date', 'tic', 'tic_id', 'group', 'group_id', 'target', 'split']]
        )

        sequences = self.preparator.create_sequences(df, sequence_length=10)

        self.assertIn('features', sequences)
        self.assertIn('stock_id', sequences)
        self.assertIn('group_id', sequences)
        self.assertIn('day', sequences)
        self.assertIn('month', sequences)
        self.assertIn('dividend_flag', sequences)

        # Check shapes
        self.assertEqual(len(sequences['features'].shape), 3)
        self.assertEqual(sequences['features'].shape[0], sequences['day'].shape[0])  # Same number of sequences
        self.assertEqual(sequences['features'].shape[1], sequences['day'].shape[1])  # Same sequence length

    def test_prepare_for_prediction(self):
        """Test complete preparation pipeline."""
        result = self.preparator.prepare_for_prediction(
            self.sample_data,
            normalize=False,
            create_seqs=False
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('tic_id', result.columns)


class TestPredictor(unittest.TestCase):
    """Test predictor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_config = load_config('main')
        self.model_config = load_config('model')

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create a simple model for testing
        self._create_test_model()

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_model(self):
        """Create a simple test model checkpoint."""
        # Create model
        model = create_lstm3_attn(
            num_features=10,
            num_stocks=5,
            num_groups=3,
            config=self.model_config
        )

        # Create dummy checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': {
                'model_type': 'lstm3_attention',
                'epoch': 1,
                'best_val_loss': 0.1
            },
            'num_features': 10,
            'num_stocks': 5,
            'num_groups': 3,
            'feature_cols': ['open', 'high', 'low', 'close', 'volume', 'ema_50', 'rsi_14', 'stochrsi_14', 'macd', 'macd_signal']
        }

        # Save checkpoint
        self.model_path = Path(self.temp_dir) / 'test_model.pt'
        torch.save(checkpoint, self.model_path)

    def _create_sample_sequences(self):
        """Create sample sequences for testing."""
        return {
            'features': np.random.randn(10, 30, 10).astype(np.float32),
            'stock_id': np.random.randint(0, 5, (10, 30)),
            'group_id': np.random.randint(0, 3, (10, 30)),
            'day': np.random.randint(1, 32, (10, 30)),
            'month': np.random.randint(1, 13, (10, 30)),
            'dividend_flag': np.random.randint(1, 3, (10, 30))
        }

    def test_predictor_init(self):
        """Test predictor initialization."""
        predictor = create_predictor(
            model_path=str(self.model_path),
            device='cpu'
        )

        self.assertIsNotNone(predictor.model)
        self.assertEqual(predictor.device, torch.device('cpu'))

    def test_predict_from_sequences(self):
        """Test prediction from sequences."""
        predictor = create_predictor(
            model_path=str(self.model_path),
            device='cpu'
        )

        sequences = self._create_sample_sequences()
        predictions = predictor.predict(sequences)

        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape[0], 10)  # 10 sequences
        self.assertEqual(predictions.shape[1], 1)  # 1 output

    def test_predict_single(self):
        """Test single row prediction."""
        predictor = create_predictor(
            model_path=str(self.model_path),
            device='cpu'
        )

        data = {
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.5,
            'volume': 50000000
        }

        # This will fail due to insufficient data for sequences, but tests the API
        result = predictor.predict_single(
            data=data,
            stock_ticker='AAPL',
            date='2024-01-15',
            group='Technology'
        )

        self.assertIn('stock_ticker', result)
        self.assertIn('prediction', result)

    def test_get_model_info(self):
        """Test getting model information."""
        predictor = create_predictor(
            model_path=str(self.model_path),
            device='cpu'
        )

        info = predictor.get_model_info()

        self.assertIn('model_type', info)
        self.assertIn('num_features', info)
        self.assertIn('num_stocks', info)
        self.assertEqual(info['model_type'], 'lstm3_attention')
        self.assertEqual(info['num_features'], 10)


class TestPredictionIntegration(unittest.TestCase):
    """Integration tests for prediction pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_config = load_config('main')
        self.model_config = load_config('model')
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_data(self):
        """Create sample stock data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=300, freq='D')

        data = []
        for ticker in ['AAPL', 'MSFT']:
            for i, date in enumerate(dates):
                price = 150 + np.cumsum(np.random.randn(len(dates)))[i] * 0.5
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

    def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        # 1. Create and preprocess data
        engineer = FeatureEngineer(self.data_config)
        df = engineer.add_all_features(self._create_sample_data(), calculate_target=True)

        preprocessor = DataPreprocessor(self.data_config)
        df = preprocessor.encode_categorical(df, fit=True)

        feature_cols = [c for c in df.columns if c not in
                       ['date', 'tic', 'tic_id', 'group', 'group_id', 'target', 'split']]

        # Create sequences (no stock_col parameter, assumes 'tic' column)
        sequences = preprocessor.create_sequences(
            df,
            feature_cols=feature_cols[:10],  # Use first 10 features for simplicity
        )

        self.assertGreater(len(sequences['features']), 0)

        # 2. Create and save model
        model = create_lstm3_attn(
            num_features=10,
            num_stocks=int(df['tic_id'].max()) + 1,
            num_groups=3,
            config=self.model_config
        )

        # Save preprocessor state
        preprocessor_path = Path(self.temp_dir) / 'preprocessor_state.joblib'
        import joblib
        joblib.dump({
            'feature_scaler': preprocessor.normalization_params,  # Use normalization_params instead of feature_scaler_params
            'stock_encoder_classes': preprocessor.stock_encoder.classes_.tolist(),
            'group_encoder_classes': preprocessor.group_encoder.classes_.tolist() if hasattr(preprocessor.group_encoder, 'classes_') else [],
            'feature_cols': feature_cols[:10]
        }, preprocessor_path)

        # 3. Create predictor
        model_path = Path(self.temp_dir) / 'test_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': {'model_type': 'lstm3_attention'},
            'num_features': 10,
            'num_stocks': int(df['tic_id'].max()) + 1,
            'num_groups': 3
        }, model_path)

        predictor = create_predictor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            device='cpu'
        )

        # 4. Test prediction
        predictions = predictor.predict(sequences)

        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape[0], len(sequences['features']))


if __name__ == '__main__':
    unittest.main()
