"""
Pytest configuration and fixtures for test_small_dataset.py.

This module provides pytest fixtures that enable the tests in test_small_dataset.py
to run correctly with pytest, while maintaining compatibility with standalone execution.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import FinancialDataset, create_data_loaders
from src.models import create_model


def create_small_dataset():
    """Create small sample dataset."""
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    data = []
    for ticker in tickers:
        price = 100
        for date in dates:
            change = np.random.randn() * 0.02
            price = price * (1 + change)

            data.append({
                'date': date,
                'tic': ticker,
                'open': price * (1 + np.random.randn() * 0.005),
                'high': price * (1 + abs(np.random.randn()) * 0.005),
                'low': price * (1 - abs(np.random.randn()) * 0.005),
                'close': price,
                'volume': np.random.randint(1000000, 10000000)
            })

    return pd.DataFrame(data)


@pytest.fixture
def df():
    """Fixture that provides a small sample dataset."""
    return create_small_dataset()


@pytest.fixture
def data_config():
    """Fixture that provides data config."""
    config = load_config('main')
    config.data.sequences.SEQUENCE_LENGTH = 20
    config.data.sequences.PREDICTION_HORIZON = 1
    return config


@pytest.fixture
def df_with_features(df, data_config):
    """Fixture that provides dataset with feature engineering applied."""
    engineer = FeatureEngineer(data_config)
    return engineer.add_all_features(df.copy(), calculate_target=True), data_config


@pytest.fixture
def sequences(df_with_features, data_config):
    """Fixture that provides preprocessed sequences."""
    df, _ = df_with_features
    preprocessor = DataPreprocessor(data_config)
    processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(df, fit=True)
    return sequences, info


@pytest.fixture
def info(sequences):
    """Fixture that provides preprocessing info."""
    return sequences[1]


@pytest.fixture
def model_config():
    """Fixture that provides model config."""
    config = load_config('model')
    config.model.training.BATCH_SIZE = 8
    config.model.training.NUM_EPOCHS = 2
    return config


@pytest.fixture
def model(info, model_config):
    """Fixture that provides a model instance."""
    return create_model(
        model_type='crnn_attention',
        num_features=info['num_features'],
        num_stocks=info['num_stocks'],
        num_groups=info['num_groups'],
        config=model_config
    )


@pytest.fixture
def loaders(sequences, model_config):
    """Fixture that provides data loaders."""
    seq, _ = sequences
    return create_data_loaders(
        train_sequences=seq['train'],
        val_sequences=seq.get('val'),
        config=model_config
    )
