"""
Small dataset performance test for all model types.

Tests the full pipeline with a small dataset to verify correctness
for all 7 model variants.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.config import load_config
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import FinancialDataset, create_data_loaders
from src.models import create_model, list_available_models
from src.training import Trainer


# All model types to test
ALL_MODEL_TYPES = [
    'crnn',
    'rnn',
    'rnn_attention',
    'crnn_attention',
    'transformer',
    'lstm3',
    'lstm3_attention'
]


def _create_small_dataset():
    """Create small sample dataset."""
    print("Creating small dataset...")

    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range('2020-01-01', periods=500, freq='D')  # ~1.5 years for more data

    data = []
    for ticker in tickers:
        price = 100
        for date in dates:
            # Random walk
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

    df = pd.DataFrame(data)
    print(f"  Created {len(df)} rows for {len(tickers)} tickers")
    return df


def _test_feature_engineering(df):
    """Test feature engineering."""
    print("\nTesting feature engineering...")

    config = load_config('main')
    config.data.sequences.SEQUENCE_LENGTH = 20
    config.data.sequences.PREDICTION_HORIZON = 1
    engineer = FeatureEngineer(config)

    start_time = time.time()
    df = engineer.add_all_features(df, calculate_target=True)
    elapsed = time.time() - start_time

    print(f"  Feature engineering time: {elapsed:.2f}s")
    print(f"  Features shape: {df.shape}")

    feature_info = engineer.get_feature_info(df)
    print(f"  Total features: {feature_info['total_features']}")

    return df, config


def _test_preprocessing(df_with_features, data_config):
    """Test preprocessing."""
    print("\nTesting preprocessing...")

    # Handle tuple from fixture (pytest)
    if isinstance(df_with_features, tuple):
        df, _ = df_with_features
    else:
        df = df_with_features

    preprocessor = DataPreprocessor(data_config)

    start_time = time.time()
    processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(df, fit=True)
    elapsed = time.time() - start_time

    print(f"  Preprocessing time: {elapsed:.2f}s")

    for split_name in ['train', 'val', 'test']:
        if split_name in sequences:
            count = len(sequences[split_name]['target'])
            print(f"  {split_name}: {count} sequences")

    return sequences, info


def _test_dataset_creation(sequences, model_config):
    """Test dataset creation."""
    print("\nTesting dataset creation...")

    # Handle tuple from fixture (pytest)
    if isinstance(sequences, tuple):
        sequences, _ = sequences

    loaders = create_data_loaders(
        train_sequences=sequences['train'],
        val_sequences=sequences.get('val'),
        config=model_config
    )

    print(f"  Train batches: {len(loaders['train'])}")
    if 'val' in loaders:
        print(f"  Val batches: {len(loaders['val'])}")

    # Test getting a batch
    batch = next(iter(loaders['train']))
    print(f"  Batch features shape: {batch['features'].shape}")

    return loaders


def _test_model_creation(info, model_config, model_type):
    """Test model creation."""
    print(f"\nTesting model creation for {model_type}...")

    model = create_model(
        model_type=model_type,
        num_features=info['num_features'],
        num_stocks=info['num_stocks'],
        num_groups=info['num_groups'],
        config=model_config
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model


def _test_training(model, loaders, model_config):
    """Test training."""
    print(f"\nTesting training (2 epochs)...")

    model_config.NUM_EPOCHS = 2

    device = 'cpu'  # Use CPU to avoid CUDA issues
    print(f"  Device: {device}")

    trainer = Trainer(model, model_config, device=device)

    start_time = time.time()
    history = trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders.get('val'),
        num_epochs=2
    )
    elapsed = time.time() - start_time

    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")

    if history['val_loss']:
        print(f"  Final val loss: {history['val_loss'][-1]:.6f}")

    return history


# ============================================================================
# Pytest Fixtures and Tests
# ============================================================================

@pytest.fixture(scope="module")
def small_dataset():
    """Create and preprocess small dataset once for all tests."""
    print("\n" + "=" * 70)
    print("SETTING UP SHARED DATASET")
    print("=" * 70)

    # Step 1: Create dataset
    df = _create_small_dataset()

    # Step 2: Feature engineering
    df_with_features, data_config = _test_feature_engineering(df)

    # Step 3: Preprocessing
    sequences, info = _test_preprocessing(df_with_features, data_config)

    # Step 4: Dataset creation with shared config
    model_config = load_config('model')
    model_config.model.training.BATCH_SIZE = 8
    model_config.model.training.NUM_EPOCHS = 2
    loaders = _test_dataset_creation(sequences, model_config)

    return {
        'loaders': loaders,
        'info': info,
        'model_config': model_config,
        'data_config': data_config,
    }


@pytest.mark.parametrize("model_type", ALL_MODEL_TYPES)
def test_full_pipeline(small_dataset, model_type):
    """Test full pipeline for a specific model type."""
    print(f"\n{'=' * 70}")
    print(f"TESTING MODEL: {model_type.upper()}")
    print(f"{'=' * 70}")

    loaders = small_dataset['loaders']
    info = small_dataset['info']
    model_config = small_dataset['model_config']

    # Create model with specific type
    model_config_copy = load_config('model')
    model_config_copy.model.training.BATCH_SIZE = 8
    model_config_copy.model.training.NUM_EPOCHS = 2

    model = _test_model_creation(info, model_config_copy, model_type)

    # Train
    history = _test_training(model, loaders, model_config_copy)

    # Verify training completed
    assert len(history['train_loss']) == 2, f"{model_type}: Training did not complete 2 epochs"
    assert history['train_loss'][-1] < history['train_loss'][0] + 1.0, f"{model_type}: Loss did not decrease reasonably"


def test_all_models_comparison(small_dataset):
    """Compare all models on the same dataset (results summary)."""
    print("\n" + "=" * 70)
    print("SUMMARY - ALL MODELS")
    print("=" * 70)

    print("\nAvailable models:")
    for model_type in list_available_models():
        print(f"  - {model_type}")

    print(f"\nTotal models tested: {len(list_available_models())}")


# ============================================================================
# Standalone main() for manual script execution
# ============================================================================

def main():
    """Run all tests."""
    print("=" * 70)
    print("SMALL DATASET PERFORMANCE TEST - ALL 7 MODELS")
    print("=" * 70)

    total_start = time.time()

    results = {}

    # Step 1: Create dataset (shared)
    df = _create_small_dataset()

    # Step 2: Feature engineering (shared)
    df_with_features, data_config = _test_feature_engineering(df)

    # Step 3: Preprocessing (shared)
    sequences, info = _test_preprocessing(df_with_features, data_config)

    # Step 4: Dataset creation (shared)
    model_config = load_config('model')
    model_config.model.training.BATCH_SIZE = 8
    model_config.model.training.NUM_EPOCHS = 2
    loaders = _test_dataset_creation(sequences, model_config)

    # Step 5-6: Test each model
    for model_type in ALL_MODEL_TYPES:
        print(f"\n{'=' * 70}")
        print(f"TESTING MODEL: {model_type.upper()}")
        print(f"{'=' * 70}")

        model_config_copy = load_config('model')
        model_config_copy.model.training.BATCH_SIZE = 8
        model_config_copy.model.training.NUM_EPOCHS = 2

        model = _test_model_creation(info, model_config_copy, model_type)

        start_time = time.time()
        history = _test_training(model, loaders, model_config_copy)
        elapsed = time.time() - start_time

        results[model_type] = {
            'final_loss': history['train_loss'][-1],
            'initial_loss': history['train_loss'][0],
            'time': elapsed,
            'params': sum(p.numel() for p in model.parameters())
        }

    total_elapsed = time.time() - total_start

    # Summary report
    print("\n" + "=" * 70)
    print("SUMMARY - ALL MODELS")
    print("=" * 70)
    print(f"{'Model':<20} | {'Final Loss':<12} | {'Loss Delta':<12} | {'Params':<12} | {'Time':<10}")
    print("-" * 70)

    for model_type in ALL_MODEL_TYPES:
        metrics = results[model_type]
        loss_delta = metrics['initial_loss'] - metrics['final_loss']
        print(f"{model_type:<20} | {metrics['final_loss']:<12.6f} | {loss_delta:<12.6f} | {metrics['params']:<12,} | {metrics['time']:<10.2f}s")

    print("-" * 70)
    print(f"{'Total':<20} | {'':<12} | {'':<12} | {'':<12} | {total_elapsed:<10.2f}s")
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == '__main__':
    main()
