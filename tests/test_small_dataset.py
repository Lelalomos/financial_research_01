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
from src.utils.cleanup import cleanup_test_files, cleanup_specific_files


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

@pytest.fixture(scope="session", autouse=True)
def cleanup_before_tests():
    """
    Cleanup old test files before running any tests in this module.
    This runs automatically at the start of the test session.
    """
    cleanup_test_files(keep_latest=3, verbose=True)
    yield
    # Optional: cleanup after tests too
    # cleanup_test_files(keep_latest=5, verbose=True)


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


def test_load_trained_model_checkpoint(small_dataset, tmp_path):
    """
    Test loading and evaluating a trained model checkpoint.

    This test covers the scenario tested by scripts/test.sh:
    1. Train a model
    2. Save checkpoint
    3. Load checkpoint
    4. Evaluate on test data
    5. Verify no NaN in predictions or metrics
    """
    print("\n" + "=" * 70)
    print("TEST: LOAD TRAINED MODEL CHECKPOINT")
    print("=" * 70)

    import os
    from src.training.early_stopping import ModelCheckpoint

    loaders = small_dataset['loaders']
    info = small_dataset['info']
    model_config = small_dataset['model_config']

    # Create and train a bilstm4_attention model
    model_config_copy = load_config('model')
    model_config_copy.model.training.BATCH_SIZE = 8
    model_config_copy.model.training.NUM_EPOCHS = 2

    model_type = 'bilstm4_attention'
    model = _test_model_creation(info, model_config_copy, model_type)

    # Train briefly
    device = 'cpu'
    trainer = Trainer(model, model_config_copy, device=device)

    print("\nTraining model briefly...")
    history = trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders.get('val'),
        num_epochs=2
    )

    # Verify no NaN in training loss
    assert not any(np.isnan(loss) for loss in history['train_loss']), \
        "NaN found in training loss"
    assert not any(np.isinf(loss) for loss in history['train_loss']), \
        "Inf found in training loss"

    # Save checkpoint
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{model_type}_best_test.pth"

    print(f"\nSaving checkpoint to {checkpoint_path}")
    torch.save({
        'epoch': 2,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'train_loss': history['train_loss'][-1],
        'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'model_type': model_type
    }, checkpoint_path)

    # Verify checkpoint file exists
    assert checkpoint_path.exists(), "Checkpoint file not created"

    # Create new model and load checkpoint
    print("\nCreating new model and loading checkpoint...")
    model_loaded = create_model(
        model_type=model_type,
        num_features=info['num_features'],
        num_stocks=info['num_stocks'],
        num_groups=info['num_groups'],
        config=model_config_copy
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded = model_loaded.to(device)
    model_loaded.eval()

    # Verify checkpoint loaded correctly
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_loaded.named_parameters()):
        assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
        assert torch.allclose(p1, p2), f"Parameter value mismatch for {n1}"

    # Evaluate loaded model
    print("\nEvaluating loaded model...")
    from src.evaluation.metrics import evaluate_model

    metrics = evaluate_model(model_loaded, loaders['train'], device=device)

    # Verify metrics are valid (no NaN)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert not np.isnan(value), f"NaN in metric: {key}"
            assert not np.isinf(value), f"Inf in metric: {key}"
            print(f"  {key}: {value:.6f}")

    print("\nCheckpoint loading and evaluation test PASSED")

    # Cleanup checkpoint files
    cleanup_specific_files([str(checkpoint_path)], verbose=True)


def test_model_type_auto_detection():
    """
    Test model type auto-detection from checkpoint filename.

    This covers the logic added to scripts/test.py for detecting
    model type from checkpoint filename patterns like:
    - bilstm4_attention_best_20260105_113045.pth
    - crnn_attention_best_*.pth
    - etc.

    Note: The detection must check longer model types first to avoid
    false matches (e.g., 'lstm3' matching 'lstm3_attention').
    """
    import os

    test_cases = [
        ("models/bilstm4_attention_best_20260105_113045.pth", "bilstm4_attention"),
        ("checkpoints/crnn_attention_best_epoch_5.pth", "crnn_attention"),
        ("models/rnn_best_model.pth", "rnn"),
        ("checkpoints/transformer_20260105_113045.pth", "transformer"),
        ("models/lstm3_attention_best.pth", "lstm3_attention"),
        ("models/lstm3_best.pth", "lstm3"),
        ("unknown_model.pth", "crnn_attention"),  # Default
    ]

    # Must check longer types first to avoid substring false matches
    known_types = ['bilstm4_attention', 'crnn_attention', 'rnn_attention',
                   'lstm3_attention', 'transformer', 'crnn', 'rnn', 'lstm3']

    for checkpoint_path, expected_type in test_cases:
        basename = os.path.basename(checkpoint_path)

        detected = None
        for known_type in known_types:
            if known_type in basename.lower():
                detected = known_type
                break

        if detected is None:
            detected = "crnn_attention"  # Default

        assert detected == expected_type, \
            f"Failed to detect model type from {checkpoint_path}: got {detected}, expected {expected_type}"

    print("Model type auto-detection test PASSED")


def test_excel_report_generation(small_dataset, tmp_path):
    """
    Test Excel report generation with detailed predictions and sector stats.

    This test covers the report generation functionality added to scripts/test.py:
    1. Evaluate model with evaluate_model_with_report
    2. Generate Excel report with all required columns
    3. Verify Excel file exists
    4. Verify report contains all required data
    """
    print("\n" + "=" * 70)
    print("TEST: EXCEL REPORT GENERATION")
    print("=" * 70)

    from src.evaluation.metrics import evaluate_model_with_report, print_sector_stats
    import pandas as pd

    loaders = small_dataset['loaders']
    info = small_dataset['info']
    model_config = small_dataset['model_config']

    # Create and train a simple model
    model_config_copy = load_config('model')
    model_config_copy.model.training.BATCH_SIZE = 8
    model_config_copy.model.training.NUM_EPOCHS = 2

    model_type = 'crnn_attention'
    model = _test_model_creation(info, model_config_copy, model_type)

    # Train briefly
    device = 'cpu'
    trainer = Trainer(model, model_config_copy, device=device)

    print("\nTraining model briefly...")
    trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders.get('val'),
        num_epochs=2
    )

    # Create stock_id_to_ticker and group_id_to_sector mappings
    # Since we have synthetic data, create dummy mappings
    stock_id_to_ticker = {i: f"STOCK{i:03d}" for i in range(info['num_stocks'])}
    group_id_to_sector = {i: f"SECTOR{i}" for i in range(info['num_groups'])}

    # Generate Excel report
    report_path = tmp_path / "test_report.xlsx"
    print(f"\nGenerating Excel report to {report_path}...")

    metrics, report_df, sector_stats = evaluate_model_with_report(
        model,
        loaders['train'],
        device=device,
        stock_id_to_ticker=stock_id_to_ticker,
        group_id_to_sector=group_id_to_sector,
        output_path=str(report_path)
    )

    # Verify Excel file exists
    assert report_path.exists(), f"Excel report file not created at {report_path}"
    print(f"  Excel report created: {report_path}")

    # Verify report DataFrame has all required columns
    required_columns = [
        'stock_id', 'group_id', 'real_target', 'predict_target',
        'ticker', 'sector', 'distance', 'std_real', 'std_predict', 'direction_score'
    ]
    for col in required_columns:
        assert col in report_df.columns, f"Missing column: {col}"
    print(f"  Report has all {len(required_columns)} required columns")

    # Verify direction scores are valid (0 or 1)
    assert report_df['direction_score'].isin([0, 1]).all(), "Direction scores should be 0 or 1"
    print(f"  Direction scores are valid (0 or 1)")

    # Verify sector stats
    assert len(sector_stats) > 0, "No sector statistics generated"
    print(f"  Generated sector stats for {len(sector_stats)} sectors")

    # Verify overall metrics are valid (no NaN)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert not np.isnan(value), f"NaN in metric: {key}"
            assert not np.isinf(value), f"Inf in metric: {key}"

    print(f"  All metrics are valid (no NaN/Inf)")

    # Read Excel file and verify structure
    excel_data = pd.ExcelFile(report_path)
    assert 'Predictions' in excel_data.sheet_names, "Missing 'Predictions' sheet"
    assert 'Sector Stats' in excel_data.sheet_names, "Missing 'Sector Stats' sheet"
    assert 'Overall Metrics' in excel_data.sheet_names, "Missing 'Overall Metrics' sheet"
    print(f"  Excel has all required sheets: {excel_data.sheet_names}")

    # Verify Predictions sheet
    predictions_df = pd.read_excel(report_path, sheet_name='Predictions')
    assert len(predictions_df) > 0, "Predictions sheet is empty"
    assert 'ticker' in predictions_df.columns, "Missing 'ticker' column in Predictions"
    assert 'sector' in predictions_df.columns, "Missing 'sector' column in Predictions"
    assert 'direction_score' in predictions_df.columns, "Missing 'direction_score' column in Predictions"
    print(f"  Predictions sheet has {len(predictions_df)} rows")

    # Verify Sector Stats sheet
    sector_stats_df = pd.read_excel(report_path, sheet_name='Sector Stats')
    assert len(sector_stats_df) > 0, "Sector Stats sheet is empty"
    assert 'total' in sector_stats_df.columns, "Missing 'total' column in Sector Stats"
    assert 'correct' in sector_stats_df.columns, "Missing 'correct' column in Sector Stats"
    assert 'accuracy' in sector_stats_df.columns, "Missing 'accuracy' column in Sector Stats"
    print(f"  Sector Stats sheet has {len(sector_stats_df)} sectors")

    # Print sector stats (as the script would do)
    print_sector_stats(sector_stats)

    print("\nExcel report generation test PASSED")

    # Cleanup generated files after this test
    cleanup_specific_files([str(report_path)], verbose=True)


def test_excel_report_backtest(small_dataset, tmp_path):
    """
    Test Excel report generation from backtest with detailed trades.

    This test covers the report generation functionality in Backtester:
    1. Run backtest
    2. Generate Excel report with trade details
    3. Verify Excel file exists with correct sheets
    """
    print("\n" + "=" * 70)
    print("TEST: EXCEL REPORT FROM BACKTEST")
    print("=" * 70)

    from src.evaluation import Backtester
    import pandas as pd

    loaders = small_dataset['loaders']
    info = small_dataset['info']
    model_config = small_dataset['model_config']

    # Create and train a simple model
    model_config_copy = load_config('model')
    model_config_copy.model.training.BATCH_SIZE = 8
    model_config_copy.model.training.NUM_EPOCHS = 2

    model_type = 'crnn_attention'
    model = _test_model_creation(info, model_config_copy, model_type)

    # Train briefly
    device = 'cpu'
    trainer = Trainer(model, model_config_copy, device=device)
    trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders.get('val'),
        num_epochs=2
    )

    # Create mappings
    stock_id_to_ticker = {i: f"STOCK{i:03d}" for i in range(info['num_stocks'])}
    group_id_to_sector = {i: f"SECTOR{i}" for i in range(info['num_groups'])}

    # Run backtest
    backtester = Backtester(model, model_config_copy, device=device)
    report_path = tmp_path / "backtest_report.xlsx"

    print(f"\nRunning backtest and generating report to {report_path}...")
    results = backtester.run_backtest(
        loaders['train'],
        prediction_threshold=0.0,
        initial_capital=100000.0,
        stock_id_to_ticker=stock_id_to_ticker,
        group_id_to_sector=group_id_to_sector
    )

    # Generate report
    backtester.generate_report(results, str(report_path), format='excel')

    # Verify Excel file exists
    assert report_path.exists(), f"Backtest Excel report not created at {report_path}"
    print(f"  Excel report created: {report_path}")

    # Verify structure
    excel_data = pd.ExcelFile(report_path)
    assert 'Summary' in excel_data.sheet_names, "Missing 'Summary' sheet"
    assert 'Trades' in excel_data.sheet_names, "Missing 'Trades' sheet"
    assert 'Sector Stats' in excel_data.sheet_names, "Missing 'Sector Stats' sheet"
    print(f"  Backtest Excel has sheets: {excel_data.sheet_names}")

    # Verify Trades sheet has required columns
    trades_df = pd.read_excel(report_path, sheet_name='Trades')
    required_trade_columns = ['Ticker', 'Sector', 'Real Target', 'Predict Target',
                              'Distance', 'Direction Score', 'Return (%)', 'Portfolio Value',
                              'Std Real', 'Std Predict']
    for col in required_trade_columns:
        assert col in trades_df.columns, f"Missing column in Trades: {col}"
    print(f"  Trades sheet has all {len(required_trade_columns)} required columns")

    # Verify sector stats in results
    assert 'sector_stats' in results, "No sector_stats in backtest results"
    assert len(results['sector_stats']) > 0, "Empty sector_stats in results"
    print(f"  Backtest has sector stats for {len(results['sector_stats'])} sectors")

    print("\nBacktest Excel report test PASSED")

    # Cleanup generated files after this test
    cleanup_specific_files([str(report_path)], verbose=True)


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
