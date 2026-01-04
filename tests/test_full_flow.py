"""
Full end-to-end flow test for the financial prediction system.

This test covers the complete workflow:
1. Data preprocessing & feature engineering
2. Train/val/test split
3. Model training with early stopping
4. Model validation
5. Model testing
6. Prediction/inference
7. Backtesting simulation

This ensures all components work together correctly.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import create_data_loaders
from src.models.lstm3_attn_model import create_model as create_lstm3_attn
from src.training import Trainer
from src.prediction.predictor import create_predictor
from src.utils.logger import get_logger
from src.data.sampling import sample_stocks_by_group, get_sampling_stats

logger = get_logger("full_flow_test", log_dir="logs")


def create_realistic_dataset(n_stocks=5, n_groups=3, n_days=400):
    """
    Create a realistic financial dataset for testing with balanced groups.

    Simulates stock prices with:
    - Trend component
    - Random walk
    - Seasonality
    - Volatility clustering
    - Balanced group distribution

    Args:
        n_stocks: Number of stocks to simulate
        n_groups: Number of groups to distribute stocks across
        n_days: Number of trading days

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Creating realistic dataset: {n_stocks} stocks, {n_groups} groups, {n_days} days")
    np.random.seed(42)

    # Calculate stocks per group (balanced)
    stocks_per_group = n_stocks // n_groups
    remaining = n_stocks % n_groups

    # Create tickers distributed across groups
    tickers = []
    ticker_to_group = {}
    for group_idx in range(n_groups):
        n_this_group = stocks_per_group + (1 if group_idx < remaining else 0)
        for i in range(n_this_group):
            ticker = f"STOCK{group_idx:02d}_{i:02d}"
            tickers.append(ticker)
            ticker_to_group[ticker] = group_idx

    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')

    data = []
    for ticker in tickers:
        group_id = ticker_to_group[ticker]

        # Initial price
        price = 100 + np.random.rand() * 50

        # Parameters for this stock
        trend = np.random.randn() * 0.001  # Daily trend
        volatility = 0.015 + np.random.rand() * 0.01  # Base volatility

        for i, date in enumerate(dates):
            # Add trend
            price = price * (1 + trend)

            # Add random walk with volatility clustering
            vol_multiplier = 1 + 0.3 * np.sin(i / 50)  # Volatility cycles
            change = np.random.randn() * volatility * vol_multiplier
            price = price * (1 + change)

            # Ensure positive price
            price = max(price, 10)

            # Generate OHLC
            high = price * (1 + abs(np.random.randn()) * 0.005)
            low = price * (1 - abs(np.random.randn()) * 0.005)
            open_price = low + (high - low) * np.random.rand()
            close = price
            volume = int(np.random.randint(1000000, 50000000))

            # Random dividend flag (20% chance of dividend)
            dividend_flag = 1 if np.random.rand() < 0.2 else 2

            data.append({
                'date': date,
                'tic': ticker,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'dividend_flag': dividend_flag,  # Include in raw data
                'group_id': group_id  # Use numeric group_id
            })

    df = pd.DataFrame(data)
    logger.info(f"Created dataset: {len(df)} rows, {len(tickers)} stocks, {n_groups} groups")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def test_full_pipeline():
    """
    Test the complete pipeline from data to predictions.

    Stages:
    1. Data preprocessing & feature engineering
    2. Train/val/test split
    3. Model training
    4. Model validation
    5. Model testing
    6. Prediction on new data
    7. Backtesting simulation
    """
    logger.info("=" * 70)
    logger.info("FULL END-TO-END PIPELINE TEST")
    logger.info("=" * 70)

    # Create temp directory for this test
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = Path(temp_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        # =====================================================================
        # STAGE 1: DATA PREPROCESSING & FEATURE ENGINEERING
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1: DATA PREPROCESSING & FEATURE ENGINEERING")
        logger.info("=" * 70)

        # Create dataset
        n_stocks = 5
        df = create_realistic_dataset(n_stocks=n_stocks, n_days=400)

        # Configure for faster training
        data_config = load_config('main')
        data_config.data.sequences.SEQUENCE_LENGTH = 30
        data_config.data.sequences.PREDICTION_HORIZON = 5
        data_config.data.splits.TRAIN_RATIO = 0.7
        data_config.data.splits.VAL_RATIO = 0.15
        data_config.data.splits.TEST_RATIO = 0.15

        model_config = load_config('model')
        model_config.model.training.NUM_EPOCHS = 3
        model_config.model.training.BATCH_SIZE = 32
        model_config.model.training.LEARNING_RATE = 0.001
        model_config.model.training.EARLY_STOPPING_PATIENCE = 2

        # Feature engineering
        engineer = FeatureEngineer(data_config)
        df = engineer.add_all_features(df, calculate_target=True)

        logger.info(f"Feature engineered data shape: {df.shape}")
        logger.info(f"Features: {engineer.get_feature_info(df)['total_features']}")

        # Preprocessing
        preprocessor = DataPreprocessor(data_config)
        df = preprocessor.encode_categorical(df, fit=True)
        df = preprocessor.normalize_features(df, fit=True)

        # Get feature columns (exclude non-numeric columns)
        feature_cols = [c for c in df.columns if c not in
                        ['date', 'tic', 'tic_id', 'group', 'group_id', 'target', 'split']]

        logger.info(f"Number of feature columns: {len(feature_cols)}")

        # =====================================================================
        # STAGE 2: TRAIN/VAL/TEST SPLIT & SEQUENCE CREATION
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: TRAIN/VAL/TEST SPLIT & SEQUENCE CREATION")
        logger.info("=" * 70)

        # Time-based split
        df_sorted = df.sort_values(['tic', 'date']).reset_index(drop=True)

        # Create splits manually for better control
        n_tickers = df_sorted['tic'].nunique()

        train_data = []
        val_data = []
        test_data = []

        for ticker in df_sorted['tic'].unique():
            ticker_df = df_sorted[df_sorted['tic'] == ticker]
            n = len(ticker_df)
            train_end = int(n * data_config.data.splits.TRAIN_RATIO)
            val_end = train_end + int(n * data_config.data.splits.VAL_RATIO)

            train_data.append(ticker_df.iloc[:train_end])
            val_data.append(ticker_df.iloc[train_end:val_end])
            test_data.append(ticker_df.iloc[val_end:])

        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)

        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Val samples: {len(val_df)}")
        logger.info(f"Test samples: {len(test_df)}")

        # Create sequences
        train_sequences = preprocessor.create_sequences(train_df, feature_cols)
        val_sequences = preprocessor.create_sequences(val_df, feature_cols)
        test_sequences = preprocessor.create_sequences(test_df, feature_cols)

        logger.info(f"Train sequences: {len(train_sequences['features'])}")
        logger.info(f"Val sequences: {len(val_sequences['features'])}")
        logger.info(f"Test sequences: {len(test_sequences['features'])}")

        # Add dividend_flag to sequences if not present
        for seq_dict in [train_sequences, val_sequences, test_sequences]:
            if 'dividend_flag' not in seq_dict:
                # Default to no dividend
                seq_len = seq_dict['features'].shape[1]
                seq_dict['dividend_flag'] = np.full(
                    (seq_dict['features'].shape[0], seq_len),
                    2,  # No dividend
                    dtype=np.int32
                )

        # =====================================================================
        # STAGE 3: MODEL TRAINING
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 3: MODEL TRAINING")
        logger.info("=" * 70)

        # Create data loaders
        data_loaders = create_data_loaders(
            train_sequences,
            val_sequences,
            test_sequences,
            config=model_config
        )

        # Create model
        num_features = train_sequences['features'].shape[2]
        num_stocks = int(train_df['tic_id'].max()) + 1
        num_groups = int(train_df['group_id'].max()) + 1

        logger.info(f"Model parameters:")
        logger.info(f"  Num features: {num_features}")
        logger.info(f"  Num stocks: {num_stocks}")
        logger.info(f"  Num groups: {num_groups}")

        model = create_lstm3_attn(
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
            config=model_config
        )

        # Create trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        trainer = Trainer(model, model_config, device=device)

        # Train
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
        )

        logger.info("Training completed!")

        # Get best validation loss (or last if early stopping didn't trigger)
        best_val_loss = min(history['val_loss']) if history['val_loss'] else history['train_loss'][-1]

        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Epochs trained: {len(history['train_loss'])}")

        # Save best model manually
        best_model_path = checkpoint_dir / "best_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': len(history['train_loss']),
            'loss': history['train_loss'][-1],
            'val_metrics': {'val_loss': history['val_loss'][-1] if history['val_loss'] else None},
            'score': best_val_loss,
        }, best_model_path)
        logger.info(f"Saved model to {best_model_path}")

        # =====================================================================
        # STAGE 4: MODEL VALIDATION
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 4: MODEL VALIDATION")
        logger.info("=" * 70)

        # Validate on validation set
        val_metrics = trainer.validate(data_loaders['val'])
        logger.info(f"Validation metrics:")
        logger.info(f"  Loss: {val_metrics['loss']:.6f}")
        logger.info(f"  MSE: {val_metrics['mse']:.6f}")
        logger.info(f"  MAE: {val_metrics['mae']:.6f}")
        logger.info(f"  RMSE: {val_metrics['rmse']:.6f}")
        logger.info(f"  Directional Accuracy: {val_metrics['directional_accuracy']:.4f}")

        # =====================================================================
        # STAGE 5: MODEL TESTING
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 5: MODEL TESTING")
        logger.info("=" * 70)

        test_metrics = trainer.validate(data_loaders['test'])
        logger.info(f"Test metrics:")
        logger.info(f"  Loss: {test_metrics['loss']:.6f}")
        logger.info(f"  MSE: {test_metrics['mse']:.6f}")
        logger.info(f"  MAE: {test_metrics['mae']:.6f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"  Directional Accuracy: {test_metrics['directional_accuracy']:.4f}")

        # =====================================================================
        # STAGE 6: PREDICTION ON NEW DATA
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 6: PREDICTION ON NEW DATA")
        logger.info("=" * 70)

        # Save model in format expected by predictor
        predictor_checkpoint_path = checkpoint_dir / "model_for_predictor.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': {
                'model_type': 'lstm3_attention',
                'epoch': len(history['train_loss']),
                'best_val_loss': best_val_loss,
                'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'val_metrics': val_metrics,
            },
            'num_features': num_features,
            'num_stocks': num_stocks,
            'num_groups': num_groups,
            'feature_cols': feature_cols,
        }, predictor_checkpoint_path)

        logger.info(f"Saved model for predictor: {predictor_checkpoint_path}")

        # Create predictor
        predictor = create_predictor(
            model_path=str(predictor_checkpoint_path),
            device=device
        )

        # Make predictions on test sequences
        test_predictions = predictor.predict(test_sequences)

        logger.info(f"Test predictions shape: {test_predictions.shape}")
        logger.info(f"Test predictions (denorm): mean={test_predictions.mean():.4f}%, std={test_predictions.std():.4f}%")

        # =====================================================================
        # STAGE 7: BACKTESTING SIMULATION
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 7: BACKTESTING SIMULATION")
        logger.info("=" * 70)

        # Simulate a simple trading strategy based on predictions
        # Use test set predictions

        # Get actual targets from test sequences
        actual_targets = test_sequences['target']

        # Ensure predictions and targets have same length
        min_len = min(len(test_predictions), len(actual_targets))
        test_predictions_aligned = test_predictions[:min_len]
        actual_targets_aligned = actual_targets[:min_len]

        # Trading strategy
        # If prediction > 0.5% -> BUY
        # If prediction < -0.5% -> SELL
        # Otherwise HOLD

        signals = np.where(test_predictions_aligned.flatten() > 0.5, 1,
                         np.where(test_predictions_aligned.flatten() < -0.5, -1, 0))

        # Calculate returns
        # Actual return is the target (percent change)
        actual_returns = actual_targets_aligned.flatten()

        # Strategy return (only when we have a position)
        strategy_returns = signals * actual_returns

        # Buy and hold return (always long)
        buy_and_hold_return = actual_returns.mean()

        # Strategy return
        strategy_return = strategy_returns.mean()

        # Hit rate (correct direction predictions)
        correct_direction = ((signals > 0) & (actual_returns > 0)) | ((signals < 0) & (actual_returns < 0))
        hit_rate = correct_direction.sum() / len(signals)

        logger.info(f"Backtesting results:")
        logger.info(f"  Buy & Hold return: {buy_and_hold_return:.4f}%")
        logger.info(f"  Strategy return: {strategy_return:.4f}%")
        logger.info(f"  Hit rate: {hit_rate:.4f}")

        # Count signals
        n_buy = (signals == 1).sum()
        n_sell = (signals == -1).sum()
        n_hold = (signals == 0).sum()

        logger.info(f"  BUY signals: {n_buy}")
        logger.info(f"  SELL signals: {n_sell}")
        logger.info(f"  HOLD signals: {n_hold}")

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("FULL PIPELINE TEST SUMMARY")
        logger.info("=" * 70)

        summary = {
            'data': {
                'n_stocks': n_stocks,
                'n_features': num_features,
                'train_sequences': len(train_sequences['features']),
                'val_sequences': len(val_sequences['features']),
                'test_sequences': len(test_sequences['features']),
            },
            'training': {
                'epochs': len(history['train_loss']),
                'best_val_loss': best_val_loss,
            },
            'validation': {
                'loss': val_metrics['loss'],
                'rmse': val_metrics['rmse'],
                'directional_accuracy': val_metrics['directional_accuracy'],
            },
            'test': {
                'loss': test_metrics['loss'],
                'rmse': test_metrics['rmse'],
                'directional_accuracy': test_metrics['directional_accuracy'],
            },
            'backtesting': {
                'buy_and_hold': buy_and_hold_return,
                'strategy_return': strategy_return,
                'hit_rate': hit_rate,
            }
        }

        # Save summary to JSON
        summary_path = Path(temp_dir) / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)

        logger.info(f"Test summary saved to: {summary_path}")
        logger.info("\nFull pipeline test completed successfully!")
        logger.info("=" * 70)

        return 0

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    sys.exit(test_full_pipeline())
