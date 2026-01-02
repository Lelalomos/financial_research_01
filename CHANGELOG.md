# Changelog

All notable changes to the financial prediction system are documented in this file.

## [Version 2.0] - 2026-01-03

### Added

#### Dividend Flag Feature
- **Dividend flag embedding** to capture dividend payment status
  - Flag 1: Stock has dividend (DividendYield > 0 or DividendShare > 0)
  - Flag 2: Stock has no dividend (DividendYield = 0 or None)
  - 8-dimensional embedding for the binary flag
- Updated `src/data/financial_metrics_loader.py` to extract dividend flag from Highlights data
- Added `EMBEDDING_DIM_DIVIDEND_FLAG` to `config/model_config.py`
- Added `dividend_flag` to `config/data_config.py` financial metrics columns

#### Prediction System
- **Complete prediction pipeline** for inference on new data
  - `src/data/prediction_prep.py`: Data preparation module for prediction
  - `src/prediction/predictor.py`: Predictor class with model loading
  - `scripts/predict.py`: CLI script for predictions
- Supports 4 prediction modes:
  - **single**: Predict for a single stock on a specific date
  - **batch**: Predict from CSV/Parquet/Excel file
  - **interactive**: Interactive CLI for manual input
  - **info**: Display model checkpoint information

#### Full End-to-End Test
- **`tests/test_full_flow.py`**: Comprehensive pipeline test covering:
  1. Data preprocessing & feature engineering
  2. Train/val/test split & sequence creation
  3. Model training with early stopping
  4. Model validation
  5. Model testing
  6. Prediction on new data
  7. Backtesting simulation

#### Permission Fix Script
- **`scripts/fix_permissions.py`**: Handles root-owned files from Docker containers
  - Tries chmod/chown to fix permissions
  - Offers option to delete and regenerate files if needed

### Changed

#### Model Updates
- Updated ALL model forward passes to accept `dividend_flag` parameter:
  - `src/models/lstm3_attn_model.py`
  - `src/models/lstm3_model.py`
  - `src/models/crnn_attention.py`
  - `src/models/crnn_model.py`
  - `src/models/rnn_attention.py`
  - `src/models/rnn_model.py`
  - `src/models/transformer_model.py`
- Updated `src/training/trainer.py` to extract and pass `dividend_flag` from batches
- Updated `src/data/dataset.py` to include `dividend_flag` in sequences
- Updated `src/data/preprocessing.py` to exclude `dividend_flag` from normalization (treated as categorical)

#### Test Updates
- Updated `tests/test_models.py` to include `dividend_flag` in sample inputs
- Added `tests/test_prediction.py` with 12 tests for the prediction system

#### Configuration Changes
- Changed default model type from `crnn_attention` to `lstm3_attention`
- Changed validation split from 10% to 15% (70/15/15 split instead of 70/10/20)
- Updated total embedding dimension calculation to include dividend flag

### Fixed

- Fixed `torch.load()` in `src/prediction/predictor.py` to use `weights_only=False` parameter for PyTorch compatibility
- Fixed `tests/test_full_flow.py` to use correct `trainer.train()` API (not `trainer.fit()`)
- Fixed history dict access in full flow test - `train()` returns `{'train_loss': [...], 'val_loss': [...]}`
- Fixed undefined `n_stocks` variable in full flow test summary

### Test Results

All 81 tests passing:
```
================ 81 passed, 28327 warnings in 77.22s =================
```

Test breakdown:
- Data pipeline tests: 16 tests
- Model tests: 40 tests
- Training tests: 12 tests
- Prediction tests: 12 tests
- Full flow test: 1 test

## [Version 1.0] - 2025-12-XX

### Initial Release

#### Features
- Multiple model architectures (CRNN, RNN, RNN+Attention, CRNN+Attention, Transformer)
- Comprehensive feature engineering (technical indicators, candlestick patterns, external data)
- Time-based data splitting (70/10/20 train/val/test)
- Configurable prediction horizon (default: 5 days)
- Docker deployment with GPU support
- Comprehensive logging and TensorBoard integration
- Backtesting with performance metrics

#### Test Coverage
67 unit tests covering data pipeline, models, and training.
