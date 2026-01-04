# Changelog

All notable changes to the financial prediction system are documented in this file.

## [Version 3.0] - 2026-01-04

### Added

#### JSON Configuration System
- **Complete migration from Python dataclass configs to JSON files**
  - `config/main.json`: Data sources, features, sequences, technical indicators
  - `config/model.json`: Model architecture with separate sections for each of 8 model types
  - `config/hyperparameter.json`: Hyperparameter search settings (NEW)
  - `config/test.json`: Testing configuration
  - `config/deploy.json`: Deployment configuration (API section removed)
  - `config/validate.json`: Validation configuration
- **`src/config/config_loader.py`**: Config class for JSON-based configuration with attribute access
- **`tests/test_dynamic_config.py`**: 9 new tests for dynamic config lists (COMMODITIES, TREASURY_YIELDS, EMA_PERIODS)

#### Dynamic Configuration Lists
- COMMODITIES dict can be modified at runtime (add/remove commodities)
- TREASURY_YIELDS list can be modified at runtime (add/remove treasury yields)
- EMA_PERIODS list can be modified at runtime (add/remove EMA periods)

#### Model Configuration Structure
- Each model type now has its own dedicated config section under `model.models.{model_type}`
- Added missing config keys: `FC_USE_BATCH_NORM`, `USE_LAYER_NORM`, `USE_ATTENTION`, `CNN_USE_BATCH_NORM`, `LSTM3_ATTENTION_HIDDEN_SIZE`, `DROPOUT_EMBEDDING`

### Changed

#### Configuration Access Patterns
- Old: `config.model.architecture.*`
- New: `config.model.models.{model_type}.*` (model-specific parameters)
- Old: `config.model.training.*` → New: `config.model.training.*` (unchanged)
- Old: `load_config('model').model.architecture.MODEL_TYPE` → New: `model_type` passed as parameter to `create_model()`

#### Hyperparameter Configuration
- Moved from `config/model.json` to separate `config/hyperparameter.json`
- Updated `src/hyperparameter/optimizer.py` to use separate `hparam_config`
- Updated `scripts/optuna_tune.py` to load from `hyperparameter.json`

#### Embedding Layer
- `EmbeddingLayer` now uses shared `config.model.embeddings.DROPOUT_EMBEDDING` instead of model-specific setting

#### Updated Files (43 total)
- All 8 model files (`src/models/*.py`)
- `src/models/__init__.py`
- `src/hyperparameter/optimizer.py`
- `src/config/config_loader.py`
- `src/config/__init__.py`
- `src/prediction/predictor.py`
- `src/data/prediction_prep.py`
- All test files referencing config
- All scripts (train.py, validate.py, test.py, backtest.py, optuna_tune.py)
- Documentation files (README.md, docs/configuration.md, docs/data_pipeline.md, docs/model_architecture.md)

### Removed
- `config/data_config.py` (migrated to `config/main.json`)
- `config/model_config.py` (migrated to `config/model.json`)
- API section from `config/deploy.json` (unused)

### Test Results

All 131 tests passing (including 9 new dynamic config tests):
```
=========== 131 passed, 2 skipped, 4821 warnings in 80.39s ===========
```

Test breakdown:
- Data pipeline tests: 16 tests
- Model tests: 52 tests
- Training tests: 12 tests
- Prediction tests: 12 tests
- Optuna/hyperparameter tests: 20 tests
- Dynamic config tests: 9 tests (NEW)
- Full flow test: 1 test
- Small dataset test: 9 tests

## [Version 2.0] - 2026-01-03

### Added

#### Dividend Flag Feature
- **Dividend flag embedding** to capture dividend payment status
  - Flag 1: Stock has dividend (DividendYield > 0 or DividendShare > 0)
  - Flag 2: Stock has no dividend (DividendYield = 0 or None)
  - 8-dimensional embedding for the binary flag
- Updated `src/data/financial_metrics_loader.py` to extract dividend flag from Highlights data
- Added `EMBEDDING_DIM_DIVIDEND_FLAG` to config
- Added `dividend_flag` to financial metrics columns

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

### Changed

#### Model Updates
- Updated ALL model forward passes to accept `dividend_flag` parameter
- Updated `src/training/trainer.py` to extract and pass `dividend_flag` from batches
- Updated `src/data/dataset.py` to include `dividend_flag` in sequences
- Updated `src/data/preprocessing.py` to exclude `dividend_flag` from normalization

#### Configuration Changes
- Changed default model type to `lstm3_attention`
- Changed validation split to 15% (70/15/15 split)
- Updated total embedding dimension calculation to include dividend flag

### Fixed

- Fixed `torch.load()` in `src/prediction/predictor.py` for PyTorch compatibility
- Fixed `tests/test_full_flow.py` to use correct `trainer.train()` API
- Fixed history dict access in full flow test

### Test Results

All 81 tests passing:
```
================ 81 passed, 28327 warnings in 77.22s =================
```

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
