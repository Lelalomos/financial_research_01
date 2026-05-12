# Multi-Model Financial Forecasting

An AI-assisted multi-model financial forecasting platform built with PyTorch. It learns patterns from S&P 500 stocks using deep learning models, technical indicators, geometric price-structure features, candlestick patterns, and external market data.

## Development Approach

**This project is developed with AI assistance (Claude Code) under human direction and command.**

The development process follows a human-AI collaborative approach:
- **Human Role**: Provides commands, requirements, architectural decisions, and high-level direction
- **AI Role**: Writes code, implements features, runs tests, and handles technical implementation details

### Key Features of Development Method

1. **Human-in-the-loop Development**
   - All code changes are initiated by human command
   - AI suggests implementations which human reviews and approves
   - Human provides testing requirements and validates results

2. **Comprehensive Testing**
   - All code changes are tested automatically after implementation
   - Tests run in isolated Docker container environment
   - Latest recorded full validation: 246 tests passed in Docker

3. **Code Quality**
   - Follows Python best practices and PEP 8 standards
   - Type hints for better code clarity
   - Comprehensive docstrings for all modules and functions
   - Proper error handling and logging

## Features

- **Multiple Model Architectures**: CRNN, RNN, RNN+Attention, CRNN+Attention, Transformer, LSTM3, LSTM3+Attention, **BiLSTM4+Attention**
- **Comprehensive Feature Engineering**:
  - Technical indicators (EMA, RSI, StochRSI, MACD)
  - Geometric / structural features (normalized ATR, ROC, Bollinger Band width, support/resistance slopes)
  - **Fibonacci retracement levels** (38.2%, 50%, 61.8%) with normalized distance features
  - 100+ candlestick patterns via TA-Lib, with optional sparse-pattern exclusion
  - External data (VIX, commodities, treasury yields)
  - Time-based features (day, month embeddings)
  - **Dividend flag feature** (1=has dividend, 2=no dividend) with embedding
  - Financial metrics (PE ratio, PEG ratio, EPS, ROE, ROI, debt ratios, current ratio)
  - Optional market regime feature (`regime_id`) fit on train dates only
- **Typed Configuration Validation** with Pydantic while preserving JSON configs
- **Log Transform Normalization** for all features
- **Time-based Data Splitting** (70% train, 10% val, 20% test)
- **Walk-forward and purged time-series split utilities**
- **Custom Financial Losses** (directional, directional-MSE, Sharpe-style)
- **Model Ensembling** with checkpoint compatibility checks
- **Meta-labeling Utilities** for offline second-stage trade-filter analysis
- **Interpretability Utilities**:
  - attention weight extraction for attention models
  - optional Captum-based feature attribution
- **Balanced Stock Sampling** (`--stocks` option) - Sample stocks evenly across all group_ids
- **Configurable Prediction Horizon** (default: 5 days)
- **Docker Deployment** with GPU support
- **Comprehensive Logging** and TensorBoard integration
- **Backtesting** with performance metrics, turnover, and transaction-cost reporting
- **Prediction System** with support for single/batch/interactive prediction
- **Hyperparameter Tuning** with Optuna (automated search for best parameters)

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository_url>
cd research_02

# Install dependencies
pip install -r requirements.txt

# Or install in editable mode
pip install -e .
```

### Usage

```bash
# Run full pipeline (preprocess -> train -> validate -> test -> backtest)
python scripts/run_all.py --model-type lstm3_attention --epochs 100

# Or run individual steps

# 1. Preprocess data (with --stocks option for balanced sampling)
python scripts/preprocess_data.py --start-date 2015-01-01
python scripts/preprocess_data.py --stocks 50  # Sample 50 stocks balanced across all groups

# 2. Train model
python scripts/train.py --model-type lstm3_attention --epochs 100

# 3. Validate
python scripts/validate.py --model best

# 4. Test
python scripts/test.py --model best

# 5. Backtest
python scripts/backtest.py --model best --output outputs/report.xlsx

# 6. Predict (single/batch/interactive)
python scripts/predict.py interactive --model models/checkpoints/best_model.pth
python scripts/predict.py batch --model models/checkpoints/best_model.pth --input data/new_data.csv
python scripts/predict.py single --model models/checkpoints/best_model.pth --ticker AAPL --date 2024-01-15 --input "open=150,high=152,low=149,close=151,volume=50000000"

# 7. Hyperparameter tuning (NEW)
bash scripts/optuna_tune.sh --model-type bilstm4_attention --n-trials 50
python scripts/optuna_tune.py --model-type bilstm4_attention --n-trials 50 --stocks 20
```

### Quick Test with Small Dataset

```bash
# Test the full pipeline with a small dataset
python tests/test_small_dataset.py

# Run comprehensive end-to-end test
python tests/test_full_flow.py

# Run all unit tests
pytest tests/ -v

# Latest recorded full validation in Docker
docker exec crnn_predictor pytest -q

# Test stock sampling (NEW)
pytest tests/test_sampling.py -v

# Test hyperparameter tuning
pytest tests/test_optuna_tune.py -v
```

### Training Monitoring

Use the shell wrapper when you want TensorBoard and optional local MLflow
during training:

```bash
./scripts/train.sh --model-type bilstm4_attention --monitor
./scripts/train.sh --model-type bilstm4_attention --mlflow
./scripts/train.sh --model-type bilstm4_attention --monitor-all
```

- `--monitor`: start TensorBoard
- `--mlflow`: start the local MLflow UI
- `--monitor-all`: start both

Default local outputs:
- TensorBoard logs: `logs/tensorboard`
- MLflow runs: `mlruns/`

Key validation health signals to watch during training:
- `val/pred_positive_rate`
- `val/pred_negative_rate`
- `val/pred_std`
- `val/pred_target_corr`
- `val/collapse_penalty`
- `val/is_collapsed`

See [docs/run_guide.md](docs/run_guide.md) for wrapper usage and
[docs/experiment_tracking.md](docs/experiment_tracking.md) for local MLflow
setup details.

## Project Structure

```
research_02/
├── config/
│   ├── main.json            # Data configuration (sources, features, sequences)
│   ├── model.json           # Model configuration (all model types)
│   ├── hyperparameter.json  # Hyperparameter search settings
│   ├── test.json            # Testing configuration
│   ├── deploy.json          # Deployment configuration
│   └── validate.json        # Validation configuration
├── src/
│   ├── config/
│   │   ├── config_loader.py # JSON config loader with Config class
│   │   ├── schemas.py       # Pydantic validation for known configs
│   │   └── __init__.py       # Convenience functions
│   ├── data/
│   │   ├── downloader.py    # Data downloading (yfinance, FRED)
│   │   ├── feature_engineering.py  # Technical indicators
│   │   ├── preprocessing.py  # Normalization, splitting
│   │   ├── financial_metrics_loader.py  # Financial metrics from JSON
│   │   ├── prediction_prep.py  # Data preparation for prediction
│   │   ├── regime.py         # Optional train-only market regime detection
│   │   ├── sampling.py       # Balanced stock sampling
│   │   ├── time_series_split.py # Walk-forward and purged splits
│   │   ├── validation.py     # Dataset column validation
│   │   └── dataset.py        # PyTorch Dataset
│   ├── models/
│   │   ├── crnn_attention.py # CNN + BiLSTM + Attention (base module)
│   │   ├── crnn_model.py     # CNN + BiLSTM
│   │   ├── rnn_model.py      # BiLSTM only
│   │   ├── rnn_attention.py  # BiLSTM + Attention
│   │   ├── lstm3_model.py    # 3-layer BiLSTM
│   │   ├── lstm3_attn_model.py # 3-layer BiLSTM + Attention
│   │   ├── bilstm4_attn_model.py # 4-layer BiLSTM + Attention
│   │   └── transformer_model.py  # Transformer
│   ├── hyperparameter/       # Hyperparameter tuning
│   │   └── optimizer.py       # Optuna optimizer
│   ├── training/
│   │   ├── trainer.py       # Training loop
│   │   ├── losses.py        # Custom financial losses
│   │   └── early_stopping.py
│   ├── prediction/
│   │   ├── predictor.py     # Prediction module
│   │   └── ensemble.py      # Optional ensemble prediction
│   ├── evaluation/
│   │   ├── metrics.py       # Evaluation metrics
│   │   ├── validator.py     # Validation
│   │   ├── backtester.py    # Backtesting
│   │   ├── meta_labeling.py # Offline meta-label generation
│   │   ├── interpretability.py # Attention reports
│   │   └── feature_attribution.py # Optional Captum attribution
│   └── utils/
│       └── logger.py        # Logging utilities
├── scripts/
│   ├── preprocess_data.py
│   ├── create_hparam_dataset.py  # Create small dataset for hparam tuning
│   ├── optuna_tune.py       # Optuna hyperparameter tuning
│   ├── optuna_tune.sh       # Shell wrapper for hparam tuning
│   ├── train.py
│   ├── test.py
│   ├── validate.py
│   ├── backtest.py
│   ├── predict.py          # Prediction CLI
│   └── run_all.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_sampling.py      # Stock sampling tests
│   ├── test_optuna_tune.py  # Hyperparameter tuning tests
│   ├── test_dynamic_config.py # Dynamic config list tests
│   ├── test_training.py
│   ├── test_prediction.py   # Prediction tests
│   ├── test_ensemble.py     # Ensemble prediction tests
│   ├── test_regime.py       # Market regime tests
│   ├── test_meta_labeling.py # Meta-labeling tests
│   ├── test_interpretability.py # Attention report tests
│   ├── test_feature_attribution.py # Attribution tests
│   ├── test_validation.py   # Dataset column validation tests
│   ├── test_data_download.py # Data download integration tests
│   ├── test_small_dataset.py
│   └── test_full_flow.py    # End-to-end test
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Data Sources

| Source | Description |
|--------|-------------|
| **S&P 500 Stocks** | yfinance (live download) |
| **Technical Indicators** | EMA 50/200, RSI, StochRSI, MACD |
| **Geometric Features** | Normalized ATR, ROC, Bollinger Band width, support/resistance slopes |
| **Candlestick Patterns** | 100+ patterns via TA-Lib |
| **Financial Metrics** | PE ratio, PEG ratio, EPS, ROE, ROI, debt ratios, current ratio, dividend yield |
| **VIX Index** | Volatility index (^VIX) |
| **Commodities** | Gold, Copper, Corn, Soybeans, Cocoa, Silver |
| **Treasury Yields** | 2Y, 10Y, 30Y from FRED |

### Financial Metrics Data

The model loads financial metrics from JSON files in `raw_data/ticket_data/us/{TICKER}.json`. These files contain:

**From Highlights Section (single values):**
- `PERatio`: Price-to-Earnings ratio
- `PEGRatio`: PEG ratio
- `DilutedEpsTTM`: Earnings Per Share (diluted, trailing twelve months)
- `DividendYield`: Annual dividend yield (used to calculate dividend_flag)

**From Financials Section (quarterly data):**
- Balance Sheet: `totalAssets`, `totalLiab`, `totalStockholderEquity`, `totalCurrentAssets`, `totalCurrentLiabilities`
- Income Statement: `netIncome`

**Calculated Metrics:**
- `ROE` = netIncome / totalStockholderEquity
- `ROI` = netIncome / totalAssets
- `Debt-to-Equity` = totalLiab / totalStockholderEquity
- `Debt-to-Asset` = totalLiab / totalAssets
- `Current Ratio` = totalCurrentAssets / totalCurrentLiabilities

The quarterly metrics are forward-filled to daily frequency to match price data.

### Dataset Column Validation

The project includes automatic column validation to ensure required data is present:

**Required Columns** (must be present):
- Identifiers: `date`, `tic`
- Price Data: `open`, `high`, `low`, `close`, `volume`
- Target: `target`

**Optional Columns** (warn if missing):
- Financial Metrics: `pe_ratio`, `peg_ratio`, `eps`, `dividend_flag`, `roe`, `roi`, `debt_to_equity`, `debt_to_asset`, `current_ratio`
- Fibonacci Features: `swing_high`, `swing_low`, `fib_range`, `fib_38`, `fib_50`, `fib_61`, `dist_fib_38`, `dist_fib_50`, `dist_fib_61`, `break_fib_61`
- Time Features: `day`, `month`, `dayofweek`
- External Data: `vix`, `bondyield`
- Market Regime: `regime_id`
- Grouping: `group`
- Encoded: `tic_id`, `group_id`

**Configuration** (`config/main.json`):
```json
{
  "data": {
    "validation": {
      "REQUIRED_COLUMNS": {
        "identifiers": ["date", "tic"],
        "price_data": ["open", "high", "low", "close", "volume"],
        "target": ["target"]
      },
      "OPTIONAL_COLUMNS": {
        "financial_metrics": ["pe_ratio", "eps", "roe", ...],
        "time_features": ["day", "month", "dayofweek"]
      },
      "VALIDATE_ON_LOAD": true,
      "WARN_ON_MISSING_OPTIONAL": true
    }
  }
}
```

**Why Financial Metrics Might Be Missing:**
1. `raw_data/ticket_data/us/` directory not mounted in Docker
2. JSON files don't exist for the tickers
3. JSON files don't contain the required `Highlights` or `Financials` sections
4. `financial_metrics` feature flag is disabled in config

To enable the `raw_data` volume mount, ensure `docker-compose.yml` includes:
```yaml
volumes:
  - ./raw_data:/app/raw_data
```

## Model Architecture

```
Input (batch, seq_len, features)
    ↓
Embedding Layer (stock, group, day, month, dividend_flag)
    ↓
CNN Feature Extraction (for CRNN models)
    ↓
BiLSTM (2-3 layers, bidirectional)
    ↓
MultiheadAttention (4 heads) (for Attention models)
    ↓
Fully Connected Layers
    ↓
Output (batch, 1) - Percent change prediction
```

### Embedding Features

All models support the following categorical embeddings:
- **stock_id**: Unique identifier for each stock (64 dimensions)
- **group_id**: Sector/group classification (32 dimensions)
- **day**: Day of month (16 dimensions)
- **month**: Month of year (16 dimensions)
- **dividend_flag**: Dividend status - 1=has dividend, 2=no dividend (8 dimensions)

## Advanced Evaluation and Interpretability

The roadmap implementation has added several optional analysis utilities:

| Utility | Purpose | Docs |
|---------|---------|------|
| Model ensembling | Average compatible checkpoint predictions | `docs/roadmap_execution_plan.md` |
| Market regime detection | Optional `regime_id` fit on train dates only | `docs/data_pipeline.md` |
| Meta-labeling | Build offline correctness labels for second-stage analysis | `docs/meta_labeling.md` |
| Attention reports | Extract and aggregate attention weights | `docs/interpretability.md` |
| Feature attribution | Optional Captum-based feature importance reports | `docs/feature_attribution.md` |

These utilities are conservative by default. Ensembling and market regimes must
be enabled in config, and Captum is imported only when attribution is requested.
Interpretability and attribution reports are diagnostic evidence, not causal
proof or automatic feature-removal decisions.

## Workflow & Process

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Download S&P 500 stock data (yfinance)                               │
│  2. Download VIX index (^VIX)                                            │
│  3. Download commodities (Gold, Silver, Copper, etc.)                    │
│  4. Download treasury yields (FRED: 2Y, 10Y, 30Y)                        │
│  5. Load financial metrics (PE, EPS, ROE, etc.) from JSON files         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPTIONAL: STOCK SAMPLING                             │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/preprocess_data.py --stocks 50                          │
│                                                                         │
│  • Automatically balances stocks across ALL group_ids (sectors)         │
│  • Example: 50 stocks across 11 groups = ~4-5 stocks per group          │
│  • Uses seed=42 for reproducibility                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Price Features:   OHLCV, returns, log_returns                          │
│  EMA:              50, 100, 200 period exponential moving averages       │
│  RSI:              14-period Relative Strength Index                     │
│  StochRSI:         Stochastic RSI                                        │
│  MACD:             (12, 26, 9) parameters                                │
│  Fibonacci:        Optional swing high/low, retracement levels,          │
│                    normalized distance features, break indicators        │
│  Candlestick:      100+ patterns (Doji, Hammer, Engulfing, etc.)        │
│  VIX:              Volatility index                                      │
│  Commodities:      Gold, Silver, Copper prices                          │
│  Treasury Yields:  2Y, 10Y, 30Y rates                                   │
│  Financial Metrics: PE, PEG, EPS, ROE, ROI, Debt ratios, Dividend       │
│  Time Features:    Day of month, Month of year (for embeddings)         │
│  Market Regime:    Optional train-only quantile regime_id from VIX       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING & NORMALIZATION                         │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Split First:     Global date split before fitting normalization      │
│  2. Log Transform:   Fit on train only, transform val/test               │
│  3. Train/Val/Test Split: 70% / 10% / 20% (time-based)                  │
│  4. Sequence Creation:  Create sliding windows (seq_len=30)              │
│  5. Optional Regime: Fit train-only thresholds, apply to all splits      │
│  6. Target Calculation: future_return = price[t+5] / price[t] - 1       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/train.py --model-type lstm3_attention --epochs 100     │
│                                                                         │
│  • Load model architecture from config/model.json                       │
│  • Create embeddings (stock, group, day, month, dividend_flag)          │
│  • Training loop with early stopping                                    │
│  • Save best checkpoint based on validation loss                        │
│  • TensorBoard logging for visualization                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/validate.py --model best                                │
│                                                                         │
│  • Evaluate on validation set (10% of data)                             │
│  • Calculate metrics: MSE, RMSE, MAE, R², Directional Accuracy          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           TESTING                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/test.py --model best                                    │
│                                                                         │
│  • Evaluate on test set (20% of data)                                   │
│  • Final performance metrics                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKTESTING                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/backtest.py --model best --output report.xlsx          │
│                                                                         │
│  • Simulate trading strategy on test data                               │
│  • Calculate: Sharpe Ratio, Sortino Ratio, Max Drawdown                 │
│  • Calculate: Win Rate, Profit Factor                                   │
│  • Generate Excel report with trade details                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       PREDICTION (INFERENCE)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/predict.py <mode> --model best                          │
│                                                                         │
│  Modes:                                                                 │
│  • single:   Predict for 1 stock on 1 date                              │
│  • batch:    Predict from CSV/Parquet/Excel file                        │
│  • interactive: Manual input via CLI                                    │
│  • info:     Display model checkpoint information                       │
│  • ensemble: Enable through config/model.json, not a separate mode       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER TUNING (OPTIONAL)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/optuna_tune.py --n-trials 50 --stocks 20               │
│                                                                         │
│  • Automated hyperparameter search with Optuna                          │
│  • Searches: learning_rate, hidden_size, dropout, seq_length, etc.      │
│  • Uses smaller dataset (20 stocks) for faster trials                   │
│  • Outputs: best_hyperparameters.json                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference Commands

```bash
# Full pipeline (one command)
python scripts/run_all.py --model-type lstm3_attention --epochs 100

# Step-by-step
python scripts/preprocess_data.py --stocks 50      # Sample + preprocess
python scripts/train.py --model-type lstm3_attention --epochs 100
python scripts/validate.py --model best
python scripts/test.py --model best
python scripts/backtest.py --model best

# Hyperparameter tuning
python scripts/optuna_tune.py --model-type bilstm4_attention --n-trials 50

# Make predictions
python scripts/predict.py interactive --model best
```

## Configuration

The project uses **JSON-based configuration** with a Python Config wrapper class for easy access.
Known configs are validated with Pydantic at load time while preserving
backward-compatible dot notation.

### Configuration Files

| File | Purpose |
|------|---------|
| `config/main.json` | Data sources, features, sequences, technical indicators |
| `config/model.json` | Model architecture with separate sections for each model type |
| `config/hyperparameter.json` | Hyperparameter search settings (Optuna) |
| `config/test.json` | Testing configuration |
| `config/validate.json` | Validation configuration |
| `config/deploy.json` | Deployment configuration |

### Loading Configuration

```python
from src.config import load_config

# Load any config file
main_config = load_config('main')
model_config = load_config('model')
hparam_config = load_config('hyperparameter')

# Access values with dot notation
start_date = main_config.data.sources.START_DATE
seq_length = main_config.data.sequences.SEQUENCE_LENGTH
learning_rate = model_config.model.training.LEARNING_RATE

# Modify values at runtime
main_config.data.sources.COMMODITIES._data['ZW=F'] = 'Wheat'  # Add commodity
main_config.data.sources.TREASURY_YIELDS.append('DGS5')        # Add treasury yield
main_config.data.technical_indicators.EMA_PERIODS.append(20)   # Add EMA period
```

### Key Configuration Options

#### Data Configuration (`config/main.json`)

```json
{
  "data": {
    "sources": {
      "START_DATE": "2000-01-01",
      "END_DATE": null,
      "COMMODITIES": {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "HG=F": "Copper"
      },
      "TREASURY_YIELDS": ["DGS10", "DGS30", "DGS2"]
    },
    "splits": {
      "TRAIN_RATIO": 0.7,
      "VAL_RATIO": 0.1,
      "TEST_RATIO": 0.2
    },
    "sequences": {
      "SEQUENCE_LENGTH": 30,
      "PREDICTION_HORIZON": 5
    },
    "technical_indicators": {
      "EMA_PERIODS": [50, 200],
      "RSI_PERIOD": 14
    },
    "fibonacci": {
      "FIBONACCI_WINDOW": 30
    },
    "regime": {
      "ENABLED": false,
      "METHOD": "quantile",
      "PROXY_COLUMN": "vix",
      "N_REGIMES": 3
    }
  }
}
```

#### Model Configuration (`config/model.json`)

```json
{
  "model": {
    "embeddings": {
      "EMBEDDING_DIM_STOCK": 64,
      "EMBEDDING_DIM_GROUP": 32,
      "EMBEDDING_DIM_DAY": 16,
      "EMBEDDING_DIM_MONTH": 16,
      "EMBEDDING_DIM_DIVIDEND_FLAG": 8,
      "DROPOUT_EMBEDDING": 0.2
    },
    "training": {
      "LEARNING_RATE": 0.0001,
      "BATCH_SIZE": 128,
      "NUM_EPOCHS": 30,
      "EARLY_STOPPING_PATIENCE": 15
    },
    "loss": {
      "LOSS_TYPE": "directional_mse",
      "DIRECTIONAL_ALPHA": 0.1
    },
    "ensemble": {
      "ENABLED": false,
      "CHECKPOINT_PATHS": [],
      "WEIGHTS": null
    },
    "models": {
      "lstm3_attention": {
        "LSTM3_HIDDEN_SIZE": 256,
        "LSTM3_NUM_LAYERS": 3,
        "LSTM3_ATTENTION_HEADS": 8
      },
      "bilstm4_attention": {
        "LSTM4_HIDDEN_SIZES": [128, 256, 512, 256],
        "LSTM4_ATTENTION_HEADS": 4
      }
      // ... other models
    }
  }
}
```

The current default training profile favors directional learning:

- `LEARNING_RATE = 0.0001`
- `BATCH_SIZE = 128`
- `NUM_EPOCHS = 30`
- `LOSS_TYPE = "directional_mse"`

Lightning is the default training backend. The saved `*_best_lightning.pth`
checkpoint now includes validation prediction-health diagnostics, and best
checkpoint selection penalizes obviously collapsed one-sided validation output.

#### Hyperparameter Configuration (`config/hyperparameter.json`)

```json
{
  "hyperparameter": {
    "N_TRIALS": 50,
    "HPARAM_STOCKS": 20,
    "LEARNING_RATE_RANGE": [0.00001, 0.001],
    "LSTM_HIDDEN_SIZE_RANGE": [64, 512],
    "SEQUENCE_LENGTH_CHOICES": [20, 30, 60, 90]
  }
}
```

### Dynamic Configuration Lists

These lists can be modified at runtime without changing the JSON file:

- **COMMODITIES**: Add/remove commodity futures (e.g., Gold, Silver, Copper)
- **TREASURY_YIELDS**: Add/remove FRED treasury yield series
- **EMA_PERIODS**: Add/remove EMA periods for technical indicators

```python
# Example: Add new commodity
config = load_config('main')
config.data.sources.COMMODITIES._data['ZW=F'] = 'Wheat'
```

## Documentation Index

| Document | Purpose |
|----------|---------|
| `docs/configuration.md` | Configuration guide |
| `docs/data_pipeline.md` | Data pipeline and optional regime feature |
| `docs/model_architecture.md` | Model architecture reference |
| `docs/meta_labeling.md` | Meta-labeling usage and leakage guidance |
| `docs/interpretability.md` | Attention report usage |
| `docs/feature_attribution.md` | Optional Captum attribution usage |
| `docs/roadmap_execution_plan.md` | Roadmap status and next tasks |
| `docs/implementation_log.md` | Completed work and validation history |
| `docs/gpu_troubleshooting.md` | GPU diagnostics and fixes |

## Docker Deployment

### Quick Start

```bash
# Build image
docker-compose build

# Run full pipeline
docker-compose run --rm crnn-prediction

# Run with GPU
docker-compose --profile gpu run --rm trainer

# Jupyter notebook
docker-compose up jupyter
# Access at http://localhost:8888
```

### GPU Setup and Testing

The project includes comprehensive GPU support with automatic detection and testing.

#### Testing GPU Activation

A dedicated script is available to test GPU activation in Docker containers:

```bash
# Test GPU in the running container (container ID or name)
./scripts/test_gpu_activation.sh
```

This script performs the following checks:
1. Container status verification
2. NVIDIA driver and nvidia-smi availability
3. CUDA library detection
4. PyTorch CUDA availability
5. GPU device detection and information
6. GPU memory allocation tests
7. Comprehensive unit tests

#### Manual GPU Verification

```bash
# Check if nvidia-smi works in container
docker exec crnn_predictor nvidia-smi

# Test PyTorch CUDA
docker exec crnn_predictor python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run GPU detection unit tests
docker exec crnn_predictor python3 -m pytest tests/test_gpu_detection.py -v
```

#### GPU Configuration

The GPU is configured through environment variables and Docker device requests:

```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0  # GPU device to use
  - TORCH_CUDA_ARCH_LIST=8.0 8.6 8.9 9.0+PTX  # CUDA compute capabilities

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

#### Common GPU Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **CUDA not available** | `torch.cuda.is_available()` returns `False` | Ensure nvidia-docker2 is installed and restart Docker |
| **No GPU detected** | `torch.cuda.device_count()` returns `0` | Check `CUDA_VISIBLE_DEVICES` and verify GPU is visible on host |
| **CUDA runtime error** | Tensor operations fail | Verify NVIDIA driver version matches CUDA version |
| **Out of memory** | `RuntimeError: CUDA out of memory` | Reduce `BATCH_SIZE` or use gradient accumulation |

#### Troubleshooting Steps

1. **Check host GPU availability**:
   ```bash
   nvidia-smi  # Should show GPU info on host
   ```

2. **Verify Docker NVIDIA runtime**:
   ```bash
   docker info | grep nvidia  # Should show nvidia runtime
   ```

3. **Test container GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
   ```

4. **Check PyTorch CUDA installation**:
   ```python
   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.version.cuda}")
   print(f"cuDNN: {torch.backends.cudnn.version()}")
   print(f"GPU Count: {torch.cuda.device_count()}")
   ```

5. **Run GPU diagnostics**:
   ```bash
   ./scripts/test_gpu_activation.sh
   ```

### Memory Limits

The docker-compose configuration includes memory limits to prevent the container from consuming all system RAM:

- **Memory Limit**: 16GB (maximum RAM the container can use)
- **Memory Reservation**: 4GB (reserved RAM for the container)
- **Memory + Swap Limit**: 18GB (triggers OOM killer when exceeded)
- **OOM Killer**: Enabled (kills processes when system RAM is low)

When system RAM drops below ~2GB, the Linux OOM (Out Of Memory) killer will terminate container processes to prevent system freeze. You can adjust these limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Adjust based on your system
    reservations:
      memory: 4G
oom_kill_disable: false
memswap_limit: 18G  # Should be slightly higher than memory limit
```

## Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/

# Run specific test file
pytest tests/test_gpu_detection.py -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::TestCRNNAttention::test_forward_pass -v
```

### GPU Testing

```bash
# Run GPU activation test script
./scripts/test_gpu_activation.sh

# Run GPU detection unit tests
pytest tests/test_gpu_detection.py -v

# Run training integration tests (includes GPU)
pytest tests/test_training_integration.py -v
```

### Integration Tests

```bash
# Small dataset performance test
python tests/test_small_dataset.py

# Full end-to-end pipeline test
python tests/test_full_flow.py

# Test prediction system
pytest tests/test_prediction.py -v

# Test hyperparameter tuning
pytest tests/test_optuna_tune.py -v
```

### Test Coverage

Latest recorded Docker validation: **246 passed, 23 warnings**. See
`docs/implementation_log.md` for the current validation history.

The test suite covers:
- **GPU detection and device utilities** (CUDA availability, memory management, device selection)
- Data pipeline (feature engineering, preprocessing, dataset creation)
  - **Fibonacci retracement features** (swing high/low, retracement levels, distance features)
  - **Market regime detection** (train-only threshold fitting)
- All model architectures (forward pass, parameter counting)
- Pydantic config validation
- Custom financial losses
- Walk-forward and purged time-series splits
- Training loop (train, validate, early stopping)
- Prediction system (single/batch/interactive modes)
- Ensemble prediction compatibility checks
- End-to-end pipeline (train → validate → test → predict → backtest)
- **Stock sampling** (balanced group sampling, edge cases)
- **Hyperparameter tuning** (Optuna optimization, dataset creation)
- **Dynamic configuration** (COMMODITIES, TREASURY_YIELDS, EMA_PERIODS modification)
- **Dataset column validation** (required/optional columns, feature consistency)
- **Data download integration** (treasury yields, VIX, commodities)
- **Training integration** (GPU training, model saving/loading, checkpointing)
- Meta-labeling, attention interpretability, and optional feature attribution utilities

## Model Variants

| Model | Architecture | Use Case |
|-------|-------------|----------|
| **RNN** | BiLSTM only | Baseline |
| **RNN + Attention** | BiLSTM + Attention | Interpretability |
| **CRNN** | CNN + BiLSTM | Feature extraction |
| **CRNN + Attention** | CNN + 4-layer BiLSTM + Attention | Rich features |
| **LSTM3** | 3-layer BiLSTM | Deeper sequential modeling |
| **LSTM3 + Attention** | 3-layer BiLSTM + Attention | **Recommended** |
| **BiLSTM4 + Attention** | 4-layer BiLSTM + Attention | Deep sequential modeling |
| **Transformer** | 4-layer BiLSTM + Transformer encoder | Hybrid architecture |

## Output Metrics

The model evaluates on:
- MSE, RMSE, MAE, R²
- Directional accuracy
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, Profit factor

## Requirements

- Python 3.10+
- PyTorch 2.0+
- TA-Lib (requires system dependencies)
- 8GB+ RAM recommended
- GPU optional (CUDA 11.0+)

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
