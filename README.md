# CRNN Financial Prediction Model

A PyTorch-based CRNN (CNN + BiLSTM + Attention) model for predicting stock price movements. The model learns patterns from S&P 500 stocks using technical indicators, candlestick patterns, and external market data.

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
   - 91 unit tests ensure code quality and correctness

3. **Code Quality**
   - Follows Python best practices and PEP 8 standards
   - Type hints for better code clarity
   - Comprehensive docstrings for all modules and functions
   - Proper error handling and logging

## Features

- **Multiple Model Architectures**: CRNN, RNN, RNN+Attention, CRNN+Attention, Transformer, LSTM3, LSTM3+Attention, **BiLSTM4+Attention**
- **Comprehensive Feature Engineering**:
  - Technical indicators (EMA, RSI, StochRSI, MACD)
  - 100+ candlestick patterns via TA-Lib
  - External data (VIX, commodities, treasury yields)
  - Time-based features (day, month embeddings)
  - **Dividend flag feature** (1=has dividend, 2=no dividend) with embedding
  - Financial metrics (PE ratio, PEG ratio, EPS, ROE, ROI, debt ratios, current ratio)
- **Log Transform Normalization** for all features
- **Time-based Data Splitting** (70% train, 15% val, 15% test)
- **Balanced Stock Sampling** (`--stocks` option) - Sample stocks evenly across all group_ids
- **Configurable Prediction Horizon** (default: 5 days)
- **Docker Deployment** with GPU support
- **Comprehensive Logging** and TensorBoard integration
- **Backtesting** with performance metrics
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
python scripts/predict.py --model models/checkpoints/best_model.pth --mode interactive
python scripts/predict.py --model models/checkpoints/best_model.pth --mode batch --input data/new_data.csv
python scripts/predict.py --model models/checkpoints/best_model.pth --mode single --tic AAPL --date 2024-01-15

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

# Test stock sampling (NEW)
pytest tests/test_sampling.py -v

# Test hyperparameter tuning
pytest tests/test_optuna_tune.py -v
```

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
│   │   └── __init__.py       # Convenience functions
│   ├── data/
│   │   ├── downloader.py    # Data downloading (yfinance, FRED)
│   │   ├── feature_engineering.py  # Technical indicators
│   │   ├── preprocessing.py  # Normalization, splitting
│   │   ├── financial_metrics_loader.py  # Financial metrics from JSON
│   │   ├── prediction_prep.py  # Data preparation for prediction
│   │   ├── sampling.py       # Balanced stock sampling
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
│   │   └── early_stopping.py
│   ├── prediction/
│   │   └── predictor.py     # Prediction module
│   ├── evaluation/
│   │   ├── metrics.py       # Evaluation metrics
│   │   ├── validator.py     # Validation
│   │   └── backtester.py    # Backtesting
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
| **Technical Indicators** | EMA 50/100/200, RSI, StochRSI, MACD |
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
- Time Features: `day`, `month`, `dayofweek`
- External Data: `vix`, `bondyield`
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
│  Candlestick:      100+ patterns (Doji, Hammer, Engulfing, etc.)        │
│  VIX:              Volatility index                                      │
│  Commodities:      Gold, Silver, Copper prices                          │
│  Treasury Yields:  2Y, 10Y, 30Y rates                                   │
│  Financial Metrics: PE, PEG, EPS, ROE, ROI, Debt ratios, Dividend       │
│  Time Features:    Day of month, Month of year (for embeddings)         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING & NORMALIZATION                         │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Log Transform:   log(x + 1) for all numeric features                │
│  2. Stock Filtering:  Keep stocks with ≥252 trading days                │
│  3. Train/Val/Test Split: 70% / 15% / 15% (time-based)                  │
│  4. Sequence Creation:  Create sliding windows (seq_len=30)              │
│  5. Target Calculation: future_return = price[t+5] / price[t] - 1       │
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
│  • Evaluate on validation set (15% of data)                             │
│  • Calculate metrics: MSE, RMSE, MAE, R², Directional Accuracy          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           TESTING                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  python scripts/test.py --model best                                    │
│                                                                         │
│  • Evaluate on test set (15% of data)                                   │
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
│  python scripts/predict.py --model best --mode <mode>                   │
│                                                                         │
│  Modes:                                                                 │
│  • single:   Predict for 1 stock on 1 date                              │
│  • batch:    Predict from CSV/Parquet/Excel file                        │
│  • interactive: Manual input via CLI                                    │
│  • info:     Display model checkpoint information                       │
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
python scripts/predict.py --model best --mode interactive
```

## Configuration

The project uses **JSON-based configuration** with a Python Config wrapper class for easy access.

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
      "VAL_RATIO": 0.15,
      "TEST_RATIO": 0.15
    },
    "sequences": {
      "SEQUENCE_LENGTH": 30,
      "PREDICTION_HORIZON": 5
    },
    "technical_indicators": {
      "EMA_PERIODS": [50, 100, 200],
      "RSI_PERIOD": 14
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
      "NUM_EPOCHS": 200,
      "EARLY_STOPPING_PATIENCE": 15
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

## Docker Deployment

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

## Testing

```bash
# Unit tests
pytest tests/

# Small dataset performance test
python tests/test_small_dataset.py

# Full end-to-end pipeline test
python tests/test_full_flow.py

# Test prediction system
pytest tests/test_prediction.py -v
```

### Test Coverage

The project has **149 tests** covering:
- Data pipeline (feature engineering, preprocessing, dataset creation)
- All model architectures (forward pass, parameter counting)
- Training loop (train, validate, early stopping)
- Prediction system (single/batch/interactive modes)
- End-to-end pipeline (train → validate → test → predict → backtest)
- **Stock sampling** (balanced group sampling, edge cases)
- **Hyperparameter tuning** (Optuna optimization, dataset creation)
- **Dynamic configuration** (COMMODITIES, TREASURY_YIELDS, EMA_PERIODS modification)
- **Dataset column validation** (required/optional columns, feature consistency)
- **Data download integration** (treasury yields, VIX, commodities)

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
