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
   - 81 unit tests ensure code quality and correctness

3. **Code Quality**
   - Follows Python best practices and PEP 8 standards
   - Type hints for better code clarity
   - Comprehensive docstrings for all modules and functions
   - Proper error handling and logging

## Features

- **Multiple Model Architectures**: CRNN, RNN, RNN+Attention, CRNN+Attention, Transformer, LSTM3, LSTM3+Attention
- **Comprehensive Feature Engineering**:
  - Technical indicators (EMA, RSI, StochRSI, MACD)
  - 100+ candlestick patterns via TA-Lib
  - External data (VIX, commodities, treasury yields)
  - Time-based features (day, month embeddings)
  - **Dividend flag feature** (1=has dividend, 2=no dividend) with embedding
  - Financial metrics (PE ratio, PEG ratio, EPS, ROE, ROI, debt ratios, current ratio)
- **Log Transform Normalization** for all features
- **Time-based Data Splitting** (70% train, 15% val, 15% test)
- **Configurable Prediction Horizon** (default: 5 days)
- **Docker Deployment** with GPU support
- **Comprehensive Logging** and TensorBoard integration
- **Backtesting** with performance metrics
- **Prediction System** with support for single/batch/interactive prediction

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

# 1. Preprocess data
python scripts/preprocess_data.py --start-date 2015-01-01

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
```

### Quick Test with Small Dataset

```bash
# Test the full pipeline with a small dataset
python tests/test_small_dataset.py

# Run comprehensive end-to-end test
python tests/test_full_flow.py

# Run all unit tests
pytest tests/ -v
```

## Project Structure

```
research_02/
├── config/
│   ├── data_config.py       # Data configuration
│   └── model_config.py      # Model configuration
├── src/
│   ├── data/
│   │   ├── downloader.py    # Data downloading (yfinance, FRED)
│   │   ├── feature_engineering.py  # Technical indicators
│   │   ├── preprocessing.py  # Normalization, splitting
│   │   ├── financial_metrics_loader.py  # Financial metrics from API
│   │   ├── prediction_prep.py  # Data preparation for prediction
│   │   └── dataset.py        # PyTorch Dataset
│   ├── models/
│   │   ├── crnn_attention.py # CNN + BiLSTM + Attention (base module)
│   │   ├── crnn_model.py     # CNN + BiLSTM
│   │   ├── rnn_model.py      # BiLSTM only
│   │   ├── rnn_attention.py  # BiLSTM + Attention
│   │   ├── lstm3_model.py    # 3-layer BiLSTM
│   │   ├── lstm3_attn_model.py # 3-layer BiLSTM + Attention
│   │   └── transformer_model.py  # Transformer
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
│   ├── train.py
│   ├── test.py
│   ├── validate.py
│   ├── backtest.py
│   ├── predict.py          # Prediction CLI
│   ├── fix_permissions.py  # Docker permission fix
│   └── run_all.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_prediction.py   # Prediction tests
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
- **day**: Day of month (8 dimensions)
- **month**: Month of year (8 dimensions)
- **dividend_flag**: Dividend status - 1=has dividend, 2=no dividend (8 dimensions)

## Configuration

### Data Configuration (`config/data_config.py`)

```python
@dataclass
class DataConfig:
    START_DATE: str = "2010-01-01"
    TRAIN_RATIO: float = 0.70
    SEQUENCE_LENGTH: int = 30      # Lookback window
    PREDICTION_HORIZON: int = 5   # Days ahead to predict

    # Feature flags for ablation
    FEATURE_FLAGS: Dict[str, bool] = {
        'price_features': True,
        'ema_features': True,
        'candlestick_patterns': True,
        'vix': True,
        'commodities': True,
        ...
    }
```

### Model Configuration (`config/model_config.py`)

```python
@dataclass
class ModelConfig:
    MODEL_TYPE: str = "lstm3_attention"

    # Embeddings
    EMBEDDING_DIM_STOCK: int = 64
    EMBEDDING_DIM_GROUP: int = 32
    EMBEDDING_DIM_DAY: int = 8
    EMBEDDING_DIM_MONTH: int = 8
    EMBEDDING_DIM_DIVIDEND_FLAG: int = 8

    # Architecture
    RNN_HIDDEN_SIZE: int = 128
    ATTENTION_HEADS: int = 4

    # Training
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 256
    NUM_EPOCHS: int = 200
    EARLY_STOPPING_PATIENCE: int = 15
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

The project has 81 unit tests covering:
- Data pipeline (feature engineering, preprocessing, dataset creation)
- All model architectures (forward pass, parameter counting)
- Training loop (train, validate, early stopping)
- Prediction system (single/batch/interactive modes)
- End-to-end pipeline (train → validate → test → predict → backtest)

## Model Variants

| Model | Architecture | Use Case |
|-------|-------------|----------|
| **RNN** | BiLSTM only | Baseline |
| **RNN + Attention** | BiLSTM + Attention | Interpretability |
| **CRNN** | CNN + BiLSTM | Feature extraction |
| **CRNN + Attention** | CNN + BiLSTM + Attention | Rich features |
| **LSTM3** | 3-layer BiLSTM | Deeper sequential modeling |
| **LSTM3 + Attention** | 3-layer BiLSTM + Attention | **Recommended** |
| **Transformer** | Transformer encoder | Alternative architecture |

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
