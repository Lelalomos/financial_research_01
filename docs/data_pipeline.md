# Data Pipeline Documentation

## Overview

The data pipeline consists of four main stages:

1. **Data Download** (`src/data/downloader.py`)
2. **Feature Engineering** (`src/data/feature_engineering.py`)
3. **Preprocessing** (`src/data/preprocessing.py`)
4. **Dataset Creation** (`src/data/dataset.py`)

## Stage 1: Data Download

### Sources

| Data | Source | Symbols |
|------|--------|---------|
| Stocks | yfinance | S&P 500 tickers |
| VIX | yfinance | ^VIX |
| Commodities | yfinance | GC=F, HG=F, ZC=F, ZS=F, CC=F, SI=F |
| Treasury Yields | FRED | DGS2, DGS10, DGS30 |

### Usage

```python
from src.data.downloader import DataDownloader
from src.config import load_config

config = DataConfig()
downloader = DataDownloader(config)

# Download all data
data = downloader.download_all(save=True)

# Or load existing data
data = downloader.load_saved_data()
```

## Stage 2: Feature Engineering

### Technical Indicators

- **EMA**: Exponential Moving Average (50, 100, 200)
- **RSI**: Relative Strength Index (14)
- **StochRSI**: Stochastic RSI (14)
- **MACD**: Moving Average Convergence Divergence

### Fibonacci Retracement Features

- **Swing High/Low**: Rolling maximum/minimum over configurable window (default: 30 days)
- **Retracement Levels**:
  - `fib_38`: 38.2% retracement level (swing_high - 0.382 * range)
  - `fib_50`: 50% retracement level (swing_high - 0.5 * range)
  - `fib_61`: 61.8% retracement level (swing_high - 0.618 * range)
- **Distance Features** (RNN-friendly normalized distances):
  - `dist_fib_38`: (close - fib_38) / range
  - `dist_fib_50`: (close - fib_50) / range
  - `dist_fib_61`: (close - fib_61) / range
- **Break Indicator**: `break_fib_61` = 1 if close < fib_61, else 0

### Candlestick Patterns

All 100+ TA-Lib patterns are included:
- Engulfing patterns
- Doji patterns
- Hammer/Hanging Man
- Morning/Evening Star
- And many more...

### Time Features

- Day of month (1-31)
- Month (1-12)
- Day of week (0-6)

### Target Calculation

```
target = (price[t+H] - price[t]) / price[t] * 100
```

Where H is the prediction horizon (default 5 days).

## Stage 3: Preprocessing

### Normalization

All features are normalized using log transform:

```python
normalized = log1p(x - min(x) + offset)
```

### Time-Based Split

Each stock is split independently by date:

- **Train**: First 70% of dates
- **Val**: Next 10% of dates
- **Test**: Last 20% of dates

This ensures no temporal leakage.

### Sequence Creation

Sliding window sequences are created:

- **Window size**: 30 days (configurable)
- **Stride**: 1 day
- **Target**: Future return at horizon

## Stage 4: Dataset

### PyTorch Dataset

```python
from src.data.dataset import FinancialDataset

dataset = FinancialDataset(sequences, config)

# Returns:
# {
#     'features': (seq_len, num_features),
#     'stock_id': (seq_len,),
#     'group_id': (seq_len,),
#     'day': (seq_len,),
#     'month': (seq_len,),
#     'target': (1,)
# }
```

### DataLoaders

```python
from src.data.dataset import create_data_loaders

loaders = create_data_loaders(
    train_sequences,
    val_sequences,
    config=config
)
```

## Configuration

All data parameters are in `config/main.json`:

```python
from src.config import load_config

config = load_config('main')

# Access sequence parameters
sequence_length = config.data.sequences.SEQUENCE_LENGTH  # 30
prediction_horizon = config.data.sequences.PREDICTION_HORIZON  # 5

# Access technical indicators
ema_periods = config.data.technical_indicators.EMA_PERIODS  # [50, 100, 200]

# Access Fibonacci configuration
fib_window = config.data.fibonacci.FIBONACCI_WINDOW  # 30

# Access feature flags
feature_flags = config.data.features.FEATURE_FLAGS
```
