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

- **EMA**: Exponential Moving Average (current default periods: 50, 200)
- **RSI**: Relative Strength Index (14)
- **StochRSI**: Stochastic RSI (14)
- **MACD**: Moving Average Convergence Divergence

### Geometric / Structural Features

Geometric features are enabled by default through
`data.features.FEATURE_FLAGS.geometric_features`.
Sub-feature toggles live under `data.geometric`.

- **ATR_14_NORM**: ATR normalized by price
- **ROC_10**: 10-period rate of change
- **BB_WIDTH_20**: Bollinger Band width
- **SLOPE_SUP_20**: Linear-regression slope of a rolling support proxy using
  `data.geometric.CHANNEL_WINDOW`
- **SLOPE_RES_20**: Linear-regression slope of a rolling resistance proxy using
  `data.geometric.CHANNEL_WINDOW`
- **CHANNEL_COMPRESSION_20**: Rolling channel width normalized by price,
  enabled by `data.geometric.ENABLE_CHANNEL_COMPRESSION`
- **CHANNEL_POSITION_20**: Relative close position inside the rolling channel,
  enabled by `data.geometric.ENABLE_CHANNEL_POSITION`
- **DIST_TO_SWING_HIGH_20** / **DIST_TO_SWING_LOW_20**: Normalized distance to
  rolling swing extremes using `data.geometric.SWING_WINDOW`, enabled by
  `data.geometric.ENABLE_SWING_DISTANCE`
- **DAYS_SINCE_SWING_HIGH_20** / **DAYS_SINCE_SWING_LOW_20**: Bars since the
  most recent rolling swing extremes, enabled by
  `data.geometric.ENABLE_SWING_TIME_DISTANCE`
- **OPT_SLOPE_SUP_30** / **OPT_SLOPE_RES_30**: Pivot-anchored optimized
  support/resistance slopes using `data.geometric.TRENDLINE_WINDOW`, enabled by
  `data.geometric.ENABLE_OPTIMIZED_TRENDLINES`
- **OPT_CHANNEL_WIDTH_30**: Width between the optimized resistance/support
  lines at the end of the trendline window, enabled by
  `data.geometric.ENABLE_OPTIMIZED_CHANNEL_WIDTH`

### Fibonacci Retracement Features

Fibonacci features are implemented but disabled by default in `config/main.json`.
Set `data.features.FEATURE_FLAGS.fibonacci_features` to `true` to include them.

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

You can also exclude specific candlestick patterns through
`data.candlestick.EXCLUDE_PATTERNS` in `config/main.json` while keeping the
rest of the candlestick feature set enabled.

The current default config excludes 29 ultra-sparse `CDL*` patterns so the
generated feature set avoids the noisiest rare-event candlestick columns.

### Time Features

- Day of month (1-31)
- Month (1-12)
- Day of week (0-6)

### Optional Polars Feature Engineering

Polars implementations are available for selected feature engineering paths but
are disabled by default:

- `data.features.FEATURE_FLAGS.polars_time_features`
- `data.features.FEATURE_FLAGS.polars_fibonacci_features`
- `data.features.FEATURE_FLAGS.polars_external_merges`

The pipeline still returns pandas DataFrames for downstream compatibility.
TA-Lib technical indicators, stockstats integration, financial metrics loading,
sklearn preprocessing, prediction APIs, and reports remain pandas-based.

See `docs/polars_migration.md` for enablement, profiling, and parity testing.

### Market Regime Feature

Market regime detection is implemented but disabled by default in
`config/main.json`. Set both `data.regime.ENABLED` and
`data.features.FEATURE_FLAGS.market_regime` to `true` to include `regime_id`.

The current implementation is quantile-based and dependency-light. During
preprocessing, thresholds are fit on the training split only using the
configured proxy column, then reused for validation and test rows. This avoids
future-data leakage from validation/test market conditions.

Default settings use `vix` as the proxy and produce three regimes:

- `0`: lower proxy values
- `1`: middle proxy values
- `2`: higher proxy values

`regime_id` is included as a regular sequence feature when enabled. It is not
normalized, and the fitted regime parameters are written to preprocessing info
for later inference/checkpoint use.

### Target Calculation

```
target = (price[t+H] - price[t]) / price[t] * 100
```

Where H is the prediction horizon (default 5 days).

## Stage 3: Preprocessing

### Normalization

Features are split first, then normalization parameters are fit on the
training split only and reused for validation/test. This avoids future-data
leakage.

The default normalization uses log transform:

```python
normalized = log1p(x - min(x) + offset)
```

### Time-Based Split

Splits use global date ranges shared by all stocks:

- **Train**: First 70% of dates
- **Val**: Next 10% of dates
- **Test**: Last 20% of dates

This avoids overlap between validation/test dates and training dates.

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
ema_periods = config.data.technical_indicators.EMA_PERIODS  # [50, 200]

# Access Fibonacci configuration
fib_window = config.data.fibonacci.FIBONACCI_WINDOW  # 30

# Access feature flags
feature_flags = config.data.features.FEATURE_FLAGS
```
