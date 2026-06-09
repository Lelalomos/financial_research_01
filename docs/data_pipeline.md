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

When preprocessing is run with `--stocks N`, the pipeline uses balanced
sampling across all groups. The per-group stock choice can now be controlled by
`data.sampling.STOCK_SELECTION_MODE`:

- `random`: current random balanced sampling
- `sorted`: largest market-cap stocks first inside each group

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

### Market Structure Features

Market structure features are part of the geometric feature family and are
controlled by `data.geometric.ENABLE_MARKET_STRUCTURE_FEATURES`.

Current generated features include:

- normalized support/resistance distance features
  - `distance_to_20d_high`
  - `distance_to_60d_high`
  - `distance_to_120d_high`
  - `distance_to_252d_high`
  - `distance_to_20d_low`
  - `distance_to_60d_low`
  - `distance_to_120d_low`
  - `distance_to_252d_low`
- breakout and breakdown features
  - `breakout_20d`
  - `breakout_60d`
  - `breakout_120d`
  - `breakdown_20d`
  - `breakdown_60d`
  - `breakdown_120d`
- structure state and rolling counts
  - `higher_high`
  - `lower_high`
  - `higher_low`
  - `lower_low`
  - `higher_high_count_20`
  - `higher_low_count_20`
  - `lower_high_count_20`
  - `lower_low_count_20`
- 52-week features
  - `distance_to_52w_high`
  - `distance_to_52w_low`
  - `near_52w_high`
  - `near_52w_low`
- volume confirmation
  - `breakout_volume_ratio`
  - `breakdown_volume_ratio`
  - `volume_spike_ratio`
  - `volume_momentum`
- volatility and trend
  - `atr_14`
  - `atr_20`
  - `atr_ratio`
  - `rolling_volatility_20`
  - `rolling_volatility_60`
  - `trend_strength_score`
  - `trend_persistence_20`
  - `trend_persistence_60`
- lagged structure signals
  - `higher_high_lag_1`
  - `higher_low_lag_1`
  - `breakout_20d_lag_1`
  - `breakdown_20d_lag_1`
  - `trend_strength_score_lag_1`
  - plus the same signal family for lags `3`, `5`, `10`, and `20`

Behavior:

- all calculations are past-only rolling calculations
- output columns are filled so the feature block does not emit NaN values
- if a market-structure column already exists in the input dataset, the
  pipeline keeps the existing values and only creates the missing features
- metadata about generated and preserved features is available from
  `src/data/market_structure_features.py`

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

The candlestick feature block now generates custom vectorized candlestick
features instead of TA-Lib `CDL*` columns.

Current generated features include:

- basic shape features
  - `body_size`
  - `candle_direction`
  - `upper_shadow`
  - `lower_shadow`
  - `high_low_range`
  - `body_ratio`
  - `upper_shadow_ratio`
  - `lower_shadow_ratio`
  - `close_position`
  - `open_position`
- gap features
  - `gap_up`
  - `gap_down`
  - `gap_size`
  - `overnight_return`
- momentum candlestick features
  - `body_size_change`
  - `body_size_ema_5`
  - `body_size_ema_20`
  - `upper_shadow_change`
  - `lower_shadow_change`
- volatility features
  - `atr`
  - `atr_14`
  - `atr_20`
  - `rolling_volatility`
  - `support_distance`
  - `resistance_distance`
  - `breakout_signal`
  - `rolling_high_low_range_5`
  - `rolling_high_low_range_20`
- return and volume features
  - `return_1d`
  - `return_5d`
  - `return_20d`
  - `volume_momentum`
  - `volume_spike`
- sequence-ready lag and rolling close features
  - `lag_1`
  - `lag_3`
  - `lag_5`
  - `lag_10`
  - `lag_20`
  - `rolling_mean_5`
  - `rolling_std_5`
  - `rolling_min_5`
  - `rolling_max_5`
- normalized candlestick features
  - `body_size_pct`
  - `upper_shadow_pct`
  - `lower_shadow_pct`
  - `range_pct`
- binary pattern flags
  - `doji`
  - `hammer`
  - `inverted_hammer`
  - `shooting_star`
  - `bullish_engulfing`
  - `bearish_engulfing`
  - `morning_star`
  - `evening_star`
- rolling statistics for `body_size`, `upper_shadow`, and `lower_shadow`
  - rolling mean
  - rolling std
  - rolling z-score
  - windows: `5`, `10`, `20`, `60`

The generated candlestick columns are filled so the candlestick feature block
does not emit NaN values.

If any of these columns already exist in the input dataset, the pipeline keeps
the existing values and only creates the missing features.

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

### Rolling Cointegration Features

Rolling cointegration features are available but disabled by default. Set
`data.features.FEATURE_FLAGS.cointegration_features` to `true` to include them.

The feature engineering stage adds these continuous features:

- `spread`
- `rolling_mean_spread`
- `rolling_std_spread`
- `spread_zscore`
- `spread_norm`
- `equilibrium_gap`
- `equilibrium_zscore`
- `equilibrium_gap_norm`
- `relative_price_vs_sector`
- `relative_price_vs_sector_norm`
- `relative_return_vs_sector`
- `spread_adf_stat`
- `spread_adf_pvalue`
- `equilibrium_adf_stat`
- `equilibrium_adf_pvalue`

Current behavior:

- pair spread features are built inside each sector using a rolling same-sector
  peer selected by highest absolute return correlation in the rolling window
- hedge ratio beta is estimated with rolling OLS
- sector equilibrium features use
  `statsmodels.tsa.vector_ar.vecm.coint_johansen` on same-sector log prices
- all calculations are past-only rolling windows, so they do not use future
  observations beyond the current timestamp
- `feature_columns.txt` will include the new feature names automatically, which
  keeps SHAP, attention maps, and attribution utilities aligned

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

Sequence creation now supports two modes through `data.dataset.MODE`:

- `precomputed_sequences`: preprocessing stops after normalized split parquet export, and training builds full in-memory sequence dictionaries from those normalized split caches before the first epoch
- `on_the_fly_sequences`: preprocessing stops after normalized split parquet export, and training uses a lazy dataset that slices sliding windows from `data/processed/.cache/normalized_splits/*.parquet` as batches are requested

Special prepared datasets such as `data/processed_chronos2` and
`data/processed_kronos_rich` may also include extra `.npy` arrays beyond the
base keys. The current dataset loader preserves these optional arrays and
passes them through in each batch so model-specific training code can use them.

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
