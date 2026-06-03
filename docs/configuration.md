# Configuration Guide

## Overview

Configuration is managed through JSON files in the `config/` directory:

1. **`config/main.json`** - Data sources, features, preprocessing parameters
2. **`config/model.json`** - Model architecture, training hyperparameters (all model types)
3. **`config/hyperparameter.json`** - Hyperparameter search settings
4. **`config/test.json`** - Testing configuration
5. **`config/deploy.json`** - Deployment configuration
6. **`config/validate.json`** - Validation configuration

## Loading Configuration

```python
from src.config import load_config

# Load main config (data settings)
main_config = load_config('main')

# Load model config
model_config = load_config('model')

# Load hyperparameter config
hparam_config = load_config('hyperparameter')
```

## Data Configuration (`config/main.json`)

### Data Sources

```json
{
  "data": {
    "sources": {
      "START_DATE": "2015-01-01",
      "END_DATE": null,
      "SP500_TICKER_SOURCE": "wikipedia",
      "USE_YFINANCE_LIVE": true,
      "COMMODITIES": {
        "GC=F": "Gold",
        "HG=F": "Copper",
        "SI=F": "Silver"
      },
      "TREASURY_YIELDS": ["DGS10", "DGS30", "DGS2"]
    }
  }
}
```

### Modifying Dynamic Lists

The following lists can be modified at runtime:

```python
from src.config import load_config

config = load_config('main')

# Add new commodity
config.data.sources.COMMODITIES._data['ZW=F'] = 'Wheat'

# Add new treasury yield
config.data.sources.TREASURY_YIELDS.append('DGS5')

# Add new EMA period
config.data.technical_indicators.EMA_PERIODS.append(20)
```

### Sequence Parameters

```json
{
  "data": {
    "sequences": {
      "SEQUENCE_LENGTH": 30,
      "PREDICTION_HORIZON": 5,
      "TARGET_THRESHOLD": 10.0,
      "NORMALIZE_TARGET": true
    }
  }
}
```

### Stock Sampling

```json
{
  "data": {
    "sampling": {
      "STOCK_SELECTION_MODE": "sorted",
      "MARKET_CAP_METADATA_DIR": "raw_data/ticket_data/us"
    }
  }
}
```

These settings are used when preprocessing runs with `--stocks N`.

- `STOCK_SELECTION_MODE = "random"`
  - keeps the current balanced random sampling inside each group
- `STOCK_SELECTION_MODE = "sorted"`
  - sorts stocks inside each group by market cap and selects the largest first

`MARKET_CAP_METADATA_DIR` points to the local ticker metadata JSON files used
to read market-cap values for `sorted` mode.

### Technical Indicators

```json
{
  "data": {
    "technical_indicators": {
      "EMA_PERIODS": [50, 200],
      "RSI_PERIOD": 14,
      "STOCHRSI_PERIOD": 14,
      "MACD_PARAMS": [12, 26, 9]
    },
    "fibonacci": {
      "FIBONACCI_WINDOW": 30
    }
  }
}
```

### Fibonacci Retracement Features

```json
{
  "data": {
    "fibonacci": {
      "FIBONACCI_WINDOW": 30
    }
  }
}
```

The Fibonacci retracement features include:
- `swing_high`: Rolling maximum of `high` over the window
- `swing_low`: Rolling minimum of `low` over the window
- `fib_range`: Difference between swing_high and swing_low
- `fib_38`, `fib_50`, `fib_61`: Fibonacci retracement levels (38.2%, 50%, 61.8%)
- `dist_fib_38`, `dist_fib_50`, `dist_fib_61`: Normalized distance features
- `break_fib_61`: Binary indicator (1 if close < fib_61)

### Geometric Feature Window

The geometric feature block supports a shared window plus per-feature flags:

```json
{
  "data": {
    "geometric": {
      "CHANNEL_WINDOW": 20,
      "SWING_WINDOW": 20,
      "TRENDLINE_WINDOW": 30,
      "TRENDLINE_TOLERANCE": 0.0001,
      "TRENDLINE_MAX_ITERATIONS": 100,
      "ENABLE_ATR_FEATURE": true,
      "ENABLE_ROC_FEATURE": true,
      "ENABLE_BB_WIDTH_FEATURE": true,
      "ENABLE_SLOPE_FEATURES": true,
      "ENABLE_CHANNEL_COMPRESSION": false,
      "ENABLE_CHANNEL_POSITION": false,
      "ENABLE_SWING_DISTANCE": false,
      "ENABLE_SWING_TIME_DISTANCE": false,
      "ENABLE_OPTIMIZED_TRENDLINES": false,
      "ENABLE_OPTIMIZED_CHANNEL_WIDTH": false
    }
  }
}
```

`CHANNEL_WINDOW` controls the rolling min/max window and the linear-regression
slope period used for support and resistance slope features. The current output
column names remain `slope_sup_20` and `slope_res_20` for backward
compatibility even if the configured window changes.

`SWING_WINDOW` controls the rolling lookback used for the structural swing
features.

`TRENDLINE_WINDOW`, `TRENDLINE_TOLERANCE`, and
`TRENDLINE_MAX_ITERATIONS` control the optional pivot-anchored optimized
trendline solver described in `docs/deep-dive-technical.md`.

`ENABLE_ATR_FEATURE`, `ENABLE_ROC_FEATURE`, and `ENABLE_BB_WIDTH_FEATURE`
control whether the existing ATR, ROC, and Bollinger-width geometric columns
are generated.

`ENABLE_SLOPE_FEATURES` controls whether support/resistance slope columns are
generated when the master `data.features.FEATURE_FLAGS.geometric_features` flag
is enabled.

`ENABLE_CHANNEL_COMPRESSION` controls generation of `channel_compression_20`,
which measures normalized channel width:

```python
(rolling_max - rolling_min) / abs(close)
```

`ENABLE_CHANNEL_POSITION` controls generation of `channel_position_20`, which
measures where the close sits inside the channel:

```python
(close - rolling_min) / (rolling_max - rolling_min)
```

Both features use epsilon guards internally to avoid divide-by-zero when price
or channel width is effectively zero.

`ENABLE_SWING_DISTANCE` controls:

- `dist_to_swing_high_20`
- `dist_to_swing_low_20`

These measure normalized distance from the close to the rolling swing high/low
defined by `SWING_WINDOW`.

`ENABLE_SWING_TIME_DISTANCE` controls:

- `days_since_swing_high_20`
- `days_since_swing_low_20`

These count bars since the most recent occurrence of the rolling swing high/low
inside the `SWING_WINDOW` lookback.

`ENABLE_OPTIMIZED_TRENDLINES` controls:

- `opt_slope_sup_30`
- `opt_slope_res_30`

These are pivot-anchored optimized support/resistance slopes fit over
`TRENDLINE_WINDOW`. The current output names remain suffixed with `_30` for
compatibility even if the configured window changes.

`ENABLE_OPTIMIZED_CHANNEL_WIDTH` controls:

- `opt_channel_width_30`

This measures the end-of-window width between the optimized resistance and
support lines over the configured trendline window.

### Feature Flags

```json
{
  "data": {
    "features": {
      "FEATURE_FLAGS": {
        "price_features": true,
        "ema_features": true,
        "rsi_features": true,
        "stochrsi_features": true,
        "macd_features": true,
        "geometric_features": true,
        "fibonacci_features": false,
        "candlestick_patterns": true,
        "vix": true,
        "commodities": true,
        "treasury_yields": true,
        "time_features": true,
        "financial_metrics": true,
        "cointegration_features": false,
        "polars_time_features": false,
        "polars_fibonacci_features": false,
        "polars_external_merges": false
      }
    }
  }
}
```

`fibonacci_features` is disabled by default. Enable it when you want the
additional retracement columns included in training features.

`geometric_features` is enabled by default. It adds normalized ATR, rate of
change, Bollinger Band width, and support/resistance slope features.

`cointegration_features` is disabled by default. Enable it when you want the
pipeline to add rolling pair-spread features, rolling Johansen sector
equilibrium features, sector-relative features, and ADF stationarity outputs.

The `polars_*` flags are disabled by default. They enable opt-in Polars
implementations for time features, Fibonacci features, and external data merges
while preserving pandas DataFrame outputs. See `docs/polars_migration.md`.

### Cointegration Configuration

```json
{
  "data": {
    "cointegration": {
      "ROLLING_WINDOW": 252,
      "NORMALIZATION_WINDOW": 252,
      "JOHANSEN_DET_ORDER": 0,
      "JOHANSEN_K_AR_DIFF": 1
    }
  }
}
```

These settings are used only when:

```json
data.features.FEATURE_FLAGS.cointegration_features = true
```

`ROLLING_WINDOW` controls how much past history is used to estimate the raw
cointegration relationship itself. It is used for:

- rolling OLS hedge ratio beta
- `spread`
- rolling Johansen equilibrium vectors
- ADF stationarity checks

`NORMALIZATION_WINDOW` controls how much past history is used to normalize the
already-built feature values. It is used for:

- `spread_norm`
- `equilibrium_gap_norm`
- `relative_price_vs_sector_norm`

Simple rule:

- `ROLLING_WINDOW` builds the raw feature
- `NORMALIZATION_WINDOW` scales the raw feature

Good starting values:

- `252` for medium-term behavior
- `504` for slower, more stable behavior

### Candlestick Configuration

Candlestick generation can be enabled globally and selectively pruned:

```json
{
  "data": {
    "candlestick": {
      "USE_CANDLESTICK_PATTERNS": true,
      "EXCLUDE_PATTERNS": ["CDL3STARSINSOUTH", "CDLKICKING"]
    }
  }
}
```

`EXCLUDE_PATTERNS` removes specific TA-Lib `CDL*` features during feature
engineering while keeping the rest of the candlestick set enabled.

The current default config excludes 29 ultra-sparse candlestick patterns based
on prior feature-sparsity analysis.

### Market Regime Detection

Market regime detection is disabled by default. Enable both the feature flag and
the regime section:

```json
{
  "data": {
    "features": {
      "FEATURE_FLAGS": {
        "market_regime": true
      }
    },
    "regime": {
      "ENABLED": true,
      "METHOD": "quantile",
      "PROXY_COLUMN": "vix",
      "N_REGIMES": 3,
      "LOW_QUANTILE": 0.33,
      "HIGH_QUANTILE": 0.66,
      "DEFAULT_REGIME": 1
    }
  }
}
```

`PROXY_COLUMN` must exist before preprocessing, usually from external data such
as `vix`. Thresholds are fit on the training split only and then reused for
validation/test. The resulting `regime_id` feature is included in model feature
columns when enabled.

## Model Configuration (`config/model.json`)

### Structure

The model config has separate sections for each model type:

```json
{
  "model": {
    "embeddings": {
      "EMBEDDING_DIM_STOCK": 64,
      "EMBEDDING_DIM_GROUP": 32,
      "EMBEDDING_DIM_DAY": 16,
      "EMBEDDING_DIM_MONTH": 16,
      "EMBEDDING_DIM_DIVIDEND_FLAG": 8,
      "DROPOUT_EMBEDDING": 0.1
    },
    "training": {
      "LEARNING_RATE": 0.0001,
      "WEIGHT_DECAY": 0.00001,
      "BATCH_SIZE": 128,
      "NUM_EPOCHS": 30,
      "EARLY_STOPPING_PATIENCE": 15,
      "OPTIMIZER": "adam",
      "SCHEDULER": "reduce_on_plateau"
    },
    "loss": {
      "LOSS_TYPE": "directional_huber",
      "HUBER_DELTA": 0.5,
      "DIRECTIONAL_ALPHA": 0.1,
      "SHARPE_EPSILON": 0.000001,
      "QUANTILE": 0.5
    },
    "models": {
      "lstm3_attention": {
        "LSTM3_HIDDEN_SIZE": 256,
        "LSTM3_NUM_LAYERS": 3,
        "LSTM3_DROPOUT": 0.2,
        "LSTM3_USE_LAYER_NORM": true,
        "LSTM3_ATTENTION_HEADS": 8,
        "LSTM3_ATTENTION_DROPOUT": 0.1,
        "FC_HIDDEN_SIZES": [256, 128],
        "FC_DROPOUT": 0.3,
        "FC_USE_BATCH_NORM": false
      },
      "bilstm4_attention": {
        "LSTM4_HIDDEN_SIZES": [128, 256, 512, 256],
        "LSTM4_DROPOUT": 0.4,
        "LSTM4_ATTENTION_HEADS": 4,
        "LSTM4_ATTENTION_DROPOUT": 0.4,
        "FC_HIDDEN_SIZES": [256, 128],
        "FC_DROPOUT": 0.3,
        "FC_USE_BATCH_NORM": false
      }
    }
  }
}
```

### Accessing Model-Specific Parameters

```python
from src.config import load_config

config = load_config('model')

# Access LSTM3 attention parameters
hidden_size = config.model.models.lstm3_attention.LSTM3_HIDDEN_SIZE
num_heads = config.model.models.lstm3_attention.LSTM3_ATTENTION_HEADS

# Access BiLSTM4 parameters
hidden_sizes = config.model.models.bilstm4_attention.LSTM4_HIDDEN_SIZES
```

### Available Model Types

- `crnn` - CNN + RNN
- `rnn` - Simple RNN
- `rnn_attention` - RNN + Attention
- `crnn_attention` - CNN + RNN + Attention
- `transformer` - Transformer model
- `lstm3` - 3-layer LSTM
- `lstm3_attention` - 3-layer LSTM + Attention
- `bilstm4_attention` - 4-layer Bidirectional LSTM + Attention
- `multi_branch_bilstm` - Multi-branch recurrent model
- `kronos` - Tokenized autoregressive generative model

Kronos credit:

- Original project: `Kronos`
- Original repository: `https://github.com/shiyu-coder/Kronos`
- Original paper: `Kronos: A Foundation Model for the Language of Financial Markets`
- Authors listed in the upstream citation:
  - Yu Shi
  - Zongliang Fu
  - Shuo Chen
  - Bohan Zhao
  - Wei Xu
  - Changshui Zhang
  - Jian Li

### Embedding Dimensions

```python
config.model.embeddings.EMBEDDING_DIM_STOCK = 64
config.model.embeddings.EMBEDDING_DIM_GROUP = 32
config.model.embeddings.EMBEDDING_DIM_DAY = 16
config.model.embeddings.EMBEDDING_DIM_MONTH = 16
config.model.embeddings.EMBEDDING_DIM_DIVIDEND_FLAG = 8
```

### Training Parameters

```python
config.model.training.LEARNING_RATE = 1e-4
config.model.training.WEIGHT_DECAY = 1e-5
config.model.training.BATCH_SIZE = 128
config.model.training.NUM_EPOCHS = 30
config.model.training.EARLY_STOPPING_PATIENCE = 15
```

### Validation Parameters

Evaluation loaders can override the training batch size:

```python
config.model.validation.VAL_BATCH_SIZE = None  # fallback to training.BATCH_SIZE
config.model.selection.DEFAULT_MODEL_TYPE = "bilstm4_attention"
```

When `VAL_BATCH_SIZE` is `null`, validation, test, and backtest scripts fall
back to `model.training.BATCH_SIZE`.

`DEFAULT_MODEL_TYPE` also controls the fallback model used by:

- `scripts/train.py`
- `scripts/test.py`
- `scripts/validate.py`
- `scripts/backtest.py`

If you set:

```python
config.model.selection.DEFAULT_MODEL_TYPE = "kronos"
```

those scripts use Kronos unless you override it with `--model-type`.

### Training Backend

Lightning is the default training backend, and the custom trainer remains the
backup path:

```json
{
  "model": {
    "training_backend": {
      "DEFAULT": "lightning",
      "FALLBACK": "custom",
      "ALLOW_CUSTOM_FALLBACK": true
    }
  }
}
```

Use `python scripts/train.py --backend custom` to force the custom trainer.
Lightning reuses the configured loss, optimizer, scheduler, and gradient
clipping settings. See `docs/lightning_backend.md` for compatibility,
checkpoint details, and the custom trainer fallback scope.

Checkpoint settings are shared by both backends:

```python
config.model.checkpointing.CHECKPOINT_DIR = "models/checkpoints"
config.model.checkpointing.SAVE_BEST_ONLY = True
config.model.checkpointing.SAVE_LAST_N = 3
config.model.checkpointing.CHECKPOINT_FREQUENCY = 1
```

Semantics:
- `SAVE_BEST_ONLY = true`: save only best-improving checkpoints to a stable
  best-file path
- `SAVE_BEST_ONLY = false`: still save best-improving checkpoints, and also
  overwrite a stable periodic checkpoint every `CHECKPOINT_FREQUENCY` epochs
- `SAVE_LAST_N`: retained for backward compatibility with older timestamped
  checkpoint workflows; stable overwrite paths do not create additional
  periodic files

### Loss Function

```python
config.model.loss.LOSS_TYPE = "directional_huber"  # e.g. 'directional_huber', 'directional_mse', 'quantile_loss', 'pinball_loss', 'multi_part_rich_loss', 'huber', 'mse', 'mae', 'smooth_l1'
config.model.loss.DIRECTIONAL_ALPHA = 0.1
config.model.loss.HUBER_DELTA = 0.5  # used when LOSS_TYPE == "huber" or "directional_huber"
config.model.loss.QUANTILE = 0.5  # used when LOSS_TYPE == "quantile_loss" or "pinball_loss"
```

Loss selection notes:

- `quantile_loss` and `pinball_loss` are equivalent in this repo
- `multi_part_rich_loss` is intended for `chronos_rich`
- `multi_part_rich_loss` uses the `chronos_rich` component settings and weights:
  - `SCALAR_LOSS_TYPE`
  - `SCALAR_LOSS_WEIGHT`
  - `OHLCV_LOSS_TYPE`
  - `OHLCV_LOSS_WEIGHT`
  - `RETURN_PATH_LOSS_TYPE`
  - `RETURN_PATH_LOSS_WEIGHT`
  - `REGIME_LOSS_TYPE`
  - `REGIME_LOSS_WEIGHT`
- `kronos_rich` has its own loss config and does not follow `chronos_rich`
- `kronos_rich` uses:
  - `RECON_LOSS_TYPE`
  - `PRE_LOSS_TYPE`
  - `TOKEN_LOSS_TYPE`
  - `RECON_LOSS_WEIGHT`
  - `PRE_LOSS_WEIGHT`
  - `TOKEN_LOSS_WEIGHT`
  - `BSQ_LOSS_WEIGHT`

### Local MLflow Experiment Tracking

Experiment tracking is disabled by default. Enable local MLflow in
`config/model.json` only when you want local run tracking:

```json
{
  "model": {
    "experiment_tracking": {
      "ENABLED": true,
      "BACKEND": "mlflow",
      "MLFLOW_TRACKING_URI": "file:./mlruns",
      "EXPERIMENT_NAME": "multi-model-financial-forecasting",
      "LOG_PARAMS": true,
      "LOG_METRICS": true,
      "LOG_ARTIFACTS": false
    }
  }
}
```

This local MLflow mode does not require an API key. See
`docs/experiment_tracking.md` for install, training, and local UI commands.

## Hyperparameter Configuration (`config/hyperparameter.json`)

```json
{
  "hyperparameter": {
    "N_TRIALS": 50,
    "TIMEOUT": null,
    "N_JOBS": 1,
    "MODEL_TYPE": "bilstm4_attention",
    "HPARAM_STOCKS": 20,
    "HPARAM_MAX_EPOCHS": 50,
    "HPARAM_ES_PATIENCE": 10,
    "LEARNING_RATE_RANGE": [0.00001, 0.001],
    "LSTM_HIDDEN_SIZE_RANGE": [64, 512],
    "LSTM_NUM_LAYERS_RANGE": [1, 4],
    "DROPOUT_RANGE": [0.1, 0.5],
    "WEIGHT_DECAY_RANGE": [0.000001, 0.001],
    "SEQUENCE_LENGTH_CHOICES": [20, 30, 60, 90],
    "BATCH_SIZE_CHOICES": [32, 64, 128]
  }
}
```

## Using Custom Configurations

### Method 1: Command Line Arguments

```bash
python scripts/train.py \
    --model-type lstm3_attention \
    --epochs 100 \
    --batch-size 128 \
    --lr 5e-5
```

### Method 2: Modify JSON Config

Edit `config/model.json`:

```json
{
  "model": {
    "training": {
      "LEARNING_RATE": 5e-5,
      "BATCH_SIZE": 128
    }
  }
}
```

### Method 3: Python Code

```python
from src.config import load_config

config = load_config('model')

# Modify training parameters
config.model.training.LEARNING_RATE = 5e-5
config.model.training.BATCH_SIZE = 128

# Use with model
from src.models import create_model

model = create_model(
    model_type="lstm3_attention",
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

## Environment Variables

```bash
# CUDA device
export CUDA_VISIBLE_DEVICES=0

# Python path
export PYTHONPATH=/path/to/research_02

# Disable logging
export PYTHONUNBUFFERED=1
```
