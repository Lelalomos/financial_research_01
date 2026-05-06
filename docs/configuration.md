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

### Technical Indicators

```json
{
  "data": {
    "technical_indicators": {
      "EMA_PERIODS": [50, 100, 200],
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
        "fibonacci_features": false,
        "candlestick_patterns": true,
        "vix": true,
        "commodities": true,
        "treasury_yields": true,
        "time_features": true,
        "financial_metrics": true,
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

The `polars_*` flags are disabled by default. They enable opt-in Polars
implementations for time features, Fibonacci features, and external data merges
while preserving pandas DataFrame outputs. See `docs/polars_migration.md`.

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
      "NUM_EPOCHS": 200,
      "EARLY_STOPPING_PATIENCE": 15,
      "OPTIMIZER": "adam",
      "SCHEDULER": "reduce_on_plateau"
    },
    "loss": {
      "LOSS_TYPE": "huber",
      "HUBER_DELTA": 0.1
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
config.model.training.BATCH_SIZE = 256
config.model.training.NUM_EPOCHS = 200
config.model.training.EARLY_STOPPING_PATIENCE = 15
```

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
See `docs/lightning_backend.md` for compatibility and checkpoint details.

### Loss Function

```python
config.model.loss.LOSS_TYPE = "huber"  # 'huber', 'mse', 'mae', 'smooth_l1'
config.model.loss.HUBER_DELTA = 0.1
```

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
      "EXPERIMENT_NAME": "crnn-financial-prediction",
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
