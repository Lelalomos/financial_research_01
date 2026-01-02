# Configuration Guide

## Overview

There are two main configuration files:

1. **`config/data_config.py`** - Data sources, features, preprocessing
2. **`config/model_config.py`** - Model architecture, training hyperparameters

## Data Configuration

### Basic Settings

```python
from config.data_config import DataConfig

config = DataConfig(
    START_DATE="2015-01-01",    # Data start date
    END_DATE=None,               # None = current date
    TRAIN_RATIO=0.70,            # Training split ratio
    TEST_RATIO=0.20,             # Test split ratio
    VAL_RATIO=0.10               # Validation split ratio
)
```

### Sequence Parameters

```python
config = DataConfig(
    SEQUENCE_LENGTH=30,           # Lookback window (days)
    PREDICTION_HORIZON=5,        # Days ahead to predict
    TARGET_THRESHOLD=10.0        # For target normalization
)
```

### Technical Indicators

```python
config = DataConfig(
    EMA_PERIODS=(50, 100, 200),  # EMA periods
    RSI_PERIOD=14,               # RSI period
    STOCHRSI_PERIOD=14,          # StochRSI period
    MACD_PARAMS=(12, 26, 9),     # MACD (fast, slow, signal)
    USE_CANDLESTICK_PATTERNS=True # Enable TA-Lib patterns
)
```

### External Data

```python
config = DataConfig(
    VIX_SYMBOL="^VIX",
    COMMODITIES={
        'GC=F': 'Gold',
        'HG=F': 'Copper',
        'ZC=F': 'Corn',
        'ZS=F': 'Soybeans',
        'CC=F': 'Cocoa',
        'SI=F': 'Silver'
    },
    TREASURY_YIELDS=('DGS10', 'DGS30', 'DGS2')
)
```

### Feature Flags (Ablation)

Disable specific feature groups:

```python
config = DataConfig(
    FEATURE_FLAGS={
        'price_features': True,
        'ema_features': True,
        'rsi_features': True,
        'stochrsi_features': True,
        'macd_features': True,
        'candlestick_patterns': True,
        'vix': True,
        'commodities': True,
        'treasury_yields': True,
        'time_features': True,
    }
)
```

### Normalization

```python
config = DataConfig(
    NORMALIZATION_METHOD="log_transform",  # or 'standard', 'minmax', 'robust'
    LOG_TRANSFORM_OFFSET=1.0
)
```

## Model Configuration

### Model Selection

```python
from config.model_config import ModelConfig

config = ModelConfig(
    MODEL_TYPE="crnn_attention"  # Options: 'crnn', 'rnn', 'rnn_attention',
                                  #         'crnn_attention', 'transformer'
)
```

### Embedding Dimensions

```python
config = ModelConfig(
    EMBEDDING_DIM_STOCK=64,
    EMBEDDING_DIM_GROUP=32,
    EMBEDDING_DIM_DAY=16,
    EMBEDDING_DIM_MONTH=16
)
```

### CNN Architecture

```python
config = ModelConfig(
    CNN_CHANNELS=(64, 128),
    CNN_KERNEL_SIZE=3,
    CNN_POOL_SIZE=2,
    CNN_USE_BATCH_NORM=False
)
```

### RNN Architecture

```python
config = ModelConfig(
    RNN_HIDDEN_SIZE=128,
    RNN_NUM_LAYERS=2,
    RNN_DROPOUT=0.2,
    USE_BIDIRECTIONAL=True
)
```

### Attention

```python
config = ModelConfig(
    USE_ATTENTION=True,
    ATTENTION_HEADS=4,
    ATTENTION_DROPOUT=0.1
)
```

### Fully Connected Layers

```python
config = ModelConfig(
    FC_HIDDEN_SIZES=(256, 128),
    FC_DROPOUT=0.3,
    FC_USE_BATCH_NORM=False
)
```

### Training Parameters

```python
config = ModelConfig(
    LEARNING_RATE=1e-4,
    WEIGHT_DECAY=1e-5,
    BATCH_SIZE=256,
    NUM_EPOCHS=200,
    EARLY_STOPPING_PATIENCE=15,
    GRADIENT_CLIP_VALUE=1.0
)
```

### Loss Function

```python
config = ModelConfig(
    LOSS_TYPE="huber",    # Options: 'huber', 'mse', 'mae', 'smooth_l1'
    HUBER_DELTA=0.1
)
```

### Optimizer

```python
config = ModelConfig(
    OPTIMIZER="adam",     # Options: 'adam', 'adamw', 'sgd', 'rmsprop'
    SCHEDULER="reduce_on_plateau",  # None, 'reduce_on_plateau', 'cosine', 'step'
    SCHEDULER_PARAMS={
        'mode': 'min',
        'factor': 0.5,
        'patience': 10
    }
)
```

### Device & Performance

```python
config = ModelConfig(
    DEVICE="cuda",         # 'cuda' or 'cpu'
    NUM_WORKERS=4,
    PIN_MEMORY=True,
    USE_MIXED_PRECISION=False
)
```

## Using Custom Configurations

### Method 1: Command Line Arguments

```bash
python scripts/train.py \
    --model-type crnn_attention \
    --epochs 100 \
    --batch-size 128 \
    --lr 5e-5
```

### Method 2: Config Override File

Create `config/custom_config.json`:

```json
{
    "RNN_HIDDEN_SIZE": 256,
    "ATTENTION_HEADS": 8,
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 5e-5
}
```

Then:

```bash
python scripts/train.py --config config/custom_config.json
```

### Method 3: Python Code

```python
from config.model_config import ModelConfig

config = ModelConfig(
    RNN_HIDDEN_SIZE=256,
    ATTENTION_HEADS=8,
    BATCH_SIZE=128
)

from src.models import create_model

model = create_model(
    model_type="crnn_attention",
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

## Preset Configurations

### Quick Training (Less Accurate)

```python
config = ModelConfig(
    MODEL_TYPE="rnn",
    RNN_HIDDEN_SIZE=64,
    RNN_NUM_LAYERS=1,
    BATCH_SIZE=512,
    NUM_EPOCHS=50
)
```

### Best Performance

```python
config = ModelConfig(
    MODEL_TYPE="crnn_attention",
    RNN_HIDDEN_SIZE=256,
    RNN_NUM_LAYERS=2,
    ATTENTION_HEADS=8,
    BATCH_SIZE=256,
    NUM_EPOCHS=200
)
```

### Low Memory

```python
config = ModelConfig(
    MODEL_TYPE="rnn",
    RNN_HIDDEN_SIZE=64,
    BATCH_SIZE=32,
    NUM_WORKERS=1,
    PIN_MEMORY=False
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
