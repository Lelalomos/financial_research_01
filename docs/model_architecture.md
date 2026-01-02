# Model Architecture Documentation

## Overview

The CRNN (CNN + BiLSTM + Attention) model combines:
1. **CNN** for local feature extraction
2. **BiLSTM** for temporal dependencies
3. **MultiheadAttention** for sequence importance
4. **Embeddings** for categorical features

## Architecture Diagram

```
Input: (batch, seq_len, num_features)
    │
    ▼
┌─────────────────────────────────┐
│  Embedding Layer                │
│  ├─ Stock Embedding (64-dim)    │
│  ├─ Group Embedding (32-dim)    │
│  ├─ Day Embedding (16-dim)      │
│  └─ Month Embedding (16-dim)    │
└─────────────────────────────────┘
    │
    ▼ Concatenate
    │
    ▼
┌─────────────────────────────────┐
│  CNN Block                      │
│  ├─ Conv1d (in→64, k=3)        │
│  ├─ LeakyReLU                   │
│  ├─ Conv1d (64→128, k=3)       │
│  ├─ LeakyReLU                   │
│  └─ MaxPool1d (stride=2)       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  BiLSTM Block                   │
│  ├─ BiLSTM (128, 2 layers)      │
│  └─ LayerNorm                   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  MultiheadAttention             │
│  └─ 4 heads, 256-dim           │
└─────────────────────────────────┘
    │
    ▼ Mean Pool
    │
    ▼
┌─────────────────────────────────┐
│  Fully Connected                │
│  ├─ Linear (256→256)            │
│  ├─ LeakyReLU + Dropout         │
│  ├─ Linear (256→128)            │
│  ├─ LeakyReLU + Dropout         │
│  └─ Linear (128→1)              │
└─────────────────────────────────┘
    │
    ▼
Output: (batch, 1) - Percent change
```

## Model Variants

### 1. CRNN (No Attention)

```python
from src.models import CRNNModel

model = CRNNModel(
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

**Use case**: Baseline with CNN feature extraction

### 2. RNN (No CNN)

```python
from src.models import RNNModel

model = RNNModel(
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

**Use case**: Simple baseline, fastest training

### 3. RNN + Attention

```python
from src.models import RNNAttentionModel

model = RNNAttentionModel(
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

**Use case**: Attention for interpretability

### 4. CRNN + Attention (Recommended)

```python
from src.models import CRNNAttentionModel

model = CRNNAttentionModel(
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

**Use case**: Best performance, feature extraction + attention

### 5. Transformer

```python
from src.models import TransformerModel

model = TransformerModel(
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

**Use case**: Alternative approach, pure attention

## Configuration

All model parameters are in `config/model_config.py`:

```python
@dataclass
class ModelConfig:
    # Model selection
    MODEL_TYPE: str = "crnn_attention"

    # Embeddings
    EMBEDDING_DIM_STOCK: int = 64
    EMBEDDING_DIM_GROUP: int = 32
    EMBEDDING_DIM_DAY: int = 16
    EMBEDDING_DIM_MONTH: int = 16

    # CNN
    CNN_CHANNELS: tuple = (64, 128)
    CNN_KERNEL_SIZE: int = 3

    # RNN
    RNN_HIDDEN_SIZE: int = 128
    RNN_NUM_LAYERS: int = 2
    USE_BIDIRECTIONAL: bool = True

    # Attention
    ATTENTION_HEADS: int = 4

    # Training
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 256
    NUM_EPOCHS: int = 200
    EARLY_STOPPING_PATIENCE: int = 15
```

## Usage Example

```python
from src.models import create_model
from config.model_config import ModelConfig

config = ModelConfig(
    MODEL_TYPE="crnn_attention",
    RNN_HIDDEN_SIZE=256,
    ATTENTION_HEADS=8
)

model = create_model(
    model_type="crnn_attention",
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)

# Forward pass
output = model(
    features=torch.randn(32, 30, 50),
    stock_id=torch.randint(0, 500, (32, 30)),
    group_id=torch.randint(0, 20, (32, 30)),
    day=torch.randint(1, 32, (32, 30)),
    month=torch.randint(1, 13, (32, 30))
)
# output: (32, 1)
```

## Model Summary

| Component | Output Shape | Parameters |
|-----------|-------------|------------|
| Embeddings | (batch, seq, 128) | ~64K |
| CNN | (batch, seq, 128) | ~100K |
| BiLSTM | (batch, seq, 256) | ~400K |
| Attention | (batch, seq, 256) | ~200K |
| FC | (batch, 1) | ~100K |
| **Total** | | ~864K |
