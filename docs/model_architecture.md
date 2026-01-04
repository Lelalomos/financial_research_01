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

### 5. Transformer (with 4-layer BiLSTM)

```python
from src.models import TransformerModel

model = TransformerModel(
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config
)
```

**Architecture**:
- 4-layer BiLSTM (128, 256, 512, 256) for sequential modeling
- Project to d_model (256)
- Positional encoding
- Transformer encoder (4 layers, 8 heads)
- Single Linear FC layer

**Use case**: Deep architecture combining LSTM sequential modeling with Transformer attention

## Configuration

All model parameters are in `config/model.json` with separate sections for each model type:

```python
from src.config import load_config

config = load_config('model')

# Access embeddings (shared across all models)
stock_emb_dim = config.model.embeddings.EMBEDDING_DIM_STOCK  # 64

# Access model-specific parameters (e.g., LSTM3+Attention)
lstm3_hidden = config.model.models.lstm3_attention.LSTM3_HIDDEN_SIZE  # 256
lstm3_heads = config.model.models.lstm3_attention.LSTM3_ATTENTION_HEADS  # 8

# Access training parameters (shared)
learning_rate = config.model.training.LEARNING_RATE  # 0.0001
batch_size = config.model.training.BATCH_SIZE  # 128
```

## Usage Example

```python
from src.models import create_model
from src.config import load_config

config = load_config('model')

# Optionally modify parameters
config.model.training.LEARNING_RATE = 1e-4

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
