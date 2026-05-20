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

### 6. BiLSTM4 + Attention

**Architecture**:
- 4-layer BiLSTM (128, 256, 512, 256)
- Multihead attention over sequence outputs
- Feed-forward MLP applied independently to each timestep after attention
- Mean pooling across timesteps
- Final linear output layer

**Use case**: Larger recurrent-attention model with extra timestep-level
nonlinearity before pooling

### 7. Multi-Branch BiLSTM

```python
from src.models import create_model

model = create_model(
    model_type="multi_branch_bilstm",
    num_features=50,
    num_stocks=500,
    num_groups=20,
    config=config,
    feature_cols=feature_cols,
)
```

**Architecture**:
- Technical branch: BiLSTM over raw price and indicator features
- Geometric branch: BiLSTM over structural features such as slopes, channel,
  swing-distance, and Fibonacci-derived columns
- Macro/financial branch: BiLSTM over external and financial-metric features
- Fusion head: concatenates pooled branch outputs and predicts the target

**Use case**: Experimental QuantAgent-inspired architecture that separates noisy
raw technical inputs from higher-level geometric context and slower macro data

### 8. Kronos

```python
from src.models import create_kronos_model, create_kronos_tokenizer

tokenizer = create_kronos_tokenizer(config)
model = create_kronos_model(config)
```

**Architecture**:
- Tokenizer stage:
  - takes continuous market rows such as `open, high, low, close, volume, amount`
  - projects them into a Transformer latent space
  - compresses the latent vectors with binary spherical quantization
  - splits each token into two parts:
    - `s1`: coarse token
    - `s2`: fine token
- Generator stage:
  - embeds `s1` and `s2` token IDs with a hierarchical embedding
  - adds calendar/time embeddings
  - can also add `stock_id` and `group_id` embeddings from the prepared-data pipeline
  - runs stacked causal Transformer blocks
  - predicts `s1` first, then predicts `s2` conditioned on `s1`
- Predictor stage:
  - normalizes input history
  - converts history into token IDs with the tokenizer
  - generates future tokens autoregressively
  - decodes generated tokens back into continuous price/volume outputs

**How it is different from the other models**:
- Most models in this repo predict one numeric target directly.
- Kronos is a generative sequence model.
- It first turns continuous data into discrete token IDs, then predicts future
  token IDs, then decodes them back to numeric values.
- Kronos can now consume prepared-data categorical context such as `stock_id`
  and `group_id`, but it still remains a token-generation model rather than a
  direct regression head.

**Current default size in this repo**:
- Tokenizer: about `3.96M` params
- Kronos generator: about `5.95M` params
- Combined: about `9.90M` params

**Use case**: Experimental tokenized time-series generation model for
autoregressive multi-feature forecasting

**Reference / credit**:
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
- Paper link: `https://arxiv.org/abs/2508.02739`
- Upstream license: `MIT`

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

# Access Kronos parameters
kronos_d_model = config.model.models.kronos.network.D_MODEL  # 256
kronos_max_context = config.model.models.kronos.predictor.MAX_CONTEXT  # 512

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

## Kronos Summary

| Component | Output Shape | Parameters |
|-----------|-------------|------------|
| Tokenizer encoder/decoder | (batch, seq, 6) | ~3.96M |
| Kronos token generator | (batch, seq, s1/s2 logits) | ~5.95M |
| Predictor wrapper | DataFrame forecast | 0 |
| **Combined** | | **~9.90M** |

This repo contains a local integration of the Kronos code. Credit for the
original model idea and upstream implementation belongs to the original Kronos
authors and repository listed above.
