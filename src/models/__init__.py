"""
Models module for CRNN Financial Prediction Model.

This module provides all model variants:
- CRNN (CNN + BiLSTM)
- RNN (BiLSTM only)
- RNN + Attention
- CRNN + Attention (recommended)
- Transformer
- LSTM3 (3-layer BiLSTM)
- LSTM3 + Attention
"""

from typing import Optional, Type

import torch.nn as nn

from config.model_config import ModelConfig
from .crnn_model import CRNNModel, create_model as create_crnn
from .rnn_model import RNNModel, create_model as create_rnn
from .rnn_attention import RNNAttentionModel, create_model as create_rnn_attention
from .crnn_attention import CRNNAttentionModel, create_model as create_crnn_attention
from .transformer_model import TransformerModel, create_model as create_transformer
from .lstm3_model import LSTM3Model, create_model as create_lstm3
from .lstm3_attn_model import LSTM3AttentionModel, create_model as create_lstm3_attention

# Map model types to their create functions
_MODEL_REGISTRY = {
    'crnn': (CRNNModel, create_crnn),
    'rnn': (RNNModel, create_rnn),
    'rnn_attention': (RNNAttentionModel, create_rnn_attention),
    'crnn_attention': (CRNNAttentionModel, create_crnn_attention),
    'transformer': (TransformerModel, create_transformer),
    'lstm3': (LSTM3Model, create_lstm3),
    'lstm3_attention': (LSTM3AttentionModel, create_lstm3_attention),
}

__all__ = [
    'CRNNModel',
    'RNNModel',
    'RNNAttentionModel',
    'CRNNAttentionModel',
    'TransformerModel',
    'LSTM3Model',
    'LSTM3AttentionModel',
    'create_model',
    'get_model_class',
    'list_available_models',
]


def create_model(
    model_type: str,
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config: Optional[ModelConfig] = None
) -> nn.Module:
    """
    Create a model by type.

    Args:
        model_type: Type of model ('crnn', 'rnn', 'rnn_attention', 'crnn_attention', 'transformer')
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config: ModelConfig instance

    Returns:
        Model instance
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )

    _, create_fn = _MODEL_REGISTRY[model_type]

    if config is None:
        config = ModelConfig(MODEL_TYPE=model_type)
    else:
        # Update config model type
        config.MODEL_TYPE = model_type

    return create_fn(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )


def get_model_class(model_type: str) -> Type[nn.Module]:
    """
    Get model class by type.

    Args:
        model_type: Type of model

    Returns:
        Model class
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )

    model_class, _ = _MODEL_REGISTRY[model_type]
    return model_class


def list_available_models() -> list:
    """Return list of available model types."""
    return list(_MODEL_REGISTRY.keys())
