"""
Models module for Multi-Model Financial Forecasting.

This module provides all model variants:
- CRNN (CNN + BiLSTM)
- RNN (BiLSTM only)
- RNN + Attention
- CRNN + Attention (recommended)
- Transformer
- LSTM3 (3-layer BiLSTM)
- LSTM3 + Attention
- BiLSTM4 + Attention (4-layer with variable hidden sizes)
"""

from typing import Optional, Type

import torch.nn as nn

from src.config import load_config
from .crnn_model import CRNNModel, create_model as create_crnn
from .rnn_model import RNNModel, create_model as create_rnn
from .rnn_attention import RNNAttentionModel, create_model as create_rnn_attention
from .crnn_attention import CRNNAttentionModel, create_model as create_crnn_attention
from .transformer_model import TransformerModel, create_model as create_transformer
from .lstm3_model import LSTM3Model, create_model as create_lstm3
from .lstm3_attn_model import LSTM3AttentionModel, create_model as create_lstm3_attention
from .bilstm4_attn_model import BiLSTM4AttentionModel, create_model as create_bilstm4_attention
from .multi_branch_bilstm import MultiBranchBiLSTMModel, create_model as create_multi_branch_bilstm

# Map model types to their create functions
_MODEL_REGISTRY = {
    'crnn': (CRNNModel, create_crnn),
    'rnn': (RNNModel, create_rnn),
    'rnn_attention': (RNNAttentionModel, create_rnn_attention),
    'crnn_attention': (CRNNAttentionModel, create_crnn_attention),
    'transformer': (TransformerModel, create_transformer),
    'lstm3': (LSTM3Model, create_lstm3),
    'lstm3_attention': (LSTM3AttentionModel, create_lstm3_attention),
    'bilstm4_attention': (BiLSTM4AttentionModel, create_bilstm4_attention),
    'multi_branch_bilstm': (MultiBranchBiLSTMModel, create_multi_branch_bilstm),
}

__all__ = [
    'CRNNModel',
    'RNNModel',
    'RNNAttentionModel',
    'CRNNAttentionModel',
    'TransformerModel',
    'LSTM3Model',
    'LSTM3AttentionModel',
    'BiLSTM4AttentionModel',
    'MultiBranchBiLSTMModel',
    'create_model',
    'get_model_class',
    'list_available_models',
]


def create_model(
    model_type: str,
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config: Optional[object] = None,
    feature_cols: Optional[list] = None,
) -> nn.Module:
    """
    Create a model by type.

    Args:
        model_type: Type of model ('crnn', 'rnn', 'rnn_attention', 'crnn_attention', 'transformer')
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config instance

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
        config = load_config('model')

    kwargs = {
        'num_features': num_features,
        'num_stocks': num_stocks,
        'num_groups': num_groups,
        'config': config,
    }
    if model_type == 'multi_branch_bilstm':
        kwargs['feature_cols'] = feature_cols

    return create_fn(**kwargs)


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
