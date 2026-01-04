"""
RNN + Attention model for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month)
- BiLSTM layers
- MultiheadAttention
- Fully connected layers
"""

import torch
import torch.nn as nn
from typing import Optional

from src.config import load_config
from .crnn_attention import EmbeddingLayer, BiLSTMBlock


class RNNAttentionModel(nn.Module):
    """
    RNN + Attention model for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month)
    2. Concatenate embeddings + features
    3. BiLSTM layers
    4. MultiheadAttention
    5. Fully connected layers
    6. Output (percent change prediction)
    """

    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config
    ):
        """
        Initialize RNN + Attention model.

        Args:
            num_features: Number of input features
            num_stocks: Number of unique stocks
            num_groups: Number of unique groups
            config instance
        """
        super().__init__()

        self.config = config

        # Embedding layer
        self.embeddings = EmbeddingLayer(
            num_stocks=num_stocks,
            num_groups=num_groups,
            config=config
        )

        # Calculate input dimension after embeddings
        embedding_dim = self.embeddings.output_dim
        rnn_input_dim = embedding_dim + num_features

        # BiLSTM block
        self.lstm = BiLSTMBlock(
            input_size=rnn_input_dim,
            hidden_size=config.model.models.rnn_attention.RNN_HIDDEN_SIZE,
            num_layers=config.model.models.rnn_attention.RNN_NUM_LAYERS,
            dropout=config.model.models.rnn_attention.RNN_DROPOUT,
            use_layer_norm=config.model.models.rnn_attention.USE_LAYER_NORM
        )

        # Multihead attention
        if config.model.models.rnn_attention.USE_ATTENTION:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.lstm.output_dim,
                num_heads=config.model.models.rnn_attention.ATTENTION_HEADS,
                dropout=config.model.models.rnn_attention.ATTENTION_DROPOUT,
                batch_first=True
            )
        else:
            self.attention = None

        # Fully connected layers
        fc_input_dim = self.lstm.output_dim

        fc_layers = []
        prev_dim = fc_input_dim

        for fc_size in config.model.models.rnn_attention.FC_HIDDEN_SIZES:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(config.model.models.rnn_attention.FC_DROPOUT)
            ])

            if config.model.models.rnn_attention.FC_USE_BATCH_NORM:
                fc_layers.append(nn.BatchNorm1d(fc_size))

            prev_dim = fc_size

        fc_layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self,
        features: torch.Tensor,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: (batch, seq_len, num_features)
            stock_id: (batch, seq_len)
            group_id: (batch, seq_len)
            day: (batch, seq_len)
            month: (batch, seq_len)
            dividend_flag: (batch, seq_len) - 1=has dividend, 2=no dividend

        Returns:
            Output: (batch, 1) - percent change prediction
        """
        # Get embeddings
        emb = self.embeddings(stock_id, group_id, day, month, dividend_flag)

        # Concatenate embeddings and features
        x = torch.cat([emb, features], dim=-1)

        # BiLSTM
        x = self.lstm(x)

        # Attention
        if self.attention is not None:
            x, _ = self.attention(x, x, x)  # Self-attention
            x = x.mean(dim=1)  # Mean pooling over sequence
        else:
            x = x[:, -1, :]  # Use last time step

        # Fully connected
        output = self.fc(x)

        return output


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config = None
) -> RNNAttentionModel:
    """
    Create RNN + Attention model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config instance

    Returns:
        RNNAttentionModel instance
    """
    if config is None:
        from src.config import load_config
        config = load_config('model')

    return RNNAttentionModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
