"""
BiLSTM4 + Attention model for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month, dividend_flag)
- 4-layer BiLSTM with different hidden sizes (128, 256, 512, 256)
- MultiheadAttention (4 heads)
- Single Linear FC layer
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.config import load_config
from .crnn_attention import EmbeddingLayer


class BiLSTM4Block(nn.Module):
    """
    4-layer bidirectional LSTM with different hidden sizes per layer.

    Architecture:
    - Layer 1: input_size -> 128 hidden (bidirectional -> 256 output)
    - Layer 2: 256 -> 256 hidden (bidirectional -> 512 output)
    - Layer 3: 512 -> 512 hidden (bidirectional -> 1024 output)
    - Layer 4: 1024 -> 256 hidden (bidirectional -> 512 output)
    - Dropout: 0.4 between layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, ...],
        dropout: float
    ):
        """
        Initialize 4-layer BiLSTM block.

        Args:
            input_size: Input feature size
            hidden_sizes: Tuple of 4 hidden sizes (one per layer)
            dropout: Dropout rate
        """
        super().__init__()

        if len(hidden_sizes) != 4:
            raise ValueError(f"hidden_sizes must have exactly 4 elements, got {len(hidden_sizes)}")

        self.hidden_sizes = hidden_sizes

        # Layer 1: input_size -> hidden_sizes[0] (bidirectional)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            batch_first=True,
            bidirectional=True
        )

        # Layer 2: hidden_sizes[0]*2 -> hidden_sizes[1] (bidirectional)
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0] * 2,
            hidden_size=hidden_sizes[1],
            batch_first=True,
            bidirectional=True
        )

        # Layer 3: hidden_sizes[1]*2 -> hidden_sizes[2] (bidirectional)
        self.lstm3 = nn.LSTM(
            input_size=hidden_sizes[1] * 2,
            hidden_size=hidden_sizes[2],
            batch_first=True,
            bidirectional=True
        )

        # Layer 4: hidden_sizes[2]*2 -> hidden_sizes[3] (bidirectional)
        self.lstm4 = nn.LSTM(
            input_size=hidden_sizes[2] * 2,
            hidden_size=hidden_sizes[3],
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            Output: (batch, seq_len, hidden_sizes[3] * 2)
        """
        # Layer 1
        x, _ = self.lstm1(x)
        x = self.dropout(x)

        # Layer 2
        x, _ = self.lstm2(x)
        x = self.dropout(x)

        # Layer 3
        x, _ = self.lstm3(x)
        x = self.dropout(x)

        # Layer 4
        x, _ = self.lstm4(x)
        x = self.dropout(x)

        return x

    @property
    def output_dim(self) -> int:
        """Output dimension (bidirectional = 2x last hidden size)."""
        return self.hidden_sizes[3] * 2


class BiLSTM4AttentionModel(nn.Module):
    """
    BiLSTM4 + Attention model for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month, dividend_flag)
    2. Concatenate embeddings + features
    3. 4-layer BiLSTM with variable hidden sizes (128, 256, 512, 256)
    4. MultiheadAttention (4 heads)
    5. Mean pooling over sequence
    6. Single Linear FC layer
    7. Output (percent change prediction)
    """

    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config
    ):
        """
        Initialize BiLSTM4 + Attention model.

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
        lstm_input_dim = embedding_dim + num_features

        # 4-layer BiLSTM block with variable hidden sizes
        self.lstm = BiLSTM4Block(
            input_size=lstm_input_dim,
            hidden_sizes=config.model.models.bilstm4_attention.LSTM4_HIDDEN_SIZES,
            dropout=config.model.models.bilstm4_attention.LSTM4_DROPOUT
        )

        # Multihead attention (4 heads, dropout 0.4)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm.output_dim,
            num_heads=config.model.models.bilstm4_attention.LSTM4_ATTENTION_HEADS,
            dropout=config.model.models.bilstm4_attention.LSTM4_ATTENTION_DROPOUT,
            batch_first=True
        )

        # Single Linear FC layer
        self.fc = nn.Linear(self.lstm.output_dim, 1)
        self.fc_dropout = nn.Dropout(config.model.models.bilstm4_attention.LSTM4_DROPOUT)

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

        # 4-layer BiLSTM
        x = self.lstm(x)

        # MultiheadAttention (self-attention)
        x, _ = self.attention(x, x, x)

        # Mean pooling over sequence
        x = x.mean(dim=1)

        # Apply dropout before FC
        x = self.fc_dropout(x)

        # Single Linear FC layer
        output = self.fc(x)

        return output


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config = None
) -> BiLSTM4AttentionModel:
    """
    Create BiLSTM4 + Attention model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config instance

    Returns:
        BiLSTM4AttentionModel instance
    """
    if config is None:
        from src.config import load_config
        config = load_config('model')

    return BiLSTM4AttentionModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
