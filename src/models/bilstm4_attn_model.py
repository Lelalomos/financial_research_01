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

from src.config import load_config
from .crnn_attention import EmbeddingLayer, BiLSTM4Block


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
