"""
LSTM3 + Attention model for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month, dividend_flag)
- 3-layer BiLSTM
- MultiheadAttention
- Fully connected layers
"""

import torch
import torch.nn as nn
from typing import Optional

from config.model_config import ModelConfig
from .crnn_attention import EmbeddingLayer, BiLSTMBlock


class LSTM3AttentionModel(nn.Module):
    """
    LSTM3 + Attention model for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month, dividend_flag)
    2. Concatenate embeddings + features
    3. 3-layer BiLSTM (using LSTM3_* config parameters)
    4. MultiheadAttention (using LSTM3_ATTENTION_* parameters)
    5. Fully connected layers
    6. Output (percent change prediction)
    """

    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config: ModelConfig
    ):
        """
        Initialize LSTM3 + Attention model.

        Args:
            num_features: Number of input features
            num_stocks: Number of unique stocks
            num_groups: Number of unique groups
            config: ModelConfig instance
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

        # 3-layer BiLSTM block
        self.lstm = BiLSTMBlock(
            input_size=lstm_input_dim,
            hidden_size=config.LSTM3_HIDDEN_SIZE,
            num_layers=config.LSTM3_NUM_LAYERS,
            dropout=config.LSTM3_DROPOUT,
            use_layer_norm=config.LSTM3_USE_LAYER_NORM
        )

        # Determine attention embedding dimension
        if config.LSTM3_ATTENTION_HIDDEN_SIZE is None:
            attn_embed_dim = self.lstm.output_dim
        else:
            attn_embed_dim = config.LSTM3_ATTENTION_HIDDEN_SIZE

        # Multihead attention (using LSTM3_ATTENTION_* parameters)
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=config.LSTM3_ATTENTION_HEADS,
            dropout=config.LSTM3_ATTENTION_DROPOUT,
            batch_first=True
        )

        # Project LSTM output to attention dimension if needed
        if attn_embed_dim != self.lstm.output_dim:
            self.attention_projection = nn.Linear(
                self.lstm.output_dim,
                attn_embed_dim
            )
        else:
            self.attention_projection = nn.Identity()

        # Fully connected layers
        fc_input_dim = attn_embed_dim

        fc_layers = []
        prev_dim = fc_input_dim

        for fc_size in config.FC_HIDDEN_SIZES:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(config.FC_DROPOUT)
            ])

            if config.FC_USE_BATCH_NORM:
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

        # 3-layer BiLSTM
        x = self.lstm(x)

        # Project to attention dimension if needed
        x = self.attention_projection(x)

        # MultiheadAttention (self-attention)
        x, _ = self.attention(x, x, x)

        # Mean pooling over sequence
        x = x.mean(dim=1)

        # Fully connected
        output = self.fc(x)

        return output


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config: Optional[ModelConfig] = None
) -> LSTM3AttentionModel:
    """
    Create LSTM3 + Attention model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config: ModelConfig instance

    Returns:
        LSTM3AttentionModel instance
    """
    if config is None:
        config = ModelConfig()

    return LSTM3AttentionModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
