"""
Transformer model with 4-layer BiLSTM for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month)
- 4-layer BiLSTM with variable hidden sizes (128, 256, 512, 256)
- Positional encoding
- Transformer encoder
- Fully connected layers
"""

import torch
import torch.nn as nn
from typing import Optional

from src.config import load_config
from .crnn_attention import EmbeddingLayer, BiLSTM4Block


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            Output: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer + 4-layer BiLSTM model for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month, dividend_flag)
    2. Concatenate embeddings + features
    3. 4-layer BiLSTM with variable hidden sizes (128, 256, 512, 256)
    4. Project to d_model
    5. Positional encoding
    6. Transformer encoder
    7. Mean pooling
    8. Single Linear FC layer
    9. Output (percent change prediction)
    """

    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config
    ):
        """
        Initialize Transformer model.

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
            hidden_sizes=config.model.models.transformer.LSTM4_HIDDEN_SIZES,
            dropout=config.model.models.transformer.LSTM4_DROPOUT
        )

        # Project LSTM output to d_model
        self.lstm_projection = nn.Linear(self.lstm.output_dim, config.model.models.transformer.TRANSFORMER_D_MODEL)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.model.models.transformer.TRANSFORMER_D_MODEL)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.models.transformer.TRANSFORMER_D_MODEL,
            nhead=config.model.models.transformer.TRANSFORMER_NUM_HEADS,
            dim_feedforward=config.model.models.transformer.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.model.models.transformer.TRANSFORMER_DROPOUT,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.models.transformer.TRANSFORMER_NUM_LAYERS
        )

        # Single Linear FC layer (like other models with BiLSTM4)
        self.fc = nn.Linear(config.model.models.transformer.TRANSFORMER_D_MODEL, 1)
        self.fc_dropout = nn.Dropout(config.model.models.transformer.TRANSFORMER_DROPOUT)

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

        # Project LSTM output to d_model
        x = self.lstm_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

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
) -> TransformerModel:
    """
    Create Transformer model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config instance

    Returns:
        TransformerModel instance
    """
    if config is None:
        from src.config import load_config
        config = load_config('model')

    return TransformerModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
