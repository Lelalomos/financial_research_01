"""
Transformer model for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month)
- Positional encoding
- Transformer encoder
- Fully connected layers
"""

import torch
import torch.nn as nn
from typing import Optional

from config.model_config import ModelConfig
from .crnn_attention import EmbeddingLayer


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
    Transformer model for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month)
    2. Concatenate embeddings + features
    3. Positional encoding
    4. Transformer encoder
    5. Mean pooling
    6. Fully connected layers
    7. Output (percent change prediction)
    """

    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config: ModelConfig
    ):
        """
        Initialize Transformer model.

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
        input_dim = embedding_dim + num_features

        # Project to d_model if needed
        self.input_projection = nn.Linear(input_dim, config.TRANSFORMER_D_MODEL)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.TRANSFORMER_D_MODEL)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NUM_HEADS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.TRANSFORMER_DROPOUT,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.TRANSFORMER_NUM_LAYERS
        )

        # Fully connected layers
        fc_input_dim = config.TRANSFORMER_D_MODEL

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

        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

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
) -> TransformerModel:
    """
    Create Transformer model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config: ModelConfig instance

    Returns:
        TransformerModel instance
    """
    if config is None:
        config = ModelConfig()

    return TransformerModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
