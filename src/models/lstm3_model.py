"""
LSTM3 model (3-layer BiLSTM) for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month)
- 3-layer BiLSTM (separate way)
- Fully connected layers
"""

import torch
import torch.nn as nn
from typing import Optional

from src.config import load_config
from .crnn_attention import EmbeddingLayer, BiLSTMBlock


class LSTM3Model(nn.Module):
    """
    LSTM3 model (3-layer BiLSTM) for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month)
    2. Concatenate embeddings + features
    3. 3-layer BiLSTM (using LSTM3_* config parameters)
    4. Fully connected layers
    5. Output (percent change prediction)
    """

    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config
    ):
        """
        Initialize LSTM3 model.

        Args:
            num_features: Number of input features
            num_stocks: Number of unique stocks
            num_groups: Number of unique groups
            config instance
        """
        super().__init__()

        self.config = config

        # Embedding layer (reusing from crnn_attention.py)
        self.embeddings = EmbeddingLayer(
            num_stocks=num_stocks,
            num_groups=num_groups,
            config=config
        )

        # Calculate input dimension after embeddings
        embedding_dim = self.embeddings.output_dim
        lstm_input_dim = embedding_dim + num_features

        # 3-layer BiLSTM block (using LSTM3_* parameters)
        self.lstm = BiLSTMBlock(
            input_size=lstm_input_dim,
            hidden_size=config.model.models.lstm3.LSTM3_HIDDEN_SIZE,
            num_layers=config.model.models.lstm3.LSTM3_NUM_LAYERS,  # Fixed at 3
            dropout=config.model.models.lstm3.LSTM3_DROPOUT,
            use_layer_norm=config.model.models.lstm3.LSTM3_USE_LAYER_NORM
        )

        # Fully connected layers (same as other models)
        fc_input_dim = self.lstm.output_dim

        fc_layers = []
        prev_dim = fc_input_dim

        for fc_size in config.model.models.lstm3.FC_HIDDEN_SIZES:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(config.model.models.lstm3.FC_DROPOUT)
            ])

            if config.model.models.lstm3.FC_USE_BATCH_NORM:
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

        # Use last time step
        x = x[:, -1, :]

        # Fully connected
        output = self.fc(x)

        return output


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config = None
) -> LSTM3Model:
    """
    Create LSTM3 model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config instance

    Returns:
        LSTM3Model instance
    """
    if config is None:
        from src.config import load_config
        config = load_config('model')

    return LSTM3Model(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
