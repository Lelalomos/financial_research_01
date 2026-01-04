"""
CRNN model (without attention) for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month)
- CNN feature extraction
- BiLSTM layers
- Fully connected layers
"""

import torch
import torch.nn as nn
from typing import Optional

from src.config import load_config
from .crnn_attention import EmbeddingLayer, CNNBlock, BiLSTMBlock


class CRNNModel(nn.Module):
    """
    CRNN model (no attention) for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month)
    2. Concatenate embeddings + features
    3. CNN feature extraction
    4. BiLSTM layers
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
        Initialize CRNN model.

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
        cnn_input_dim = embedding_dim + num_features

        # CNN block
        self.cnn = CNNBlock(
            input_dim=cnn_input_dim,
            channels=config.model.models.crnn.CNN_CHANNELS,
            kernel_size=config.model.models.crnn.CNN_KERNEL_SIZE,
            pool_size=config.model.models.crnn.CNN_POOL_SIZE,
            use_batch_norm=config.model.models.crnn.CNN_USE_BATCH_NORM
        )

        # BiLSTM block
        self.lstm = BiLSTMBlock(
            input_size=self.cnn.output_dim,
            hidden_size=config.model.models.crnn.RNN_HIDDEN_SIZE,
            num_layers=config.model.models.crnn.RNN_NUM_LAYERS,
            dropout=config.model.models.crnn.RNN_DROPOUT,
            use_layer_norm=config.model.models.crnn.USE_LAYER_NORM
        )

        # Fully connected layers
        fc_input_dim = self.lstm.output_dim

        fc_layers = []
        prev_dim = fc_input_dim

        for fc_size in config.model.models.crnn.FC_HIDDEN_SIZES:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(config.model.models.crnn.FC_DROPOUT)
            ])

            if config.model.models.crnn.FC_USE_BATCH_NORM:
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

        # CNN feature extraction
        x = self.cnn(x)

        # BiLSTM
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
) -> CRNNModel:
    """
    Create CRNN model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config instance

    Returns:
        CRNNModel instance
    """
    if config is None:
        from src.config import load_config
        config = load_config('model')

    return CRNNModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
