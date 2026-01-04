"""
CRNN + Attention model for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month, dividend_flag)
- CNN feature extraction
- BiLSTM layers
- MultiheadAttention
- Fully connected layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.config import load_config


class EmbeddingLayer(nn.Module):
    """Embedding layer for categorical features."""

    def __init__(
        self,
        num_stocks: int,
        num_groups: int,
        config
    ):
        """
        Initialize embedding layer.

        Args:
            num_stocks: Number of unique stocks
            num_groups: Number of unique groups
            config instance
        """
        super().__init__()

        self.stock_embedding = nn.Embedding(
            num_embeddings=num_stocks,
            embedding_dim=config.model.embeddings.EMBEDDING_DIM_STOCK
        )

        self.group_embedding = nn.Embedding(
            num_embeddings=num_groups,
            embedding_dim=config.model.embeddings.EMBEDDING_DIM_GROUP
        )

        self.day_embedding = nn.Embedding(
            num_embeddings=32,  # Days 1-31 + padding
            embedding_dim=config.model.embeddings.EMBEDDING_DIM_DAY
        )

        self.month_embedding = nn.Embedding(
            num_embeddings=13,  # Months 1-12 + padding
            embedding_dim=config.model.embeddings.EMBEDDING_DIM_MONTH
        )

        self.dividend_flag_embedding = nn.Embedding(
            num_embeddings=3,  # 0=padding, 1=has dividend, 2=no dividend
            embedding_dim=config.model.embeddings.EMBEDDING_DIM_DIVIDEND_FLAG
        )

        self.dropout = nn.Dropout(config.model.embeddings.DROPOUT_EMBEDDING)

    def forward(
        self,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            stock_id: (batch, seq_len)
            group_id: (batch, seq_len)
            day: (batch, seq_len)
            month: (batch, seq_len)
            dividend_flag: (batch, seq_len) - 1=has dividend, 2=no dividend

        Returns:
            embeddings: (batch, seq_len, total_embedding_dim)
        """
        stock_emb = self.stock_embedding(stock_id)
        group_emb = self.group_embedding(group_id)
        day_emb = self.day_embedding(day)
        month_emb = self.month_embedding(month)
        dividend_emb = self.dividend_flag_embedding(dividend_flag)

        # Concatenate all embeddings
        embeddings = torch.cat([stock_emb, group_emb, day_emb, month_emb, dividend_emb], dim=-1)
        embeddings = self.dropout(embeddings)

        return embeddings

    @property
    def output_dim(self) -> int:
        """Total output dimension."""
        return (
            self.stock_embedding.embedding_dim +
            self.group_embedding.embedding_dim +
            self.day_embedding.embedding_dim +
            self.month_embedding.embedding_dim +
            self.dividend_flag_embedding.embedding_dim
        )


class CNNBlock(nn.Module):
    """CNN feature extraction block."""

    def __init__(
        self,
        input_dim: int,
        channels: tuple,
        kernel_size: int,
        pool_size: int,
        use_batch_norm: bool = False
    ):
        """
        Initialize CNN block.

        Args:
            input_dim: Input feature dimension
            channels: Tuple of output channels for each conv layer
            kernel_size: Kernel size for conv layers
            pool_size: Pooling size
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        layers = []
        in_channels = input_dim

        for out_channels in channels:
            layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.LeakyReLU(0.1),
            ])

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))

            in_channels = out_channels

        layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=1))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, features) or (batch, features, seq_len)

        Returns:
            Output: (batch, seq_len, channels[-1])
        """
        # Input is (batch, seq_len, features)
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        x = self.cnn(x)

        # Transpose back to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        return x

    @property
    def output_dim(self) -> int:
        """Output channel dimension."""
        # Get from last conv layer (before MaxPool1d)
        for layer in reversed(self.cnn):
            if isinstance(layer, nn.Conv1d):
                return layer.out_channels
        return 0


class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM block."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_layer_norm: bool = False
    ):
        """
        Initialize BiLSTM block.

        Args:
            input_size: Input feature size
            hidden_size: Hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            Output: (batch, seq_len, hidden_size * 2)
        """
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x

    @property
    def output_dim(self) -> int:
        """Output dimension (bidirectional = 2x hidden)."""
        return self.lstm.hidden_size * 2


class CRNNAttentionModel(nn.Module):
    """
    CRNN + Attention model for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month)
    2. Concatenate embeddings + features
    3. CNN feature extraction
    4. BiLSTM layers
    5. MultiheadAttention
    6. Fully connected layers
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
        Initialize CRNN + Attention model.

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
            channels=config.model.models.crnn_attention.CNN_CHANNELS,
            kernel_size=config.model.models.crnn_attention.CNN_KERNEL_SIZE,
            pool_size=config.model.models.crnn_attention.CNN_POOL_SIZE,
            use_batch_norm=config.model.models.crnn_attention.CNN_USE_BATCH_NORM
        )

        # BiLSTM block
        self.lstm = BiLSTMBlock(
            input_size=self.cnn.output_dim,
            hidden_size=config.model.models.crnn_attention.RNN_HIDDEN_SIZE,
            num_layers=config.model.models.crnn_attention.RNN_NUM_LAYERS,
            dropout=config.model.models.crnn_attention.RNN_DROPOUT,
            use_layer_norm=config.model.models.crnn_attention.USE_LAYER_NORM
        )

        # Multihead attention
        if config.model.models.crnn_attention.USE_ATTENTION:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.lstm.output_dim,
                num_heads=config.model.models.crnn_attention.ATTENTION_HEADS,
                dropout=config.model.models.crnn_attention.ATTENTION_DROPOUT,
                batch_first=True
            )
        else:
            self.attention = None

        # Fully connected layers
        fc_input_dim = self.lstm.output_dim

        fc_layers = []
        prev_dim = fc_input_dim

        for fc_size in config.model.models.crnn_attention.FC_HIDDEN_SIZES:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(config.model.models.crnn_attention.FC_DROPOUT)
            ])

            if config.model.models.crnn_attention.FC_USE_BATCH_NORM:
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

        # Attention
        if self.attention is not None:
            x, _ = self.attention(x, x, x)  # Self-attention
            x = x.mean(dim=1)  # Mean pooling over sequence
        else:
            x = x[:, -1, :]  # Use last time step

        # Fully connected
        output = self.fc(x)

        return output

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config = None
) -> CRNNAttentionModel:
    """
    Create CRNN + Attention model.

    Args:
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        config instance

    Returns:
        CRNNAttentionModel instance
    """
    if config is None:
        from src.config import load_config
        config = load_config('model')

    return CRNNAttentionModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config
    )
