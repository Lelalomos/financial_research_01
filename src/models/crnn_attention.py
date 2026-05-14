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


def init_weights_xavier_uniform(module):
    """
    Initialize weights using Xavier uniform initialization.

    This helps prevent vanishing/exploding gradients during training.
    Applies appropriate initialization for different layer types.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0 (Jozefowicz et al., 2015)
                # Prevents vanishing gradients for long financial sequences
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_embeddings_xavier(module):
    """
    Initialize embeddings using Xavier uniform initialization.

    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


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

        # Initialize embeddings with Xavier to prevent vanishing variance
        for module in self.modules():
            init_embeddings_xavier(module)

    @staticmethod
    def _prepare_embedding_input(
        tensor: torch.Tensor,
        name: str,
        max_value: int
    ) -> torch.Tensor:
        """Normalize categorical tensors before embedding lookup."""
        if tensor.dim() == 3 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        if tensor.dim() != 2:
            raise ValueError(f"{name} must have shape (batch, seq_len), got {tuple(tensor.shape)}")

        tensor = tensor.long()
        if torch.any((tensor < 0) | (tensor > max_value)):
            raise ValueError(f"{name} values must be between 0 and {max_value}")
        return tensor

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
        stock_id = self._prepare_embedding_input(stock_id, "stock_id", self.stock_embedding.num_embeddings - 1)
        group_id = self._prepare_embedding_input(group_id, "group_id", self.group_embedding.num_embeddings - 1)
        day = self._prepare_embedding_input(day, "day", self.day_embedding.num_embeddings - 1)
        month = self._prepare_embedding_input(month, "month", self.month_embedding.num_embeddings - 1)
        dividend_flag = self._prepare_embedding_input(
            dividend_flag,
            "dividend_flag",
            self.dividend_flag_embedding.num_embeddings - 1
        )

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


class BiLSTM4Block(nn.Module):
    """
    4-layer bidirectional LSTM with different hidden sizes per layer.

    Architecture:
    - Layer 1: input_size -> 128 hidden (bidirectional -> 256 output)
    - Layer 2: 256 -> 256 hidden (bidirectional -> 512 output)
    - Layer 3: 512 -> 512 hidden (bidirectional -> 1024 output)
    - Layer 4: 1024 -> 256 hidden (bidirectional -> 512 output)
    - Dropout between layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple,
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

        self.norm1 = nn.LayerNorm(hidden_sizes[0] * 2)
        self.norm2 = nn.LayerNorm(hidden_sizes[1] * 2)
        self.norm3 = nn.LayerNorm(hidden_sizes[2] * 2)
        self.norm4 = nn.LayerNorm(hidden_sizes[3] * 2)

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
        x = self.norm1(x)
        x = self.dropout(x)

        # Layer 2
        x, _ = self.lstm2(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # Layer 3
        x, _ = self.lstm3(x)
        x = self.norm3(x)
        x = self.dropout(x)

        # Layer 4
        x, _ = self.lstm4(x)
        x = self.norm4(x)
        x = self.dropout(x)

        return x

    @property
    def output_dim(self) -> int:
        """Output dimension (bidirectional = 2x last hidden size)."""
        return self.hidden_sizes[3] * 2


class BiLSTMBlock(nn.Module):
    """
    Standard bidirectional LSTM block (for backward compatibility).

    Used by CRNNModel which needs a simple multi-layer BiLSTM.
    """

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

        # 4-layer BiLSTM block with variable hidden sizes
        self.lstm = BiLSTM4Block(
            input_size=self.cnn.output_dim,
            hidden_sizes=config.model.models.crnn_attention.LSTM4_HIDDEN_SIZES,
            dropout=config.model.models.crnn_attention.LSTM4_DROPOUT
        )

        # Multihead attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm.output_dim,
            num_heads=config.model.models.crnn_attention.LSTM4_ATTENTION_HEADS,
            dropout=config.model.models.crnn_attention.LSTM4_ATTENTION_DROPOUT,
            batch_first=True
        )

        # Single Linear FC layer (like bilstm4_attention)
        self.fc = nn.Linear(self.lstm.output_dim, 1)
        self.fc_dropout = nn.Dropout(config.model.models.crnn_attention.LSTM4_DROPOUT)

        # Apply weight initialization to all layers
        self.apply(init_weights_xavier_uniform)
        # Keep output variance large enough to avoid near-constant predictions.
        nn.init.xavier_uniform_(self.fc.weight, gain=1.0)
        nn.init.zeros_(self.fc.bias)

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

    def forward_with_attention(
        self,
        features: torch.Tensor,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor
    ):
        """
        Forward pass that also returns per-head attention weights.

        Returns:
            Tuple of (output, attention_weights). Attention weights have shape
            (batch, num_heads, reduced_seq_len, reduced_seq_len) because the CNN
            block may reduce sequence length.
        """
        emb = self.embeddings(stock_id, group_id, day, month, dividend_flag)
        x = torch.cat([emb, features], dim=-1)
        x = self.cnn(x)
        x = self.lstm(x)
        x, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False
        )
        x = x.mean(dim=1)
        x = self.fc_dropout(x)
        output = self.fc(x)
        return output, attention_weights

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
