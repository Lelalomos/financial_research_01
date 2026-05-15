"""
BiLSTM4 + Attention model for financial prediction.

Model architecture:
- Embeddings (stock, group, day, month, dividend_flag)
- 4-layer BiLSTM with different hidden sizes (128, 256, 512, 256)
- MultiheadAttention (4 heads)
- Timestep-wise MLP after attention
- Mean pooling and final output layer
"""

import torch
import torch.nn as nn

from src.config import load_config
from .crnn_attention import EmbeddingLayer, BiLSTM4Block, init_weights_xavier_uniform


class BiLSTM4AttentionModel(nn.Module):
    """
    BiLSTM4 + Attention model for financial prediction.

    Architecture:
    1. Embedding layer (stock, group, day, month, dividend_flag)
    2. Concatenate embeddings + features
    3. 4-layer BiLSTM with variable hidden sizes (128, 256, 512, 256)
    4. MultiheadAttention (4 heads)
    5. Feed-forward network applied to every timestep
    6. Mean pooling over sequence
    7. Final output layer
    8. Output (percent change prediction)
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

        # Timestep-wise MLP applied to the attention output before pooling.
        self.timestep_mlp = nn.ModuleList()
        prev_dim = self.lstm.output_dim
        use_batch_norm = config.model.models.bilstm4_attention.FC_USE_BATCH_NORM

        for fc_size in config.model.models.bilstm4_attention.FC_HIDDEN_SIZES:
            self.timestep_mlp.append(
                nn.ModuleDict({
                    "linear": nn.Linear(prev_dim, fc_size),
                    "batch_norm": nn.BatchNorm1d(fc_size) if use_batch_norm else nn.Identity(),
                    "activation": nn.LeakyReLU(0.1),
                    "dropout": nn.Dropout(config.model.models.bilstm4_attention.FC_DROPOUT),
                })
            )
            prev_dim = fc_size

        self.output_layer = nn.Linear(prev_dim, 1)

        # Apply weight initialization to all layers
        self.apply(init_weights_xavier_uniform)
        # Keep output variance large enough to avoid near-constant predictions
        nn.init.xavier_uniform_(self.output_layer.weight, gain=1.0)
        nn.init.zeros_(self.output_layer.bias)

    def _apply_timestep_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured MLP independently to each timestep."""
        for block in self.timestep_mlp:
            x = block["linear"](x)
            if not isinstance(block["batch_norm"], nn.Identity):
                batch_size, seq_len, hidden_dim = x.shape
                x = x.reshape(batch_size * seq_len, hidden_dim)
                x = block["batch_norm"](x)
                x = x.reshape(batch_size, seq_len, hidden_dim)
            x = block["activation"](x)
            x = block["dropout"](x)
        return x

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

        # MLP on every timestep after attention
        x = self._apply_timestep_mlp(x)

        # Mean pooling over sequence
        x = x.mean(dim=1)

        # Final output layer
        output = self.output_layer(x)

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
            (batch, num_heads, seq_len, seq_len).
        """
        emb = self.embeddings(stock_id, group_id, day, month, dividend_flag)
        x = torch.cat([emb, features], dim=-1)
        x = self.lstm(x)
        x, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False
        )
        x = self._apply_timestep_mlp(x)
        x = x.mean(dim=1)
        output = self.output_layer(x)
        return output, attention_weights


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
