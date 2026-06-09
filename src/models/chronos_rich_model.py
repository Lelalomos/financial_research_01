"""
Chronos-rich adapter model for multi-target market forecasting.

This keeps the Chronos2-style patch encoder, but predicts richer future outputs:
- future OHLCV path
- future return path
- future regime label

It also exposes a scalar `prediction` equal to the final return in the predicted
return path so the existing direct-model evaluation and backtest flow can still
work.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from src.config import load_config
from src.training.losses import create_loss_module
from .chronos2_model import Chronos2EmbeddingLayer, Patchify
from .kronos_module import RMSNorm


class ChronosRichMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
        activation_name: str,
        use_bias: bool,
    ):
        super().__init__()
        self.activation_name = activation_name.strip().lower()
        self.is_gated = self.activation_name in {"geglu", "swiglu"}
        self.input_projection = nn.Linear(input_dim, hidden_dim * 2 if self.is_gated else hidden_dim, bias=use_bias)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

        if self.activation_name == "geglu":
            self.activation = nn.GELU()
        elif self.activation_name == "swiglu":
            self.activation = nn.SiLU()
        else:
            self.activation = ChronosRichModel._build_activation(self.activation_name)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(hidden_states)
        if self.is_gated:
            value, gate = projected.chunk(2, dim=-1)
            activated = value * self.activation(gate)
        else:
            activated = self.activation(projected)
        return self.output_projection(self.dropout(activated))


class ChronosRichTimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float, norm_type: str, use_bias: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
            bias=use_bias,
        )
        self.norm = ChronosRichModel._build_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        normed_hidden = self.norm(hidden_states)
        attn_output, _ = self.attn(
            normed_hidden,
            normed_hidden,
            normed_hidden,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)


class ChronosRichGroupSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float, norm_type: str, use_bias: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
            bias=use_bias,
        )
        self.norm = ChronosRichModel._build_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.num_heads = int(num_heads)

    def _build_group_attn_mask(
        self,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_tokens = attention_mask.shape
        group_mask = group_ids[:, None] == group_ids[None, :]
        key_valid = attention_mask.transpose(0, 1).unsqueeze(1)
        allowed = group_mask.unsqueeze(0) & key_valid

        query_valid = attention_mask.transpose(0, 1)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=group_ids.device).unsqueeze(0)
        allowed = allowed | ((~query_valid).unsqueeze(-1) & diag_mask)

        blocked = ~allowed
        return blocked.unsqueeze(1).expand(num_tokens, self.num_heads, batch_size, batch_size).reshape(
            num_tokens * self.num_heads,
            batch_size,
            batch_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        group_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_valid = attention_mask.any(dim=1)
        safe_group_ids = torch.where(batch_valid, group_ids, torch.arange(group_ids.shape[0], device=group_ids.device))

        hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        normed_hidden = self.norm(hidden_states)
        attn_mask = self._build_group_attn_mask(safe_group_ids, attention_mask)
        attn_output, _ = self.attn(
            normed_hidden,
            normed_hidden,
            normed_hidden,
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = rearrange(hidden_states, "time batch d -> batch time d")
        return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)


class ChronosRichFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, activation_name: str, norm_type: str, use_bias: bool):
        super().__init__()
        self.norm = ChronosRichModel._build_norm(norm_type, d_model)
        self.ff = ChronosRichMLP(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            dropout_rate=dropout_rate,
            activation_name=activation_name,
            use_bias=use_bias,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ff_output = self.ff(self.norm(hidden_states))
        hidden_states = hidden_states + self.dropout(ff_output)
        return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)


class ChronosRichEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout_rate: float, activation_name: str, norm_type: str, use_bias: bool):
        super().__init__()
        self.time_attention = ChronosRichTimeSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            use_bias=use_bias,
        )
        self.group_attention = ChronosRichGroupSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            use_bias=use_bias,
        )
        self.feed_forward = ChronosRichFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation_name=activation_name,
            norm_type=norm_type,
            use_bias=use_bias,
        )

    def forward(self, hidden_states: torch.Tensor, group_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.time_attention(hidden_states, attention_mask)
        hidden_states = self.group_attention(hidden_states, group_ids, attention_mask)
        return self.feed_forward(hidden_states, attention_mask)


class ChronosRichModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config,
        feature_cols: Optional[list[str]] = None,
    ):
        super().__init__()

        self.config = config
        self.data_config = load_config("main")
        self.feature_cols = list(feature_cols or [])
        self.close_index = self.feature_cols.index("close") if "close" in self.feature_cols else 0
        self.prediction_horizon = int(self.data_config.data.sequences.PREDICTION_HORIZON)
        self.ohlcv_output_dim = 5
        self.close_output_index = 3
        self.num_regimes = 3

        chronos_cfg = config.model.models.chronos_rich
        activation_name = str(getattr(chronos_cfg, "ACTIVATION", "relu"))
        norm_type = str(getattr(chronos_cfg, "NORM_TYPE", "layernorm"))
        use_bias = bool(getattr(chronos_cfg, "USE_BIAS", True))
        self.quantiles = [float(q) for q in chronos_cfg.QUANTILES]
        self.num_quantiles = len(self.quantiles)
        self.median_quantile_index = min(
            range(self.num_quantiles),
            key=lambda idx: abs(self.quantiles[idx] - 0.5),
        )

        self.ohlcv_loss_weight = float(getattr(chronos_cfg, "OHLCV_LOSS_WEIGHT", 1.0))
        self.return_path_loss_weight = float(getattr(chronos_cfg, "RETURN_PATH_LOSS_WEIGHT", 1.0))
        self.regime_loss_weight = float(getattr(chronos_cfg, "REGIME_LOSS_WEIGHT", 0.5))
        self.scalar_loss_weight = float(getattr(chronos_cfg, "SCALAR_LOSS_WEIGHT", 1.0))
        self.scalar_loss_fn = self._build_regression_loss(
            loss_type=str(getattr(chronos_cfg, "SCALAR_LOSS_TYPE", "directional_huber")),
            huber_delta=float(getattr(chronos_cfg, "SCALAR_HUBER_DELTA", 0.5)),
            directional_alpha=float(getattr(chronos_cfg, "SCALAR_DIRECTIONAL_ALPHA", 0.1)),
            quantile=float(getattr(chronos_cfg, "SCALAR_QUANTILE", 0.5)),
        )
        self.ohlcv_loss_fn = self._build_regression_loss(
            loss_type=str(getattr(chronos_cfg, "OHLCV_LOSS_TYPE", "mse")),
            huber_delta=float(getattr(chronos_cfg, "OHLCV_HUBER_DELTA", 1.0)),
            directional_alpha=float(getattr(chronos_cfg, "OHLCV_DIRECTIONAL_ALPHA", 0.1)),
            quantile=float(getattr(chronos_cfg, "OHLCV_QUANTILE", 0.5)),
        )
        self.return_path_loss_fn = self._build_regression_loss(
            loss_type=str(getattr(chronos_cfg, "RETURN_PATH_LOSS_TYPE", "mse")),
            huber_delta=float(getattr(chronos_cfg, "RETURN_PATH_HUBER_DELTA", 1.0)),
            directional_alpha=float(getattr(chronos_cfg, "RETURN_PATH_DIRECTIONAL_ALPHA", 0.1)),
            quantile=float(getattr(chronos_cfg, "RETURN_PATH_QUANTILE", 0.5)),
        )
        self.regime_loss_fn = self._build_classification_loss(
            loss_type=str(getattr(chronos_cfg, "REGIME_LOSS_TYPE", "cross_entropy")),
            label_smoothing=float(getattr(chronos_cfg, "REGIME_LABEL_SMOOTHING", 0.0)),
        )

        self.embeddings = Chronos2EmbeddingLayer(
            num_stocks=num_stocks,
            num_groups=num_groups,
            config=config,
        )
        self.patchify = Patchify(
            patch_size=int(chronos_cfg.INPUT_PATCH_SIZE),
            patch_stride=int(chronos_cfg.INPUT_PATCH_STRIDE),
        )

        patch_size = int(chronos_cfg.INPUT_PATCH_SIZE)
        self.patch_projection = nn.Linear(patch_size * 3, int(chronos_cfg.D_MODEL), bias=use_bias)
        self.input_dropout = nn.Dropout(float(chronos_cfg.DROPOUT_RATE))
        self.encoder = nn.ModuleList(
            [
                ChronosRichEncoderBlock(
                    d_model=int(chronos_cfg.D_MODEL),
                    d_ff=int(chronos_cfg.D_FF),
                    num_heads=int(chronos_cfg.NUM_HEADS),
                    dropout_rate=float(chronos_cfg.DROPOUT_RATE),
                    activation_name=activation_name,
                    norm_type=norm_type,
                    use_bias=use_bias,
                )
                for _ in range(int(chronos_cfg.NUM_LAYERS))
            ]
        )
        self.encoder_norm = self._build_norm(norm_type, int(chronos_cfg.D_MODEL))

        self.forecast_head = ChronosRichMLP(
            input_dim=int(chronos_cfg.D_MODEL),
            hidden_dim=int(chronos_cfg.D_FF),
            output_dim=self.num_quantiles * self.prediction_horizon,
            dropout_rate=float(chronos_cfg.DROPOUT_RATE),
            activation_name=activation_name,
            use_bias=use_bias,
        )

        shared_input_dim = (num_features * 2) + self.embeddings.output_dim + (self.prediction_horizon * 3)
        hidden_sizes = list(chronos_cfg.HEAD_HIDDEN_SIZES)
        head_dropout = float(chronos_cfg.HEAD_DROPOUT)
        shared_layers: list[nn.Module] = []
        in_dim = shared_input_dim
        for hidden_dim in hidden_sizes:
            shared_layers.extend(
                [
                    ChronosRichMLP(
                        input_dim=in_dim,
                        hidden_dim=int(hidden_dim),
                        output_dim=int(hidden_dim),
                        dropout_rate=head_dropout,
                        activation_name=activation_name,
                        use_bias=use_bias,
                    ),
                ]
            )
            in_dim = int(hidden_dim)
        self.shared_head = nn.Sequential(*shared_layers)
        self.shared_output_dim = in_dim

        self.future_ohlcv_head = nn.Linear(
            self.shared_output_dim,
            self.prediction_horizon * self.ohlcv_output_dim,
            bias=use_bias,
        )
        self.future_regime_head = nn.Linear(self.shared_output_dim, self.num_regimes, bias=use_bias)

        self._init_weights()

    @staticmethod
    def _build_regression_loss(
        *,
        loss_type: str,
        huber_delta: float,
        directional_alpha: float,
        quantile: float,
    ) -> nn.Module:
        if loss_type == "cross_entropy":
            raise ValueError("cross_entropy is not valid for ChronosRich regression targets")
        return create_loss_module(
            loss_type,
            huber_delta=huber_delta,
            directional_alpha=directional_alpha,
            quantile=quantile,
        )

    @staticmethod
    def _build_classification_loss(*, loss_type: str, label_smoothing: float) -> nn.Module:
        if loss_type != "cross_entropy":
            raise ValueError("ChronosRich regime loss must use cross_entropy")
        return create_loss_module(loss_type, label_smoothing=label_smoothing)

    @staticmethod
    def _build_activation(activation_name: str) -> nn.Module:
        normalized = activation_name.strip().lower()
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU,
            "geglu": nn.GELU,
            "swiglu": nn.SiLU,
        }
        activation_cls = activations.get(normalized)
        if activation_cls is None:
            allowed = ", ".join(sorted(activations))
            raise ValueError(f"Unsupported ChronosRich activation '{activation_name}'. Allowed: {allowed}")
        return activation_cls()

    @staticmethod
    def _build_norm(norm_type: str, dim: int) -> nn.Module:
        normalized = norm_type.strip().lower()
        if normalized == "layernorm":
            return nn.LayerNorm(dim)
        if normalized == "rmsnorm":
            return RMSNorm(dim)
        raise ValueError(f"Unsupported ChronosRich norm type '{norm_type}'. Allowed: layernorm, rmsnorm")

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def _make_patch_inputs(self, close_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches, patch_mask = self.patchify(close_context)
        batch_size, num_patches, patch_size = patches.shape

        time_encoding = torch.linspace(
            -1.0,
            0.0,
            steps=num_patches * patch_size,
            device=close_context.device,
            dtype=close_context.dtype,
        ).view(1, num_patches, patch_size).expand(batch_size, -1, -1)

        patch_inputs = torch.cat([patches, patch_mask, time_encoding], dim=-1)
        attention_mask = patch_mask.sum(dim=-1) > 0
        return patch_inputs, attention_mask

    def _forecast_quantiles(self, close_context: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        patch_inputs, attention_mask = self._make_patch_inputs(close_context)
        x = self.patch_projection(patch_inputs)
        x = self.input_dropout(x)

        encoded = x
        for encoder_block in self.encoder:
            encoded = encoder_block(encoded, group_ids=group_ids, attention_mask=attention_mask)
        encoded = self.encoder_norm(encoded)
        encoded = encoded * attention_mask.unsqueeze(-1).to(encoded.dtype)

        valid_counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (encoded * attention_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        quantiles = self.forecast_head(pooled)
        return quantiles.view(close_context.shape[0], self.num_quantiles, self.prediction_horizon)

    def _summarize_forecast(self, quantile_preds: torch.Tensor) -> torch.Tensor:
        median_path = quantile_preds[:, self.median_quantile_index, :]
        mean_path = quantile_preds.mean(dim=1)
        std_path = quantile_preds.std(dim=1, unbiased=False)
        return torch.cat([median_path, mean_path, std_path], dim=-1)

    def _build_shared_state(
        self,
        features: torch.Tensor,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        close_context = features[:, :, self.close_index]
        sample_group_ids = group_id[:, -1].to(torch.long)
        quantile_preds = self._forecast_quantiles(close_context, group_ids=sample_group_ids)
        forecast_summary = self._summarize_forecast(quantile_preds)

        embeddings = self.embeddings(stock_id, group_id, day, month, dividend_flag)
        embedding_summary = embeddings[:, -1, :]
        feature_last = features[:, -1, :]
        feature_mean = features.mean(dim=1)

        head_input = torch.cat(
            [feature_last, feature_mean, embedding_summary, forecast_summary],
            dim=-1,
        )
        if len(self.shared_head) > 0:
            shared_state = self.shared_head(head_input)
        else:
            shared_state = head_input
        return shared_state, close_context[:, -1]

    def forward(
        self,
        features: torch.Tensor,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        shared_state, last_close = self._build_shared_state(
            features,
            stock_id,
            group_id,
            day,
            month,
            dividend_flag,
        )
        future_ohlcv = self.future_ohlcv_head(shared_state).view(
            features.shape[0],
            self.prediction_horizon,
            self.ohlcv_output_dim,
        )
        future_close_path = future_ohlcv[:, :, self.close_output_index]
        safe_last_close = torch.where(last_close.abs() < 1e-8, torch.ones_like(last_close), last_close)
        future_return_path = ((future_close_path - safe_last_close.unsqueeze(-1)) / safe_last_close.unsqueeze(-1)) * 100.0
        future_regime_logits = self.future_regime_head(shared_state)
        prediction = future_return_path[:, -1:].contiguous()

        return {
            "prediction": prediction,
            "future_ohlcv": future_ohlcv,
            "future_return_path": future_return_path,
            "future_regime_logits": future_regime_logits,
            "future_regime": future_regime_logits.argmax(dim=-1),
        }

    def compute_loss(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], criterion) -> torch.Tensor:
        if getattr(criterion, "expects_structured_output", False):
            return criterion(output, batch)

        loss = self.scalar_loss_weight * self.scalar_loss_fn(output["prediction"], batch["target"])

        if "future_ohlcv" in batch:
            loss = loss + self.ohlcv_loss_weight * self.ohlcv_loss_fn(output["future_ohlcv"], batch["future_ohlcv"])

        if "future_return_path" in batch:
            loss = loss + self.return_path_loss_weight * self.return_path_loss_fn(
                output["future_return_path"],
                batch["future_return_path"],
            )

        if "future_regime" in batch:
            loss = loss + self.regime_loss_weight * self.regime_loss_fn(
                output["future_regime_logits"],
                batch["future_regime"],
            )

        return loss


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config=None,
    feature_cols: Optional[list[str]] = None,
) -> ChronosRichModel:
    if config is None:
        config = load_config("model")

    return ChronosRichModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config,
        feature_cols=feature_cols,
    )
