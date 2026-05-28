"""
Chronos-2-style adapter model for financial return prediction.

This implementation keeps the key design ideas that matter for this repo:

- patch the historical close context
- encode patch embeddings with a Transformer stack
- predict multiple future quantiles across the configured horizon
- fuse those forecast summaries with the repo's categorical embeddings and
  engineered features to predict the scalar return target

It is intentionally self-contained so train/test/backtest can run inside the
current project container without depending on an external Chronos package
mount.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import load_config
from .crnn_attention import EmbeddingLayer


class Chronos2EmbeddingLayer(nn.Module):
    """Chronos2-specific categorical embedding layer."""

    def __init__(
        self,
        num_stocks: int,
        num_groups: int,
        config,
    ):
        super().__init__()
        chronos_cfg = config.model.models.chronos2
        shared_cfg = config.model.embeddings

        self.use_stock_embedding = bool(getattr(chronos_cfg, "USE_STOCK_EMBEDDING", True))
        self.use_group_embedding = bool(getattr(chronos_cfg, "USE_GROUP_EMBEDDING", True))

        stock_dim = int(getattr(chronos_cfg, "STOCK_EMB_DIM", shared_cfg.EMBEDDING_DIM_STOCK))
        group_dim = int(getattr(chronos_cfg, "GROUP_EMB_DIM", shared_cfg.EMBEDDING_DIM_GROUP))
        day_dim = int(getattr(chronos_cfg, "DAY_EMB_DIM", shared_cfg.EMBEDDING_DIM_DAY))
        month_dim = int(getattr(chronos_cfg, "MONTH_EMB_DIM", shared_cfg.EMBEDDING_DIM_MONTH))
        dividend_dim = int(
            getattr(chronos_cfg, "DIVIDEND_FLAG_EMB_DIM", shared_cfg.EMBEDDING_DIM_DIVIDEND_FLAG)
        )
        embedding_dropout = float(
            getattr(chronos_cfg, "DROPOUT_EMBEDDING", shared_cfg.DROPOUT_EMBEDDING)
        )

        self.stock_embedding = (
            nn.Embedding(num_embeddings=num_stocks, embedding_dim=stock_dim)
            if self.use_stock_embedding
            else None
        )
        self.group_embedding = (
            nn.Embedding(num_embeddings=num_groups, embedding_dim=group_dim)
            if self.use_group_embedding
            else None
        )
        self.day_embedding = nn.Embedding(num_embeddings=32, embedding_dim=day_dim)
        self.month_embedding = nn.Embedding(num_embeddings=13, embedding_dim=month_dim)
        self.dividend_flag_embedding = nn.Embedding(num_embeddings=3, embedding_dim=dividend_dim)
        self.dropout = nn.Dropout(embedding_dropout)

        self._base_layer = EmbeddingLayer(
            num_stocks=max(num_stocks, 1),
            num_groups=max(num_groups, 1),
            config=config,
        )

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        if self.stock_embedding is not None:
            stock_id = self._base_layer._prepare_embedding_input(
                stock_id,
                "stock_id",
                self.stock_embedding.num_embeddings - 1,
            )
            parts.append(self.stock_embedding(stock_id))

        if self.group_embedding is not None:
            group_id = self._base_layer._prepare_embedding_input(
                group_id,
                "group_id",
                self.group_embedding.num_embeddings - 1,
            )
            parts.append(self.group_embedding(group_id))

        day = self._base_layer._prepare_embedding_input(day, "day", self.day_embedding.num_embeddings - 1)
        month = self._base_layer._prepare_embedding_input(month, "month", self.month_embedding.num_embeddings - 1)
        dividend_flag = self._base_layer._prepare_embedding_input(
            dividend_flag,
            "dividend_flag",
            self.dividend_flag_embedding.num_embeddings - 1,
        )
        parts.extend(
            [
                self.day_embedding(day),
                self.month_embedding(month),
                self.dividend_flag_embedding(dividend_flag),
            ]
        )
        return self.dropout(torch.cat(parts, dim=-1))

    @property
    def output_dim(self) -> int:
        total = (
            self.day_embedding.embedding_dim
            + self.month_embedding.embedding_dim
            + self.dividend_flag_embedding.embedding_dim
        )
        if self.stock_embedding is not None:
            total += self.stock_embedding.embedding_dim
        if self.group_embedding is not None:
            total += self.group_embedding.embedding_dim
        return total


class Patchify(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        if seq_len < self.patch_size:
            pad = self.patch_size - seq_len
            x = F.pad(x, (pad, 0), value=0.0)
            mask = F.pad(torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype), (pad, 0), value=0.0)
        else:
            remainder = (seq_len - self.patch_size) % self.patch_stride
            pad = (self.patch_stride - remainder) % self.patch_stride
            x = F.pad(x, (pad, 0), value=0.0)
            mask = F.pad(torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype), (pad, 0), value=0.0)

        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        patch_mask = mask.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return patches, patch_mask


class Chronos2ForecastModel(nn.Module):
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

        chronos_cfg = config.model.models.chronos2
        self.quantiles = [float(q) for q in chronos_cfg.QUANTILES]
        self.num_quantiles = len(self.quantiles)
        self.median_quantile_index = min(
            range(self.num_quantiles),
            key=lambda idx: abs(self.quantiles[idx] - 0.5),
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

        # patch values + patch mask + simple relative time encoding
        self.patch_projection = nn.Linear(patch_size * 3, int(chronos_cfg.D_MODEL))
        self.input_dropout = nn.Dropout(float(chronos_cfg.DROPOUT_RATE))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(chronos_cfg.D_MODEL),
            nhead=int(chronos_cfg.NUM_HEADS),
            dim_feedforward=int(chronos_cfg.D_FF),
            dropout=float(chronos_cfg.DROPOUT_RATE),
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(chronos_cfg.NUM_LAYERS),
        )
        self.encoder_norm = nn.LayerNorm(int(chronos_cfg.D_MODEL))

        self.forecast_head = nn.Sequential(
            nn.Linear(int(chronos_cfg.D_MODEL), int(chronos_cfg.D_FF)),
            nn.ReLU(),
            nn.Dropout(float(chronos_cfg.DROPOUT_RATE)),
            nn.Linear(int(chronos_cfg.D_FF), self.num_quantiles * self.prediction_horizon),
        )

        summary_dim = (num_features * 2) + self.embeddings.output_dim + (self.prediction_horizon * 3)
        hidden_sizes = list(chronos_cfg.HEAD_HIDDEN_SIZES)
        head_dropout = float(chronos_cfg.HEAD_DROPOUT)

        layers: list[nn.Module] = []
        in_dim = summary_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(head_dropout),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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

    def _forecast_quantiles(self, close_context: torch.Tensor) -> torch.Tensor:
        patch_inputs, attention_mask = self._make_patch_inputs(close_context)
        x = self.patch_projection(patch_inputs)
        x = self.input_dropout(x)

        encoded = self.encoder(
            x,
            src_key_padding_mask=~attention_mask,
        )
        encoded = self.encoder_norm(encoded)

        valid_counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (encoded * attention_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        quantiles = self.forecast_head(pooled)
        return quantiles.view(close_context.shape[0], self.num_quantiles, self.prediction_horizon)

    def _summarize_forecast(self, quantile_preds: torch.Tensor) -> torch.Tensor:
        median_path = quantile_preds[:, self.median_quantile_index, :]
        mean_path = quantile_preds.mean(dim=1)
        std_path = quantile_preds.std(dim=1, unbiased=False)
        return torch.cat([median_path, mean_path, std_path], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor,
    ) -> torch.Tensor:
        close_context = features[:, :, self.close_index]
        quantile_preds = self._forecast_quantiles(close_context)
        forecast_summary = self._summarize_forecast(quantile_preds)

        embeddings = self.embeddings(stock_id, group_id, day, month, dividend_flag)
        embedding_summary = embeddings[:, -1, :]
        feature_last = features[:, -1, :]
        feature_mean = features.mean(dim=1)

        head_input = torch.cat(
            [feature_last, feature_mean, embedding_summary, forecast_summary],
            dim=-1,
        )
        return self.head(head_input)


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config=None,
    feature_cols: Optional[list[str]] = None,
) -> Chronos2ForecastModel:
    if config is None:
        config = load_config("model")

    return Chronos2ForecastModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config,
        feature_cols=feature_cols,
    )
