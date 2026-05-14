"""
Multi-branch BiLSTM model for financial prediction.

This model splits numeric sequence features into technical, geometric, and
macro/financial branches using checkpoint-aligned feature column metadata. Each
branch is encoded independently before fusion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.config import load_config
from .crnn_attention import BiLSTMBlock, EmbeddingLayer


class MultiBranchBiLSTMModel(nn.Module):
    """BiLSTM model with dedicated branches for different feature groups."""

    def __init__(
        self,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        config,
        feature_cols: Sequence[str],
    ):
        super().__init__()

        if feature_cols is None:
            raise ValueError("feature_cols are required for multi_branch_bilstm")
        if len(feature_cols) != num_features:
            raise ValueError(
                f"feature_cols length {len(feature_cols)} must match num_features {num_features}"
            )

        self.config = config
        self.feature_cols = list(feature_cols)
        model_cfg = config.model.models.multi_branch_bilstm
        self.use_embeddings = model_cfg.USE_EMBEDDINGS
        self.branch_pooling = model_cfg.BRANCH_POOLING

        if self.branch_pooling not in {"mean", "last"}:
            raise ValueError("BRANCH_POOLING must be 'mean' or 'last'")

        if self.use_embeddings:
            self.embeddings = EmbeddingLayer(
                num_stocks=num_stocks,
                num_groups=num_groups,
                config=config,
            )
            embedding_dim = self.embeddings.output_dim
        else:
            self.embeddings = None
            embedding_dim = 0

        branch_indices, unassigned = self._resolve_feature_groups(self.feature_cols, model_cfg)
        if unassigned and not model_cfg.INCLUDE_UNASSIGNED_IN_TECHNICAL:
            raise ValueError(
                "Unassigned feature columns for multi_branch_bilstm: "
                + ", ".join(unassigned)
            )
        if unassigned:
            branch_indices["technical"].extend(
                self.feature_cols.index(column) for column in unassigned
            )

        self.register_buffer(
            "technical_feature_indices",
            torch.tensor(sorted(branch_indices["technical"]), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "geometric_feature_indices",
            torch.tensor(sorted(branch_indices["geometric"]), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "macro_feature_indices",
            torch.tensor(sorted(branch_indices["macro"]), dtype=torch.long),
            persistent=True,
        )

        technical_dim = len(branch_indices["technical"]) + embedding_dim
        geometric_dim = len(branch_indices["geometric"])
        macro_dim = len(branch_indices["macro"])

        self.technical_branch = self._create_branch(
            input_dim=technical_dim,
            hidden_size=model_cfg.TECHNICAL_HIDDEN_SIZE,
            num_layers=model_cfg.TECHNICAL_NUM_LAYERS,
            dropout=model_cfg.TECHNICAL_DROPOUT,
            use_layer_norm=model_cfg.TECHNICAL_USE_LAYER_NORM,
        )
        self.geometric_branch = self._create_branch(
            input_dim=geometric_dim,
            hidden_size=model_cfg.GEOMETRIC_HIDDEN_SIZE,
            num_layers=model_cfg.GEOMETRIC_NUM_LAYERS,
            dropout=model_cfg.GEOMETRIC_DROPOUT,
            use_layer_norm=model_cfg.GEOMETRIC_USE_LAYER_NORM,
        )
        self.macro_branch = self._create_branch(
            input_dim=macro_dim,
            hidden_size=model_cfg.MACRO_HIDDEN_SIZE,
            num_layers=model_cfg.MACRO_NUM_LAYERS,
            dropout=model_cfg.MACRO_DROPOUT,
            use_layer_norm=model_cfg.MACRO_USE_LAYER_NORM,
        )

        fusion_input_dim = sum(
            branch.output_dim
            for branch in [self.technical_branch, self.geometric_branch, self.macro_branch]
            if branch is not None
        )
        if fusion_input_dim == 0:
            raise ValueError("multi_branch_bilstm requires at least one active branch input")

        fusion_layers: List[nn.Module] = []
        prev_dim = fusion_input_dim
        for hidden_size in model_cfg.FUSION_HIDDEN_SIZES:
            fusion_layers.append(nn.Linear(prev_dim, hidden_size))
            if model_cfg.FUSION_USE_BATCH_NORM:
                fusion_layers.append(nn.BatchNorm1d(hidden_size))
            fusion_layers.extend(
                [
                    nn.LeakyReLU(0.1),
                    nn.Dropout(model_cfg.FUSION_DROPOUT),
                ]
            )
            prev_dim = hidden_size
        fusion_layers.append(nn.Linear(prev_dim, 1))
        self.fusion_head = nn.Sequential(*fusion_layers)

    @staticmethod
    def _matches(column: str, exact: Sequence[str], prefixes: Sequence[str]) -> bool:
        return column in exact or any(column.startswith(prefix) for prefix in prefixes)

    def _resolve_feature_groups(
        self,
        feature_cols: Sequence[str],
        model_cfg,
    ) -> Tuple[Dict[str, List[int]], List[str]]:
        technical_exact = list(model_cfg.TECHNICAL_EXACT_FEATURES)
        technical_prefix = list(model_cfg.TECHNICAL_PREFIX_FEATURES)
        geometric_exact = list(model_cfg.GEOMETRIC_EXACT_FEATURES)
        geometric_prefix = list(model_cfg.GEOMETRIC_PREFIX_FEATURES)
        macro_exact = list(model_cfg.MACRO_FINANCIAL_EXACT_FEATURES)
        macro_prefix = list(model_cfg.MACRO_FINANCIAL_PREFIX_FEATURES)

        groups = {"technical": [], "geometric": [], "macro": []}
        unassigned: List[str] = []

        for idx, column in enumerate(feature_cols):
            if self._matches(column, technical_exact, technical_prefix):
                groups["technical"].append(idx)
            elif self._matches(column, geometric_exact, geometric_prefix):
                groups["geometric"].append(idx)
            elif self._matches(column, macro_exact, macro_prefix):
                groups["macro"].append(idx)
            else:
                unassigned.append(column)

        return groups, unassigned

    @staticmethod
    def _create_branch(
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_layer_norm: bool,
    ) -> Optional[BiLSTMBlock]:
        if input_dim <= 0:
            return None
        return BiLSTMBlock(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def _pool_branch(self, x: torch.Tensor) -> torch.Tensor:
        if self.branch_pooling == "mean":
            return x.mean(dim=1)
        return x[:, -1, :]

    @staticmethod
    def _select_features(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.index_select(features, dim=2, index=indices.to(features.device))

    def _run_branch(
        self,
        branch: Optional[BiLSTMBlock],
        branch_features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if branch is None or branch_features is None:
            return None
        encoded = branch(branch_features)
        return self._pool_branch(encoded)

    def forward(
        self,
        features: torch.Tensor,
        stock_id: torch.Tensor,
        group_id: torch.Tensor,
        day: torch.Tensor,
        month: torch.Tensor,
        dividend_flag: torch.Tensor,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []

        embedding_tensor = None
        if self.embeddings is not None:
            embedding_tensor = self.embeddings(stock_id, group_id, day, month, dividend_flag)

        technical_features = None
        if self.technical_feature_indices.numel() > 0:
            technical_features = self._select_features(features, self.technical_feature_indices)
        if embedding_tensor is not None:
            technical_features = (
                embedding_tensor
                if technical_features is None
                else torch.cat([embedding_tensor, technical_features], dim=-1)
            )
        technical_output = self._run_branch(self.technical_branch, technical_features)
        if technical_output is not None:
            outputs.append(technical_output)

        geometric_output = self._run_branch(
            self.geometric_branch,
            self._select_features(features, self.geometric_feature_indices)
            if self.geometric_feature_indices.numel() > 0
            else None,
        )
        if geometric_output is not None:
            outputs.append(geometric_output)

        macro_output = self._run_branch(
            self.macro_branch,
            self._select_features(features, self.macro_feature_indices)
            if self.macro_feature_indices.numel() > 0
            else None,
        )
        if macro_output is not None:
            outputs.append(macro_output)

        if not outputs:
            raise ValueError("multi_branch_bilstm produced no active branch outputs")

        fused = torch.cat(outputs, dim=-1)
        return self.fusion_head(fused)


def create_model(
    num_features: int,
    num_stocks: int,
    num_groups: int,
    config=None,
    feature_cols: Optional[Sequence[str]] = None,
) -> MultiBranchBiLSTMModel:
    if config is None:
        config = load_config("model")

    return MultiBranchBiLSTMModel(
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        config=config,
        feature_cols=feature_cols,
    )
