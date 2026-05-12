"""
Shared runtime configuration helpers for train/eval scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch


def get_eval_batch_size(config) -> int:
    """
    Return the evaluation batch size.

    Prefer config.model.validation.VAL_BATCH_SIZE when present, otherwise fall
    back to the training batch size so evaluation stays aligned with config.
    """
    validation_cfg = getattr(config.model, "validation", None)
    if validation_cfg is not None:
        val_batch_size = validation_cfg.get("VAL_BATCH_SIZE")
        if val_batch_size is not None:
            return int(val_batch_size)
    return int(config.model.training.BATCH_SIZE)


def load_checkpoint_metadata(checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict:
    """Load a checkpoint in weights-only-safe mode."""
    return torch.load(checkpoint_path, map_location=map_location, weights_only=True)


def infer_model_type_from_checkpoint(
    checkpoint_path: str,
    available_model_types: Iterable[str],
    fallback_model_type: Optional[str] = None,
) -> str:
    """
    Infer the model type from checkpoint metadata, then filename, then fallback.
    """
    checkpoint = load_checkpoint_metadata(checkpoint_path, map_location="cpu")
    model_type = checkpoint.get("model_type")
    metadata = checkpoint.get("metadata", {})

    if not model_type and isinstance(metadata, dict):
        model_type = metadata.get("model_type")

    available = set(available_model_types)
    if model_type in available:
        return model_type

    basename = Path(checkpoint_path).name.lower()
    for known_type in sorted(available, key=len, reverse=True):
        if known_type in basename:
            return known_type

    if fallback_model_type is not None:
        return fallback_model_type

    raise ValueError(
        f"Could not infer model type from checkpoint: {checkpoint_path}"
    )
