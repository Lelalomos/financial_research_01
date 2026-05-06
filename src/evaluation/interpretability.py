"""
Interpretability utilities for attention-based models.

Attention weights are diagnostic signals about model internals. They should not
be treated as proof of causality or standalone feature importance.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch


def attention_to_numpy(attention_weights) -> np.ndarray:
    """
    Convert attention weights to a 4D NumPy array.

    Accepted shapes:
    - (batch, heads, query_len, key_len)
    - (batch, query_len, key_len), which is treated as one head
    """
    if isinstance(attention_weights, torch.Tensor):
        weights = attention_weights.detach().cpu().numpy()
    else:
        weights = np.asarray(attention_weights)

    if weights.ndim == 3:
        weights = weights[:, np.newaxis, :, :]
    if weights.ndim != 4:
        raise ValueError(
            "attention_weights must have shape (batch, heads, query_len, key_len) "
            "or (batch, query_len, key_len)"
        )
    return weights


def aggregate_attention_by_position(
    attention_weights,
    positions: Optional[Iterable] = None,
    average_heads: bool = True,
) -> pd.DataFrame:
    """
    Aggregate attention received by each key position.

    Args:
        attention_weights: Attention tensor/array.
        positions: Optional labels for key positions.
        average_heads: If True, average across heads. If False, return one row
            per head and position.

    Returns:
        DataFrame with attention summaries by sequence position.
    """
    weights = attention_to_numpy(attention_weights)
    _, num_heads, _, key_len = weights.shape

    if positions is None:
        position_labels = list(range(key_len))
    else:
        position_labels = list(positions)
        if len(position_labels) != key_len:
            raise ValueError("positions length must match attention key length")

    # Mean over batch and query positions. Result: (heads, key_len)
    per_head = weights.mean(axis=(0, 2))

    if average_heads:
        scores = per_head.mean(axis=0)
        return pd.DataFrame({
            "position": position_labels,
            "attention_score": scores,
        })

    rows = []
    for head_idx in range(num_heads):
        for position_idx, score in enumerate(per_head[head_idx]):
            rows.append({
                "head": head_idx,
                "position": position_labels[position_idx],
                "attention_score": score,
            })
    return pd.DataFrame(rows)


def summarize_attention(
    attention_weights,
    positions: Optional[Iterable] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return attention summary sorted from highest to lowest attention score.
    """
    summary = aggregate_attention_by_position(
        attention_weights,
        positions=positions,
        average_heads=True,
    )
    summary = summary.sort_values("attention_score", ascending=False).reset_index(drop=True)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        summary = summary.head(top_k).reset_index(drop=True)
    return summary


def get_attention_report(model, batch: dict, device: Optional[str] = None) -> dict:
    """
    Run a model's forward_with_attention method and summarize attention.

    Args:
        model: Model implementing forward_with_attention.
        batch: Batch dictionary with features, stock_id, group_id, day, month,
            and dividend_flag.
        device: Optional device override.

    Returns:
        Dictionary containing predictions, raw attention weights, and summary.
    """
    if not hasattr(model, "forward_with_attention"):
        raise ValueError("Model does not implement forward_with_attention")

    if device is None:
        device = next(model.parameters()).device

    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
        for key, value in batch.items()
    }

    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model.forward_with_attention(
            features=tensor_batch["features"],
            stock_id=tensor_batch["stock_id"],
            group_id=tensor_batch["group_id"],
            day=tensor_batch["day"],
            month=tensor_batch["month"],
            dividend_flag=tensor_batch["dividend_flag"],
        )

    if attention_weights is None:
        raise ValueError("Model did not return attention weights")

    attention_array = attention_to_numpy(attention_weights)
    return {
        "predictions": predictions.detach().cpu().numpy(),
        "attention_weights": attention_array,
        "summary": summarize_attention(attention_array),
    }
