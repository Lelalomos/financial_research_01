"""
Helpers for printing compact previews of processed sequence samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def load_ticker_mapping(data_dir: Path) -> Dict[int, str]:
    """
    Best-effort stock_id -> ticker mapping from pre-normalized parquet data.
    """
    mapping: Dict[int, str] = {}
    pre_normalized_path = data_dir.parent / "pre_normalized.parquet"
    if not pre_normalized_path.exists():
        return mapping

    try:
        import pandas as pd

        df = pd.read_parquet(pre_normalized_path, columns=["tic_id", "tic"])
        for _, row in df.drop_duplicates().iterrows():
            mapping[int(row["tic_id"])] = str(row["tic"])
    except Exception:
        return {}

    return mapping


def log_sequence_preview(
    logger,
    sequences: Dict[str, np.ndarray],
    feature_cols: Optional[Iterable[str]] = None,
    ticker_map: Optional[Dict[int, str]] = None,
    split_name: str = "train",
    max_rows: int = 10,
    max_feature_preview: int = 5,
) -> None:
    """
    Log a compact, one-time preview of processed sequence samples.
    """
    if not sequences or "target" not in sequences or len(sequences["target"]) == 0:
        logger.info(f"No {split_name} sequences available for preview.")
        return

    ticker_map = ticker_map or {}
    feature_names = list(feature_cols or [])
    preview_count = min(max_rows, len(sequences["target"]))

    logger.info("=" * 60)
    logger.info(f"{split_name.upper()} SAMPLE PREVIEW ({preview_count} rows)")
    logger.info("=" * 60)

    for idx in range(preview_count):
        stock_id = int(sequences["stock_id"][idx][0]) if "stock_id" in sequences else -1
        ticker = ticker_map.get(stock_id, f"stock_id={stock_id}")
        target = float(np.asarray(sequences["target"][idx]).reshape(-1)[0])
        seq_len = int(sequences["features"][idx].shape[0])

        start_day = int(sequences["day"][idx][0]) if "day" in sequences else -1
        end_day = int(sequences["day"][idx][-1]) if "day" in sequences else -1
        start_month = int(sequences["month"][idx][0]) if "month" in sequences else -1
        end_month = int(sequences["month"][idx][-1]) if "month" in sequences else -1

        logger.info(
            f"sample {idx} | ticker={ticker} | target={target:+.6f} | "
            f"seq_len={seq_len} | start={start_month}/{start_day} | end={end_month}/{end_day}"
        )

        last_step = np.asarray(sequences["features"][idx][-1]).reshape(-1)
        preview_pairs = []
        feature_limit = min(max_feature_preview, len(last_step))
        for feature_idx in range(feature_limit):
            feature_name = (
                feature_names[feature_idx]
                if feature_idx < len(feature_names)
                else f"feature_{feature_idx}"
            )
            preview_pairs.append(f"{feature_name}={last_step[feature_idx]:+.4f}")

        logger.info("  last_step_features: " + ", ".join(preview_pairs))

