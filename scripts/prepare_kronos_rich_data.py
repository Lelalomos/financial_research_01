#!/usr/bin/env python
"""
Prepare Kronos-rich training data from normalized split caches.

This script creates a separate dataset for richer Kronos experiments with:
- future OHLCV rows
- future close path
- future cumulative return path
- future realized volatility
- future regime label
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils.logger import get_logger

from scripts.prepare_chronos2_data import (
    ensure_base_preprocessing,
    load_normalized_split_cache,
    load_preprocessing_metadata,
)


KRONOS_RICH_ARRAY_KEYS = [
    "features",
    "stock_id",
    "group_id",
    "day",
    "month",
    "dividend_flag",
    "target",
    "future_ohlcv",
    "future_ohlcv_mask",
    "future_close_path",
    "future_return_path",
    "future_volatility",
    "future_regime",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Kronos-rich multi-target data")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--stock-limit", type=int, default=None)
    parser.add_argument("--stocks", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-scalar-target", action="store_true")
    parser.add_argument("--skip-return-path", action="store_true")
    parser.add_argument("--skip-regime-label", action="store_true")
    parser.add_argument("--export-pre-normalize", type=str, default="data/pre_normalized.parquet")
    parser.add_argument("--export-normalized", type=str, default="data/normalized_data.parquet")
    parser.add_argument("--no-resume-cache", action="store_true")
    return parser.parse_args()


def resolve_kronos_rich_prep_settings(config, args) -> Dict[str, object]:
    prep_cfg = getattr(config.data, "kronos_rich_preparation", None)
    processed_dir = Path(
        args.processed_dir
        or getattr(config.data.paths, "PROCESSED_DATA_PATH", "data/processed")
    )
    output_dir = Path(
        args.output_dir
        or (
            getattr(prep_cfg, "OUTPUT_DIR", None)
            if prep_cfg is not None
            else None
        )
        or "data/processed_kronos_rich"
    )
    ohlcv_columns = list(
        getattr(prep_cfg, "OHLCV_COLUMNS", ["open", "high", "low", "close", "volume"])
        if prep_cfg is not None
        else ["open", "high", "low", "close", "volume"]
    )
    include_scalar_target = not args.skip_scalar_target
    include_return_path = not args.skip_return_path
    include_regime_label = not args.skip_regime_label
    if prep_cfg is not None:
        include_scalar_target = bool(getattr(prep_cfg, "INCLUDE_SCALAR_TARGET", True)) and include_scalar_target
        include_return_path = bool(getattr(prep_cfg, "INCLUDE_RETURN_PATH", True)) and include_return_path
        include_regime_label = bool(getattr(prep_cfg, "INCLUDE_REGIME_LABEL", True)) and include_regime_label

    return {
        "processed_dir": processed_dir,
        "output_dir": output_dir,
        "ohlcv_columns": ohlcv_columns,
        "include_scalar_target": include_scalar_target,
        "include_return_path": include_return_path,
        "include_regime_label": include_regime_label,
        "regime_source": (
            getattr(prep_cfg, "REGIME_SOURCE", "column_or_realized_volatility")
            if prep_cfg is not None
            else "column_or_realized_volatility"
        ),
        "volatility_low_quantile": float(
            getattr(prep_cfg, "VOLATILITY_LOW_QUANTILE", 0.33)
            if prep_cfg is not None
            else 0.33
        ),
        "volatility_high_quantile": float(
            getattr(prep_cfg, "VOLATILITY_HIGH_QUANTILE", 0.66)
            if prep_cfg is not None
            else 0.66
        ),
    }


def _validate_required_columns(df: pd.DataFrame, feature_cols: List[str], ohlcv_columns: List[str]) -> None:
    missing_ohlcv = [col for col in ohlcv_columns if col not in df.columns]
    if missing_ohlcv:
        raise ValueError(f"Missing OHLCV columns for Kronos-rich preparation: {missing_ohlcv}")
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in split DataFrame: {missing_features}")


def assign_future_regime_labels(
    future_volatility: np.ndarray,
    existing_regime: np.ndarray | None,
    include_regime_label: bool,
    regime_source: str,
    low_quantile: float,
    high_quantile: float,
) -> np.ndarray:
    if not include_regime_label:
        return np.zeros(len(future_volatility), dtype=np.int64)
    if regime_source == "column_or_realized_volatility" and existing_regime is not None:
        return existing_regime.astype(np.int64, copy=False)

    low_cut = float(np.quantile(future_volatility, low_quantile))
    high_cut = float(np.quantile(future_volatility, high_quantile))
    labels = np.ones(len(future_volatility), dtype=np.int64)
    labels[future_volatility <= low_cut] = 0
    labels[future_volatility >= high_cut] = 2
    return labels


def build_kronos_rich_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int,
    prediction_horizon: int,
    ohlcv_columns: List[str],
    include_scalar_target: bool = True,
    include_return_path: bool = True,
    include_regime_label: bool = True,
    regime_source: str = "column_or_realized_volatility",
    volatility_low_quantile: float = 0.33,
    volatility_high_quantile: float = 0.66,
) -> Dict[str, np.ndarray]:
    if df is None or df.empty:
        return {key: np.array([]) for key in KRONOS_RICH_ARRAY_KEYS}

    _validate_required_columns(df, feature_cols, ohlcv_columns)
    stock_col = "tic_id" if "tic_id" in df.columns else "tic"
    sequences = {key: [] for key in KRONOS_RICH_ARRAY_KEYS}
    fallback_regimes: List[int] = []

    for _, stock_df in df.groupby(stock_col, sort=True):
        ordered = stock_df.sort_values("date").reset_index(drop=True)
        if len(ordered) < sequence_length + prediction_horizon:
            continue

        feature_matrix = ordered[feature_cols].to_numpy(dtype=np.float32, copy=True)
        ohlcv_matrix = ordered[ohlcv_columns].to_numpy(dtype=np.float32, copy=True)
        stock_id = ordered["tic_id"].to_numpy(dtype=np.int64, copy=True)
        group_id = (
            ordered["group_id"].to_numpy(dtype=np.int64, copy=True)
            if "group_id" in ordered.columns
            else np.zeros(len(ordered), dtype=np.int64)
        )
        day = ordered["day"].to_numpy(dtype=np.int32, copy=True)
        month = ordered["month"].to_numpy(dtype=np.int32, copy=True)
        dividend_flag = (
            ordered["dividend_flag"].to_numpy(dtype=np.int32, copy=True)
            if "dividend_flag" in ordered.columns
            else np.ones(len(ordered), dtype=np.int32)
        )
        scalar_target = ordered["target"].to_numpy(dtype=np.float32, copy=True)
        existing_regime = (
            ordered["regime_id"].to_numpy(dtype=np.int64, copy=True)
            if "regime_id" in ordered.columns
            else None
        )

        close_index = ohlcv_columns.index("close")
        valid_window_count = len(ordered) - sequence_length - prediction_horizon + 1
        for start in range(valid_window_count):
            end = start + sequence_length
            future_end = end + prediction_horizon

            seq_features = feature_matrix[start:end]
            if np.isnan(seq_features).any():
                continue

            future_ohlcv = ohlcv_matrix[end:future_end]
            if np.isnan(future_ohlcv).any():
                continue

            last_close = float(ohlcv_matrix[end - 1, close_index])
            future_close_path = future_ohlcv[:, close_index].astype(np.float32, copy=False)
            if abs(last_close) < 1e-8:
                continue
            future_return_path = ((future_close_path - last_close) / last_close) * 100.0
            step_returns = np.diff(np.concatenate([[last_close], future_close_path])) / last_close
            future_volatility = float(np.std(step_returns))

            target_idx = future_end - 1
            target_value = scalar_target[target_idx]
            if include_scalar_target and np.isnan(target_value):
                continue

            sequences["features"].append(seq_features)
            sequences["stock_id"].append(stock_id[start:end])
            sequences["group_id"].append(group_id[start:end])
            sequences["day"].append(day[start:end])
            sequences["month"].append(month[start:end])
            sequences["dividend_flag"].append(dividend_flag[start:end])
            sequences["target"].append(target_value if include_scalar_target else 0.0)
            sequences["future_ohlcv"].append(future_ohlcv)
            sequences["future_ohlcv_mask"].append(np.ones(prediction_horizon, dtype=np.float32))
            sequences["future_close_path"].append(future_close_path)
            sequences["future_return_path"].append(
                future_return_path if include_return_path else np.zeros(prediction_horizon, dtype=np.float32)
            )
            sequences["future_volatility"].append(future_volatility)
            if existing_regime is not None and include_regime_label:
                fallback_regimes.append(int(existing_regime[target_idx]))
            else:
                fallback_regimes.append(-1)

    if not sequences["features"]:
        return {key: np.array([]) for key in KRONOS_RICH_ARRAY_KEYS}

    future_volatility = np.asarray(sequences["future_volatility"], dtype=np.float32)
    existing_regime = None
    if include_regime_label and any(value >= 0 for value in fallback_regimes):
        existing_regime = np.asarray(fallback_regimes, dtype=np.int64)
        existing_regime = np.where(existing_regime >= 0, existing_regime, 1)
    future_regime = assign_future_regime_labels(
        future_volatility=future_volatility,
        existing_regime=existing_regime,
        include_regime_label=include_regime_label,
        regime_source=regime_source,
        low_quantile=volatility_low_quantile,
        high_quantile=volatility_high_quantile,
    )

    return {
        "features": np.asarray(sequences["features"], dtype=np.float32),
        "stock_id": np.asarray(sequences["stock_id"], dtype=np.int64),
        "group_id": np.asarray(sequences["group_id"], dtype=np.int64),
        "day": np.asarray(sequences["day"], dtype=np.int32),
        "month": np.asarray(sequences["month"], dtype=np.int32),
        "dividend_flag": np.asarray(sequences["dividend_flag"], dtype=np.int32),
        "target": np.asarray(sequences["target"], dtype=np.float32),
        "future_ohlcv": np.asarray(sequences["future_ohlcv"], dtype=np.float32),
        "future_ohlcv_mask": np.asarray(sequences["future_ohlcv_mask"], dtype=np.float32),
        "future_close_path": np.asarray(sequences["future_close_path"], dtype=np.float32),
        "future_return_path": np.asarray(sequences["future_return_path"], dtype=np.float32),
        "future_volatility": future_volatility,
        "future_regime": future_regime.astype(np.int64, copy=False),
    }


def save_kronos_rich_split(output_dir: Path, split_name: str, sequences: Dict[str, np.ndarray], logger) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, array in sequences.items():
        np.save(split_dir / f"{key}.npy", array, allow_pickle=False)
    logger.info(f"Saved Kronos-rich {split_name} split with {len(sequences['target'])} samples")


def write_kronos_rich_metadata(
    output_dir: Path,
    metadata: Dict[str, object],
    feature_cols: List[str],
    logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / "info.json"
    payload = dict(metadata)
    payload["feature_cols"] = feature_cols
    info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "feature_columns.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    logger.info(f"Saved Kronos-rich metadata to {info_path}")


def main() -> int:
    args = parse_args()
    config = load_config("main")
    logger = get_logger("prepare_kronos_rich_data", log_dir="logs")

    settings = resolve_kronos_rich_prep_settings(config, args)
    processed_dir = Path(settings["processed_dir"])
    output_dir = Path(settings["output_dir"])

    logger.info("Preparing Kronos-rich training data...")
    ensure_base_preprocessing(args, processed_dir, logger)
    metadata = load_preprocessing_metadata(processed_dir)
    splits = load_normalized_split_cache(processed_dir)
    feature_cols = list(metadata["feature_cols"])

    for split_name, split_df in splits.items():
        sequences = build_kronos_rich_sequences(
            df=split_df,
            feature_cols=feature_cols,
            sequence_length=int(metadata["sequence_length"]),
            prediction_horizon=int(metadata["prediction_horizon"]),
            ohlcv_columns=list(settings["ohlcv_columns"]),
            include_scalar_target=bool(settings["include_scalar_target"]),
            include_return_path=bool(settings["include_return_path"]),
            include_regime_label=bool(settings["include_regime_label"]),
            regime_source=str(settings["regime_source"]),
            volatility_low_quantile=float(settings["volatility_low_quantile"]),
            volatility_high_quantile=float(settings["volatility_high_quantile"]),
        )
        save_kronos_rich_split(output_dir, split_name, sequences, logger)

    write_kronos_rich_metadata(
        output_dir=output_dir,
        metadata={
            "num_features": int(metadata["num_features"]),
            "num_stocks": int(metadata["num_stocks"]),
            "num_groups": int(metadata["num_groups"]),
            "sequence_length": int(metadata["sequence_length"]),
            "prediction_horizon": int(metadata["prediction_horizon"]),
            "normalize_target": metadata.get("normalize_target"),
            "target_threshold": metadata.get("target_threshold"),
            "ohlcv_columns": list(settings["ohlcv_columns"]),
            "include_scalar_target": bool(settings["include_scalar_target"]),
            "include_return_path": bool(settings["include_return_path"]),
            "include_regime_label": bool(settings["include_regime_label"]),
            "regime_source": str(settings["regime_source"]),
            "volatility_low_quantile": float(settings["volatility_low_quantile"]),
            "volatility_high_quantile": float(settings["volatility_high_quantile"]),
            "source_processed_dir": str(processed_dir),
        },
        feature_cols=feature_cols,
        logger=logger,
    )

    logger.info("Kronos-rich data preparation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
