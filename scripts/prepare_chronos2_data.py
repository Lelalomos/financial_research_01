#!/usr/bin/env python
"""
Prepare Chronos2-specific training data from normalized split caches.

This script does not replace the existing scalar-target preprocessing flow.
Instead it builds a separate dataset with future target paths so Chronos2-style
quantile training can be experimented with safely.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils.logger import get_logger


CHRONOS2_SEQUENCE_ARRAY_KEYS = [
    "features",
    "stock_id",
    "group_id",
    "day",
    "month",
    "dividend_flag",
    "target",
    "future_target",
    "future_target_mask",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Chronos2 path-target data")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date override for the base preprocessing stage",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date override for the base preprocessing stage",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="Explicit ticker list for the base preprocessing stage",
    )
    parser.add_argument(
        "--stock-limit",
        type=int,
        default=None,
        help="Limit number of downloaded stocks during base preprocessing",
    )
    parser.add_argument(
        "--stocks",
        type=int,
        default=None,
        help="Balanced stock sampling count for the base preprocessing stage",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Reserved config override path for compatibility with preprocess wrappers",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing downloaded market data during base preprocessing",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Existing processed dataset directory with .cache/normalized_splits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Chronos2-prepared arrays",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Column used to build Chronos2 future target paths",
    )
    parser.add_argument(
        "--skip-scalar-target",
        action="store_true",
        help="Do not save the repo scalar target array alongside future_target",
    )
    parser.add_argument(
        "--export-pre-normalize",
        type=str,
        default="data/pre_normalized.parquet",
        help="Path to export pre-normalized data during base preprocessing",
    )
    parser.add_argument(
        "--export-normalized",
        type=str,
        default="data/normalized_data.parquet",
        help="Path to export normalized data during base preprocessing",
    )
    parser.add_argument(
        "--no-resume-cache",
        action="store_true",
        help="Disable resume cache for the base preprocessing stage",
    )
    return parser.parse_args()


def resolve_chronos2_prep_settings(config, args) -> Dict[str, object]:
    prep_cfg = getattr(config.data, "chronos2_preparation", None)
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
        or "data/processed_chronos2"
    )
    target_column = (
        args.target_column
        or (
            getattr(prep_cfg, "TARGET_COLUMN", None)
            if prep_cfg is not None
            else None
        )
        or "close"
    )
    include_scalar_target = not args.skip_scalar_target
    if prep_cfg is not None:
        include_scalar_target = bool(
            getattr(prep_cfg, "INCLUDE_SCALAR_TARGET", True)
        ) and include_scalar_target
    target_mode = (
        getattr(prep_cfg, "TARGET_MODE", None)
        if prep_cfg is not None
        else None
    ) or "trend_extension"
    trend_lookback = int(
        getattr(prep_cfg, "TREND_LOOKBACK", 7)
        if prep_cfg is not None
        else 7
    )
    trend_method = (
        getattr(prep_cfg, "TREND_METHOD", None)
        if prep_cfg is not None
        else None
    ) or "mean_gap"

    return {
        "processed_dir": processed_dir,
        "output_dir": output_dir,
        "target_column": target_column,
        "include_scalar_target": include_scalar_target,
        "target_mode": target_mode,
        "trend_lookback": trend_lookback,
        "trend_method": trend_method,
    }


def _load_preprocess_module():
    script_path = Path(__file__).with_name("preprocess_data.py")
    spec = importlib.util.spec_from_file_location("chronos2_base_preprocess", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_base_preprocessing(args, processed_dir: Path, logger) -> None:
    if getattr(args, "no_resume_cache", False):
        logger.info("Chronos2 base preprocessing will ignore resume cache because --no-resume-cache was requested")
    else:
        logger.info(
            "Chronos2 preparation requires normalized split cache; "
            "running base preprocessing so fingerprint-based cache validation can decide whether reuse is safe..."
        )
    preprocess_module = _load_preprocess_module()
    preprocess_args = SimpleNamespace(
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=args.tickers,
        stock_limit=args.stock_limit,
        stocks=args.stocks,
        config=args.config,
        skip_download=args.skip_download,
        output_dir=str(processed_dir),
        export_pre_normalize=args.export_pre_normalize,
        export_normalized=args.export_normalized,
        no_resume_cache=args.no_resume_cache,
        skip_sequences=True,
    )
    original_parse_args = preprocess_module.parse_args
    try:
        preprocess_module.parse_args = lambda: preprocess_args
        result = preprocess_module.main()
    finally:
        preprocess_module.parse_args = original_parse_args

    if result != 0:
        raise RuntimeError(f"Base preprocessing failed with exit code {result}")


def load_preprocessing_metadata(processed_dir: Path) -> Dict[str, object]:
    info_path = processed_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing metadata at {info_path}"
        )
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_normalized_split_cache(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    split_cache_dir = processed_dir / ".cache" / "normalized_splits"
    if not split_cache_dir.exists():
        raise FileNotFoundError(
            f"Missing normalized split cache at {split_cache_dir}"
        )

    splits: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_path = split_cache_dir / f"{split_name}.parquet"
        if split_path.exists():
            splits[split_name] = pd.read_parquet(split_path)
    return splits


def build_chronos2_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int,
    prediction_horizon: int,
    target_column: str,
    include_scalar_target: bool = True,
    target_mode: str = "trend_extension",
    trend_lookback: int = 7,
    trend_method: str = "mean_gap",
) -> Dict[str, np.ndarray]:
    if df is None or df.empty:
        return {
            "features": np.array([]),
            "stock_id": np.array([]),
            "group_id": np.array([]),
            "day": np.array([]),
            "month": np.array([]),
            "dividend_flag": np.array([]),
            "target": np.array([]),
            "future_target": np.array([]),
            "future_target_mask": np.array([]),
        }

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in split DataFrame")

    stock_col = "tic_id" if "tic_id" in df.columns else "tic"

    sequences = {key: [] for key in CHRONOS2_SEQUENCE_ARRAY_KEYS}
    for _, stock_df in df.groupby(stock_col, sort=True):
        ordered = stock_df.sort_values("date").reset_index(drop=True)
        if len(ordered) < sequence_length + prediction_horizon:
            continue

        feature_matrix = ordered[feature_cols].to_numpy(dtype=np.float32, copy=True)
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
        trend_series = ordered[target_column].to_numpy(dtype=np.float32, copy=True)

        valid_window_count = len(ordered) - sequence_length - prediction_horizon + 1
        for start in range(valid_window_count):
            end = start + sequence_length
            future_end = end + prediction_horizon

            seq_features = feature_matrix[start:end]
            if np.isnan(seq_features).any():
                continue

            recent_window_start = max(start, end - max(2, trend_lookback))
            recent_values = trend_series[recent_window_start:end]
            if np.isnan(recent_values).any() or recent_values.shape[0] < 2:
                continue

            future_target = generate_future_target(
                recent_values=recent_values,
                prediction_horizon=prediction_horizon,
                target_mode=target_mode,
                trend_method=trend_method,
            )

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
            sequences["future_target"].append(future_target)
            sequences["future_target_mask"].append(np.ones(prediction_horizon, dtype=np.float32))

    if not sequences["features"]:
        return {
            "features": np.array([]),
            "stock_id": np.array([]),
            "group_id": np.array([]),
            "day": np.array([]),
            "month": np.array([]),
            "dividend_flag": np.array([]),
            "target": np.array([]),
            "future_target": np.array([]),
            "future_target_mask": np.array([]),
        }

    return {
        "features": np.asarray(sequences["features"], dtype=np.float32),
        "stock_id": np.asarray(sequences["stock_id"], dtype=np.int64),
        "group_id": np.asarray(sequences["group_id"], dtype=np.int64),
        "day": np.asarray(sequences["day"], dtype=np.int32),
        "month": np.asarray(sequences["month"], dtype=np.int32),
        "dividend_flag": np.asarray(sequences["dividend_flag"], dtype=np.int32),
        "target": np.asarray(sequences["target"], dtype=np.float32),
        "future_target": np.asarray(sequences["future_target"], dtype=np.float32),
        "future_target_mask": np.asarray(sequences["future_target_mask"], dtype=np.float32),
    }


def generate_future_target(
    recent_values: np.ndarray,
    prediction_horizon: int,
    target_mode: str = "trend_extension",
    trend_method: str = "mean_gap",
) -> np.ndarray:
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    if recent_values.shape[0] < 2:
        raise ValueError("recent_values must contain at least two observations")
    if target_mode != "trend_extension":
        raise ValueError(f"Unsupported Chronos2 target mode: {target_mode}")
    if trend_method != "mean_gap":
        raise ValueError(f"Unsupported Chronos2 trend method: {trend_method}")

    gaps = np.diff(recent_values.astype(np.float32, copy=False))
    mean_gap = float(np.mean(gaps))
    last_value = float(recent_values[-1])
    steps = np.arange(prediction_horizon, dtype=np.float32)
    return last_value + (steps * mean_gap)


def save_chronos2_split(output_dir: Path, split_name: str, sequences: Dict[str, np.ndarray], logger) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, array in sequences.items():
        np.save(split_dir / f"{key}.npy", array, allow_pickle=False)
    logger.info(f"Saved Chronos2 {split_name} split with {len(sequences['target'])} samples")


def write_chronos2_metadata(
    output_dir: Path,
    metadata: Dict[str, object],
    feature_cols: List[str],
    logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / "info.json"
    info_payload = dict(metadata)
    info_payload["feature_cols"] = feature_cols
    info_path.write_text(json.dumps(info_payload, indent=2), encoding="utf-8")
    logger.info(f"Saved Chronos2 metadata to {info_path}")

    feature_cols_path = output_dir / "feature_columns.txt"
    feature_cols_path.write_text("\n".join(feature_cols), encoding="utf-8")
    logger.info(f"Saved Chronos2 feature columns to {feature_cols_path}")


def main() -> int:
    args = parse_args()
    config = load_config("main")
    logger = get_logger("prepare_chronos2_data", log_dir="logs")

    settings = resolve_chronos2_prep_settings(config, args)
    processed_dir = settings["processed_dir"]
    output_dir = settings["output_dir"]
    target_column = settings["target_column"]
    include_scalar_target = settings["include_scalar_target"]
    target_mode = settings["target_mode"]
    trend_lookback = settings["trend_lookback"]
    trend_method = settings["trend_method"]

    logger.info("Preparing Chronos2-specific training data...")
    ensure_base_preprocessing(args, Path(processed_dir), logger)
    metadata = load_preprocessing_metadata(processed_dir)
    splits = load_normalized_split_cache(processed_dir)
    feature_cols = list(metadata["feature_cols"])
    sequence_length = int(metadata["sequence_length"])
    prediction_horizon = int(metadata["prediction_horizon"])

    for split_name, split_df in splits.items():
        sequences = build_chronos2_sequences(
            df=split_df,
            feature_cols=feature_cols,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_column=str(target_column),
            include_scalar_target=bool(include_scalar_target),
            target_mode=str(target_mode),
            trend_lookback=int(trend_lookback),
            trend_method=str(trend_method),
        )
        save_chronos2_split(Path(output_dir), split_name, sequences, logger)

    write_chronos2_metadata(
        Path(output_dir),
        metadata={
            "num_features": int(metadata["num_features"]),
            "num_stocks": int(metadata["num_stocks"]),
            "num_groups": int(metadata["num_groups"]),
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
            "normalize_target": metadata.get("normalize_target"),
            "target_threshold": metadata.get("target_threshold"),
            "target_column": target_column,
            "include_scalar_target": bool(include_scalar_target),
            "target_mode": target_mode,
            "trend_lookback": int(trend_lookback),
            "trend_method": trend_method,
            "source_processed_dir": str(processed_dir),
        },
        feature_cols=feature_cols,
        logger=logger,
    )

    logger.info("Chronos2 data preparation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
