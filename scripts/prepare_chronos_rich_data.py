#!/usr/bin/env python
"""
Prepare Chronos-rich multi-target training data from normalized split caches.

This reuses the same rich target structure as Kronos-rich, but writes to a
Chronos-specific processed directory so the new Chronos-based model can train
without sharing artifacts with the Kronos branch.
"""

import argparse
from pathlib import Path
from typing import Dict

from src.config import load_config
from src.utils.logger import get_logger

from scripts.prepare_chronos2_data import ensure_base_preprocessing, load_normalized_split_cache, load_preprocessing_metadata
from scripts.prepare_kronos_rich_data import (
    build_kronos_rich_sequences,
    save_kronos_rich_split,
    write_kronos_rich_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Chronos-rich multi-target data")
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


def resolve_chronos_rich_prep_settings(config, args) -> Dict[str, object]:
    prep_cfg = getattr(config.data, "chronos_rich_preparation", None)
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
        or "data/processed_chronos_rich"
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


def main() -> int:
    args = parse_args()
    config = load_config("main")
    logger = get_logger("prepare_chronos_rich_data", log_dir="logs")

    settings = resolve_chronos_rich_prep_settings(config, args)
    processed_dir = Path(settings["processed_dir"])
    output_dir = Path(settings["output_dir"])

    logger.info("Preparing Chronos-rich training data...")
    ensure_base_preprocessing(args, processed_dir, logger)
    metadata = load_preprocessing_metadata(processed_dir)
    splits = load_normalized_split_cache(processed_dir)
    feature_cols = list(metadata["feature_cols"])

    for split_name, split_df in splits.items():
        sequences = build_kronos_rich_sequences(
            df=split_df,
            feature_cols=feature_cols,
            sequence_length=metadata["sequence_length"],
            prediction_horizon=metadata["prediction_horizon"],
            ohlcv_columns=list(settings["ohlcv_columns"]),
            include_scalar_target=bool(settings["include_scalar_target"]),
            include_return_path=bool(settings["include_return_path"]),
            include_regime_label=bool(settings["include_regime_label"]),
            regime_source=str(settings["regime_source"]),
            volatility_low_quantile=float(settings["volatility_low_quantile"]),
            volatility_high_quantile=float(settings["volatility_high_quantile"]),
        )
        save_kronos_rich_split(output_dir, split_name, sequences, logger)

    metadata["data_dir"] = str(output_dir)
    metadata["model_type"] = "chronos_rich"
    write_kronos_rich_metadata(output_dir, metadata, feature_cols, logger)
    logger.info("Chronos-rich preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
