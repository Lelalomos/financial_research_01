#!/usr/bin/env python
"""
Data preprocessing script.

Downloads data, calculates features, normalizes, and creates train/val/test splits.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.downloader import DataDownloader
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.utils.data_preview import log_sequence_preview
from src.utils.logger import get_logger

SEQUENCE_ARRAY_KEYS = ["features", "stock_id", "group_id", "day", "month", "target"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess financial data')

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD), default from config'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), default from config'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=None,
        help='List of tickers to download (default: S&P 500)'
    )

    parser.add_argument(
        '--stock-limit',
        type=int,
        default=None,
        help='Limit number of stocks from index (e.g., 400 for first 400 stocks)'
    )

    parser.add_argument(
        '--stocks',
        type=int,
        default=None,
        help='Sample N stocks balanced across ALL group_ids (default: all stocks)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file override'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, use existing data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--export-pre-normalize',
        type=str,
        default='data/pre_normalized.parquet',
        help='Path to export pre-normalization data (parquet format)'
    )

    parser.add_argument(
        '--export-normalized',
        type=str,
        default='data/normalized_data.parquet',
        help='Path to export normalized data (parquet format)'
    )

    parser.add_argument(
        '--no-resume-cache',
        action='store_true',
        help='Disable preprocessing stage cache and rebuild from scratch'
    )

    parser.add_argument(
        '--skip-sequences',
        action='store_true',
        help='Skip saving sequence arrays; training must build sequences from normalized splits'
    )

    return parser.parse_args()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_pipeline_fingerprint(args, config) -> str:
    root = Path(__file__).resolve().parent.parent
    payload = {
        "args": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "tickers": args.tickers,
            "stock_limit": args.stock_limit,
            "stocks": args.stocks,
            "skip_download": args.skip_download,
        },
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "code_hashes": {
            "preprocess_data": _hash_file(Path(__file__).resolve()),
            "feature_engineering": _hash_file(root / "src/data/feature_engineering.py"),
            "feature_engineering_polars": _hash_file(root / "src/data/feature_engineering_polars.py"),
            "preprocessing": _hash_file(root / "src/data/preprocessing.py"),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _stage_is_valid(manifest: dict, fingerprint: str, stage_name: str, required_paths: list[Path]) -> bool:
    if manifest.get("fingerprint") != fingerprint:
        return False
    completed = manifest.get("completed_stages", {})
    if not completed.get(stage_name):
        return False
    return all(path.exists() for path in required_paths)


def _save_split_cache(split_cache_dir: Path, splits: dict) -> None:
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        split_df.to_parquet(split_cache_dir / f"{split_name}.parquet", index=False)


def _load_split_cache(split_cache_dir: Path) -> dict:
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_path = split_cache_dir / f"{split_name}.parquet"
        if split_path.exists():
            import pandas as pd
            splits[split_name] = pd.read_parquet(split_path)
    return splits


def _expected_sequence_paths(output_dir: Path, splits: dict) -> list[Path]:
    required_paths = [
        output_dir / "info.json",
        output_dir / "feature_columns.txt",
    ]
    for split_name, split_df in splits.items():
        if split_df.empty:
            continue
        split_dir = output_dir / split_name
        required_paths.extend(split_dir / f"{key}.npy" for key in SEQUENCE_ARRAY_KEYS)
    return required_paths


def _load_sequence_cache(output_dir: Path, splits: dict) -> dict:
    import numpy as np

    sequences = {}
    for split_name, split_df in splits.items():
        if split_df.empty:
            continue
        split_dir = output_dir / split_name
        sequences[split_name] = {
            key: np.load(split_dir / f"{key}.npy", allow_pickle=False)
            for key in SEQUENCE_ARRAY_KEYS
        }
    return sequences


def _write_preprocessing_metadata(output_dir: Path, info: dict, logger) -> tuple[Path, Path]:
    info_path = output_dir / 'info.json'
    with open(info_path, 'w') as f:
        json.dump({
            'num_stocks': info['num_stocks'],
            'num_groups': info['num_groups'],
            'num_features': info['num_features'],
            'sequence_length': info['sequence_length'],
            'prediction_horizon': info['prediction_horizon'],
            'feature_cols': info['feature_cols'],
            'regime_params': info.get('regime_params'),
            'normalize_target': info.get('normalize_target'),
            'target_threshold': info.get('target_threshold'),
        }, f, indent=2)
    logger.info(f"Saved info to {info_path}")

    feature_cols_path = output_dir / 'feature_columns.txt'
    with open(feature_cols_path, 'w') as f:
        f.write('\n'.join(info['feature_cols']))
    logger.info(f"Saved feature columns to {feature_cols_path}")
    return info_path, feature_cols_path


def main():
    """Main preprocessing pipeline."""
    args = parse_args()

    # Load config
    config = load_config('main')
    if args.start_date:
        config.data.sources.START_DATE = args.start_date
    if args.end_date:
        config.data.sources.END_DATE = args.end_date

    logger = get_logger("preprocess_data", log_dir="logs")

    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / ".cache"
    split_cache_dir = cache_dir / "normalized_splits"
    sampled_cache_path = cache_dir / "sampled_input.parquet"
    feature_cache_path = cache_dir / "feature_engineered.parquet"
    info_cache_path = cache_dir / "normalized_info.json"
    manifest_path = cache_dir / "preprocess_manifest.json"
    manifest = _load_manifest(manifest_path)
    data_mode = getattr(getattr(config.data, "dataset", None), "MODE", "precomputed_sequences")
    skip_sequence_arrays = args.skip_sequences or data_mode in {"precomputed_sequences", "on_the_fly_sequences"}

    pipeline_fingerprint = _build_pipeline_fingerprint(args, config)
    if args.no_resume_cache or manifest.get("fingerprint") != pipeline_fingerprint:
        manifest = {
            "fingerprint": pipeline_fingerprint,
            "completed_stages": {},
        }
        if args.no_resume_cache:
            logger.info("Resume cache disabled for this run; rebuilding all preprocessing stages.")
        else:
            logger.info("Preprocessing fingerprint changed; cached stages will be rebuilt.")
    else:
        logger.info("Preprocessing fingerprint matched cache manifest; resumable stages are enabled.")

    # Check if we can write to the output directory
    # If there are root-owned files from previous runs, we need to handle them
    can_write = True
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            # Check if we can write to this directory
            test_file = split_dir / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError):
                logger.warning(f"Cannot write to {split_dir} - permission denied")
                can_write = False
                break

    if not can_write:
        logger.warning("Cannot write to existing output directory due to permission issues.")
        logger.warning("This usually happens when the directory contains files owned by root from previous Docker runs.")
        logger.warning("Please run from the host machine to clean up:")
        logger.warning(f"  docker exec -u root crnn_predictor rm -rf {output_dir}")
        logger.warning("Or use a different output directory with --output-dir")
        return 1

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(exist_ok=True)

    # Step 1: Download/load and sample raw data
    sampled_from_cache = _stage_is_valid(manifest, pipeline_fingerprint, "sampled_input", [sampled_cache_path])
    if sampled_from_cache:
        logger.info(f"Resuming from cached sampled input: {sampled_cache_path}")
        import pandas as pd
        stock_df = pd.read_parquet(sampled_cache_path)
        vix_df = commodities_df = treasury_df = None
    else:
        if not args.skip_download:
            logger.info("Step 1: Downloading data...")
            downloader = DataDownloader(config)

            existing_data = downloader.load_saved_data()

            if existing_data.get('stocks') is not None:
                logger.info("Found existing data, skipping download")
                stock_df = existing_data['stocks']
                vix_df = existing_data.get('vix')
                commodities_df = existing_data.get('commodities')
                treasury_df = existing_data.get('treasury_yields')
            else:
                downloaded_data = downloader.download_all(tickers=args.tickers, stock_limit=args.stock_limit, save=True)
                stock_df = downloaded_data['stocks']
                vix_df = downloaded_data.get('vix')
                commodities_df = downloaded_data.get('commodities')
                treasury_df = downloaded_data.get('treasury_yields')
        else:
            logger.info("Loading existing data...")
            downloader = DataDownloader(config)
            existing_data = downloader.load_saved_data()

            if existing_data.get('stocks') is None:
                logger.error("No existing data found. Run without --skip-download first.")
                return 1

            stock_df = existing_data['stocks']
            vix_df = existing_data.get('vix')
            commodities_df = existing_data.get('commodities')
            treasury_df = existing_data.get('treasury_yields')

        if args.stocks:
            logger.info(f"Sampling {args.stocks} stocks balanced across all group_ids...")
            from src.data.sampling import sample_stocks_by_group, get_sampling_stats

            if 'group_id' not in stock_df.columns and 'group' in stock_df.columns:
                stock_df['group_id'] = stock_df['group']
            elif 'group_id' not in stock_df.columns and 'GICS Sector' in stock_df.columns:
                stock_df['group_id'] = stock_df['GICS Sector']
            elif 'group_id' not in stock_df.columns:
                logger.warning("No group_id column found, using single group")
                stock_df['group_id'] = 0

            selected_tickers = sample_stocks_by_group(stock_df, args.stocks, seed=42)
            stock_df = stock_df[stock_df['tic'].isin(selected_tickers)].copy()

            stats = get_sampling_stats(stock_df, selected_tickers)
            logger.info(f"Sampled {stats['total_selected']} stocks from {stats['total_groups']} groups:")
            for group_id, count in stats['stocks_per_group'].items():
                logger.info(f"  Group {group_id}: {count} stocks")

        sampled_cache_path.parent.mkdir(parents=True, exist_ok=True)
        stock_df.to_parquet(sampled_cache_path, index=False)
        manifest["completed_stages"]["sampled_input"] = True
        _save_manifest(manifest_path, manifest)
        logger.info(f"Saved sampled input cache to {sampled_cache_path}")

    # Step 2: Feature engineering
    if _stage_is_valid(manifest, pipeline_fingerprint, "feature_engineered", [feature_cache_path]):
        logger.info(f"Resuming from cached feature-engineered dataset: {feature_cache_path}")
        import pandas as pd
        feature_df = pd.read_parquet(feature_cache_path)
    else:
        logger.info("Step 2: Engineering features...")
        if sampled_from_cache:
            downloader = DataDownloader(config)
            existing_data = downloader.load_saved_data()
            vix_df = existing_data.get('vix')
            commodities_df = existing_data.get('commodities')
            treasury_df = existing_data.get('treasury_yields')
        engineer = FeatureEngineer(config)

        feature_df = engineer.add_all_features(
            stock_df,
            vix_df=vix_df,
            commodities_df=commodities_df,
            treasury_df=treasury_df,
            calculate_target=True
        )
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(feature_cache_path, index=False)
        manifest["completed_stages"]["feature_engineered"] = True
        _save_manifest(manifest_path, manifest)
        logger.info(f"Saved feature-engineered cache to {feature_cache_path}")

    # Log feature info
    engineer = FeatureEngineer(config)
    feature_info = engineer.get_feature_info(feature_df)
    logger.info(f"Feature engineering complete:")
    logger.info(f"  Total features: {feature_info['total_features']}")
    logger.info(f"  Price features: {feature_info['price_features']}")
    logger.info(f"  EMA features: {feature_info['ema_features']}")
    logger.info(f"  RSI features: {feature_info['rsi_features']}")
    logger.info(f"  Candlestick patterns: {feature_info['candlestick_patterns']}")

    # Step 3: Preprocessing through normalized splits
    preprocessor = DataPreprocessor(config)
    normalized_cache_files = [split_cache_dir / f"{name}.parquet" for name in ["train", "val", "test"]] + [info_cache_path]
    if _stage_is_valid(manifest, pipeline_fingerprint, "normalized_splits", normalized_cache_files):
        logger.info(f"Resuming from cached normalized splits in {split_cache_dir}")
        splits = _load_split_cache(split_cache_dir)
        info = json.loads(info_cache_path.read_text())
        info.setdefault("normalize_target", config.data.sequences.NORMALIZE_TARGET)
        info.setdefault("target_threshold", config.data.sequences.TARGET_THRESHOLD)
        processed_df = None
    else:
        logger.info("Step 3: Preprocessing data...")
        processed_df, splits, info = preprocessor.preprocess_tabular(
            feature_df,
            fit=True,
            export_pre_normalize=args.export_pre_normalize,
            export_normalized=args.export_normalized
        )
        info["normalize_target"] = config.data.sequences.NORMALIZE_TARGET
        info["target_threshold"] = config.data.sequences.TARGET_THRESHOLD
        _save_split_cache(split_cache_dir, splits)
        info_cache_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        manifest["completed_stages"]["normalized_splits"] = True
        _save_manifest(manifest_path, manifest)
        logger.info(f"Saved normalized split cache to {split_cache_dir}")

    _write_preprocessing_metadata(output_dir, info, logger)

    if skip_sequence_arrays:
        logger.info("Skipping sequence array creation.")
        logger.info(f"  Data mode: {data_mode}")
        logger.info(f"  Normalized split cache: {split_cache_dir}")
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info("Sequence arrays were intentionally skipped.")
        return 0

    # Step 4: Create or load sequences
    sequence_cache_files = _expected_sequence_paths(output_dir, splits)
    if _stage_is_valid(manifest, pipeline_fingerprint, "sequence_arrays", sequence_cache_files):
        logger.info(f"Resuming from cached sequence arrays in {output_dir}")
        sequences = _load_sequence_cache(output_dir, splits)
    else:
        logger.info("Step 4: Creating sequences...")
        sequences = {}
        for split_name, split_df in splits.items():
            if not split_df.empty:
                logger.info(f"Creating sequences for {split_name} split...")
                sequences[split_name] = preprocessor.create_sequences(
                    split_df,
                    feature_cols=info['feature_cols']
                )

        # Step 5: Save processed data
        logger.info("Step 5: Saving processed data...")

        import numpy as np

        # Save sequences
        for split_name in ['train', 'val', 'test']:
            if split_name in sequences:
                split_dir = output_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)

                for key, data in sequences[split_name].items():
                    np.save(split_dir / f'{key}.npy', data)

                logger.info(f"Saved {split_name} sequences: {len(sequences[split_name]['target'])} samples")

        manifest["completed_stages"]["sequence_arrays"] = True
        _save_manifest(manifest_path, manifest)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  Train samples: {len(sequences.get('train', {}).get('target', []))}")
    logger.info(f"  Val samples: {len(sequences.get('val', {}).get('target', []))}")
    logger.info(f"  Test samples: {len(sequences.get('test', {}).get('target', []))}")
    ticker_map = {}
    if 'tic_id' in feature_df.columns and 'tic' in feature_df.columns:
        ticker_map = {
            int(row['tic_id']): str(row['tic'])
            for _, row in feature_df[['tic_id', 'tic']].drop_duplicates().iterrows()
        }
    log_sequence_preview(
        logger=logger,
        sequences=sequences.get('train', {}),
        feature_cols=info.get('feature_cols'),
        ticker_map=ticker_map,
        split_name='train',
        max_rows=10,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
