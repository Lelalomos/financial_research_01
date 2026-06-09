#!/usr/bin/env python
"""
Testing script for CRNN models.

Generates Excel report with:
- Ticker name, real target, predict target, distance
- Std of real, std of predict
- Direction score (1 if same sign, 0 if different)
- Sector name
- Direction accuracy by sector in terminal output
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset import FinancialDataset
from src.data.preprocessing import DataPreprocessor
from src.models import create_model
from src.evaluation import evaluate_model, print_metrics, evaluate_model_with_report, print_sector_stats
from src.evaluation.kronos import (
    build_kronos_report,
    build_kronos_sequence_metadata,
    compute_kronos_metrics,
    generate_kronos_predictions,
    is_kronos_family,
    load_kronos_checkpoint,
    resolve_kronos_embedding_sizes,
)
from src.utils.device import resolve_device, get_device_info
from src.utils.logger import get_logger
from src.utils.postgres_logging import make_run_logger, make_run_uuid
from src.training import (
    find_checkpoint_path,
    get_eval_batch_size,
    infer_model_type_from_checkpoint,
    load_checkpoint_metadata,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test CRNN model')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint or alias ("best" or "final")'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        help='Model type. Defaults to config.model.selection.DEFAULT_MODEL_TYPE.'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data'
    )

    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default=None,
        help='Directory with raw data (for ticker/sector mapping). If not provided, will use IDs only.'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Data split to evaluate'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (e.g. cuda, cuda:0, cpu). Defaults to robust auto-detect.'
    )

    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )

    parser.add_argument(
        '--excel-report',
        type=str,
        default=None,
        help='Output path for Excel report (e.g., reports/test_report.xlsx)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional limit for evaluation samples. Useful for quick smoke tests.'
    )

    return parser.parse_args()


def load_sequences(data_dir: Path, split: str, max_samples: int = None):
    """Load sequences from directory."""
    split_dir = data_dir / split

    if not split_dir.exists():
        return None

    sequences = {}
    for file_path in sorted(split_dir.glob("*.npy")):
        sequences[file_path.stem] = np.load(file_path)

    if len(sequences) == 0:
        return None

    if max_samples is not None:
        limit = max(int(max_samples), 0)
        sequences = {key: value[:limit] for key, value in sequences.items()}

    return sequences


def load_id_mappings(data_dir: Path, raw_data_dir: Path = None):
    """
    Load stock_id to ticker and group_id to sector mappings from preprocessed data.

    Args:
        data_dir: Processed data directory
        raw_data_dir: Raw data directory (optional, for fallback mapping)

    Returns:
        Tuple of (stock_id_to_ticker, group_id_to_sector) dictionaries
    """
    stock_id_to_ticker = {}
    group_id_to_sector = {}

    import pandas as pd

    # First try: Check for pre_normalized.parquet in the data directory
    pre_normalized_path = data_dir.parent / 'pre_normalized.parquet'
    if pre_normalized_path.exists():
        try:
            df = pd.read_parquet(pre_normalized_path)

            # Extract stock_id -> ticker mapping
            if 'tic_id' in df.columns and 'tic' in df.columns:
                for _, row in df[['tic_id', 'tic']].drop_duplicates().iterrows():
                    stock_id_to_ticker[int(row['tic_id'])] = row['tic']

            # Extract group_id -> sector mapping
            if 'group_id' in df.columns and 'group' in df.columns:
                for _, row in df[['group_id', 'group']].drop_duplicates().iterrows():
                    group_id_to_sector[int(row['group_id'])] = row['group']

            if stock_id_to_ticker and group_id_to_sector:
                return stock_id_to_ticker, group_id_to_sector
        except Exception as e:
            pass

    # Second try: Check for parquet files in processed split directories
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split

        # Check for parquet files that might contain the mappings
        for parquet_file in split_dir.glob('*.parquet'):
            try:
                df = pd.read_parquet(parquet_file)

                # Extract stock_id -> ticker mapping
                if 'tic_id' in df.columns and 'tic' in df.columns:
                    for _, row in df[['tic_id', 'tic']].drop_duplicates().iterrows():
                        stock_id_to_ticker[int(row['tic_id'])] = row['tic']

                # Extract group_id -> sector mapping
                if 'group_id' in df.columns and 'group' in df.columns:
                    for _, row in df[['group_id', 'group']].drop_duplicates().iterrows():
                        group_id_to_sector[int(row['group_id'])] = row['group']

                break  # Found mappings, no need to check other files
            except Exception as e:
                continue

    # If found mappings, return them
    if stock_id_to_ticker and group_id_to_sector:
        return stock_id_to_ticker, group_id_to_sector

    # Third try: If no mappings found in parquet files, try to create from raw data
    if not stock_id_to_ticker and raw_data_dir:
        logger = get_logger("test", log_dir="logs")
        logger.warning("No ticker/sector mappings found in processed data. Attempting to create from raw data...")

        try:
            import pandas as pd
            from src.data.feature_engineering import FeatureEngineer

            config = load_config('main')
            engineer = FeatureEngineer(config)

            # Create preprocessor to get encoders
            preprocessor = DataPreprocessor(config)

            # Load raw data to create encoders
            raw_path = Path(raw_data_dir)
            if raw_path.exists():
                all_files = list(raw_path.rglob('*.parquet'))[:5]  # Limit files for speed
                if all_files:
                    dfs = []
                    for f in all_files:
                        try:
                            dfs.append(pd.read_parquet(f))
                        except:
                            continue

                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=True)

                        # Fit encoders
                        preprocessor.encode_categorical(combined_df, fit=True)

                        # Create mappings
                        for i, ticker in enumerate(preprocessor.stock_encoder.classes_):
                            stock_id_to_ticker[i] = ticker

                        if hasattr(preprocessor.group_encoder, 'classes_'):
                            for i, group in enumerate(preprocessor.group_encoder.classes_):
                                group_id_to_sector[i] = group

                        logger.info(f"Created mappings for {len(stock_id_to_ticker)} stocks and {len(group_id_to_sector)} sectors")
        except Exception as e:
            logger.warning(f"Could not create mappings from raw data: {e}")

    return stock_id_to_ticker, group_id_to_sector


def main():
    """Main testing function."""
    args = parse_args()

    logger = get_logger("test", log_dir="logs")
    config = load_config('model')
    run_logger = make_run_logger(
        logger,
        enabled=bool(getattr(getattr(config.model, "postgres_logging", None), "ENABLED", True)),
    )
    run_uuid = make_run_uuid()
    base_payload = {
        "run_uuid": run_uuid,
        "cli_args_json": vars(args),
        "data_dir": args.data_dir,
        "split": args.split,
        "raw_data_dir": args.raw_data_dir,
        "output_json_path": args.output,
        "excel_report_path": args.excel_report,
        "max_samples": args.max_samples,
    }

    def log_db(status, extra=None, error_message=None):
        payload = dict(base_payload)
        payload["status"] = status
        if error_message is not None:
            payload["error_message"] = error_message
        if extra:
            payload.update(extra)
        run_logger.log_run("test_runs", payload)

    logger.info("=" * 60)
    logger.info("TESTING SCRIPT")
    logger.info("=" * 60)
    device = resolve_device(requested_device=args.device, force_cpu=args.force_cpu, verbose=True)
    device_info = get_device_info(verbose=False)
    logger.info(f"Resolved device: {device}")
    logger.info(f"CUDA available: {device_info['cuda_available']}")
    logger.info(f"CUDA working: {device_info.get('cuda_working', False)}")
    if device_info.get('cuda_working'):
        logger.info(f"GPU: {device_info.get('gpu_name', 'Unknown')}")

    try:
        main_config = load_config('main')
        base_payload.update(
            {
                "device": str(device),
                "main_config_json": main_config.to_dict(),
                "model_config_json": config.to_dict(),
            }
        )

        data_dir = Path(args.data_dir)
        raw_data_dir = Path(args.raw_data_dir) if args.raw_data_dir else None

        logger.info(f"Loading {args.split} data from {data_dir}...")

        sequences = load_sequences(data_dir, args.split, max_samples=args.max_samples)

        if sequences is None:
            message = f"No {args.split} data found"
            logger.error(message)
            log_db("failed", error_message=message)
            return 1

        sample_count = len(sequences['target'])
        logger.info(f"Loaded {sample_count} samples")

        stock_id_to_ticker, group_id_to_sector = load_id_mappings(data_dir, raw_data_dir)

        if stock_id_to_ticker:
            logger.info(f"Loaded {len(stock_id_to_ticker)} ticker mappings")
        else:
            logger.warning("No ticker mappings available, will use stock IDs")

        if group_id_to_sector:
            logger.info(f"Loaded {len(group_id_to_sector)} sector mappings")
        else:
            logger.warning("No sector mappings available, will use group IDs")

        default_model_type = config.get_default_model_type()
        model_type = args.model_type or default_model_type
        available_model_types = config.get_available_model_types()
        if model_type not in available_model_types:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {available_model_types}"
            )

        info_path = data_dir / 'info.json'
        with open(info_path, 'r') as f:
            info = json.load(f)

        dataset = FinancialDataset(sequences, config)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=get_eval_batch_size(config),
            shuffle=False,
            num_workers=config.model.device.NUM_WORKERS
        )

        embedding_sizes = dataset.get_embedding_sizes()
        resolved_num_stocks, resolved_num_groups = resolve_kronos_embedding_sizes(info, embedding_sizes)

        checkpoint_path = find_checkpoint_path(
            model_input=args.model,
            checkpoint_dir=config.model.checkpointing.CHECKPOINT_DIR,
            model_type=model_type,
            num_features=dataset.num_features,
            num_stocks=resolved_num_stocks,
            num_groups=resolved_num_groups,
        )

        if args.model_type is None and checkpoint_path:
            model_type = infer_model_type_from_checkpoint(
                checkpoint_path,
                available_model_types,
                fallback_model_type=default_model_type,
            )
            logger.info(f"Auto-detected model type: {model_type}")

        resolved_checkpoint = Path(checkpoint_path)
        logger.info(f"Resolved model type: {model_type}")
        logger.info(f"Selected checkpoint file: {resolved_checkpoint.name}")
        logger.info(f"Selected checkpoint path: {resolved_checkpoint}")

        db_extra = {
            "model_type": model_type,
            "checkpoint_path": str(resolved_checkpoint),
            "dataset_info_json": info,
            "feature_cols_json": info.get('feature_cols') or [],
            "num_features": dataset.num_features,
            "num_stocks": resolved_num_stocks,
            "num_groups": resolved_num_groups,
            "sequence_length": info.get('sequence_length'),
            "prediction_horizon": info.get('prediction_horizon'),
            "normalize_target": bool(info.get('normalize_target', False)),
            "target_threshold": float(info.get('target_threshold', 1.0)),
            "sample_count": sample_count,
        }

        if is_kronos_family(model_type):
            logger.info("Creating Kronos tokenizer/model for evaluation...")
            tokenizer, model, checkpoint = load_kronos_checkpoint(
                checkpoint_path=checkpoint_path,
                config=config,
                num_features=dataset.num_features,
                num_stocks=resolved_num_stocks,
                num_groups=resolved_num_groups,
                device=device,
                model_type=model_type,
            )
            checkpoint_epoch = checkpoint.get('epoch')
            logger.info(f"Checkpoint from epoch {checkpoint_epoch or 'unknown'}")

            metadata = build_kronos_sequence_metadata(
                data_dir=data_dir,
                split=args.split,
                feature_cols=info.get('feature_cols') or [],
                sequence_length=info['sequence_length'],
                prediction_horizon=info['prediction_horizon'],
                normalize_target=bool(info.get('normalize_target', False)),
                target_threshold=float(info.get('target_threshold', 1.0)),
                expected_samples=sample_count,
                max_samples=args.max_samples,
            )
            predictions, targets, sample_stock_ids, sample_group_ids, raw_predictions, raw_targets = generate_kronos_predictions(
                sequences=sequences,
                metadata=metadata,
                data_dir=data_dir,
                config=config,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=get_eval_batch_size(config),
                normalize_target=bool(info.get('normalize_target', False)),
                target_threshold=float(info.get('target_threshold', 1.0)),
                feature_cols=info.get('feature_cols') or [],
                model_type=model_type,
            )
            metrics = compute_kronos_metrics(predictions, targets)
            print_metrics(metrics, prefix=f"{args.split.upper()} - ")

            if args.excel_report:
                logger.info("Generating Kronos detailed report...")
                report_df, sector_stats = build_kronos_report(
                    predictions,
                    targets,
                    sample_stock_ids,
                    sample_group_ids,
                    raw_predictions=raw_predictions,
                    raw_targets=raw_targets,
                    normalize_target=bool(info.get('normalize_target', False)),
                    target_threshold=float(info.get('target_threshold', 1.0)),
                    stock_id_to_ticker=stock_id_to_ticker if stock_id_to_ticker else None,
                    group_id_to_sector=group_id_to_sector if group_id_to_sector else None,
                )
                report_df.to_excel(args.excel_report, index=False)
                print_sector_stats(sector_stats)
                logger.info(f"Excel report saved to {args.excel_report}")
        else:
            logger.info(f"Creating {model_type} model...")

            model = create_model(
                model_type=model_type,
                num_features=dataset.num_features,
                num_stocks=resolved_num_stocks,
                num_groups=resolved_num_groups,
                config=config,
                feature_cols=info.get('feature_cols'),
            )

            logger.info(f"Loading checkpoint from {checkpoint_path}")

            checkpoint = load_checkpoint_metadata(checkpoint_path, map_location=device)
            checkpoint_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)

            logger.info(f"Checkpoint from epoch {checkpoint_epoch}")

            if args.excel_report:
                logger.info("Evaluating model with detailed report...")

                metrics, report_df, sector_stats = evaluate_model_with_report(
                    model,
                    loader,
                    device=str(device),
                    stock_id_to_ticker=stock_id_to_ticker if stock_id_to_ticker else None,
                    group_id_to_sector=group_id_to_sector if group_id_to_sector else None,
                    output_path=args.excel_report
                )

                print_metrics(metrics, prefix=f"{args.split.upper()} - ")
                print_sector_stats(sector_stats)

                logger.info(f"Excel report saved to {args.excel_report}")
            else:
                logger.info("Evaluating model...")
                metrics = evaluate_model(model, loader, device=str(device))
                print_metrics(metrics, prefix=f"{args.split.upper()} - ")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Results saved to {args.output}")

        db_extra["checkpoint_epoch"] = checkpoint_epoch
        db_extra.update(metrics)
        log_db("completed", extra=db_extra)
        return 0
    except Exception as exc:
        log_db("failed", error_message=str(exc))
        raise


if __name__ == '__main__':
    sys.exit(main())
