#!/usr/bin/env python
"""
Validation script for CRNN models.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset import FinancialDataset
from src.models import create_model
from src.evaluation import Validator, evaluate_model_with_report, print_metrics, print_sector_stats
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
    parser = argparse.ArgumentParser(description='Validate CRNN model')

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
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='val',
        help='Data split to validate'
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
        help='Output file for validation results (JSON)'
    )

    parser.add_argument(
        '--excel-report',
        type=str,
        default=None,
        help='Output path for Excel validation report (e.g., outputs/validate_report.xlsx)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional limit for validation samples. Useful for quick smoke tests.'
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


def main():
    """Main validation function."""
    args = parse_args()

    logger = get_logger("validate", log_dir="logs")
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
        run_logger.log_run("validation_runs", payload)

    logger.info("=" * 60)
    logger.info("VALIDATION SCRIPT")
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
        default_model_type = config.get_default_model_type()
        available_model_types = config.get_available_model_types()
        requested_model_type = args.model_type or default_model_type
        if requested_model_type not in available_model_types:
            raise ValueError(
                f"Unknown model type: {requested_model_type}. "
                f"Available models: {available_model_types}"
            )

        data_dir = Path(args.data_dir)
        logger.info(f"Loading {args.split} data from {data_dir}...")

        sequences = load_sequences(data_dir, args.split, max_samples=args.max_samples)

        if sequences is None:
            message = f"No {args.split} data found"
            logger.error(message)
            log_db("failed", error_message=message)
            return 1

        sample_count = len(sequences['target'])
        logger.info(f"Loaded {sample_count} samples")

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
            model_type=requested_model_type,
            num_features=dataset.num_features,
            num_stocks=resolved_num_stocks,
            num_groups=resolved_num_groups,
        )

        model_type = requested_model_type
        if args.model_type is None:
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
            logger.info("Creating Kronos tokenizer/model for validation...")
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
                report_df, sector_stats = build_kronos_report(
                    predictions,
                    targets,
                    sample_stock_ids,
                    sample_group_ids,
                    raw_predictions=raw_predictions,
                    raw_targets=raw_targets,
                    normalize_target=bool(info.get('normalize_target', False)),
                    target_threshold=float(info.get('target_threshold', 1.0)),
                )
                report_df.to_excel(args.excel_report, index=False)
                print_sector_stats(sector_stats)
                logger.info(f"Excel report saved to {args.excel_report}")

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Results saved to {args.output}")
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
            logger.info("Validating model...")

            validator = Validator(model, config, device=str(device))

            if args.excel_report:
                logger.info("Evaluating model with detailed validation report...")
                metrics, report_df, sector_stats = evaluate_model_with_report(
                    model,
                    loader,
                    device=str(device),
                    output_path=args.excel_report,
                )
                print_metrics(metrics, prefix=f"{args.split.upper()} - ")
                print_sector_stats(sector_stats)

                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    logger.info(f"Results saved to {args.output}")

                logger.info(f"Excel report saved to {args.excel_report}")
            else:
                metrics = validator.validate(loader, log_file=args.output)

        db_extra["checkpoint_epoch"] = checkpoint_epoch
        db_extra.update(metrics)
        log_db("completed", extra=db_extra)
        return 0
    except Exception as exc:
        log_db("failed", error_message=str(exc))
        raise


if __name__ == '__main__':
    sys.exit(main())
