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
    load_kronos_checkpoint,
)
from src.utils.device import resolve_device, get_device_info
from src.utils.logger import get_logger
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
    for file in ['features', 'stock_id', 'group_id', 'day', 'month', 'dividend_flag', 'target']:
        file_path = split_dir / f'{file}.npy'
        if file_path.exists():
            sequences[file] = np.load(file_path)

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

    # Load config
    config = load_config('model')
    default_model_type = config.get_default_model_type()
    available_model_types = config.get_available_model_types()
    requested_model_type = args.model_type or default_model_type
    if requested_model_type not in available_model_types:
        raise ValueError(
            f"Unknown model type: {requested_model_type}. "
            f"Available models: {available_model_types}"
        )

    # Load data
    data_dir = Path(args.data_dir)

    logger.info(f"Loading {args.split} data from {data_dir}...")

    sequences = load_sequences(data_dir, args.split, max_samples=args.max_samples)

    if sequences is None:
        logger.error(f"No {args.split} data found")
        return 1

    logger.info(f"Loaded {len(sequences['target'])} samples")

    # Load info
    info_path = data_dir / 'info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Create dataset
    dataset = FinancialDataset(sequences, config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=get_eval_batch_size(config),
        shuffle=False,
        num_workers=config.model.device.NUM_WORKERS
    )

    # Get embedding sizes
    embedding_sizes = dataset.get_embedding_sizes()

    # Resolve checkpoint and model type before model creation
    checkpoint_path = find_checkpoint_path(
        model_input=args.model,
        checkpoint_dir=config.model.checkpointing.CHECKPOINT_DIR,
        model_type=requested_model_type,
        num_features=dataset.num_features,
        num_stocks=embedding_sizes['num_stocks'],
        num_groups=embedding_sizes['num_groups'],
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

    if model_type == 'kronos':
        logger.info("Creating Kronos tokenizer/model for validation...")
        tokenizer, model, checkpoint = load_kronos_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
            num_features=dataset.num_features,
            num_stocks=embedding_sizes['num_stocks'],
            num_groups=embedding_sizes['num_groups'],
            device=device,
        )
        logger.info(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        metadata = build_kronos_sequence_metadata(
            data_dir=data_dir,
            split=args.split,
            feature_cols=info.get('feature_cols') or [],
            sequence_length=info['sequence_length'],
            prediction_horizon=info['prediction_horizon'],
            normalize_target=bool(info.get('normalize_target', False)),
            target_threshold=float(info.get('target_threshold', 1.0)),
            expected_samples=len(sequences['target']),
            max_samples=args.max_samples,
        )
        predictions, targets, sample_stock_ids, sample_group_ids = generate_kronos_predictions(
            sequences=sequences,
            metadata=metadata,
            config=config,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=get_eval_batch_size(config),
            normalize_target=bool(info.get('normalize_target', False)),
            target_threshold=float(info.get('target_threshold', 1.0)),
            feature_cols=info.get('feature_cols') or [],
        )
        metrics = compute_kronos_metrics(predictions, targets)
        print_metrics(metrics, prefix=f"{args.split.upper()} - ")

        if args.excel_report:
            report_df, sector_stats = build_kronos_report(
                predictions,
                targets,
                sample_stock_ids,
                sample_group_ids,
            )
            report_df.to_excel(args.excel_report, index=False)
            print_sector_stats(sector_stats)
            logger.info(f"Excel report saved to {args.excel_report}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    else:
        # Create model
        logger.info(f"Creating {model_type} model...")

        model = create_model(
            model_type=model_type,
            num_features=dataset.num_features,
            num_stocks=embedding_sizes['num_stocks'],
            num_groups=embedding_sizes['num_groups'],
            config=config,
            feature_cols=info.get('feature_cols'),
        )

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = load_checkpoint_metadata(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")

        # Validate
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

    return 0


if __name__ == '__main__':
    sys.exit(main())
