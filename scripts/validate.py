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
from src.evaluation import Validator
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
        help='Path to model checkpoint or "best" for best model'
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
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for validation results (JSON)'
    )

    return parser.parse_args()


def load_sequences(data_dir: Path, split: str):
    """Load sequences from directory."""
    split_dir = data_dir / split

    if not split_dir.exists():
        return None

    sequences = {}
    for file in ['features', 'stock_id', 'group_id', 'day', 'month', 'target']:
        file_path = split_dir / f'{file}.npy'
        if file_path.exists():
            sequences[file] = np.load(file_path)

    if len(sequences) == 0:
        return None

    return sequences


def main():
    """Main validation function."""
    args = parse_args()

    logger = get_logger("validate", log_dir="logs")

    logger.info("=" * 60)
    logger.info("VALIDATION SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Using device: {args.device}")

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

    sequences = load_sequences(data_dir, args.split)

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

    # Create model
    logger.info(f"Creating {model_type} model...")

    model = create_model(
        model_type=model_type,
        num_features=dataset.num_features,
        num_stocks=embedding_sizes['num_stocks'],
        num_groups=embedding_sizes['num_groups'],
        config=config
    )

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = load_checkpoint_metadata(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")

    # Validate
    logger.info("Validating model...")

    validator = Validator(model, config, device=args.device)

    metrics = validator.validate(loader, log_file=args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
