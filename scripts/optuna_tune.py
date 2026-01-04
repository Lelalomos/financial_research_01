#!/usr/bin/env python
"""
Optuna hyperparameter tuning script.

This script performs hyperparameter optimization using Optuna:
- Creates small balanced dataset (optional)
- Runs Optuna study with configurable trials
- Outputs best hyperparameters to JSON
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import HyperparameterSearchConfig, get_config_for_model
from src.hyperparameter import OptunaOptimizer
from src.data import FinancialDataset
from src.utils.logger import get_logger

logger = get_logger("optuna_tune")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning')

    parser.add_argument(
        '--model-type',
        type=str,
        default='bilstm4_attention',
        choices=['crnn', 'n', 'rnn_attention', 'crnn_attention',
                 'transformer', 'lstm3', 'lstm3_attention', 'bilstm4_attention'],
        help='Model type to tune (default: bilstm4_attention)'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials'
    )

    parser.add_argument(
        '--stocks',
        type=int,
        default=20,
        help='Number of stocks for small dataset'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=None,
        help='Number of years (default: ALL years)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed_hparam',
        help='Directory with small dataset'
    )

    parser.add_argument(
        '--create-dataset',
        action='store_true',
        help='Create small dataset before tuning'
    )

    parser.add_argument(
        '--pre-normalized-path',
        type=str,
        default='data/pre_normalized.parquet',
        help='Path to pre-normalized data for creating small dataset'
    )

    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50,
        help='Max epochs per trial'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for best hyperparameters (default: best_hyperparameters_{model_type}.json)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    return parser.parse_args()


def create_data_loaders(data_dir: str, config):
    """Create train and validation data loaders."""
    logger.info(f"Loading data from {data_dir}...")

    train_dataset = FinancialDataset(Path(data_dir) / 'train')
    val_dataset = FinancialDataset(Path(data_dir) / 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def main():
    """Main function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    # Create hyperparameter config
    hparam_config = HyperparameterSearchConfig()
    hparam_config.MODEL_TYPE = args.model_type
    hparam_config.N_TRIALS = args.n_trials
    hparam_config.HPARAM_STOCKS = args.stocks
    hparam_config.HPARAM_YEARS = args.years
    hparam_config.HPARAM_ALL_YEARS = (args.years is None)
    hparam_config.HPARAM_MAX_EPOCHS = args.max_epochs

    if args.output:
        hparam_config.BEST_PARAMS_PATH = args.output

    # Create small dataset if requested
    if args.create_dataset:
        logger.info("Creating small dataset...")

        import subprocess
        cmd = [
            sys.executable,
            'scripts/create_hparam_dataset.py',
            '--n-stocks', str(args.stocks),
            '--output-dir', args.data_dir
        ]

        if args.years:
            cmd.extend(['--years', str(args.years)])

        cmd.extend(['--pre-normalized-path', args.pre_normalized_path])

        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            logger.error("Failed to create small dataset")
            return 1

    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.data_dir, hparam_config)

    # Get dataset info
    train_dataset = train_loader.dataset
    num_features = train_dataset.num_features
    num_stocks = train_dataset.num_stocks
    num_groups = train_dataset.num_groups

    logger.info(f"Num features: {num_features}")
    logger.info(f"Num stocks: {num_stocks}")
    logger.info(f"Num groups: {num_groups}")

    # Create device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create optimizer
    optimizer = OptunaOptimizer(hparam_config=hparam_config)

    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    result = optimizer.optimize(
        train_loader=train_loader,
        val_loader=val_loader,
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
        device=device
    )

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best value: {result['best_value']:.6f}")
    logger.info(f"Best params:")
    for key, value in result['best_params'].items():
        logger.info(f"  {key}: {value}")

    # Print model-specific config snippet
    logger.info("=" * 60)
    logger.info("To use these hyperparameters, update your config:")
    logger.info(f"python scripts/train.py --model-type {args.model_type} \\")
    for key, value in result['best_params'].items():
        logger.info(f"  --{key} {value} \\")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
