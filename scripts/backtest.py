#!/usr/bin/env python
"""
Backtesting script for CRNN models.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import get_config_for_model
from src.data.dataset import FinancialDataset
from src.models import create_model
from src.evaluation import Backtester
from src.utils.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest CRNN model')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint or "best" for best model'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        choices=['crnn', 'rnn', 'rnn_attention', 'crnn_attention', 'transformer'],
        default='crnn_attention',
        help='Model type'
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
        default='test',
        help='Data split to backtest'
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
        default='outputs/backtest_report.xlsx',
        help='Output file for backtest report'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        choices=['excel', 'csv', 'json'],
        default='excel',
        help='Output format'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Prediction threshold for taking positions'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital for backtest'
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
    """Main backtesting function."""
    args = parse_args()

    logger = get_logger("backtest", log_dir="logs")

    logger.info("=" * 60)
    logger.info("BACKTESTING SCRIPT")
    logger.info("=" * 60)

    # Load config
    config = get_config_for_model(args.model_type)

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
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # Get embedding sizes
    embedding_sizes = dataset.get_embedding_sizes()

    # Create model
    logger.info(f"Creating {args.model_type} model...")

    model = create_model(
        model_type=args.model_type,
        num_features=dataset.num_features,
        num_stocks=embedding_sizes['num_stocks'],
        num_groups=embedding_sizes['num_groups'],
        config=config
    )

    # Load checkpoint
    checkpoint_path = args.model
    if checkpoint_path == 'best':
        checkpoint_path = Path(config.CHECKPOINT_DIR) / 'best_model.pth'

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")

    # Run backtest
    logger.info("Running backtest...")

    backtester = Backtester(model, config, device=args.device)

    results = backtester.run_backtest(
        loader,
        prediction_threshold=args.threshold,
        initial_capital=args.initial_capital
    )

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backtester.generate_report(results, str(output_path), format=args.output_format)

    logger.info(f"Backtest report saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
