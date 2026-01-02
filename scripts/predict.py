#!/usr/bin/env python
"""
Prediction script for CRNN models.

Makes predictions on new data.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import get_config_for_model
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.models import create_model
from src.utils.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with CRNN model')

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
        '--tickers',
        type=str,
        nargs='+',
        required=True,
        help='Ticker symbols to predict'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data (for feature info)'
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
        help='Output file for predictions (JSON)'
    )

    return parser.parse_args()


def download_latest_data(tickers: list, lookback_days: int = 500):
    """Download latest data for tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)  # Extra for weekends

    data = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False
    )

    return data


def main():
    """Main prediction function."""
    args = parse_args()

    logger = get_logger("predict", log_dir="logs")

    logger.info("=" * 60)
    logger.info("PREDICTION SCRIPT")
    logger.info("=" * 60)

    # Load config
    config = get_config_for_model(args.model_type)

    # Load feature info
    data_dir = Path(args.data_dir)
    info_path = data_dir / 'info.json'

    if not info_path.exists():
        logger.error("Feature info not found. Run preprocess_data.py first.")
        return 1

    with open(info_path, 'r') as f:
        info = json.load(f)

    sequence_length = info['sequence_length']
    feature_cols = info['feature_cols']
    num_features = info['num_features']

    # Load preprocessor for normalization
    # In practice, you'd want to save/load the preprocessor state
    preprocessor = DataPreprocessor(config)

    # Download data
    logger.info(f"Downloading data for {args.tickers}...")

    data = yf.download(
        args.tickers,
        period='2y',
        progress=False
    )

    # Create model
    logger.info(f"Creating {args.model_type} model...")

    # Note: We need to get actual stock/group encodings from the training data
    # For now, use defaults
    model = create_model(
        model_type=args.model_type,
        num_features=num_features,
        num_stocks=500,  # Default, will need to match training
        num_groups=20,
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
    model.eval()

    logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")

    logger.info("=" * 60)
    logger.info("NOTE: This is a simplified prediction script.")
    logger.info("For production use, you need to:")
    logger.info("1. Save/load the preprocessor state (scalers, encoders)")
    logger.info("2. Save/load feature engineering state")
    logger.info("3. Handle missing features correctly")
    logger.info("4. Match stock/group IDs from training data")
    logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
