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
from src.utils.logger import get_logger
from src.training import find_checkpoint_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test CRNN model')

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
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
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

    return parser.parse_args()


def load_sequences(data_dir: Path, split: str):
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

    logger.info("=" * 60)
    logger.info("TESTING SCRIPT")
    logger.info("=" * 60)

    # Load config
    config = load_config('model')

    # Load data
    data_dir = Path(args.data_dir)
    raw_data_dir = Path(args.raw_data_dir) if args.raw_data_dir else None

    logger.info(f"Loading {args.split} data from {data_dir}...")

    sequences = load_sequences(data_dir, args.split)

    if sequences is None:
        logger.error(f"No {args.split} data found")
        return 1

    logger.info(f"Loaded {len(sequences['target'])} samples")

    # Load ID mappings for ticker names and sectors
    stock_id_to_ticker, group_id_to_sector = load_id_mappings(data_dir, raw_data_dir)

    if stock_id_to_ticker:
        logger.info(f"Loaded {len(stock_id_to_ticker)} ticker mappings")
    else:
        logger.warning("No ticker mappings available, will use stock IDs")

    if group_id_to_sector:
        logger.info(f"Loaded {len(group_id_to_sector)} sector mappings")
    else:
        logger.warning("No sector mappings available, will use group IDs")

    # Load info
    info_path = data_dir / 'info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Create dataset
    dataset = FinancialDataset(sequences, config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.model.training.BATCH_SIZE,
        shuffle=False,
        num_workers=config.model.device.NUM_WORKERS
    )

    # Get embedding sizes
    embedding_sizes = dataset.get_embedding_sizes()

    # Auto-detect model type from checkpoint filename if not specified
    model_type = args.model_type
    checkpoint_path = find_checkpoint_path(
        model_input=args.model,
        checkpoint_dir=config.model.checkpointing.CHECKPOINT_DIR,
        model_type=model_type
    )

    # If model_type is still default and checkpoint path contains a known model type
    if model_type == 'crnn_attention' and checkpoint_path:
        import os
        basename = os.path.basename(checkpoint_path)
        # Must check longer types first to avoid substring false matches
        for known_type in ['bilstm4_attention', 'crnn_attention', 'rnn_attention',
                           'lstm3_attention', 'transformer', 'crnn', 'rnn', 'lstm3']:
            if known_type in basename.lower():
                model_type = known_type
                logger.info(f"Auto-detected model type: {model_type} from checkpoint filename")
                break

    # Create model
    logger.info(f"Creating {model_type} model...")

    model = create_model(
        model_type=model_type,
        num_features=dataset.num_features,
        num_stocks=embedding_sizes['num_stocks'],
        num_groups=embedding_sizes['num_groups'],
        config=config
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate with report if excel-report is specified
    if args.excel_report:
        logger.info("Evaluating model with detailed report...")

        metrics, report_df, sector_stats = evaluate_model_with_report(
            model,
            loader,
            device=args.device,
            stock_id_to_ticker=stock_id_to_ticker if stock_id_to_ticker else None,
            group_id_to_sector=group_id_to_sector if group_id_to_sector else None,
            output_path=args.excel_report
        )

        print_metrics(metrics, prefix=f"{args.split.upper()} - ")
        print_sector_stats(sector_stats)

        logger.info(f"Excel report saved to {args.excel_report}")
    else:
        # Standard evaluation
        logger.info("Evaluating model...")

        metrics = evaluate_model(model, loader, device=args.device)

        print_metrics(metrics, prefix=f"{args.split.upper()} - ")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
