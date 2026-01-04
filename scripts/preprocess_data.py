#!/usr/bin/env python
"""
Data preprocessing script.

Downloads data, calculates features, normalizes, and creates train/val/test splits.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.downloader import DataDownloader
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.utils.logger import get_logger


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

    return parser.parse_args()


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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download data
    if not args.skip_download:
        logger.info("Step 1: Downloading data...")
        downloader = DataDownloader(config)

        # Try to load existing data first
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

    # Sample stocks if requested (BEFORE feature engineering for performance)
    if args.stocks:
        logger.info(f"Sampling {args.stocks} stocks balanced across all group_ids...")
        from src.data.sampling import sample_stocks_by_group, get_sampling_stats

        # Ensure group_id column exists
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

    # Step 2: Feature engineering
    logger.info("Step 2: Engineering features...")
    engineer = FeatureEngineer(config)

    feature_df = engineer.add_all_features(
        stock_df,
        vix_df=vix_df,
        commodities_df=commodities_df,
        treasury_df=treasury_df,
        calculate_target=True
    )

    # Log feature info
    feature_info = engineer.get_feature_info(feature_df)
    logger.info(f"Feature engineering complete:")
    logger.info(f"  Total features: {feature_info['total_features']}")
    logger.info(f"  Price features: {feature_info['price_features']}")
    logger.info(f"  EMA features: {feature_info['ema_features']}")
    logger.info(f"  RSI features: {feature_info['rsi_features']}")
    logger.info(f"  Candlestick patterns: {feature_info['candlestick_patterns']}")

    # Step 3: Preprocessing
    logger.info("Step 3: Preprocessing data...")
    preprocessor = DataPreprocessor(config)

    processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(
        feature_df,
        fit=True,
        export_pre_normalize=args.export_pre_normalize,
        export_normalized=args.export_normalized
    )

    # Step 4: Save processed data
    logger.info("Step 4: Saving processed data...")

    import numpy as np

    # Save sequences
    for split_name in ['train', 'val', 'test']:
        if split_name in sequences:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for key, data in sequences[split_name].items():
                np.save(split_dir / f'{key}.npy', data)

            logger.info(f"Saved {split_name} sequences: {len(sequences[split_name]['target'])} samples")

    # Save info
    import json
    info_path = output_dir / 'info.json'
    with open(info_path, 'w') as f:
        json.dump({
            'num_stocks': info['num_stocks'],
            'num_groups': info['num_groups'],
            'num_features': info['num_features'],
            'sequence_length': info['sequence_length'],
            'prediction_horizon': info['prediction_horizon'],
            'feature_cols': info['feature_cols'],
        }, f, indent=2)

    logger.info(f"Saved info to {info_path}")

    # Save feature columns
    feature_cols_path = output_dir / 'feature_columns.txt'
    with open(feature_cols_path, 'w') as f:
        f.write('\n'.join(info['feature_cols']))

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  Train samples: {len(sequences.get('train', {}).get('target', []))}")
    logger.info(f"  Val samples: {len(sequences.get('val', {}).get('target', []))}")
    logger.info(f"  Test samples: {len(sequences.get('test', {}).get('target', []))}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
