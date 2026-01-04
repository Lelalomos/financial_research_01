#!/usr/bin/env python
"""
Create small balanced dataset for hyperparameter tuning.

This script creates a small, balanced dataset with:
- ALL group_ids represented (no group is missed)
- Randomly sampled stocks within each group
- ALL years of data by default (can limit with --years)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.data_config import DataConfig
from src.utils.logger import get_logger

logger = get_logger("create_hparam_dataset")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create small dataset for hyperparameter tuning')

    parser.add_argument(
        '--n-stocks',
        type=int,
        default=20,
        help='Total number of stocks to sample (default: 20)'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=None,
        help='Number of years to use (default: ALL years)'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed',
        help='Input directory with processed data'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed_hparam',
        help='Output directory for small dataset'
    )

    parser.add_argument(
        '--pre-normalized-path',
        type=str,
        default='data/pre_normalized.parquet',
        help='Path to pre-normalized data'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def load_data(input_dir: str, pre_normalized_path: str) -> pd.DataFrame:
    """
    Load data from processed directory or pre-normalized file.

    Args:
        input_dir: Directory with processed data
        pre_normalized_path: Path to pre-normalized parquet file

    Returns:
        DataFrame with all data
    """
    # Try pre-normalized first
    pre_norm_path = Path(pre_normalized_path)
    if pre_norm_path.exists():
        logger.info(f"Loading data from {pre_norm_path}")
        df = pd.read_parquet(pre_norm_path)
        return df

    # Try loading from processed directory
    input_path = Path(input_dir)
    if input_path.exists():
        # Load and combine train/val/test data
        dfs = []
        for split in ['train', 'val', 'test']:
            split_path = input_path / split
            if split_path.exists():
                # Load sequences
                import numpy as np
                features_path = split_path / 'features.npy'
                if features_path.exists():
                    features = np.load(features_path)
                    # This is sequence data, need original source
                    logger.warning("Sequence data found, but need original data for sampling")
                    logger.info("Please use pre-normalized data or run with --pre-normalized-path")
                    raise ValueError("Cannot sample from sequence data. Use pre-normalized data.")

    raise FileNotFoundError(f"No data found at {input_dir} or {pre_normalized_path}")


def sample_stocks_by_group(
    df: pd.DataFrame,
    n_stocks: int,
    seed: int
) -> list:
    """
    Sample stocks ensuring ALL groups are represented.

    Args:
        df: DataFrame with 'tic' and 'group_id' columns
        n_stocks: Total number of stocks to sample
        seed: Random seed

    Returns:
        List of selected stock tickers
    """
    np.random.seed(seed)

    # Get unique groups and stocks per group
    unique_groups = df['group_id'].unique()
    n_groups = len(unique_groups)

    logger.info(f"Found {n_groups} unique groups in dataset")
    logger.info(f"Groups: {sorted(unique_groups)}")

    # Calculate stocks per group
    stocks_per_group = max(1, n_stocks // n_groups)
    remaining = n_stocks - (stocks_per_group * n_groups)

    logger.info(f"Sampling {stocks_per_group} stocks per group + {remaining} extra")

    selected_stocks = []

    # Sample stocks from each group
    for group_id in sorted(unique_groups):
        group_stocks = df[df['group_id'] == group_id]['tic'].unique().tolist()

        if len(group_stocks) == 0:
            logger.warning(f"Group {group_id} has no stocks, skipping")
            continue

        # Sample from this group
        n_sample = min(stocks_per_group, len(group_stocks))
        sampled = np.random.choice(group_stocks, size=n_sample, replace=False).tolist()
        selected_stocks.extend(sampled)

        logger.info(f"  Group {group_id}: sampled {n_sample} stocks from {len(group_stocks)} available")

    # Add remaining stocks randomly from any group
    if remaining > 0 and len(selected_stocks) < n_stocks:
        available_stocks = [s for s in df['tic'].unique() if s not in selected_stocks]
        if len(available_stocks) > 0:
            extra = np.random.choice(available_stocks, size=min(remaining, len(available_stocks)), replace=False).tolist()
            selected_stocks.extend(extra)

    # Ensure we don't exceed requested count
    selected_stocks = list(set(selected_stocks))[:n_stocks]

    logger.info(f"Total selected stocks: {len(selected_stocks)}")
    logger.info(f"Selected stocks: {sorted(selected_stocks)}")

    # Verify all groups are represented
    final_groups = df[df['tic'].isin(selected_stocks)]['group_id'].unique()
    logger.info(f"Groups represented in sample: {sorted(final_groups)}")

    return selected_stocks


def filter_by_date_range(
    df: pd.DataFrame,
    years: int = None,
    all_years: bool = True
) -> pd.DataFrame:
    """
    Filter data by date range.

    Args:
        df: Input DataFrame
        years: Number of years to keep (None = all)
        all_years: If True, use all years

    Returns:
        Filtered DataFrame
    """
    if all_years or years is None:
        logger.info("Using ALL years of data")
        return df

    # Calculate cutoff date
    max_date = pd.to_datetime(df['date'].max())
    cutoff_date = max_date - timedelta(days=years * 365)

    logger.info(f"Filtering data from {cutoff_date.date()} to {max_date.date()} ({years} years)")

    df_filtered = df[pd.to_datetime(df['date']) >= cutoff_date].copy()

    logger.info(f"Filtered data: {len(df_filtered)} rows (from {len(df)} rows)")

    return df_filtered


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    prediction_horizon: int,
    output_dir: Path
) -> None:
    """
    Create train/val/test sequences from filtered data.

    Args:
        df: Filtered DataFrame
        sequence_length: Lookback window size
        prediction_horizon: Prediction horizon
        output_dir: Output directory
    """
    logger.info("Creating sequences...")

    # Sort by date
    df = df.sort_values(['tic', 'date']).reset_index(drop=True)

    # Get unique stocks and groups
    unique_stocks = df['tic'].unique()
    unique_groups = df['group_id'].unique()

    # Create encoders
    stock_encoder = LabelEncoder()
    group_encoder = LabelEncoder()

    stock_encoder.fit(unique_stocks)
    group_encoder.fit(unique_groups)

    df['stock_id'] = stock_encoder.transform(df['tic'])
    df['group_id_encoded'] = group_encoder.transform(df['group_id'])

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in [
        'tic', 'date', 'group_id', 'target', 'stock_id', 'group_id_encoded'
    ]]

    logger.info(f"Feature columns: {len(feature_cols)}")

    # Time-based split
    dates = pd.to_datetime(df['date'])
    unique_dates = dates.unique()

    n_total = len(unique_dates)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.20)

    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train:n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]

    logger.info(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    logger.info(f"Val dates: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")
    logger.info(f"Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Create sequences for each split
    for split_name, split_dates in [('train', train_dates), ('val', val_dates), ('test', test_dates)]:
        split_df = df[dates.isin(split_dates)].copy()

        sequences = []
        targets = []
        stock_ids = []
        group_ids = []
        days = []
        months = []
        dividend_flags = []

        for stock_id in range(len(unique_stocks)):
            stock_data = split_df[split_df['stock_id'] == stock_id].sort_values('date')

            if len(stock_data) < sequence_length + prediction_horizon:
                continue

            features = stock_data[feature_cols].values

            # Create sliding windows
            for i in range(len(features) - sequence_length - prediction_horizon + 1):
                sequences.append(features[i:i + sequence_length])
                targets.append(features[i + sequence_length + prediction_horizon - 1, 0])  # Use close as target
                stock_ids.append(stock_id)
                group_ids.append(stock_data['group_id_encoded'].iloc[i + sequence_length])
                days.append(stock_data['day'].iloc[i + sequence_length])
                months.append(stock_data['month'].iloc[i + sequence_length])
                dividend_flags.append(stock_data.get('dividend_flag', 1).iloc[i + sequence_length])

        # Save sequences
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        import numpy as np

        np.save(split_dir / 'features.npy', np.array(sequences))
        np.save(split_dir / 'target.npy', np.array(targets))
        np.save(split_dir / 'stock_id.npy', np.array(stock_ids))
        np.save(split_dir / 'group_id.npy', np.array(group_ids))
        np.save(split_dir / 'day.npy', np.array(days))
        np.save(split_dir / 'month.npy', np.array(months))
        np.save(split_dir / 'dividend_flag.npy', np.array(dividend_flags))

        logger.info(f"{split_name}: {len(sequences)} sequences")

    # Save metadata
    metadata = {
        'num_stocks': len(unique_stocks),
        'num_groups': len(unique_groups),
        'num_features': len(feature_cols),
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'stocks': unique_stocks.tolist(),
        'groups': unique_groups.tolist(),
        'feature_columns': feature_cols,
        'created_at': datetime.now().isoformat()
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")


def main():
    """Main function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CREATING HPARAM DATASET")
    logger.info("=" * 60)

    # Load data
    df = load_data(args.input_dir, args.pre_normalized_path)
    logger.info(f"Loaded data: {len(df)} rows")

    # Check required columns
    required_cols = ['tic', 'date', 'group_id']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        # Check for alternative names
        if 'group' in df.columns and 'group_id' not in df.columns:
            df['group_id'] = df['group']
        else:
            raise ValueError(f"Missing required columns: {missing}")

    # Add day/month if not present
    if 'day' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month

    # Sample stocks ensuring ALL groups are represented
    selected_stocks = sample_stocks_by_group(df, args.n_stocks, args.seed)
    df_filtered = df[df['tic'].isin(selected_stocks)].copy()

    # Filter by date range
    df_filtered = filter_by_date_range(df_filtered, args.years, all_years=(args.years is None))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sequences
    create_sequences(
        df=df_filtered,
        sequence_length=30,
        prediction_horizon=5,
        output_dir=output_dir
    )

    logger.info("=" * 60)
    logger.info("DONE!")
    logger.info(f"Dataset created at {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
