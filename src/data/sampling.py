"""
Stock sampling utilities for creating balanced datasets.

This module provides functions to sample stocks ensuring all groups are
represented and balanced across group_ids.

Typical usage:
    from src.data.sampling import sample_stocks_by_group

    # Sample 50 stocks balanced across all groups
    selected = sample_stocks_by_group(df, n_stocks=50, seed=42)
    df_sampled = df[df['tic'].isin(selected)]
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def sample_stocks_by_group(
    df: pd.DataFrame,
    n_stocks: int,
    seed: int = 42,
    group_col: str = 'group_id',
    tic_col: str = 'tic'
) -> List[str]:
    """
    Sample stocks ensuring ALL groups are represented and balanced.

    Strategy:
    1. Get ALL unique group_ids from the dataset
    2. Calculate stocks_per_group = n_stocks // n_groups
    3. For EACH group_id, randomly sample stocks_per_group stocks
    4. Add remaining stocks (n_stocks % n_groups) randomly
    5. Return list of selected stock tickers

    Edge case: If n_stocks < n_groups, randomly select n_stocks groups
    and sample 1 stock from each.

    Args:
        df: DataFrame with stock data containing group_col and tic_col
        n_stocks: Total number of stocks to sample
        seed: Random seed for reproducibility (default: 42)
        group_col: Column name for group_id (default: 'group_id')
        tic_col: Column name for ticker symbol (default: 'tic')

    Returns:
        List of selected stock tickers

    Raises:
        ValueError: If df is empty or required columns are missing

    Examples:
        >>> # If 10 groups exist and n_stocks=50:
        >>> # 50 // 10 = 5 stocks per group (minimum)
        >>> # 50 % 10 = 0 extra stocks
        >>> # Result: 5 stocks from each of 10 groups = 50 total
        >>> selected = sample_stocks_by_group(df, n_stocks=50)

        >>> # If 10 groups exist and n_stocks=55:
        >>> # 55 // 10 = 5 stocks per group (minimum)
        >>> # 55 % 10 = 5 extra stocks
        >>> # Result: 5 stocks from each group + 1 extra for 5 groups = 55 total
        >>> selected = sample_stocks_by_group(df, n_stocks=55)
    """
    # Validate inputs
    if df.empty:
        raise ValueError("DataFrame is empty")

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")

    if tic_col not in df.columns:
        raise ValueError(f"Column '{tic_col}' not found in DataFrame")

    if n_stocks <= 0:
        raise ValueError(f"n_stocks must be positive, got {n_stocks}")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Get unique groups
    unique_groups = sorted(df[group_col].unique())
    n_groups = len(unique_groups)

    if n_groups == 0:
        raise ValueError(f"No groups found in column '{group_col}'")

    # Edge case: more groups than stocks
    # Randomly select n_stocks groups, sample 1 from each
    if n_stocks < n_groups:
        selected_groups = np.random.choice(unique_groups, size=n_stocks, replace=False)
        selected_stocks = []
        for group_id in selected_groups:
            group_stocks = df[df[group_col] == group_id][tic_col].unique().tolist()
            if len(group_stocks) == 0:
                continue
            selected_stocks.append(np.random.choice(group_stocks))
        return selected_stocks

    # Calculate base allocation
    stocks_per_group = max(1, n_stocks // n_groups)
    remaining = n_stocks - (stocks_per_group * n_groups)

    selected_stocks = []

    # Sample from each group
    for group_id in unique_groups:
        group_stocks = df[df[group_col] == group_id][tic_col].unique().tolist()

        if len(group_stocks) == 0:
            continue

        # Sample from this group
        n_sample = min(stocks_per_group, len(group_stocks))
        sampled = np.random.choice(group_stocks, size=n_sample, replace=False).tolist()
        selected_stocks.extend(sampled)

    # Add remaining stocks randomly from any group
    if remaining > 0:
        available = [s for s in df[tic_col].unique() if s not in selected_stocks]
        if len(available) > 0:
            extra = np.random.choice(
                available,
                size=min(remaining, len(available)),
                replace=False
            ).tolist()
            selected_stocks.extend(extra)

    # Remove duplicates and limit to n_stocks
    selected_stocks = list(set(selected_stocks))[:n_stocks]

    return selected_stocks


def get_sampling_stats(
    df: pd.DataFrame,
    selected_stocks: List[str],
    group_col: str = 'group_id',
    tic_col: str = 'tic'
) -> dict:
    """
    Get statistics about the sampling result.

    Args:
        df: Original DataFrame
        selected_stocks: List of selected stock tickers
        group_col: Column name for group_id (default: 'group_id')
        tic_col: Column name for ticker symbol (default: 'tic')

    Returns:
        Dictionary with sampling statistics
    """
    selected_df = df[df[tic_col].isin(selected_stocks)]

    stats = {
        'total_selected': len(selected_stocks),
        'total_groups': selected_df[group_col].nunique(),
        'groups': sorted(selected_df[group_col].unique().tolist()),
        'stocks_per_group': {}
    }

    for group_id in sorted(selected_df[group_col].unique()):
        group_stocks = selected_df[selected_df[group_col] == group_id][tic_col].nunique()
        stats['stocks_per_group'][int(group_id)] = group_stocks

    return stats
