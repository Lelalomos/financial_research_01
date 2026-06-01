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

from typing import Dict, List, Optional
from pathlib import Path
import json
import pandas as pd
import numpy as np


def _safe_market_cap(value) -> Optional[float]:
    """Convert a raw market-cap value to float when possible."""
    try:
        if value is None:
            return None
        market_cap = float(value)
        if np.isfinite(market_cap):
            return market_cap
    except (TypeError, ValueError):
        return None
    return None


def load_market_caps_for_tickers(
    tickers: List[str],
    metadata_dir: Optional[str],
) -> Dict[str, Optional[float]]:
    """Load market capitalization values from local ticker metadata files."""
    if not metadata_dir:
        return {ticker: None for ticker in tickers}

    base_dir = Path(metadata_dir)
    market_caps: Dict[str, Optional[float]] = {}

    for ticker in tickers:
        metadata_path = base_dir / f"{ticker}.json"
        market_caps[ticker] = None
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        highlights = payload.get("Highlights", {})
        market_caps[ticker] = (
            _safe_market_cap(highlights.get("MarketCapitalization"))
            or _safe_market_cap(highlights.get("MarketCap"))
            or _safe_market_cap(payload.get("General", {}).get("MarketCap"))
        )

    return market_caps


def _unique_preserve_order(values: List[str]) -> List[str]:
    """Remove duplicates while keeping the first occurrence."""
    return list(dict.fromkeys(values))


def _sorted_group_stocks(
    group_df: pd.DataFrame,
    tic_col: str,
    market_caps: Dict[str, Optional[float]],
) -> List[str]:
    """Return group tickers sorted by descending market cap and ticker name."""
    tickers = _unique_preserve_order(group_df[tic_col].tolist())
    return sorted(
        tickers,
        key=lambda ticker: (
            market_caps.get(ticker) is None,
            -(market_caps.get(ticker) or 0.0),
            ticker,
        ),
    )


def sample_stocks_by_group(
    df: pd.DataFrame,
    n_stocks: int,
    seed: int = 42,
    group_col: str = 'group_id',
    tic_col: str = 'tic',
    selection_mode: str = 'random',
    market_cap_metadata_dir: Optional[str] = None,
) -> List[str]:
    """
    Sample stocks ensuring ALL groups are represented and balanced.

    Strategy:
    1. Get ALL unique group_ids from the dataset
    2. Calculate stocks_per_group = n_stocks // n_groups
    3. For EACH group_id, either randomly sample stocks_per_group stocks or
       take the top stocks by market cap
    4. Add remaining stocks using the same selection mode
    5. Return list of selected stock tickers

    Edge case: If n_stocks < n_groups, choose n_stocks groups and sample
    1 stock from each using the selected mode.

    Args:
        df: DataFrame with stock data containing group_col and tic_col
        n_stocks: Total number of stocks to sample
        seed: Random seed for reproducibility (default: 42)
        group_col: Column name for group_id (default: 'group_id')
        tic_col: Column name for ticker symbol (default: 'tic')
        selection_mode: Sampling mode, either 'random' or 'sorted'
        market_cap_metadata_dir: Directory containing per-ticker JSON metadata
            with market-cap fields. Used only when selection_mode='sorted'.

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

    if selection_mode not in {'random', 'sorted'}:
        raise ValueError(
            f"selection_mode must be 'random' or 'sorted', got {selection_mode!r}"
        )

    rng = np.random.default_rng(seed)

    # Get unique groups
    unique_groups = sorted(df[group_col].unique())
    n_groups = len(unique_groups)

    if n_groups == 0:
        raise ValueError(f"No groups found in column '{group_col}'")

    unique_tickers = sorted(df[tic_col].unique().tolist())
    market_caps = load_market_caps_for_tickers(unique_tickers, market_cap_metadata_dir)
    group_to_stocks = {
        group_id: _sorted_group_stocks(df[df[group_col] == group_id], tic_col, market_caps)
        for group_id in unique_groups
    }

    # Edge case: more groups than stocks
    if n_stocks < n_groups:
        if selection_mode == 'random':
            selected_groups = rng.choice(unique_groups, size=n_stocks, replace=False).tolist()
        else:
            selected_groups = sorted(
                unique_groups,
                key=lambda group_id: (
                    group_to_stocks[group_id][0] if group_to_stocks[group_id] else "",
                ),
            )
            selected_groups = sorted(
                selected_groups,
                key=lambda group_id: (
                    market_caps.get(group_to_stocks[group_id][0]) is None if group_to_stocks[group_id] else True,
                    -(market_caps.get(group_to_stocks[group_id][0]) or 0.0) if group_to_stocks[group_id] else 0.0,
                    str(group_id),
                ),
            )[:n_stocks]

        selected_stocks = []
        for group_id in selected_groups:
            group_stocks = group_to_stocks[group_id]
            if len(group_stocks) == 0:
                continue
            if selection_mode == 'random':
                selected_stocks.append(rng.choice(group_stocks).item())
            else:
                selected_stocks.append(group_stocks[0])
        return selected_stocks

    # Calculate base allocation
    stocks_per_group = max(1, n_stocks // n_groups)
    remaining = n_stocks - (stocks_per_group * n_groups)

    selected_stocks = []

    # Sample from each group
    for group_id in unique_groups:
        group_stocks = group_to_stocks[group_id]

        if len(group_stocks) == 0:
            continue

        # Sample from this group
        n_sample = min(stocks_per_group, len(group_stocks))
        if selection_mode == 'random':
            sampled = rng.choice(group_stocks, size=n_sample, replace=False).tolist()
        else:
            sampled = group_stocks[:n_sample]
        selected_stocks.extend(sampled)

    # Add remaining stocks using the selected mode
    if remaining > 0:
        available = [ticker for ticker in unique_tickers if ticker not in selected_stocks]
        if len(available) > 0:
            if selection_mode == 'random':
                extra = rng.choice(
                    available,
                    size=min(remaining, len(available)),
                    replace=False
                ).tolist()
            else:
                extra = sorted(
                    available,
                    key=lambda ticker: (
                        market_caps.get(ticker) is None,
                        -(market_caps.get(ticker) or 0.0),
                        ticker,
                    ),
                )[:min(remaining, len(available))]
            selected_stocks.extend(extra)

    # Remove duplicates while keeping order and limit to n_stocks
    selected_stocks = _unique_preserve_order(selected_stocks)[:n_stocks]

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
        try:
            stats['stocks_per_group'][int(group_id)] = group_stocks
        except (TypeError, ValueError):
            stats['stocks_per_group'][str(group_id)] = group_stocks

    return stats
