"""
Time-series validation utilities.

Financial samples can overlap through sequence windows and prediction horizons.
These helpers create chronological folds and optional purge gaps so validation
periods are separated from training observations.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesFold:
    """Indices and boundaries for one chronological validation fold."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def _sorted_unique_dates(df: pd.DataFrame, date_col: str) -> np.ndarray:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    dates = pd.to_datetime(df[date_col])
    unique_dates = np.array(sorted(dates.dropna().unique()))
    if len(unique_dates) == 0:
        raise ValueError("No valid dates available for time-series split")
    return unique_dates


def walk_forward_split(
    df: pd.DataFrame,
    train_window: int,
    val_window: int,
    step: Optional[int] = None,
    purge_gap: int = 0,
    date_col: str = 'date'
) -> Iterator[TimeSeriesFold]:
    """
    Yield rolling walk-forward folds by date.

    Args:
        df: DataFrame containing a date column.
        train_window: Number of unique dates in each training window.
        val_window: Number of unique dates in each validation window.
        step: Number of unique dates to advance per fold. Defaults to val_window.
        purge_gap: Number of unique dates between train and validation windows.
        date_col: Date column name.
    """
    if train_window <= 0 or val_window <= 0:
        raise ValueError("train_window and val_window must be positive")
    if purge_gap < 0:
        raise ValueError("purge_gap must be non-negative")
    if step is None:
        step = val_window
    if step <= 0:
        raise ValueError("step must be positive")

    dates = _sorted_unique_dates(df, date_col)
    date_series = pd.to_datetime(df[date_col])
    total_window = train_window + purge_gap + val_window

    for start in range(0, len(dates) - total_window + 1, step):
        train_dates = dates[start:start + train_window]
        val_start_idx = start + train_window + purge_gap
        val_dates = dates[val_start_idx:val_start_idx + val_window]

        train_mask = date_series.isin(train_dates)
        val_mask = date_series.isin(val_dates)

        yield TimeSeriesFold(
            train_indices=df.index[train_mask].to_numpy(),
            val_indices=df.index[val_mask].to_numpy(),
            train_start=pd.Timestamp(train_dates[0]),
            train_end=pd.Timestamp(train_dates[-1]),
            val_start=pd.Timestamp(val_dates[0]),
            val_end=pd.Timestamp(val_dates[-1]),
        )


def purged_time_series_split(
    df: pd.DataFrame,
    n_splits: int,
    purge_gap: int,
    min_train_window: Optional[int] = None,
    date_col: str = 'date'
) -> Iterator[TimeSeriesFold]:
    """
    Yield expanding-window folds with a purge gap before validation.

    Args:
        df: DataFrame containing a date column.
        n_splits: Number of validation folds.
        purge_gap: Number of unique dates to exclude between train and val.
        min_train_window: Minimum number of dates in the first train window.
        date_col: Date column name.
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if purge_gap < 0:
        raise ValueError("purge_gap must be non-negative")

    dates = _sorted_unique_dates(df, date_col)
    if min_train_window is None:
        min_train_window = max(1, len(dates) // (n_splits + 1))
    if min_train_window <= 0:
        raise ValueError("min_train_window must be positive")

    remaining_dates = len(dates) - min_train_window - purge_gap
    if remaining_dates < n_splits:
        raise ValueError("Not enough dates for requested purged split configuration")

    val_window = remaining_dates // n_splits
    if val_window <= 0:
        raise ValueError("Validation window would be empty")

    date_series = pd.to_datetime(df[date_col])

    for split_idx in range(n_splits):
        val_start_idx = min_train_window + purge_gap + split_idx * val_window
        val_end_idx = val_start_idx + val_window
        if split_idx == n_splits - 1:
            val_end_idx = len(dates)

        train_dates = dates[:val_start_idx - purge_gap]
        val_dates = dates[val_start_idx:val_end_idx]
        if len(train_dates) == 0 or len(val_dates) == 0:
            continue

        train_mask = date_series.isin(train_dates)
        val_mask = date_series.isin(val_dates)

        yield TimeSeriesFold(
            train_indices=df.index[train_mask].to_numpy(),
            val_indices=df.index[val_mask].to_numpy(),
            train_start=pd.Timestamp(train_dates[0]),
            train_end=pd.Timestamp(train_dates[-1]),
            val_start=pd.Timestamp(val_dates[0]),
            val_end=pd.Timestamp(val_dates[-1]),
        )
