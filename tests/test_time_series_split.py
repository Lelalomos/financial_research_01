"""
Unit tests for walk-forward and purged time-series split utilities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.time_series_split import purged_time_series_split, walk_forward_split


def _sample_frame(n_dates: int = 20, n_tickers: int = 2) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    for date in dates:
        for ticker_idx in range(n_tickers):
            rows.append({"date": date, "tic": f"T{ticker_idx}", "value": np.random.randn()})
    return pd.DataFrame(rows)


def test_walk_forward_split_respects_windows_and_purge_gap():
    df = _sample_frame(n_dates=12, n_tickers=2)
    folds = list(walk_forward_split(df, train_window=4, val_window=2, step=2, purge_gap=1))

    assert len(folds) == 3

    first = folds[0]
    train_dates = set(df.loc[first.train_indices, "date"])
    val_dates = set(df.loc[first.val_indices, "date"])

    assert len(train_dates) == 4
    assert len(val_dates) == 2
    assert max(train_dates) < min(val_dates)
    assert (min(val_dates) - max(train_dates)).days == 2


def test_purged_time_series_split_has_no_overlap_and_gap():
    df = _sample_frame(n_dates=18, n_tickers=3)
    folds = list(purged_time_series_split(df, n_splits=3, purge_gap=2, min_train_window=6))

    assert len(folds) == 3
    for fold in folds:
        train_dates = set(df.loc[fold.train_indices, "date"])
        val_dates = set(df.loc[fold.val_indices, "date"])

        assert train_dates.isdisjoint(val_dates)
        assert max(train_dates) < min(val_dates)
        assert (min(val_dates) - max(train_dates)).days >= 3


def test_time_series_split_rejects_invalid_parameters():
    df = _sample_frame(n_dates=5)

    with pytest.raises(ValueError):
        list(walk_forward_split(df, train_window=0, val_window=1))

    with pytest.raises(ValueError):
        list(purged_time_series_split(df, n_splits=10, purge_gap=2, min_train_window=4))
