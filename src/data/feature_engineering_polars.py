"""
Optional Polars implementations for isolated feature engineering hotspots.

These helpers are opt-in and keep pandas as the default pipeline path.
"""

from time import perf_counter
from typing import Dict, Optional

import pandas as pd


FIBONACCI_COLUMNS = [
    "swing_high",
    "swing_low",
    "fib_range",
    "fib_38",
    "fib_50",
    "fib_61",
    "dist_fib_38",
    "dist_fib_50",
    "dist_fib_61",
    "break_fib_61",
]

TIME_FEATURE_COLUMNS = ["day", "month", "dayofweek"]


class PolarsFeatureEngineeringError(RuntimeError):
    """Raised when optional Polars feature engineering cannot run."""


def _require_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise PolarsFeatureEngineeringError(
            "Polars is required for the opt-in Polars feature engineering path. "
            "Install it with `pip install polars`, or disable "
            "the selected data.features.FEATURE_FLAGS.polars_* option."
        ) from exc
    return pl


def add_time_features_polars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date-derived time features using Polars.

    Returns a pandas DataFrame to preserve existing pipeline contracts.
    """
    if "date" not in df.columns:
        raise ValueError("Missing required column for time features: date")

    pl = _require_polars()
    row_id = "__row_id"

    result = (
        pl.from_pandas(df.reset_index(drop=True))
        .with_row_index(row_id)
        .with_columns([
            pl.col("date").cast(pl.Datetime, strict=False).alias("date"),
        ])
        .with_columns([
            pl.col("date").dt.day().cast(pl.Int64).alias("day"),
            pl.col("date").dt.month().cast(pl.Int64).alias("month"),
            (pl.col("date").dt.weekday() - 1).cast(pl.Int64).alias("dayofweek"),
        ])
        .sort(row_id)
        .drop(row_id)
    )
    return _to_pandas_with_datetime(result, ["date"])


def add_fibonacci_features_polars(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add Fibonacci retracement features using Polars grouped rolling operations.

    Args:
        df: pandas DataFrame with `tic`, `date`, `high`, `low`, and `close`.
        window: Rolling window size.

    Returns:
        pandas DataFrame with the same row order as input plus Fibonacci columns.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    required_columns = {"tic", "date", "high", "low", "close"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for Fibonacci features: {sorted(missing)}")

    pl = _require_polars()
    row_id = "__row_id"

    polars_df = pl.from_pandas(df.reset_index(drop=True)).with_row_index(row_id)
    result = (
        polars_df
        .sort(["tic", "date"])
        .with_columns([
            pl.col("high").rolling_max(window_size=window).over("tic").alias("swing_high"),
            pl.col("low").rolling_min(window_size=window).over("tic").alias("swing_low"),
        ])
        .with_columns([
            (pl.col("swing_high") - pl.col("swing_low")).alias("fib_range"),
        ])
        .with_columns([
            (pl.col("swing_high") - 0.382 * pl.col("fib_range")).alias("fib_38"),
            (pl.col("swing_high") - 0.5 * pl.col("fib_range")).alias("fib_50"),
            (pl.col("swing_high") - 0.618 * pl.col("fib_range")).alias("fib_61"),
        ])
        .with_columns([
            ((pl.col("close") - pl.col("fib_38")) / pl.col("fib_range")).alias("dist_fib_38"),
            ((pl.col("close") - pl.col("fib_50")) / pl.col("fib_range")).alias("dist_fib_50"),
            ((pl.col("close") - pl.col("fib_61")) / pl.col("fib_range")).alias("dist_fib_61"),
            (pl.col("close") < pl.col("fib_61")).fill_null(False).cast(pl.Int64).alias("break_fib_61"),
        ])
        .sort(row_id)
        .drop(row_id)
    )

    return _to_pandas_with_datetime(result, ["date"])


def merge_external_data_polars(
    stock_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
    commodities_df: Optional[pd.DataFrame] = None,
    treasury_df: Optional[pd.DataFrame] = None,
    include_vix: bool = False,
    commodity_columns: Optional[list] = None,
    include_treasury: bool = False,
) -> pd.DataFrame:
    """
    Merge external data using Polars left joins.

    The output is converted back to pandas to preserve downstream contracts.
    """
    if "date" not in stock_df.columns:
        raise ValueError("Missing required column for external data merge: date")

    pl = _require_polars()
    row_id = "__row_id"

    result = (
        pl.from_pandas(stock_df.reset_index(drop=True))
        .with_row_index(row_id)
        .with_columns(pl.col("date").cast(pl.Datetime, strict=False))
    )

    if vix_df is not None and include_vix:
        result = result.join(
            _select_external_columns(pl, vix_df, ["date", "vix"]),
            on="date",
            how="left",
        )

    if commodities_df is not None and commodity_columns:
        result = result.join(
            _select_external_columns(pl, commodities_df, ["date", *commodity_columns]),
            on="date",
            how="left",
        )

    if treasury_df is not None and include_treasury:
        result = result.join(
            _select_external_columns(pl, treasury_df, ["date", "bondyield"]),
            on="date",
            how="left",
        )

    return _to_pandas_with_datetime(result.sort(row_id).drop(row_id), ["date"])


def _select_external_columns(pl, df: pd.DataFrame, columns: list):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing external data columns: {missing}")
    return (
        pl.from_pandas(df[columns])
        .with_columns(pl.col("date").cast(pl.Datetime, strict=False))
    )


def _to_pandas_with_datetime(polars_df, datetime_columns: list) -> pd.DataFrame:
    result = polars_df.to_pandas()
    for column in datetime_columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    return result


def profile_fibonacci_implementations(
    df: pd.DataFrame,
    window: int,
    pandas_callable,
    repeat: int = 3,
) -> Dict[str, Optional[float]]:
    """
    Profile pandas and Polars Fibonacci implementations on the same input.

    The caller supplies the existing pandas callable to avoid importing the
    main FeatureEngineer here and creating circular dependencies.
    """
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    timings = {"pandas_seconds": [], "polars_seconds": []}
    for _ in range(repeat):
        start = perf_counter()
        pandas_callable(df.copy())
        timings["pandas_seconds"].append(perf_counter() - start)

        start = perf_counter()
        add_fibonacci_features_polars(df.copy(), window)
        timings["polars_seconds"].append(perf_counter() - start)

    return {
        "pandas_seconds": min(timings["pandas_seconds"]),
        "polars_seconds": min(timings["polars_seconds"]),
        "repeat": repeat,
        "rows": len(df),
        "window": window,
    }
