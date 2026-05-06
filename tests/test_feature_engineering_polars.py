"""
Parity tests for optional Polars feature engineering paths.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.config_loader import Config
from src.data.feature_engineering import FeatureEngineer
from src.data.feature_engineering_polars import (
    FIBONACCI_COLUMNS,
    PolarsFeatureEngineeringError,
    TIME_FEATURE_COLUMNS,
    add_fibonacci_features_polars,
    add_time_features_polars,
    merge_external_data_polars,
    profile_fibonacci_implementations,
)


pytest.importorskip("polars")


def _config(use_polars=False):
    return Config({
        "data": {
            "sources": {
                "RAW_DATA_INDEX_PATH": "does-not-exist",
                "INDEX_FILE": "missing.json",
                "COMMODITIES": {},
                "TREASURY_YIELDS": [],
            },
            "features": {
                "FEATURE_FLAGS": {
                    "financial_metrics": False,
                    "fibonacci_features": True,
                    "polars_fibonacci_features": use_polars,
                    "polars_time_features": use_polars,
                    "polars_external_merges": use_polars,
                    "vix": True,
                    "commodities": True,
                    "treasury_yields": True,
                }
            },
            "fibonacci": {"FIBONACCI_WINDOW": 5},
            "sources": {
                "COMMODITIES": {
                    "GC=F": "Gold",
                    "SI=F": "Silver",
                },
                "RAW_DATA_INDEX_PATH": "does-not-exist",
                "INDEX_FILE": "missing.json",
            },
        }
    })


def _sample_ohlcv():
    rng = np.random.default_rng(42)
    rows = []
    for ticker in ["MSFT", "AAPL", "GOOGL"]:
        base = 100 + rng.normal(0, 1, 18).cumsum()
        for idx, date in enumerate(pd.date_range("2023-01-01", periods=18, freq="D")):
            close = base[idx]
            rows.append({
                "date": date,
                "tic": ticker,
                "open": close * (1 + rng.normal(0, 0.01)),
                "high": close + abs(rng.normal(0, 2)),
                "low": close - abs(rng.normal(0, 2)),
                "close": close,
                "volume": int(rng.integers(1000, 10000)),
            })

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)


def test_polars_fibonacci_matches_pandas_output_with_tolerance():
    df = _sample_ohlcv()
    pandas_engineer = FeatureEngineer(_config(use_polars=False))
    pandas_result = pandas_engineer.add_fibonacci_features(df)
    polars_result = add_fibonacci_features_polars(df, window=5)

    pd.testing.assert_series_equal(polars_result["date"], pandas_result["date"], check_names=False)
    pd.testing.assert_series_equal(polars_result["tic"], pandas_result["tic"], check_names=False)

    for column in FIBONACCI_COLUMNS:
        if column == "break_fib_61":
            pd.testing.assert_series_equal(
                polars_result[column].astype("Int64"),
                pandas_result[column].astype("Int64"),
                check_names=False,
            )
        else:
            np.testing.assert_allclose(
                polars_result[column].to_numpy(dtype=float),
                pandas_result[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )


def test_feature_engineer_uses_polars_only_when_enabled():
    df = _sample_ohlcv()
    pandas_config = _config(use_polars=False)
    polars_config = _config(use_polars=True)

    pandas_result = FeatureEngineer(pandas_config).add_fibonacci_features(df)
    polars_result = FeatureEngineer(polars_config).add_fibonacci_features(df)

    for column in FIBONACCI_COLUMNS:
        if column == "break_fib_61":
            pd.testing.assert_series_equal(
                polars_result[column].astype("Int64"),
                pandas_result[column].astype("Int64"),
                check_names=False,
            )
        else:
            np.testing.assert_allclose(
                polars_result[column].to_numpy(dtype=float),
                pandas_result[column].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )


def test_polars_time_features_match_pandas_output():
    df = _sample_ohlcv()
    pandas_result = FeatureEngineer(_config(use_polars=False)).add_time_features(df)
    polars_result = add_time_features_polars(df)

    pd.testing.assert_series_equal(polars_result["date"], pandas_result["date"], check_names=False)
    for column in TIME_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            polars_result[column].astype("int64"),
            pandas_result[column].astype("int64"),
            check_names=False,
        )


def test_polars_external_merges_match_pandas_output():
    stock_df = _sample_ohlcv()[["date", "tic", "close"]].head(12).copy()
    dates = pd.DataFrame({"date": sorted(stock_df["date"].unique())})
    vix_df = dates.assign(vix=np.arange(len(dates), dtype=float) + 10.0)
    commodities_df = dates.assign(
        Gold=np.arange(len(dates), dtype=float) + 100.0,
        Silver=np.arange(len(dates), dtype=float) + 20.0,
    )
    treasury_df = dates.assign(bondyield=np.arange(len(dates), dtype=float) + 3.0)

    engineer = FeatureEngineer(_config(use_polars=False))
    pandas_result = engineer.merge_external_data(
        stock_df,
        vix_df=vix_df,
        commodities_df=commodities_df,
        treasury_df=treasury_df,
    )
    polars_result = merge_external_data_polars(
        stock_df=stock_df,
        vix_df=vix_df,
        commodities_df=commodities_df,
        treasury_df=treasury_df,
        include_vix=True,
        commodity_columns=["Gold", "Silver"],
        include_treasury=True,
    )

    pd.testing.assert_frame_equal(
        polars_result[pandas_result.columns],
        pandas_result,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_feature_engineer_uses_polars_external_merge_when_enabled():
    stock_df = _sample_ohlcv()[["date", "tic", "close"]].head(12).copy()
    dates = pd.DataFrame({"date": sorted(stock_df["date"].unique())})
    vix_df = dates.assign(vix=np.arange(len(dates), dtype=float) + 10.0)

    pandas_result = FeatureEngineer(_config(use_polars=False)).merge_external_data(stock_df, vix_df=vix_df)
    polars_result = FeatureEngineer(_config(use_polars=True)).merge_external_data(stock_df, vix_df=vix_df)

    pd.testing.assert_frame_equal(
        polars_result[pandas_result.columns],
        pandas_result,
        check_dtype=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_polars_fibonacci_validates_required_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        add_fibonacci_features_polars(pd.DataFrame({"tic": ["AAPL"]}), window=5)


def test_polars_fibonacci_profile_returns_timings():
    df = _sample_ohlcv()
    pandas_engineer = FeatureEngineer(_config(use_polars=False))

    profile = profile_fibonacci_implementations(
        df=df,
        window=5,
        pandas_callable=pandas_engineer.add_fibonacci_features,
        repeat=1,
    )

    assert profile["rows"] == len(df)
    assert profile["window"] == 5
    assert profile["repeat"] == 1
    assert profile["pandas_seconds"] >= 0
    assert profile["polars_seconds"] >= 0


def test_polars_error_is_clear_when_dependency_missing(monkeypatch):
    import src.data.feature_engineering_polars as polars_module

    def raise_missing():
        raise PolarsFeatureEngineeringError("Polars is required")

    monkeypatch.setattr(polars_module, "_require_polars", raise_missing)

    with pytest.raises(PolarsFeatureEngineeringError, match="Polars is required"):
        polars_module.add_fibonacci_features_polars(_sample_ohlcv(), window=5)
