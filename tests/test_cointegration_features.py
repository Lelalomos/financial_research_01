import copy

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.config import load_config
from src.config.config_loader import Config
from src.data.feature_engineering import FeatureEngineer


def _make_config(enabled: bool = True) -> Config:
    config = Config(copy.deepcopy(load_config("main").to_dict()))
    flags = config.data.features.FEATURE_FLAGS._data
    flags["cointegration_features"] = enabled
    flags["financial_metrics"] = False
    flags["candlestick_patterns"] = False
    flags["vix"] = False
    flags["commodities"] = False
    flags["treasury_yields"] = False
    flags["market_regime"] = False
    config.data.regime._data["ENABLED"] = False
    config.data.cointegration._data["ROLLING_WINDOW"] = 20
    config.data.cointegration._data["NORMALIZATION_WINDOW"] = 20
    config.data.cointegration._data["JOHANSEN_DET_ORDER"] = 0
    config.data.cointegration._data["JOHANSEN_K_AR_DIFF"] = 1
    return config


def _make_pair_data(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=periods, freq="D")
    base = 100 + np.linspace(0, 6, periods)
    peer = 50 + np.linspace(0, 3, periods)
    finance = 200 + np.linspace(0, 4, periods)

    rows = []
    for idx, date in enumerate(dates):
        pair_values = {
            "AAA": base[idx],
            "BBB": peer[idx],
            "FIN": finance[idx],
        }
        for ticker, close in pair_values.items():
            rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + idx,
                }
            )
    return pd.DataFrame(rows)


def _make_sector_data(periods: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=periods, freq="D")
    common = 100 + np.cumsum(rng.normal(0.2, 0.05, periods))
    tech_series = {
        "AAA": common + 2.0 + rng.normal(0.0, 0.03, periods),
        "BBB": 0.8 * common + 1.0 + rng.normal(0.0, 0.03, periods),
        "CCC": 1.2 * common - 1.5 + rng.normal(0.0, 0.03, periods),
        "FIN": 200 + np.cumsum(rng.normal(0.1, 0.2, periods)),
    }

    rows = []
    for idx, date in enumerate(dates):
        for ticker, series in tech_series.items():
            close = float(series[idx])
            rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 2_000_000 + idx,
                }
            )
    return pd.DataFrame(rows)


PAIR_MAPPING = {"AAA": "Tech", "BBB": "Tech", "FIN": "Finance"}
SECTOR_MAPPING = {"AAA": "Tech", "BBB": "Tech", "CCC": "Tech", "FIN": "Finance"}


def test_cointegration_feature_flag_can_disable_columns():
    config = _make_config(enabled=False)
    engineer = FeatureEngineer(config, sector_mapping=PAIR_MAPPING)
    with_groups = engineer.add_group_from_sector(_make_pair_data())
    result = engineer.add_cointegration_features(with_groups)

    assert "spread" not in result.columns
    assert "equilibrium_gap" not in result.columns


def test_pair_spread_and_zscore_match_rolling_ols_definition():
    config = _make_config(enabled=True)
    engineer = FeatureEngineer(config, sector_mapping=PAIR_MAPPING)
    with_groups = engineer.add_group_from_sector(_make_pair_data())
    result = engineer.add_cointegration_features(with_groups)

    target_date = pd.Timestamp("2021-01-20")
    stock_rows = with_groups[with_groups["tic"].isin(["AAA", "BBB"])].copy()
    close_pivot = stock_rows.pivot(index="date", columns="tic", values="close").sort_index()
    window = close_pivot.loc[:target_date].tail(20)

    beta = FeatureEngineer._rolling_ols_beta(
        window["AAA"].to_numpy(dtype=np.float64),
        window["BBB"].to_numpy(dtype=np.float64),
    )
    spread_window = window["AAA"].to_numpy(dtype=np.float64) - beta * window["BBB"].to_numpy(dtype=np.float64)
    expected_spread = float(spread_window[-1])
    expected_mean = float(np.mean(spread_window))
    expected_std = float(np.std(spread_window, ddof=0))
    expected_zscore = (expected_spread - expected_mean) / expected_std

    row = result[(result["tic"] == "AAA") & (result["date"] == target_date)].iloc[0]
    assert row["pair_beta"] == pytest.approx(beta, rel=1e-6)
    assert row["spread"] == pytest.approx(expected_spread, rel=1e-6)
    assert row["rolling_mean_spread"] == pytest.approx(expected_mean, rel=1e-6)
    assert row["rolling_std_spread"] == pytest.approx(expected_std, rel=1e-6)
    assert row["spread_zscore"] == pytest.approx(expected_zscore, rel=1e-6)


def test_johansen_equilibrium_gap_matches_first_eigenvector_projection():
    config = _make_config(enabled=True)
    engineer = FeatureEngineer(config, sector_mapping=SECTOR_MAPPING)
    with_groups = engineer.add_group_from_sector(_make_sector_data())
    result = engineer.add_cointegration_features(with_groups)

    target_date = pd.Timestamp("2021-01-25")
    tech_rows = with_groups[with_groups["group"] == "Tech"].copy()
    close_pivot = tech_rows.pivot(index="date", columns="tic", values="close").sort_index()
    log_window = np.log(close_pivot.loc[:target_date].tail(20))
    johansen_result = coint_johansen(log_window.to_numpy(dtype=np.float64), det_order=0, k_ar_diff=1)
    beta_vector = johansen_result.evec[:, 0]
    equilibrium_series = log_window.to_numpy(dtype=np.float64) @ beta_vector
    expected_gap = float(equilibrium_series[-1])
    expected_zscore = (equilibrium_series[-1] - equilibrium_series.mean()) / equilibrium_series.std(ddof=0)

    row = result[(result["tic"] == "AAA") & (result["date"] == target_date)].iloc[0]
    assert row["equilibrium_gap"] == pytest.approx(expected_gap, rel=1e-6)
    assert row["equilibrium_zscore"] == pytest.approx(expected_zscore, rel=1e-6)


def test_features_use_only_past_data_without_lookahead():
    config = _make_config(enabled=True)
    engineer = FeatureEngineer(config, sector_mapping=PAIR_MAPPING)
    original = engineer.add_cointegration_features(engineer.add_group_from_sector(_make_pair_data()))

    modified_input = _make_pair_data()
    modified_input.loc[
        (modified_input["tic"] == "BBB") & (modified_input["date"] >= pd.Timestamp("2021-01-30")),
        "close",
    ] *= 10.0
    modified = engineer.add_cointegration_features(engineer.add_group_from_sector(modified_input))

    cutoff = pd.Timestamp("2021-01-25")
    feature_cols = ["spread", "spread_zscore", "equilibrium_gap", "relative_price_vs_sector"]
    original_slice = original[(original["tic"] == "AAA") & (original["date"] <= cutoff)][feature_cols].reset_index(drop=True)
    modified_slice = modified[(modified["tic"] == "AAA") & (modified["date"] <= cutoff)][feature_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_slice, modified_slice)


def test_rolling_window_and_alignment_behavior():
    config = _make_config(enabled=True)
    engineer = FeatureEngineer(config, sector_mapping=PAIR_MAPPING)
    with_groups = engineer.add_group_from_sector(_make_pair_data())
    result = engineer.add_cointegration_features(with_groups)

    early_row = result[(result["tic"] == "AAA") & (result["date"] == pd.Timestamp("2021-01-10"))].iloc[0]
    assert np.isnan(early_row["spread"])
    assert np.isnan(early_row["equilibrium_gap"])

    assert len(result) == len(with_groups)
    pd.testing.assert_series_equal(result["date"], with_groups["date"], check_names=False)
    pd.testing.assert_series_equal(result["tic"], with_groups["tic"], check_names=False)


def test_sector_grouping_and_missing_value_handling():
    config = _make_config(enabled=True)
    input_df = _make_pair_data()
    input_df.loc[(input_df["tic"] == "BBB") & (input_df["date"] == pd.Timestamp("2021-01-19")), "close"] = np.nan
    engineer = FeatureEngineer(config, sector_mapping=PAIR_MAPPING)
    result = engineer.add_cointegration_features(engineer.add_group_from_sector(input_df))

    finance_row = result[(result["tic"] == "FIN") & (result["date"] == pd.Timestamp("2021-01-20"))].iloc[0]
    assert np.isnan(finance_row["spread"])
    assert np.isnan(finance_row["equilibrium_gap"])

    missing_window_row = result[(result["tic"] == "AAA") & (result["date"] == pd.Timestamp("2021-01-20"))].iloc[0]
    assert np.isnan(missing_window_row["spread"])
