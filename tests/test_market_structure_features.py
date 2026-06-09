import copy

import numpy as np
import pandas as pd

from src.config import load_config
from src.config.config_loader import Config
from src.data.dataset import FinancialDataset
from src.data.feature_engineering import FeatureEngineer
from src.data.market_structure_features import MarketStructureFeatureBuilder
from src.data.preprocessing import DataPreprocessor


def _config():
    config = Config(copy.deepcopy(load_config("main").to_dict()))
    flags = config.data.features.FEATURE_FLAGS._data
    flags["financial_metrics"] = False
    flags["market_regime"] = False
    flags["cointegration_features"] = False
    config.data.regime._data["ENABLED"] = False
    config.data.features.FEATURE_FLAGS._data["candlestick_patterns"] = False
    config.data.candlestick._data["USE_CANDLESTICK_PATTERNS"] = False
    geometric = config.data.geometric._data
    geometric["ENABLE_MARKET_STRUCTURE_FEATURES"] = True
    geometric["MARKET_STRUCTURE_WINDOWS"] = [20, 60, 120, 252]
    geometric["BREAKOUT_WINDOWS"] = [20, 60, 120]
    geometric["MARKET_STRUCTURE_COUNT_WINDOW"] = 20
    geometric["NEAR_52W_THRESHOLD"] = 0.05
    geometric["VOLUME_CONFIRMATION_WINDOW"] = 20
    geometric["ATR_WINDOWS"] = [14, 20]
    geometric["TREND_WINDOWS"] = [20, 60]
    geometric["MARKET_STRUCTURE_LAGS"] = [1, 3, 5, 10, 20]
    return config


def _long_market_structure_df():
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    rows = []
    close = 100.0
    for idx, date in enumerate(dates):
        close += 0.12
        if idx == 40:
            close += 6.0
        if idx == 120:
            close -= 10.0
        high = close + 0.6
        low = close - 0.7
        rows.append(
            {
                "date": date,
                "tic": "AAPL",
                "open": close - 0.2,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000 + idx * 2500 + (150_000 if idx in {40, 120} else 0),
            }
        )
    return pd.DataFrame(rows)


def _structure_logic_df():
    rows = [
        {"date": "2024-01-01", "tic": "AAPL", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.2, "volume": 1000},
        {"date": "2024-01-02", "tic": "AAPL", "open": 100.1, "high": 102.0, "low": 99.5, "close": 101.5, "volume": 1100},
        {"date": "2024-01-03", "tic": "AAPL", "open": 101.4, "high": 101.8, "low": 100.2, "close": 100.8, "volume": 1200},
        {"date": "2024-01-04", "tic": "AAPL", "open": 100.7, "high": 103.5, "low": 100.4, "close": 103.1, "volume": 1600},
        {"date": "2024-01-05", "tic": "AAPL", "open": 103.0, "high": 103.1, "low": 98.8, "close": 99.0, "volume": 1700},
    ]
    return pd.DataFrame(rows).assign(date=lambda df: pd.to_datetime(df["date"]))


def test_market_structure_feature_columns_exist():
    builder = MarketStructureFeatureBuilder.from_config(_config().data.geometric)
    result = builder.transform(_long_market_structure_df()).dataframe

    expected_columns = [
        "distance_to_20d_high",
        "distance_to_60d_high",
        "distance_to_120d_high",
        "distance_to_252d_high",
        "distance_to_20d_low",
        "distance_to_60d_low",
        "distance_to_120d_low",
        "distance_to_252d_low",
        "breakout_20d",
        "breakout_60d",
        "breakout_120d",
        "breakdown_20d",
        "breakdown_60d",
        "breakdown_120d",
        "higher_high",
        "lower_high",
        "higher_low",
        "lower_low",
        "higher_high_count_20",
        "higher_low_count_20",
        "lower_high_count_20",
        "lower_low_count_20",
        "distance_to_52w_high",
        "distance_to_52w_low",
        "near_52w_high",
        "near_52w_low",
        "breakout_volume_ratio",
        "breakdown_volume_ratio",
        "volume_spike_ratio",
        "volume_momentum",
        "atr_14",
        "atr_20",
        "atr_ratio",
        "rolling_volatility_20",
        "rolling_volatility_60",
        "trend_strength_score",
        "trend_persistence_20",
        "trend_persistence_60",
    ]
    for signal in [
        "higher_high",
        "lower_high",
        "higher_low",
        "lower_low",
        "breakout_20d",
        "breakdown_20d",
        "trend_strength_score",
    ]:
        for lag in [1, 3, 5, 10, 20]:
            expected_columns.append(f"{signal}_lag_{lag}")

    for column in expected_columns:
        assert column in result.columns, f"Missing market structure column: {column}"


def test_market_structure_features_have_no_nan_values():
    original = _long_market_structure_df()
    builder = MarketStructureFeatureBuilder.from_config(_config().data.geometric)
    result = builder.transform(original).dataframe
    added_columns = [column for column in result.columns if column not in original.columns]

    assert added_columns
    assert result[added_columns].isna().sum().sum() == 0
    assert np.isfinite(result[added_columns].to_numpy(dtype=float)).all()


def test_market_structure_breakout_and_breakdown_logic():
    builder = MarketStructureFeatureBuilder.from_config(_config().data.geometric)
    result = builder.transform(_long_market_structure_df()).dataframe.reset_index(drop=True)

    assert int(result.loc[40, "breakout_20d"]) == 1
    assert float(result.loc[40, "breakout_volume_ratio"]) > 1.0
    assert int(result.loc[120, "breakdown_20d"]) == 1
    assert float(result.loc[120, "breakdown_volume_ratio"]) > 1.0


def test_market_structure_higher_high_and_lower_low_logic():
    builder = MarketStructureFeatureBuilder.from_config(_config().data.geometric)
    result = builder.transform(_structure_logic_df()).dataframe.reset_index(drop=True)

    assert int(result.loc[1, "higher_high"]) == 1
    assert int(result.loc[1, "higher_low"]) == 1
    assert int(result.loc[2, "lower_high"]) == 1
    assert int(result.loc[4, "lower_low"]) == 1
    assert float(result.loc[4, "lower_low_count_20"]) >= 1.0


def test_market_structure_support_resistance_columns_are_not_overwritten():
    df = _long_market_structure_df()
    df["distance_to_20d_high"] = 9.0
    df["distance_to_20d_low"] = 8.0

    builder = MarketStructureFeatureBuilder.from_config(_config().data.geometric)
    result = builder.transform(df)

    assert (result.dataframe["distance_to_20d_high"] == 9.0).all()
    assert (result.dataframe["distance_to_20d_low"] == 8.0).all()
    assert "distance_to_20d_high" in result.metadata["preserved_features"]
    assert "distance_to_20d_low" in result.metadata["preserved_features"]


def test_market_structure_metadata_tracks_generated_features():
    builder = MarketStructureFeatureBuilder.from_config(_config().data.geometric)
    result = builder.transform(_long_market_structure_df())

    assert result.metadata["enabled"] is True
    assert "distance_to_20d_high" in result.metadata["generated_features"]
    assert "trend_strength_score" in result.metadata["signal_columns"]
    assert result.metadata["windows"]["market_structure"] == [20, 60, 120, 252]


def test_market_structure_outputs_support_pytorch_dataset():
    config = _config()
    config.data.sequences.SEQUENCE_LENGTH = 10
    config.data.sequences.PREDICTION_HORIZON = 1

    rows = []
    for ticker_idx, ticker in enumerate(["AAPL", "MSFT"]):
        price = 100.0 + ticker_idx * 20.0
        for day_idx, date in enumerate(pd.date_range("2023-01-01", periods=120, freq="D")):
            price = price * (1.0 + 0.0015 * np.cos(day_idx / 3.0))
            rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "open": price * 0.996,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 2_000_000 + (day_idx * 3000),
                }
            )

    df = pd.DataFrame(rows)
    engineered = FeatureEngineer(config).add_all_features(df, calculate_target=True)
    preprocessor = DataPreprocessor(config)
    _processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(engineered, fit=True)

    assert "train" in splits
    assert "val" in splits
    assert not splits["train"].empty
    dataset = FinancialDataset(sequences["train"], load_config("model"))
    sample = dataset[0]
    assert sample["features"].shape[0] == config.data.sequences.SEQUENCE_LENGTH
    assert sample["features"].shape[1] == info["num_features"]
