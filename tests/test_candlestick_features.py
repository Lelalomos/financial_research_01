import copy

import numpy as np
import pandas as pd

from src.config import load_config
from src.config.config_loader import Config
from src.data.dataset import FinancialDataset
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.models import create_model


_PATTERN_ROWS = {
    "doji": 1,
    "hammer": 2,
    "inverted_hammer": 3,
    "shooting_star": 4,
    "bullish_engulfing": 6,
    "bearish_engulfing": 8,
    "morning_star": 11,
    "evening_star": 14,
}


def _config():
    config = Config(copy.deepcopy(load_config("main").to_dict()))
    config.data.features.FEATURE_FLAGS._data["candlestick_patterns"] = True
    config.data.candlestick._data["USE_CANDLESTICK_PATTERNS"] = True
    config.data.features.FEATURE_FLAGS._data["financial_metrics"] = False
    config.data.features.FEATURE_FLAGS._data["market_regime"] = False
    config.data.regime._data["ENABLED"] = False
    return config


def _sample_df():
    rows = [
        {"date": "2024-01-01", "tic": "AAPL", "open": 10.0, "high": 10.4, "low": 9.8, "close": 10.2, "volume": 1000},
        {"date": "2024-01-02", "tic": "AAPL", "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.02, "volume": 1001},
        {"date": "2024-01-03", "tic": "AAPL", "open": 10.0, "high": 10.25, "low": 9.5, "close": 10.2, "volume": 1002},
        {"date": "2024-01-04", "tic": "AAPL", "open": 10.0, "high": 11.0, "low": 9.95, "close": 10.2, "volume": 1003},
        {"date": "2024-01-05", "tic": "AAPL", "open": 10.3, "high": 11.0, "low": 10.0, "close": 10.1, "volume": 1004},
        {"date": "2024-01-06", "tic": "AAPL", "open": 10.5, "high": 10.55, "low": 9.9, "close": 10.0, "volume": 1005},
        {"date": "2024-01-07", "tic": "AAPL", "open": 9.9, "high": 10.8, "low": 9.8, "close": 10.7, "volume": 1006},
        {"date": "2024-01-08", "tic": "AAPL", "open": 10.0, "high": 10.55, "low": 9.95, "close": 10.5, "volume": 1007},
        {"date": "2024-01-09", "tic": "AAPL", "open": 10.6, "high": 10.7, "low": 9.7, "close": 9.8, "volume": 1008},
        {"date": "2024-01-10", "tic": "AAPL", "open": 11.0, "high": 11.1, "low": 9.9, "close": 10.0, "volume": 1009},
        {"date": "2024-01-11", "tic": "AAPL", "open": 9.8, "high": 10.0, "low": 9.7, "close": 9.85, "volume": 1010},
        {"date": "2024-01-12", "tic": "AAPL", "open": 9.9, "high": 10.8, "low": 9.8, "close": 10.7, "volume": 1011},
        {"date": "2024-01-13", "tic": "AAPL", "open": 10.0, "high": 11.1, "low": 9.9, "close": 11.0, "volume": 1012},
        {"date": "2024-01-14", "tic": "AAPL", "open": 11.1, "high": 11.2, "low": 11.0, "close": 11.05, "volume": 1013},
        {"date": "2024-01-15", "tic": "AAPL", "open": 11.0, "high": 11.1, "low": 10.1, "close": 10.2, "volume": 1014},
    ]
    return pd.DataFrame(rows).assign(date=lambda df: pd.to_datetime(df["date"]))


def test_candlestick_feature_columns_exist():
    result = FeatureEngineer(_config()).add_candlestick_patterns(_sample_df())

    expected_columns = [
        "body_size",
        "candle_direction",
        "upper_shadow",
        "lower_shadow",
        "high_low_range",
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "close_position",
        "open_position",
        "gap_up",
        "gap_down",
        "gap_size",
        "overnight_return",
        "body_size_change",
        "body_size_ema_5",
        "body_size_ema_20",
        "upper_shadow_change",
        "lower_shadow_change",
        "atr_14",
        "atr_20",
        "atr",
        "rolling_volatility",
        "support_distance",
        "resistance_distance",
        "breakout_signal",
        "volume_momentum",
        "volume_spike",
        "return_1d",
        "return_5d",
        "return_20d",
        "rolling_high_low_range_5",
        "rolling_high_low_range_20",
        "body_size_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "range_pct",
        "doji",
        "hammer",
        "inverted_hammer",
        "shooting_star",
        "bullish_engulfing",
        "bearish_engulfing",
        "morning_star",
        "evening_star",
    ]
    for lag in [1, 3, 5, 10, 20]:
        expected_columns.append(f"lag_{lag}")
    for window in [5, 10, 20, 60]:
        expected_columns.extend(
            [
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_min_{window}",
                f"rolling_max_{window}",
            ]
        )
    for window in [5, 10, 20, 60]:
        for base_col in ["body_size", "upper_shadow", "lower_shadow"]:
            expected_columns.extend(
                [
                    f"{base_col}_rolling_mean_{window}",
                    f"{base_col}_rolling_std_{window}",
                    f"{base_col}_rolling_zscore_{window}",
                ]
            )

    for column in expected_columns:
        assert column in result.columns, f"Missing candlestick column: {column}"


def test_candlestick_features_have_no_nan_values():
    original = _sample_df()
    result = FeatureEngineer(_config()).add_candlestick_patterns(original)
    added_columns = [column for column in result.columns if column not in original.columns]

    assert added_columns
    assert result[added_columns].isna().sum().sum() == 0
    assert np.isfinite(result[added_columns].to_numpy(dtype=float)).all()


def test_candlestick_pattern_detection_logic():
    result = FeatureEngineer(_config()).add_candlestick_patterns(_sample_df())

    for pattern_name, row_idx in _PATTERN_ROWS.items():
        assert int(result.loc[row_idx, pattern_name]) == 1, f"{pattern_name} should be detected at row {row_idx}"

    zero_checks = {
        "doji": 0,
        "hammer": 1,
        "inverted_hammer": 2,
        "shooting_star": 3,
        "bullish_engulfing": 5,
        "bearish_engulfing": 7,
        "morning_star": 10,
        "evening_star": 13,
    }
    for pattern_name, row_idx in zero_checks.items():
        assert int(result.loc[row_idx, pattern_name]) == 0, f"{pattern_name} should not trigger at row {row_idx}"


def test_existing_dataset_features_are_not_overwritten():
    df = _sample_df()
    df["body_size"] = 999.0
    df["return_1d"] = -123.0
    df["lag_1"] = 77.0

    result = FeatureEngineer(_config()).add_candlestick_patterns(df)

    assert (result["body_size"] == 999.0).all()
    assert (result["return_1d"] == -123.0).all()
    assert (result["lag_1"] == 77.0).all()
    assert "upper_shadow" in result.columns
    assert "rolling_volatility" in result.columns


def test_sequence_ready_outputs_support_transformer_dataset():
    config = _config()
    config.data.sequences.SEQUENCE_LENGTH = 5
    config.data.sequences.PREDICTION_HORIZON = 1

    rows = []
    for ticker_idx, ticker in enumerate(["AAPL", "MSFT"]):
        price = 100.0 + ticker_idx * 10.0
        for day_idx, date in enumerate(pd.date_range("2024-01-01", periods=40, freq="D")):
            price = price * (1.0 + 0.002 * np.sin(day_idx))
            rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "open": price * 0.995,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000 + (day_idx * 1000),
                }
            )

    df = pd.DataFrame(rows)
    engineered = FeatureEngineer(config).add_all_features(df, calculate_target=True)
    preprocessor = DataPreprocessor(config)
    _processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(engineered, fit=True)

    assert "train" in splits
    assert "val" in splits
    assert not splits["train"].empty
    assert not splits["val"].empty
    assert "train" in sequences
    assert len(sequences["train"]["target"]) > 0

    model_config = load_config("model")
    model_config.model.training.BATCH_SIZE = 4
    dataset = FinancialDataset(sequences["train"], model_config)
    sample = dataset[0]
    assert sample["features"].shape[0] == config.data.sequences.SEQUENCE_LENGTH
    assert sample["features"].shape[1] == info["num_features"]

    model = create_model(
        model_type="transformer",
        num_features=info["num_features"],
        num_stocks=info["num_stocks"],
        num_groups=info["num_groups"],
        config=model_config,
    )
    output = model(
        sample["features"].unsqueeze(0),
        sample["stock_id"].unsqueeze(0),
        sample["group_id"].unsqueeze(0),
        sample["day"].unsqueeze(0),
        sample["month"].unsqueeze(0),
        sample["dividend_flag"].unsqueeze(0),
    )
    assert output.shape == (1, 1)
