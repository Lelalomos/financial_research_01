"""
Unit tests for market regime detection utilities.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.config.config_loader import Config
from src.data.preprocessing import DataPreprocessor
from src.data.regime import MarketRegimeDetector


def _make_regime_config():
    data = copy.deepcopy(load_config("main").to_dict())
    data["data"]["regime"]["ENABLED"] = True
    data["data"]["regime"]["PROXY_COLUMN"] = "vix"
    data["data"]["regime"]["N_REGIMES"] = 3
    data["data"]["regime"]["LOW_QUANTILE"] = 0.33
    data["data"]["regime"]["HIGH_QUANTILE"] = 0.66
    data["data"]["regime"]["DEFAULT_REGIME"] = 1
    data["data"]["features"]["FEATURE_FLAGS"]["market_regime"] = True
    data["data"]["sequences"]["SEQUENCE_LENGTH"] = 5
    data["data"]["sequences"]["PREDICTION_HORIZON"] = 1
    data["data"]["sequences"]["STRIDE"] = 1
    data["data"]["splits"]["TRAIN_RATIO"] = 0.6
    data["data"]["splits"]["VAL_RATIO"] = 0.2
    data["data"]["splits"]["TEST_RATIO"] = 0.2
    data["data"]["validation"]["WARN_ON_MISSING_OPTIONAL"] = False
    return Config(data)


def _make_price_frame(n_days=40):
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        for idx, date in enumerate(dates):
            close = 100.0 + idx
            rows.append({
                "date": date,
                "tic": ticker,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000 + idx,
                "target": 0.01,
                "vix": 10.0 + idx,
                "group": "Technology",
                "day": date.day,
                "month": date.month,
                "dayofweek": date.dayofweek,
            })
    return pd.DataFrame(rows)


def test_quantile_regime_assignment_is_deterministic():
    df = pd.DataFrame({"vix": [10.0, 20.0, 30.0, 40.0, 50.0]})
    detector = MarketRegimeDetector(proxy_column="vix", n_regimes=3)

    result = detector.fit(df).transform(df)

    assert set(result["regime_id"].unique()) <= {0, 1, 2}
    np.testing.assert_array_equal(result["regime_id"].to_numpy(), np.array([0, 0, 1, 2, 2]))


def test_regime_thresholds_fit_on_train_only():
    train = pd.DataFrame({"vix": [10.0, 11.0, 12.0, 13.0, 14.0]})
    val = pd.DataFrame({"vix": [1000.0]})
    detector = MarketRegimeDetector(proxy_column="vix", n_regimes=3)

    splits = detector.fit_transform_splits({"train": train, "val": val})

    assert max(detector.thresholds) < 1000.0
    assert splits["val"]["regime_id"].iloc[0] == 2


def test_regime_detector_rejects_missing_proxy_column():
    detector = MarketRegimeDetector(proxy_column="vix", n_regimes=3)

    with pytest.raises(ValueError, match="Missing regime proxy column"):
        detector.fit(pd.DataFrame({"close": [1.0, 2.0]}))


def test_preprocessing_adds_regime_feature_when_enabled():
    config = _make_regime_config()
    df = _make_price_frame()

    preprocessor = DataPreprocessor(config)
    processed_df, splits, sequences, info = preprocessor.preprocess_pipeline(df, fit=True)

    assert "regime_id" in processed_df.columns
    assert "regime_id" in info["feature_cols"]
    assert info["regime_params"]["proxy_column"] == "vix"
    assert set(processed_df["regime_id"].unique()) <= {0, 1, 2}
    assert "train" in sequences
    assert sequences["train"]["features"].shape[2] == info["num_features"]
