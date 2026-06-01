import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_kronos_rich_data import (
    assign_future_regime_labels,
    build_kronos_rich_sequences,
    resolve_kronos_rich_prep_settings,
    save_kronos_rich_split,
    write_kronos_rich_metadata,
)
from src.config import load_config


def _make_split_df():
    dates = pd.date_range("2024-01-01", periods=9, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "tic": ["AAA"] * len(dates),
            "tic_id": [0] * len(dates),
            "group_id": [1] * len(dates),
            "day": dates.day.astype(np.int32),
            "month": dates.month.astype(np.int32),
            "dividend_flag": np.ones(len(dates), dtype=np.int32),
            "open": np.array([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.float32),
            "high": np.array([11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float32),
            "low": np.array([9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float32),
            "close": np.array([10, 11, 12, 13, 14, 15, 17, 16, 18], dtype=np.float32),
            "volume": np.array([100, 101, 103, 104, 108, 112, 115, 120, 125], dtype=np.float32),
            "target": np.array([0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32),
            "regime_id": np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64),
        }
    )


def test_resolve_kronos_rich_prep_settings_reads_config_defaults():
    config = load_config("main")
    args = type(
        "Args",
        (),
        {
            "processed_dir": None,
            "output_dir": None,
            "skip_scalar_target": False,
            "skip_return_path": False,
            "skip_regime_label": False,
        },
    )()

    settings = resolve_kronos_rich_prep_settings(config, args)

    assert settings["processed_dir"] == Path("data/processed")
    assert settings["output_dir"] == Path("data/processed_kronos_rich")
    assert settings["ohlcv_columns"] == ["open", "high", "low", "close", "volume"]
    assert settings["include_return_path"] is True
    assert settings["include_regime_label"] is True


def test_build_kronos_rich_sequences_creates_all_outputs():
    df = _make_split_df()
    sequences = build_kronos_rich_sequences(
        df=df,
        feature_cols=["open", "high", "low", "close", "volume"],
        sequence_length=3,
        prediction_horizon=2,
        ohlcv_columns=["open", "high", "low", "close", "volume"],
    )

    assert sequences["features"].shape == (5, 3, 5)
    assert sequences["future_ohlcv"].shape == (5, 2, 5)
    assert sequences["future_close_path"].shape == (5, 2)
    assert sequences["future_return_path"].shape == (5, 2)
    assert sequences["future_volatility"].shape == (5,)
    assert sequences["future_regime"].shape == (5,)
    assert np.allclose(sequences["future_close_path"][0], [13.0, 14.0])
    assert np.allclose(sequences["future_ohlcv_mask"], 1.0)


def test_build_kronos_rich_sequences_can_disable_optional_targets():
    df = _make_split_df()
    sequences = build_kronos_rich_sequences(
        df=df,
        feature_cols=["open", "high", "low", "close", "volume"],
        sequence_length=3,
        prediction_horizon=2,
        ohlcv_columns=["open", "high", "low", "close", "volume"],
        include_scalar_target=False,
        include_return_path=False,
        include_regime_label=False,
    )

    assert np.allclose(sequences["target"], 0.0)
    assert np.allclose(sequences["future_return_path"], 0.0)
    assert np.all(sequences["future_regime"] == 0)


def test_assign_future_regime_labels_uses_realized_volatility_fallback():
    labels = assign_future_regime_labels(
        future_volatility=np.array([0.01, 0.03, 0.09], dtype=np.float32),
        existing_regime=None,
        include_regime_label=True,
        regime_source="column_or_realized_volatility",
        low_quantile=0.33,
        high_quantile=0.66,
    )

    assert np.array_equal(labels, np.array([0, 1, 2], dtype=np.int64))


def test_kronos_rich_prep_writes_outputs(tmp_path):
    output_dir = tmp_path / "processed_kronos_rich"
    sequences = build_kronos_rich_sequences(
        df=_make_split_df(),
        feature_cols=["open", "high", "low", "close", "volume"],
        sequence_length=3,
        prediction_horizon=2,
        ohlcv_columns=["open", "high", "low", "close", "volume"],
    )
    logger = type("Logger", (), {"info": lambda self, msg: None})()

    save_kronos_rich_split(output_dir, "train", sequences, logger)
    write_kronos_rich_metadata(
        output_dir=output_dir,
        metadata={
            "num_features": 5,
            "num_stocks": 1,
            "num_groups": 2,
            "sequence_length": 3,
            "prediction_horizon": 2,
            "ohlcv_columns": ["open", "high", "low", "close", "volume"],
            "include_return_path": True,
        },
        feature_cols=["open", "high", "low", "close", "volume"],
        logger=logger,
    )

    assert (output_dir / "train" / "future_ohlcv.npy").exists()
    assert (output_dir / "train" / "future_return_path.npy").exists()
    assert (output_dir / "train" / "future_regime.npy").exists()
    info = json.loads((output_dir / "info.json").read_text(encoding="utf-8"))
    assert info["ohlcv_columns"] == ["open", "high", "low", "close", "volume"]
