import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_chronos2_data import (
    build_chronos2_sequences,
    ensure_base_preprocessing,
    generate_future_target,
    load_normalized_split_cache,
    load_preprocessing_metadata,
    resolve_chronos2_prep_settings,
    save_chronos2_split,
    write_chronos2_metadata,
)
from src.config import load_config


def _make_split_df():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "tic": ["AAA"] * len(dates),
            "tic_id": [0] * len(dates),
            "group_id": [1] * len(dates),
            "day": dates.day.astype(np.int32),
            "month": dates.month.astype(np.int32),
            "dividend_flag": np.ones(len(dates), dtype=np.int32),
            "close": np.array([10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0], dtype=np.float32),
            "volume": np.arange(100, 108, dtype=np.float32),
            "target": np.array([0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3], dtype=np.float32),
        }
    )


def test_resolve_chronos2_prep_settings_reads_config_defaults():
    config = load_config("main")
    args = type(
        "Args",
        (),
        {
            "processed_dir": None,
            "output_dir": None,
            "target_column": None,
            "skip_scalar_target": False,
        },
    )()

    settings = resolve_chronos2_prep_settings(config, args)

    assert settings["processed_dir"] == Path("data/processed")
    assert settings["output_dir"] == Path("data/processed_chronos2")
    assert settings["target_column"] == "close"
    assert settings["include_scalar_target"] is True
    assert settings["target_mode"] == "trend_extension"
    assert settings["trend_lookback"] == 7
    assert settings["trend_method"] == "mean_gap"


def test_build_chronos2_sequences_creates_future_target_paths():
    df = _make_split_df()

    sequences = build_chronos2_sequences(
        df=df,
        feature_cols=["close", "volume"],
        sequence_length=3,
        prediction_horizon=2,
        target_column="close",
        include_scalar_target=True,
    )

    assert sequences["features"].shape == (4, 3, 2)
    assert sequences["future_target"].shape == (4, 2)
    assert sequences["future_target_mask"].shape == (4, 2)
    assert np.allclose(sequences["future_target"][0], [12.0, 13.0])
    assert np.isclose(sequences["target"][0], 1.0)
    assert np.all(sequences["future_target_mask"] == 1.0)


def test_build_chronos2_sequences_can_skip_scalar_target():
    df = _make_split_df()

    sequences = build_chronos2_sequences(
        df=df,
        feature_cols=["close", "volume"],
        sequence_length=3,
        prediction_horizon=2,
        target_column="close",
        include_scalar_target=False,
    )

    assert np.allclose(sequences["target"], 0.0)


def test_generate_future_target_extends_positive_trend():
    future_target = generate_future_target(
        recent_values=np.array([3, 4, 5, 6, 7, 8, 9], dtype=np.float32),
        prediction_horizon=4,
    )

    assert np.allclose(future_target, [9.0, 10.0, 11.0, 12.0])


def test_generate_future_target_extends_negative_trend():
    future_target = generate_future_target(
        recent_values=np.array([9, 8, 7, 6, 5], dtype=np.float32),
        prediction_horizon=5,
    )

    assert np.allclose(future_target, [5.0, 4.0, 3.0, 2.0, 1.0])


def test_ensure_base_preprocessing_runs_preprocess_when_cache_missing(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    calls = {}

    class FakeModule:
        def __init__(self):
            self.parse_args = None

        def main(self):
            args = self.parse_args()
            calls["output_dir"] = args.output_dir
            calls["skip_sequences"] = args.skip_sequences
            processed_path = Path(args.output_dir)
            (processed_path / ".cache" / "normalized_splits").mkdir(parents=True, exist_ok=True)
            (processed_path / ".cache" / "normalized_splits" / "train.parquet").write_bytes(b"ok")
            (processed_path / "info.json").write_text("{}", encoding="utf-8")
            return 0

    monkeypatch.setattr("scripts.prepare_chronos2_data._load_preprocess_module", lambda: FakeModule())

    args = type(
        "Args",
        (),
        {
            "start_date": "2000-01-01",
            "end_date": None,
            "tickers": None,
            "stock_limit": None,
            "stocks": None,
            "config": None,
            "skip_download": True,
            "export_pre_normalize": "pre.parquet",
            "export_normalized": "norm.parquet",
            "no_resume_cache": False,
        },
    )()
    logger = type("Logger", (), {"info": lambda self, msg: None})()

    ensure_base_preprocessing(args, processed_dir, logger)

    assert calls["output_dir"] == str(processed_dir)
    assert calls["skip_sequences"] is True
    assert (processed_dir / "info.json").exists()


def test_ensure_base_preprocessing_honors_no_resume_cache(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    split_cache_dir = processed_dir / ".cache" / "normalized_splits"
    split_cache_dir.mkdir(parents=True)
    (processed_dir / "info.json").write_text("{}", encoding="utf-8")
    (split_cache_dir / "train.parquet").write_bytes(b"old")
    calls = {"count": 0}

    class FakeModule:
        def __init__(self):
            self.parse_args = None

        def main(self):
            args = self.parse_args()
            calls["count"] += 1
            processed_path = Path(args.output_dir)
            (processed_path / ".cache" / "normalized_splits" / "train.parquet").write_bytes(b"new")
            (processed_path / "info.json").write_text("{}", encoding="utf-8")
            return 0

    monkeypatch.setattr("scripts.prepare_chronos2_data._load_preprocess_module", lambda: FakeModule())

    args = type(
        "Args",
        (),
        {
            "start_date": "2000-01-01",
            "end_date": None,
            "tickers": ["AAPL", "MSFT"],
            "stock_limit": None,
            "stocks": None,
            "config": None,
            "skip_download": True,
            "export_pre_normalize": "pre.parquet",
            "export_normalized": "norm.parquet",
            "no_resume_cache": True,
        },
    )()
    logger = type("Logger", (), {"info": lambda self, msg: None})()

    ensure_base_preprocessing(args, processed_dir, logger)

    assert calls["count"] == 1
    assert (split_cache_dir / "train.parquet").read_bytes() == b"new"


def test_ensure_base_preprocessing_revalidates_existing_cache_via_preprocess_manifest(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    split_cache_dir = processed_dir / ".cache" / "normalized_splits"
    split_cache_dir.mkdir(parents=True)
    (processed_dir / "info.json").write_text("{}", encoding="utf-8")
    (split_cache_dir / "train.parquet").write_bytes(b"old")
    calls = {"count": 0}

    class FakeModule:
        def __init__(self):
            self.parse_args = None

        def main(self):
            args = self.parse_args()
            calls["count"] += 1
            processed_path = Path(args.output_dir)
            (processed_path / ".cache" / "normalized_splits" / "train.parquet").write_bytes(b"fresh")
            (processed_path / "info.json").write_text("{}", encoding="utf-8")
            return 0

    monkeypatch.setattr("scripts.prepare_chronos2_data._load_preprocess_module", lambda: FakeModule())

    args = type(
        "Args",
        (),
        {
            "start_date": "2015-01-01",
            "end_date": "2020-01-01",
            "tickers": ["AAPL"],
            "stock_limit": 10,
            "stocks": 5,
            "config": None,
            "skip_download": True,
            "export_pre_normalize": "pre.parquet",
            "export_normalized": "norm.parquet",
            "no_resume_cache": False,
        },
    )()
    logger = type("Logger", (), {"info": lambda self, msg: None})()

    ensure_base_preprocessing(args, processed_dir, logger)

    assert calls["count"] == 1
    assert (split_cache_dir / "train.parquet").read_bytes() == b"fresh"


def test_chronos2_prep_reads_cache_and_writes_outputs(tmp_path):
    processed_dir = tmp_path / "processed"
    split_cache_dir = processed_dir / ".cache" / "normalized_splits"
    split_cache_dir.mkdir(parents=True)

    info = {
        "num_stocks": 1,
        "num_groups": 2,
        "num_features": 2,
        "sequence_length": 3,
        "prediction_horizon": 2,
        "feature_cols": ["close", "volume"],
        "normalize_target": True,
        "target_threshold": 2.0,
    }
    (processed_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    df = _make_split_df()
    df.to_parquet(split_cache_dir / "train.parquet", index=False)

    metadata = load_preprocessing_metadata(processed_dir)
    splits = load_normalized_split_cache(processed_dir)
    sequences = build_chronos2_sequences(
        splits["train"],
        feature_cols=metadata["feature_cols"],
        sequence_length=metadata["sequence_length"],
        prediction_horizon=metadata["prediction_horizon"],
        target_column="close",
        include_scalar_target=True,
    )

    output_dir = tmp_path / "chronos2_processed"
    logger = type("Logger", (), {"info": lambda self, msg: None})()
    save_chronos2_split(output_dir, "train", sequences, logger)
    write_chronos2_metadata(
        output_dir,
        metadata={
            "num_features": 2,
            "num_stocks": 1,
            "num_groups": 2,
            "sequence_length": 3,
            "prediction_horizon": 2,
            "normalize_target": True,
            "target_threshold": 2.0,
            "target_column": "close",
            "include_scalar_target": True,
            "source_processed_dir": str(processed_dir),
        },
        feature_cols=["close", "volume"],
        logger=logger,
    )

    assert (output_dir / "train" / "future_target.npy").exists()
    assert (output_dir / "train" / "future_target_mask.npy").exists()
    assert (output_dir / "info.json").exists()

    future_target = np.load(output_dir / "train" / "future_target.npy", allow_pickle=False)
    assert future_target.shape == (4, 2)
