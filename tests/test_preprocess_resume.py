import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.config.config_loader import Config


def _load_preprocess_script():
    script_path = Path(__file__).parent.parent / "scripts" / "preprocess_data.py"
    spec = importlib.util.spec_from_file_location("test_preprocess_data_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _make_args(tmp_path: Path):
    return SimpleNamespace(
        start_date="2000-01-01",
        end_date=None,
        tickers=None,
        stock_limit=None,
        stocks=None,
        config=None,
        skip_download=True,
        output_dir=str(tmp_path / "processed"),
        export_pre_normalize=str(tmp_path / "pre_normalized.parquet"),
        export_normalized=str(tmp_path / "normalized.parquet"),
        no_resume_cache=False,
        skip_sequences=False,
    )


def _make_stock_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "tic": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "open": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            "high": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3],
            "low": [0.9, 1.0, 1.1, 1.9, 2.0, 2.1],
            "close": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            "volume": [10, 11, 12, 20, 21, 22],
        }
    )


def _make_feature_df():
    df = _make_stock_df().copy()
    df["day"] = pd.to_datetime(df["date"]).dt.day.astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype(int)
    df["feat1"] = np.linspace(0.1, 0.6, len(df))
    df["target"] = np.linspace(-0.2, 0.3, len(df))
    df["tic_id"] = [0, 0, 0, 1, 1, 1]
    df["group_id"] = [0, 0, 0, 0, 0, 0]
    df["dividend_flag"] = [1, 1, 1, 1, 1, 1]
    return df


def _make_splits():
    feature_df = _make_feature_df()
    train = feature_df.iloc[:2].copy()
    val = feature_df.iloc[2:4].copy()
    test = feature_df.iloc[4:].copy()
    return {"train": train, "val": val, "test": test}


def _make_sequences():
    return {
        "features": np.ones((1, 2, 1), dtype=np.float32),
        "stock_id": np.zeros((1, 2), dtype=np.int64),
        "group_id": np.zeros((1, 2), dtype=np.int64),
        "day": np.array([[1, 2]], dtype=np.int32),
        "month": np.array([[1, 1]], dtype=np.int32),
        "dividend_flag": np.ones((1, 2), dtype=np.int32),
        "target": np.array([0.5], dtype=np.float32),
    }


@pytest.mark.parametrize(
    "failure_stage,expected_after_second_run",
    [
        ("feature", {"downloader": 2, "feature": 2, "preprocess_tabular": 1}),
        ("normalized", {"downloader": 1, "feature": 1, "preprocess_tabular": 2}),
        ("sequence", {"downloader": 1, "feature": 1, "preprocess_tabular": 1}),
    ],
)
def test_preprocess_resume_recovers_from_failed_stage(tmp_path, monkeypatch, failure_stage, expected_after_second_run):
    module = _load_preprocess_script()
    args = _make_args(tmp_path)
    main_config = Config(copy.deepcopy(load_config("main").to_dict()))
    main_config.data.features.FEATURE_FLAGS._data["market_regime"] = False
    main_config.data.regime._data["ENABLED"] = False
    main_config.data.dataset._data["MODE"] = "legacy_sequence_arrays"

    counts = {
        "downloader": 0,
        "feature": 0,
        "preprocess_tabular": 0,
        "create_sequences": 0,
    }
    state = {"failed_once": False}

    class FakeDownloader:
        def __init__(self, _config):
            pass

        def load_saved_data(self):
            counts["downloader"] += 1
            return {
                "stocks": _make_stock_df(),
                "vix": pd.DataFrame(),
                "commodities": pd.DataFrame(),
                "treasury_yields": pd.DataFrame(),
            }

    class FakeFeatureEngineer:
        def __init__(self, _config):
            pass

        def add_all_features(self, *args_, **kwargs_):
            counts["feature"] += 1
            if failure_stage == "feature" and not state["failed_once"]:
                state["failed_once"] = True
                raise RuntimeError("feature stage failure")
            return _make_feature_df()

        def get_feature_info(self, _df):
            return {
                "total_features": 1,
                "price_features": 1,
                "ema_features": 0,
                "rsi_features": 0,
                "candlestick_patterns": 0,
            }

    class FakePreprocessor:
        def __init__(self, _config):
            pass

        def preprocess_tabular(self, *args_, **kwargs_):
            counts["preprocess_tabular"] += 1
            if failure_stage == "normalized" and not state["failed_once"]:
                state["failed_once"] = True
                raise RuntimeError("normalized stage failure")
            return _make_feature_df(), _make_splits(), {
                "num_stocks": 2,
                "num_groups": 1,
                "num_features": 1,
                "sequence_length": 2,
                "prediction_horizon": 1,
                "feature_cols": ["feat1"],
                "regime_params": None,
            }

        def create_sequences(self, split_df, feature_cols):
            counts["create_sequences"] += 1
            if failure_stage == "sequence" and not state["failed_once"]:
                state["failed_once"] = True
                raise RuntimeError("sequence stage failure")
            return _make_sequences()

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", lambda _name: main_config)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "log_sequence_preview", lambda *a, **k: None)
    monkeypatch.setattr(module, "DataDownloader", FakeDownloader)
    monkeypatch.setattr(module, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(module, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(module, "_build_pipeline_fingerprint", lambda _args, _config: "fixed-fingerprint")

    with pytest.raises(RuntimeError):
        module.main()

    manifest_path = Path(args.output_dir) / ".cache" / "preprocess_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    completed = manifest["completed_stages"]

    if failure_stage == "feature":
        assert completed["sampled_input"] is True
        assert "feature_engineered" not in completed
    elif failure_stage == "normalized":
        assert completed["sampled_input"] is True
        assert completed["feature_engineered"] is True
        assert "normalized_splits" not in completed
    else:
        assert completed["sampled_input"] is True
        assert completed["feature_engineered"] is True
        assert completed["normalized_splits"] is True
        assert "sequence_arrays" not in completed

    result = module.main()
    assert result == 0

    assert counts["downloader"] == expected_after_second_run["downloader"]
    assert counts["feature"] == expected_after_second_run["feature"]
    assert counts["preprocess_tabular"] == expected_after_second_run["preprocess_tabular"]

    final_info = Path(args.output_dir) / "info.json"
    final_feature_cols = Path(args.output_dir) / "feature_columns.txt"
    final_train_features = Path(args.output_dir) / "train" / "features.npy"
    assert final_info.exists()
    assert final_feature_cols.exists()
    assert final_train_features.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["completed_stages"]["sequence_arrays"] is True


def test_preprocess_resume_skips_all_completed_stages_on_rerun(tmp_path, monkeypatch):
    module = _load_preprocess_script()
    args = _make_args(tmp_path)
    main_config = Config(copy.deepcopy(load_config("main").to_dict()))
    main_config.data.features.FEATURE_FLAGS._data["market_regime"] = False
    main_config.data.regime._data["ENABLED"] = False
    main_config.data.dataset._data["MODE"] = "legacy_sequence_arrays"

    counts = {
        "downloader": 0,
        "feature": 0,
        "preprocess_tabular": 0,
        "create_sequences": 0,
    }

    class FakeDownloader:
        def __init__(self, _config):
            pass

        def load_saved_data(self):
            counts["downloader"] += 1
            return {
                "stocks": _make_stock_df(),
                "vix": pd.DataFrame(),
                "commodities": pd.DataFrame(),
                "treasury_yields": pd.DataFrame(),
            }

    class FakeFeatureEngineer:
        def __init__(self, _config):
            pass

        def add_all_features(self, *args_, **kwargs_):
            counts["feature"] += 1
            return _make_feature_df()

        def get_feature_info(self, _df):
            return {
                "total_features": 1,
                "price_features": 1,
                "ema_features": 0,
                "rsi_features": 0,
                "candlestick_patterns": 0,
            }

    class FakePreprocessor:
        def __init__(self, _config):
            pass

        def preprocess_tabular(self, *args_, **kwargs_):
            counts["preprocess_tabular"] += 1
            return _make_feature_df(), _make_splits(), {
                "num_stocks": 2,
                "num_groups": 1,
                "num_features": 1,
                "sequence_length": 2,
                "prediction_horizon": 1,
                "feature_cols": ["feat1"],
                "regime_params": None,
            }

        def create_sequences(self, split_df, feature_cols):
            counts["create_sequences"] += 1
            return _make_sequences()

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", lambda _name: main_config)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "log_sequence_preview", lambda *a, **k: None)
    monkeypatch.setattr(module, "DataDownloader", FakeDownloader)
    monkeypatch.setattr(module, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(module, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(module, "_build_pipeline_fingerprint", lambda _args, _config: "fixed-fingerprint")

    assert module.main() == 0
    assert counts == {
        "downloader": 1,
        "feature": 1,
        "preprocess_tabular": 1,
        "create_sequences": 3,
    }

    assert module.main() == 0
    assert counts == {
        "downloader": 1,
        "feature": 1,
        "preprocess_tabular": 1,
        "create_sequences": 3,
    }


def test_preprocess_skip_sequences_writes_metadata_only(tmp_path, monkeypatch):
    module = _load_preprocess_script()
    args = _make_args(tmp_path)
    args.skip_sequences = True
    main_config = Config(copy.deepcopy(load_config("main").to_dict()))
    main_config.data.features.FEATURE_FLAGS._data["market_regime"] = False
    main_config.data.regime._data["ENABLED"] = False

    counts = {
        "downloader": 0,
        "feature": 0,
        "preprocess_tabular": 0,
        "create_sequences": 0,
    }

    class FakeDownloader:
        def __init__(self, _config):
            pass

        def load_saved_data(self):
            counts["downloader"] += 1
            return {
                "stocks": _make_stock_df(),
                "vix": pd.DataFrame(),
                "commodities": pd.DataFrame(),
                "treasury_yields": pd.DataFrame(),
            }

    class FakeFeatureEngineer:
        def __init__(self, _config):
            pass

        def add_all_features(self, *args_, **kwargs_):
            counts["feature"] += 1
            return _make_feature_df()

        def get_feature_info(self, _df):
            return {
                "total_features": 1,
                "price_features": 1,
                "ema_features": 0,
                "rsi_features": 0,
                "candlestick_patterns": 0,
            }

    class FakePreprocessor:
        def __init__(self, _config):
            pass

        def preprocess_tabular(self, *args_, **kwargs_):
            counts["preprocess_tabular"] += 1
            return _make_feature_df(), _make_splits(), {
                "num_stocks": 2,
                "num_groups": 1,
                "num_features": 1,
                "sequence_length": 2,
                "prediction_horizon": 1,
                "feature_cols": ["feat1"],
                "regime_params": None,
            }

        def create_sequences(self, split_df, feature_cols):
            counts["create_sequences"] += 1
            return _make_sequences()

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", lambda _name: main_config)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "log_sequence_preview", lambda *a, **k: None)
    monkeypatch.setattr(module, "DataDownloader", FakeDownloader)
    monkeypatch.setattr(module, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(module, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(module, "_build_pipeline_fingerprint", lambda _args, _config: "fixed-fingerprint")

    assert module.main() == 0

    assert counts == {
        "downloader": 1,
        "feature": 1,
        "preprocess_tabular": 1,
        "create_sequences": 0,
    }


def test_preprocess_on_the_fly_mode_skips_sequence_arrays(tmp_path, monkeypatch):
    module = _load_preprocess_script()
    args = _make_args(tmp_path)
    main_config = Config(copy.deepcopy(load_config("main").to_dict()))
    main_config.data.features.FEATURE_FLAGS._data["market_regime"] = False
    main_config.data.regime._data["ENABLED"] = False
    main_config.data.dataset._data["MODE"] = "on_the_fly_sequences"

    counts = {
        "downloader": 0,
        "feature": 0,
        "preprocess_tabular": 0,
        "create_sequences": 0,
    }

    class FakeDownloader:
        def __init__(self, _config):
            pass

        def load_saved_data(self):
            counts["downloader"] += 1
            return {
                "stocks": _make_stock_df(),
                "vix": pd.DataFrame(),
                "commodities": pd.DataFrame(),
                "treasury_yields": pd.DataFrame(),
            }

    class FakeFeatureEngineer:
        def __init__(self, _config):
            pass

        def add_all_features(self, *args_, **kwargs_):
            counts["feature"] += 1
            return _make_feature_df()

        def get_feature_info(self, _df):
            return {
                "total_features": 1,
                "price_features": 1,
                "ema_features": 0,
                "rsi_features": 0,
                "candlestick_patterns": 0,
            }

    class FakePreprocessor:
        def __init__(self, _config):
            pass

        def preprocess_tabular(self, *args_, **kwargs_):
            counts["preprocess_tabular"] += 1
            return _make_feature_df(), _make_splits(), {
                "num_stocks": 2,
                "num_groups": 1,
                "num_features": 1,
                "sequence_length": 2,
                "prediction_horizon": 1,
                "feature_cols": ["feat1"],
                "regime_params": None,
            }

        def create_sequences(self, split_df, feature_cols):
            counts["create_sequences"] += 1
            return _make_sequences()

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", lambda _name: main_config)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "log_sequence_preview", lambda *a, **k: None)
    monkeypatch.setattr(module, "DataDownloader", FakeDownloader)
    monkeypatch.setattr(module, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(module, "DataPreprocessor", FakePreprocessor)
    monkeypatch.setattr(module, "_build_pipeline_fingerprint", lambda _args, _config: "fixed-fingerprint")

    assert module.main() == 0
    assert counts == {
        "downloader": 1,
        "feature": 1,
        "preprocess_tabular": 1,
        "create_sequences": 0,
    }

    output_dir = Path(args.output_dir)
    assert (output_dir / "info.json").exists()
    assert not (output_dir / "train" / "features.npy").exists()

    output_dir = Path(args.output_dir)
    assert (output_dir / "info.json").exists()
    assert (output_dir / "feature_columns.txt").exists()
    assert not (output_dir / "train" / "features.npy").exists()

    manifest_path = output_dir / ".cache" / "preprocess_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["completed_stages"]["normalized_splits"] is True
    assert "sequence_arrays" not in manifest["completed_stages"]

    assert module.main() == 0
    assert counts == {
        "downloader": 1,
        "feature": 1,
        "preprocess_tabular": 1,
        "create_sequences": 0,
    }
