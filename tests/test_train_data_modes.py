import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config import load_config
from src.config.config_loader import Config


def _load_train_script():
    script_path = Path(__file__).parent.parent / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("test_train_script", script_path)
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
        model_type="bilstm4_attention",
        data_dir=str(tmp_path / "processed"),
        config=None,
        epochs=None,
        batch_size=None,
        lr=None,
        device="cpu",
        force_cpu=True,
        resume=None,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        stocks=None,
        fine_tune=None,
        freeze_embeddings=False,
        backend="lightning",
    )


def _make_sequences():
    import numpy as np

    return {
        "features": np.ones((2, 3, 1), dtype=np.float32),
        "stock_id": np.zeros((2, 3), dtype=np.int64),
        "group_id": np.zeros((2, 3), dtype=np.int64),
        "day": np.ones((2, 3), dtype=np.int32),
        "month": np.ones((2, 3), dtype=np.int32),
        "dividend_flag": np.ones((2, 3), dtype=np.int32),
        "target": np.array([0.1, 0.2], dtype=np.float32),
    }


def test_train_uses_precomputed_sequences_mode_as_eager_build(tmp_path, monkeypatch):
    module = _load_train_script()
    args = _make_args(tmp_path)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "feature_cols": ["feat1"],
        "num_features": 1,
        "num_stocks": 2,
        "num_groups": 1,
        "normalize_target": True,
        "target_threshold": 2.0,
    }
    (data_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    model_config = Config(copy.deepcopy(load_config("model").to_dict()))
    main_config = Config(copy.deepcopy(load_config("main").to_dict()))
    main_config.data.dataset._data["MODE"] = "precomputed_sequences"

    called = {"eager_build": 0, "classic_loaders": 0}

    class DummyDataset:
        num_features = 1

        @staticmethod
        def get_embedding_sizes():
            return {
                "num_stocks": 2,
                "num_groups": 1,
                "num_days": 32,
                "num_months": 13,
                "num_dividend_flags": 3,
            }

    class DummyLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    def fake_load_config(name):
        if name == "model":
            return model_config
        if name == "main":
            return main_config
        raise AssertionError(f"Unexpected config name: {name}")

    def fake_load_sequences(*_args, **_kwargs):
        return None

    def fail_lazy_loader(**_kwargs):
        raise AssertionError("create_lazy_data_loaders should not be called in precomputed mode")

    def fake_build_sequences_from_normalized_splits(**_kwargs):
        called["eager_build"] += 1
        sequences = _make_sequences()
        return sequences, sequences

    def fake_create_data_loaders(**_kwargs):
        called["classic_loaders"] += 1
        return {"train": DummyLoader(), "val": DummyLoader()}

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", fake_load_config)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "get_device", lambda **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(module, "get_device_info", lambda **_kwargs: {"cuda_available": False})
    monkeypatch.setattr(module, "build_sequences_from_normalized_splits", fake_build_sequences_from_normalized_splits)
    monkeypatch.setattr(module, "create_data_loaders", fake_create_data_loaders)
    monkeypatch.setattr(module, "create_lazy_data_loaders", fail_lazy_loader)
    monkeypatch.setattr(module, "load_sequences", fake_load_sequences)
    monkeypatch.setattr(module, "load_ticker_mapping", lambda _data_dir: {})
    monkeypatch.setattr(module, "log_sequence_preview", lambda *a, **k: None)
    monkeypatch.setattr(module, "create_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        module,
        "train_with_lightning",
        lambda **_kwargs: {"best_score": 0.1, "best_model_path": "best.pth", "trainer": object(), "module": object()},
    )
    monkeypatch.setattr(module, "save_final_lightning_checkpoint", lambda **_kwargs: "final.pth")

    assert module.main() == 0
    assert called["eager_build"] == 1
    assert called["classic_loaders"] == 1


def test_train_uses_on_the_fly_sequences_mode_as_lazy_streaming(tmp_path, monkeypatch):
    module = _load_train_script()
    args = _make_args(tmp_path)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / ".cache" / "normalized_splits").mkdir(parents=True, exist_ok=True)

    info = {
        "feature_cols": ["feat1"],
        "num_features": 1,
        "num_stocks": 2,
        "num_groups": 1,
        "normalize_target": True,
        "target_threshold": 2.0,
    }
    (data_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    model_config = Config(copy.deepcopy(load_config("model").to_dict()))
    main_config = Config(copy.deepcopy(load_config("main").to_dict()))
    main_config.data.dataset._data["MODE"] = "on_the_fly_sequences"

    called = {"load_splits": 0, "lazy_loaders": 0}

    class DummyDataset:
        num_features = 1

        @staticmethod
        def get_embedding_sizes():
            return {
                "num_stocks": 2,
                "num_groups": 1,
                "num_days": 32,
                "num_months": 13,
                "num_dividend_flags": 3,
            }

    class DummyLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    class DummyFrame:
        def __init__(self, rows=2):
            self._rows = rows
            self.empty = False

        def __len__(self):
            return self._rows

    def fake_load_config(name):
        if name == "model":
            return model_config
        if name == "main":
            return main_config
        raise AssertionError(f"Unexpected config name: {name}")

    def fail_load_sequences(*_args, **_kwargs):
        raise AssertionError("load_sequences should not be called in lazy on_the_fly mode")

    def fail_build_sequences_from_normalized_splits(**_kwargs):
        raise AssertionError("build_sequences_from_normalized_splits should not be called in on_the_fly mode")

    def fake_load_normalized_splits_for_training(**_kwargs):
        called["load_splits"] += 1
        frame = DummyFrame()
        return frame, frame

    def fake_create_lazy_data_loaders(**_kwargs):
        called["lazy_loaders"] += 1
        return {"train": DummyLoader(), "val": DummyLoader()}

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", fake_load_config)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "get_device", lambda **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(module, "get_device_info", lambda **_kwargs: {"cuda_available": False})
    monkeypatch.setattr(module, "load_normalized_splits_for_training", fake_load_normalized_splits_for_training)
    monkeypatch.setattr(module, "build_sequences_from_normalized_splits", fail_build_sequences_from_normalized_splits)
    monkeypatch.setattr(module, "create_lazy_data_loaders", fake_create_lazy_data_loaders)
    monkeypatch.setattr(module, "load_sequences", fail_load_sequences)
    monkeypatch.setattr(module, "load_ticker_mapping", lambda _data_dir: {})
    monkeypatch.setattr(module, "log_sequence_preview", lambda *a, **k: None)
    monkeypatch.setattr(module, "create_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        module,
        "train_with_lightning",
        lambda **_kwargs: {"best_score": 0.1, "best_model_path": "best.pth", "trainer": object(), "module": object()},
    )
    monkeypatch.setattr(module, "save_final_lightning_checkpoint", lambda **_kwargs: "final.pth")

    assert module.main() == 0
    assert called["load_splits"] == 1
    assert called["lazy_loaders"] == 1


def test_train_routes_kronos_through_unified_branch(tmp_path, monkeypatch):
    module = _load_train_script()
    args = _make_args(tmp_path)
    args.model_type = "kronos"
    args.backend = "custom"
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "feature_cols": ["feat1"],
        "num_features": 1,
        "num_stocks": 2,
        "num_groups": 1,
        "normalize_target": True,
        "target_threshold": 2.0,
    }
    (data_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    model_config = Config(copy.deepcopy(load_config("model").to_dict()))
    main_config = Config(copy.deepcopy(load_config("main").to_dict()))
    main_config.data.dataset._data["MODE"] = "precomputed_sequences"

    called = {"kronos": 0, "create_model": 0}

    class DummyDataset:
        num_features = 1

        @staticmethod
        def get_embedding_sizes():
            return {
                "num_stocks": 2,
                "num_groups": 1,
                "num_days": 32,
                "num_months": 13,
                "num_dividend_flags": 3,
            }

    class DummyLoader:
        def __init__(self):
            self.dataset = DummyDataset()

    def fake_load_config(name):
        if name == "model":
            return model_config
        if name == "main":
            return main_config
        raise AssertionError(f"Unexpected config name: {name}")

    def fake_load_sequences(*_args, **_kwargs):
        return _make_sequences()

    def fake_create_data_loaders(**_kwargs):
        return {"train": DummyLoader(), "val": DummyLoader()}

    def fake_train_kronos(**_kwargs):
        called["kronos"] += 1
        return {
            "best_score": 0.2,
            "best_model_path": "kronos_best.pth",
            "final_model_path": "kronos_final.pth",
        }

    def fail_create_model(**_kwargs):
        called["create_model"] += 1
        raise AssertionError("Generic create_model should not be called for kronos")

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", fake_load_config)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "get_device", lambda **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(module, "get_device_info", lambda **_kwargs: {"cuda_available": False})
    monkeypatch.setattr(module, "load_sequences", fake_load_sequences)
    monkeypatch.setattr(module, "create_data_loaders", fake_create_data_loaders)
    monkeypatch.setattr(module, "load_ticker_mapping", lambda _data_dir: {})
    monkeypatch.setattr(module, "log_sequence_preview", lambda *a, **k: None)
    monkeypatch.setattr(module, "train_kronos", fake_train_kronos)
    monkeypatch.setattr(module, "create_model", fail_create_model)

    assert module.main() == 0
    assert called["kronos"] == 1
    assert called["create_model"] == 0
