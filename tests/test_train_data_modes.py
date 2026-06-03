import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _load_script(script_name: str, module_name: str):
    script_path = Path(__file__).parent.parent / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
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


class _DummyEvalDataset:
    num_features = 1

    def __init__(self, *_args, **_kwargs):
        pass

    @staticmethod
    def get_embedding_sizes():
        return {
            "num_stocks": 1,
            "num_groups": 2,
            "num_days": 32,
            "num_months": 13,
            "num_dividend_flags": 3,
        }


class _DummyEvalModel:
    def load_state_dict(self, _state):
        return None

    def to(self, _device):
        return self


class _DummyBacktester:
    def __init__(self, *_args, **_kwargs):
        pass

    def run_backtest(self, *_args, **_kwargs):
        return {"summary": {}}

    def _print_backtest_summary(self, *_args, **_kwargs):
        return None

    def _print_sector_stats(self, *_args, **_kwargs):
        return None

    def generate_report(self, *_args, **_kwargs):
        return None


def test_limited_loader_stops_after_requested_batches(tmp_path):
    module = _load_train_script()

    class DummyLoader:
        dataset = object()

        def __iter__(self):
            for idx in range(5):
                yield idx

        def __len__(self):
            return 5

    loader = module.LimitedLoader(DummyLoader(), limit=2)
    assert list(loader) == [0, 1]
    assert len(loader) == 2


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


@pytest.mark.parametrize("model_type", ["kronos", "kronos_rich"])
def test_train_routes_kronos_through_unified_branch(tmp_path, monkeypatch, model_type):
    module = _load_train_script()
    args = _make_args(tmp_path)
    args.model_type = model_type
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
            "best_model_path": f"{model_type}_best.pth",
            "final_model_path": f"{model_type}_final.pth",
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


def test_train_kronos_uses_shared_optimizer_factory(tmp_path, monkeypatch):
    module = _load_train_script()
    args = _make_args(tmp_path)
    args.model_type = "kronos"
    args.backend = "custom"
    args.max_train_batches = 1
    args.max_val_batches = 1

    model_config = Config(copy.deepcopy(load_config("model").to_dict()))
    model_config.model.training._data["NUM_EPOCHS"] = 1
    model_config.model.training._data["EARLY_STOPPING_PATIENCE"] = 1
    model_config.model.checkpointing._data["CHECKPOINT_DIR"] = str(tmp_path / "checkpoints")

    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, _device):
            return self

    class DummyScheduler:
        def step(self, *_args, **_kwargs):
            return None

    class DummyEarlyStopping:
        def __init__(self, *args, **kwargs):
            self.early_stop = False

        def __call__(self, *args, **kwargs):
            return False

    checkpoint_calls = []
    captured = {}

    monkeypatch.setattr(module, "create_kronos_tokenizer", lambda config: DummyModule())

    def fake_create_kronos_model(config, num_stocks=None, num_groups=None):
        captured["num_stocks"] = num_stocks
        captured["num_groups"] = num_groups
        return DummyModule()

    monkeypatch.setattr(module, "create_kronos_model", fake_create_kronos_model)

    def fake_create_optimizer(params, config):
        params = list(params)
        captured["optimizer_param_count"] = len(params)
        return torch.optim.Adam(params, lr=config.model.training.LEARNING_RATE)

    monkeypatch.setattr(module, "create_optimizer_for_params", fake_create_optimizer)
    monkeypatch.setattr(module, "create_scheduler", lambda optimizer, config: DummyScheduler())
    monkeypatch.setattr(module, "EarlyStopping", DummyEarlyStopping)
    monkeypatch.setattr(module, "_train_kronos_epoch", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(module, "_validate_kronos_epoch", lambda *args, **kwargs: 0.4)
    monkeypatch.setattr(
        module,
        "_save_kronos_checkpoint",
        lambda checkpoint_path, tokenizer, model, optimizer, epoch, metric, model_type, checkpoint_metadata=None, **kwargs: checkpoint_calls.append(
            (str(checkpoint_path), metric, model_type)
        ),
    )

    result = module.train_kronos(
        loaders={"train": object(), "val": object()},
        config=model_config,
        device=torch.device("cpu"),
        model_type="kronos",
        checkpoint_metadata={"feature_cols": ["close"]},
        logger=_FakeLogger(),
        num_features=5,
        embedding_sizes={"num_stocks": 7, "num_groups": 3},
        args=args,
    )

    assert captured["num_stocks"] == 7


@pytest.mark.parametrize(
    ("script_name", "split"),
    [("test.py", "test"), ("validate.py", "val"), ("backtest.py", "test")],
)
def test_eval_scripts_route_kronos_rich_to_kronos_loader(tmp_path, monkeypatch, script_name, split):
    module = _load_script(script_name, f"test_{script_name.replace('.py', '')}_kronos_rich")
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "info.json").write_text(
        json.dumps(
            {
                "feature_cols": ["close"],
                "num_features": 1,
                "num_stocks": 3,
                "num_groups": 2,
                "sequence_length": 3,
                "prediction_horizon": 2,
                "normalize_target": False,
                "target_threshold": 1.0,
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        model="best",
        model_type="kronos_rich",
        data_dir=str(data_dir),
        raw_data_dir=None,
        split=split,
        device="cpu",
        force_cpu=True,
        excel_report=None,
        output=None if script_name != "backtest.py" else str(tmp_path / "report.xlsx"),
        max_samples=4,
        output_format="excel",
        threshold=0.0,
        initial_capital=1000.0,
    )

    captured = {}

    class DummyLoader:
        dataset = _DummyEvalDataset()

    model_config = Config(copy.deepcopy(load_config("model").to_dict()))

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_config", lambda name: model_config if name == "model" else load_config(name))
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "resolve_device", lambda **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(module, "get_device_info", lambda **_kwargs: {"cuda_available": False})
    monkeypatch.setattr(
        module,
        "load_sequences",
        lambda *_args, **_kwargs: {
            "features": torch.ones((2, 3, 1), dtype=torch.float32).numpy(),
            "stock_id": torch.zeros((2, 3), dtype=torch.int64).numpy(),
            "group_id": torch.zeros((2, 3), dtype=torch.int64).numpy(),
            "day": torch.ones((2, 3), dtype=torch.int32).numpy(),
            "month": torch.ones((2, 3), dtype=torch.int32).numpy(),
            "dividend_flag": torch.ones((2, 3), dtype=torch.int32).numpy(),
            "target": torch.tensor([0.1, -0.2], dtype=torch.float32).numpy(),
        },
    )
    monkeypatch.setattr(module, "FinancialDataset", _DummyEvalDataset)
    monkeypatch.setattr(module.torch.utils.data, "DataLoader", lambda *a, **k: DummyLoader())
    monkeypatch.setattr(module, "find_checkpoint_path", lambda **_kwargs: str(tmp_path / "kronos_rich_best.pth"))
    monkeypatch.setattr(module, "build_kronos_sequence_metadata", lambda **_kwargs: {"x_dates": np.empty((2, 3), dtype=object), "y_dates": np.empty((2, 2), dtype=object)})

    def fake_load_kronos_checkpoint(**kwargs):
        captured["model_type"] = kwargs["model_type"]
        return object(), object(), {"epoch": 1}

    monkeypatch.setattr(module, "load_kronos_checkpoint", fake_load_kronos_checkpoint)
    monkeypatch.setattr(
        module,
        "generate_kronos_predictions",
        lambda **kwargs: (
            np.array([0.1, -0.1], dtype=np.float32),
            np.array([0.2, -0.2], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0.1, -0.1], dtype=np.float32),
            np.array([0.2, -0.2], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(module, "compute_kronos_metrics", lambda predictions, targets: {"mae": 0.1}, raising=False)
    monkeypatch.setattr(module, "print_metrics", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(module, "build_kronos_report", lambda *args, **kwargs: (_DummyReport(), {}), raising=False)
    monkeypatch.setattr(module, "print_sector_stats", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(module, "load_checkpoint_metadata", lambda *args, **kwargs: {"model_state_dict": {}, "epoch": 1})
    monkeypatch.setattr(module, "create_model", lambda **_kwargs: _DummyEvalModel())
    monkeypatch.setattr(module, "Backtester", _DummyBacktester, raising=False)
    monkeypatch.setattr(module, "compute_kronos_backtest_results", lambda **_kwargs: {"sector_stats": {}, "final_capital": 1000.0}, raising=False)
    monkeypatch.setattr(module, "load_id_mappings", lambda *args, **kwargs: ({}, {}), raising=False)

    class _DummyReport:
        def to_excel(self, *_args, **_kwargs):
            return None

    assert module.main() == 0
    assert captured["model_type"] == "kronos_rich"


def test_test_script_prefers_metadata_embedding_sizes_for_generic_models(tmp_path, monkeypatch):
    module = _load_script("test.py", "test_eval_script")
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "info.json").write_text(
        json.dumps(
            {
                "feature_cols": ["close"],
                "num_features": 1,
                "num_stocks": 150,
                "num_groups": 11,
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        model="best",
        model_type="chronos2",
        data_dir=str(data_dir),
        raw_data_dir=None,
        split="test",
        device="cpu",
        force_cpu=True,
        excel_report=None,
        output=None,
        max_samples=16,
    )

    captured = {}
    model_config = Config(copy.deepcopy(load_config("model").to_dict()))

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "resolve_device", lambda **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(module, "get_device_info", lambda **_kwargs: {"cuda_available": False, "cuda_working": False})
    monkeypatch.setattr(module, "load_config", lambda _name: model_config)
    monkeypatch.setattr(module, "load_sequences", lambda *_args, **_kwargs: _make_sequences())
    monkeypatch.setattr(module, "load_id_mappings", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(module, "FinancialDataset", _DummyEvalDataset)
    monkeypatch.setattr(module.torch.utils.data, "DataLoader", lambda dataset, **_kwargs: {"dataset": dataset})

    def fake_find_checkpoint_path(**kwargs):
        captured["find_checkpoint_path"] = kwargs
        return str(tmp_path / "chronos2_best.pth")

    def fake_create_model(**kwargs):
        captured["create_model"] = kwargs
        return _DummyEvalModel()

    monkeypatch.setattr(module, "find_checkpoint_path", fake_find_checkpoint_path)
    monkeypatch.setattr(module, "create_model", fake_create_model)
    monkeypatch.setattr(module, "load_checkpoint_metadata", lambda *_args, **_kwargs: {"model_state_dict": {}, "epoch": 1})
    monkeypatch.setattr(module, "evaluate_model", lambda *_args, **_kwargs: {"loss": 0.1})
    monkeypatch.setattr(module, "print_metrics", lambda *_args, **_kwargs: None)

    assert module.main() == 0
    assert captured["find_checkpoint_path"]["num_stocks"] == 150
    assert captured["find_checkpoint_path"]["num_groups"] == 11
    assert captured["create_model"]["num_stocks"] == 150
    assert captured["create_model"]["num_groups"] == 11


def test_backtest_script_prefers_metadata_embedding_sizes_for_generic_models(tmp_path, monkeypatch):
    module = _load_script("backtest.py", "test_backtest_script")
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "info.json").write_text(
        json.dumps(
            {
                "feature_cols": ["close"],
                "num_features": 1,
                "num_stocks": 150,
                "num_groups": 11,
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        model="best",
        model_type="chronos2",
        data_dir=str(data_dir),
        raw_data_dir=None,
        split="test",
        device="cpu",
        force_cpu=True,
        output=str(tmp_path / "backtest.json"),
        output_format="json",
        threshold=0.1,
        initial_capital=10000.0,
        max_samples=16,
    )

    captured = {}
    model_config = Config(copy.deepcopy(load_config("model").to_dict()))

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(module, "resolve_device", lambda **_kwargs: torch.device("cpu"))
    monkeypatch.setattr(module, "get_device_info", lambda **_kwargs: {"cuda_available": False, "cuda_working": False})
    monkeypatch.setattr(module, "load_config", lambda _name: model_config)
    monkeypatch.setattr(module, "load_sequences", lambda *_args, **_kwargs: _make_sequences())
    monkeypatch.setattr(module, "load_id_mappings", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(module, "FinancialDataset", _DummyEvalDataset)
    monkeypatch.setattr(module.torch.utils.data, "DataLoader", lambda dataset, **_kwargs: {"dataset": dataset})

    def fake_find_checkpoint_path(**kwargs):
        captured["find_checkpoint_path"] = kwargs
        return str(tmp_path / "chronos2_best.pth")

    def fake_create_model(**kwargs):
        captured["create_model"] = kwargs
        return _DummyEvalModel()

    monkeypatch.setattr(module, "find_checkpoint_path", fake_find_checkpoint_path)
    monkeypatch.setattr(module, "create_model", fake_create_model)
    monkeypatch.setattr(module, "load_checkpoint_metadata", lambda *_args, **_kwargs: {"model_state_dict": {}, "epoch": 1})
    monkeypatch.setattr(module, "Backtester", _DummyBacktester)

    assert module.main() == 0
    assert captured["find_checkpoint_path"]["num_stocks"] == 150
    assert captured["find_checkpoint_path"]["num_groups"] == 11
    assert captured["create_model"]["num_stocks"] == 150
    assert captured["create_model"]["num_groups"] == 11


def test_kronos_epoch_loops_update_progress_bar(monkeypatch):
    module = _load_train_script()
    config = Config(copy.deepcopy(load_config("model").to_dict()))
    config.model.training._data["GRADIENT_CLIP_VALUE"] = 0.0

    class ProgressStub:
        instances = []

        def __init__(self, iterable, total=None, desc=None, leave=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.leave = leave
            self.postfixes = []
            ProgressStub.instances.append(self)

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, values):
            self.postfixes.append(values)

    class DummyTokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def train(self):
            return self

        def eval(self):
            return self

        def forward(self, features):
            z_pre = features * self.scale
            z_full = features * self.scale
            bsq_loss = (self.scale - 1.0).pow(2).sum()
            return (z_pre, z_full), bsq_loss, None, None

        def encode(self, features, half=True):
            batch, seq_len, _ = features.shape
            device = features.device
            zeros = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
            return zeros, zeros

    class DummyHead:
        def compute_loss(self, s1_logits, s2_logits, target_s1, target_s2, loss_fn=None):
            loss = s1_logits.mean() + s2_logits.mean()
            return loss, None, None

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.ones(1))
            self.head = DummyHead()

        def train(self):
            return self

        def eval(self):
            return self

        def forward(self, input_s1, input_s2, **_kwargs):
            batch, seq_len = input_s1.shape
            device = input_s1.device
            logits = self.bias * torch.ones((batch, seq_len, 2), device=device)
            return logits, logits

    monkeypatch.setattr(module, "tqdm", ProgressStub)

    batch = {
        "features": torch.ones((2, 4, 3), dtype=torch.float32),
        "stock_id": torch.zeros((2, 4), dtype=torch.long),
        "group_id": torch.zeros((2, 4), dtype=torch.long),
        "day": torch.ones((2, 4), dtype=torch.long),
        "month": torch.ones((2, 4), dtype=torch.long),
        "dividend_flag": torch.ones((2, 4), dtype=torch.long),
    }
    loader = [batch, batch]

    tokenizer = DummyTokenizer()
    model = DummyModel()
    optimizer = torch.optim.Adam(list(tokenizer.parameters()) + list(model.parameters()), lr=1e-3)

    train_loss = module._train_kronos_epoch(
        tokenizer=tokenizer,
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        config=config,
        max_batches=2,
        model_type="kronos",
    )

    val_loss = module._validate_kronos_epoch(
        tokenizer=tokenizer,
        model=model,
        val_loader=loader,
        device=torch.device("cpu"),
        config=config,
        max_batches=2,
        model_type="kronos",
    )

    assert train_loss > 0.0
    assert val_loss > 0.0
    assert len(ProgressStub.instances) == 2
    assert ProgressStub.instances[0].desc == "Kronos train"
    assert ProgressStub.instances[1].desc == "Kronos val"
    assert len(ProgressStub.instances[0].postfixes) == 2
    assert len(ProgressStub.instances[1].postfixes) == 2


def test_kronos_rich_uses_own_configured_loss_modules(tmp_path):
    module = _load_train_script()
    config = Config(copy.deepcopy(load_config("model").to_dict()))
    config.model.models.kronos_rich.RECON_LOSS_TYPE = "mae"
    config.model.models.kronos_rich.RECON_LOSS_WEIGHT = 2.0
    config.model.models.kronos_rich.PRE_LOSS_TYPE = "huber"
    config.model.models.kronos_rich.PRE_HUBER_DELTA = 0.25
    config.model.models.kronos_rich.PRE_LOSS_WEIGHT = 3.0
    config.model.models.kronos_rich.TOKEN_LOSS_TYPE = "cross_entropy"
    config.model.models.kronos_rich.TOKEN_LABEL_SMOOTHING = 0.2
    config.model.models.kronos_rich.TOKEN_LOSS_WEIGHT = 4.0
    config.model.models.kronos_rich.BSQ_LOSS_WEIGHT = 5.0

    loss_spec = module._build_kronos_loss_modules(config, "kronos_rich")

    assert loss_spec["recon_loss_fn"].__class__.__name__ == "L1Loss"
    assert loss_spec["pre_loss_fn"].__class__.__name__ == "HuberLoss"
    assert loss_spec["token_loss_fn"].__class__.__name__ == "CrossEntropyLossModule"
    assert loss_spec["token_loss_fn"].label_smoothing == 0.2
    assert loss_spec["recon_loss_weight"] == 2.0
    assert loss_spec["pre_loss_weight"] == 3.0
    assert loss_spec["token_loss_weight"] == 4.0
    assert loss_spec["bsq_loss_weight"] == 5.0
