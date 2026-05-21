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
        lambda checkpoint_path, tokenizer, model, optimizer, epoch, metric, model_type, checkpoint_metadata=None: checkpoint_calls.append(
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
    assert captured["num_groups"] == 3
    assert captured["optimizer_param_count"] == 2
    assert result["best_score"] == pytest.approx(0.4)
    assert len(checkpoint_calls) == 2


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
        def compute_loss(self, s1_logits, s2_logits, target_s1, target_s2):
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
    )

    val_loss = module._validate_kronos_epoch(
        tokenizer=tokenizer,
        model=model,
        val_loader=loader,
        device=torch.device("cpu"),
        max_batches=2,
    )

    assert train_loss > 0.0
    assert val_loss > 0.0
    assert len(ProgressStub.instances) == 2
    assert ProgressStub.instances[0].desc == "Kronos train"
    assert ProgressStub.instances[1].desc == "Kronos val"
    assert len(ProgressStub.instances[0].postfixes) == 2
    assert len(ProgressStub.instances[1].postfixes) == 2
