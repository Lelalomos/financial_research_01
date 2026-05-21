import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from src.config.config_loader import Config


def _load_kronos_small_script():
    script_path = Path(__file__).parent.parent / "scripts" / "test_kronos_small.py"
    spec = importlib.util.spec_from_file_location("test_kronos_small_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_raw_model_config():
    config_path = Path(__file__).parent.parent / "config" / "model.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _make_sequences(samples=4, seq_len=5, features=6):
    return {
        "features": np.ones((samples, seq_len, features), dtype=np.float32),
        "stock_id": np.zeros((samples, seq_len), dtype=np.int64),
        "group_id": np.zeros((samples, seq_len), dtype=np.int64),
        "day": np.ones((samples, seq_len), dtype=np.int32),
        "month": np.ones((samples, seq_len), dtype=np.int32),
        "dividend_flag": np.ones((samples, seq_len), dtype=np.int32),
        "target": np.zeros(samples, dtype=np.float32),
    }


def test_build_local_model_config_sets_kronos_vocab_sizes():
    module = _load_kronos_small_script()
    sequences = _make_sequences()
    sequences["stock_id"][0, 0] = 7
    sequences["group_id"][0, 0] = 3

    config = module.build_local_model_config(sequences)

    assert "NUM_STOCKS" not in config.model.models.kronos.network
    assert "NUM_GROUPS" not in config.model.models.kronos.network
    assert config.model.models.kronos.tokenizer.D_IN == sequences["features"].shape[-1]


def test_infer_kronos_vocab_sizes_uses_sequence_ids():
    module = _load_kronos_small_script()
    sequences = _make_sequences()
    sequences["stock_id"][0, 0] = 7
    sequences["group_id"][0, 0] = 3

    num_stocks, num_groups = module.infer_kronos_vocab_sizes(sequences)

    assert num_stocks == 8
    assert num_groups == 4


def test_predict_next_step_returns_single_decoded_row():
    module = _load_kronos_small_script()
    sample = _make_sequences(samples=1, seq_len=4, features=6)

    class DummyTokenizer:
        def eval(self):
            return self

        def encode(self, features, half=True):
            batch, seq_len, _ = features.shape
            return (
                torch.zeros((batch, seq_len), dtype=torch.long, device=features.device),
                torch.zeros((batch, seq_len), dtype=torch.long, device=features.device),
            )

        def decode(self, token_ids, half=True):
            batch = token_ids[0].shape[0]
            seq_len = token_ids[0].shape[1]
            return torch.ones((batch, seq_len, 6), dtype=torch.float32, device=token_ids[0].device)

    class DummyModel:
        def eval(self):
            return self

        def decode_s1(self, s1_ids, s2_ids, **_kwargs):
            batch, seq_len = s1_ids.shape
            logits = torch.zeros((batch, seq_len, 4), dtype=torch.float32, device=s1_ids.device)
            context = torch.zeros((batch, seq_len, 8), dtype=torch.float32, device=s1_ids.device)
            return logits, context

        def decode_s2(self, context, s1_ids, padding_mask=None):
            batch, seq_len, _ = context.shape
            return torch.zeros((batch, seq_len, 4), dtype=torch.float32, device=context.device)

    prediction = module.predict_next_step(
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        sample_sequences=sample,
        device=torch.device("cpu"),
    )

    assert prediction.shape == (1, 6)
    assert np.allclose(prediction, 1.0)


def test_main_smoke_flow_with_monkeypatched_train_and_predict(tmp_path, monkeypatch):
    module = _load_kronos_small_script()
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        data_dir=str(data_dir),
        device="cpu",
        train_samples=4,
        val_samples=2,
        batch_size=2,
        epochs=1,
        max_batches=1,
        learning_rate=1e-4,
    )

    train_sequences = _make_sequences(samples=4)
    val_sequences = _make_sequences(samples=2)

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "get_logger", lambda *a, **k: _FakeLogger())
    monkeypatch.setattr(
        module,
        "load_split_sequences",
        lambda _data_dir, split: train_sequences if split == "train" else val_sequences,
    )
    monkeypatch.setattr(
        module,
        "train_kronos_small",
        lambda **_kwargs: (object(), object(), {"train_loss": 1.0, "val_loss": 0.5}),
    )
    monkeypatch.setattr(
        module,
        "predict_next_step",
        lambda **_kwargs: np.zeros((1, 6), dtype=np.float32),
    )

    assert module.main() == 0


def test_train_kronos_small_uses_configured_optimizer(monkeypatch):
    module = _load_kronos_small_script()
    config = Config(_load_raw_model_config())
    config.model.training._data["OPTIMIZER"] = "adamw"

    class DummyTokenizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    captured = {}
    real_create_optimizer = module.create_optimizer_for_params

    def tracking_create_optimizer(params, optimizer_config):
        optimizer = real_create_optimizer(params, optimizer_config)
        captured["optimizer_type"] = type(optimizer)
        captured["lr"] = optimizer.param_groups[0]["lr"]
        captured["param_count"] = len(optimizer.param_groups[0]["params"])
        captured["optimizer_name"] = optimizer_config.model.training.OPTIMIZER
        return optimizer

    def fake_create_kronos_model(config, num_stocks=None, num_groups=None):
        captured["num_stocks"] = num_stocks
        captured["num_groups"] = num_groups
        return DummyModel()

    monkeypatch.setattr(module, "create_kronos_tokenizer", lambda config: DummyTokenizer())
    monkeypatch.setattr(module, "create_kronos_model", fake_create_kronos_model)
    monkeypatch.setattr(module, "make_dataloader", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "create_optimizer_for_params", tracking_create_optimizer)

    module.train_kronos_small(
        train_sequences=_make_sequences(),
        val_sequences=None,
        config=config,
        device=torch.device("cpu"),
        batch_size=2,
        epochs=1,
        max_batches=0,
        learning_rate=2e-4,
    )

    assert captured["optimizer_type"] is torch.optim.AdamW
    assert captured["optimizer_name"] == "adamw"
    assert captured["lr"] == pytest.approx(2e-4)
    assert captured["param_count"] == 2
    assert captured["num_stocks"] == 1
    assert captured["num_groups"] == 1
