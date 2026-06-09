"""
Tests for the Lightning training backend.
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from src.config.config_loader import Config
from src.config.schemas import validate_config_data
from src.models.lstm3_attn_model import create_model as create_lstm3_attention_model
from src.prediction.predictor import Predictor
from src.training import Trainer
from src.training.lightning_module import (
    CustomFormatCheckpointCallback,
    FinancialLightningModule,
    LightningDependencyError,
    create_lightning_trainer,
    _require_lightning,
    save_final_lightning_checkpoint,
    train_with_lightning,
)
from scripts import train as train_script


class TinyFinancialModel(torch.nn.Module):
    def __init__(self, num_features=3):
        super().__init__()
        self.fc = torch.nn.Linear(num_features, 1)

    def forward(self, features, stock_id, group_id, day, month, dividend_flag):
        return self.fc(features.mean(dim=1))


def _load_raw_model_config():
    path = Path(__file__).parent.parent / "config" / "model.json"
    with open(path, "r") as f:
        return json.load(f)


def _config():
    data = _load_raw_model_config()
    data["model"]["training"]["SCHEDULER"] = None
    data["model"]["training"]["USE_MIXED_PRECISION"] = False
    data["model"]["logging"]["TENSORBOARD_DIR"] = None
    return Config(data)


def _batch(batch_size=4, seq_len=5, num_features=3):
    return {
        "features": torch.randn(batch_size, seq_len, num_features),
        "stock_id": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "group_id": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "day": torch.ones(batch_size, seq_len, dtype=torch.long),
        "month": torch.ones(batch_size, seq_len, dtype=torch.long),
        "dividend_flag": torch.ones(batch_size, seq_len, dtype=torch.long),
        "target": torch.randn(batch_size, 1),
    }


def _clone_batch(batch):
    return {key: value.clone() for key, value in batch.items()}


def test_model_config_accepts_lightning_default_backend():
    data = _load_raw_model_config()
    validate_config_data("model", data)
    assert data["model"]["training_backend"]["DEFAULT"] == "lightning"
    assert data["model"]["selection"]["DEFAULT_MODEL_TYPE"] in data["model"]["models"]


def test_model_config_rejects_unknown_training_backend():
    data = copy.deepcopy(_load_raw_model_config())
    data["model"]["training_backend"]["DEFAULT"] = "unknown"

    with pytest.raises(Exception, match="lightning|custom"):
        validate_config_data("model", data)


def test_lightning_dependency_is_available_or_clear():
    try:
        _require_lightning()
    except LightningDependencyError as exc:
        assert "pip install lightning" in str(exc)


def test_lightning_module_forward_training_and_validation_steps():
    _require_lightning()
    module = FinancialLightningModule(
        model=TinyFinancialModel(),
        config=_config(),
        model_type="tiny",
    )
    batch = _batch()

    output = module(batch)
    loss = module.training_step(batch, batch_idx=0)
    val_metrics = module.validation_step(batch, batch_idx=0)
    optimizer = module.configure_optimizers()

    assert output.shape == (4, 1)
    assert loss.ndim == 0
    assert "val/loss" in val_metrics
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_lightning_training_smoke_saves_custom_checkpoint(tmp_path):
    _require_lightning()
    config = _config()
    config.model.training._data["NUM_EPOCHS"] = 1
    config.model.training._data["BATCH_SIZE"] = 2
    config.model.checkpointing._data["CHECKPOINT_DIR"] = str(tmp_path)
    config.model.checkpointing._data["SAVE_BEST_ONLY"] = True

    train_loader = torch.utils.data.DataLoader([_batch(batch_size=2), _batch(batch_size=2)], batch_size=None)
    val_loader = torch.utils.data.DataLoader([_batch(batch_size=2)], batch_size=None)

    result = train_with_lightning(
        model=TinyFinancialModel(),
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        model_type="tiny",
        checkpoint_metadata={"feature_cols": ["a", "b", "c"], "num_features": 3},
    )

    checkpoint_path = Path(result["best_model_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert result["backend"] == "lightning"
    assert checkpoint_path.exists()
    assert checkpoint["model_type"] == "tiny"
    assert checkpoint["metadata"]["training_backend"] == "lightning"
    assert checkpoint["feature_cols"] == ["a", "b", "c"]
    assert "model_state_dict" in checkpoint
    assert "val_metrics" in checkpoint


def test_lightning_checkpoint_penalizes_one_sided_validation_predictions(tmp_path):
    _require_lightning()
    callback = CustomFormatCheckpointCallback(
        save_dir=str(tmp_path),
        model_type="tiny",
        checkpoint_metadata={},
        save_best_only=True,
    )

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 1)

    class DummyModule:
        def __init__(self):
            self.model = DummyModel()

    class DummyTrainer:
        def __init__(self, metrics, epoch):
            self.callback_metrics = metrics
            self.current_epoch = epoch
            self.optimizers = [torch.optim.SGD(DummyModel().parameters(), lr=0.1)]

    pl_module = DummyModule()

    collapsed_metrics = {
        "val/loss": torch.tensor(0.10),
        "val/collapse_penalty": torch.tensor(1050.0),
        "val/is_collapsed": torch.tensor(1.0),
        "val/directional_accuracy": torch.tensor(0.54),
        "val/pred_positive_rate": torch.tensor(1.0),
        "val/pred_negative_rate": torch.tensor(0.0),
        "val/pred_std": torch.tensor(1e-6),
    }
    healthy_metrics = {
        "val/loss": torch.tensor(0.20),
        "val/collapse_penalty": torch.tensor(0.0),
        "val/is_collapsed": torch.tensor(0.0),
        "val/directional_accuracy": torch.tensor(0.56),
        "val/pred_positive_rate": torch.tensor(0.62),
        "val/pred_negative_rate": torch.tensor(0.38),
        "val/pred_std": torch.tensor(0.02),
    }

    callback.on_validation_epoch_end(DummyTrainer(collapsed_metrics, epoch=0), pl_module)
    first_score = callback.best_selection_score
    callback.on_validation_epoch_end(DummyTrainer(healthy_metrics, epoch=1), pl_module)

    assert callback.best_selection_score < first_score
    checkpoint = torch.load(Path(callback.best_path), map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["is_collapsed"] is False
    assert checkpoint["selection_score"] < first_score


def test_lightning_checkpoint_frequency_controls_periodic_saves(tmp_path):
    _require_lightning()
    callback = CustomFormatCheckpointCallback(
        save_dir=str(tmp_path),
        model_type="tiny",
        checkpoint_metadata={},
        save_best_only=False,
        save_last_n=2,
        checkpoint_frequency=2,
    )

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 1)

    class DummyModule:
        def __init__(self):
            self.model = DummyModel()

    class DummyTrainer:
        def __init__(self, metrics, epoch):
            self.callback_metrics = metrics
            self.current_epoch = epoch
            self.optimizers = [torch.optim.SGD(DummyModel().parameters(), lr=0.1)]

    pl_module = DummyModule()

    for epoch, loss in enumerate([0.5, 0.6, 0.7, 0.8]):
        metrics = {
            "val/loss": torch.tensor(loss),
            "val/collapse_penalty": torch.tensor(0.0),
            "val/is_collapsed": torch.tensor(0.0),
        }
        callback.on_validation_epoch_end(DummyTrainer(metrics, epoch=epoch), pl_module)

    periodic_path = tmp_path / "tiny_latest_periodic_lightning.pth"
    assert periodic_path.exists()
    checkpoint = torch.load(periodic_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 4


def test_trainer_save_model_logs_checkpoint(tmp_path):
    config = _config()
    config.model.checkpointing._data["CHECKPOINT_DIR"] = str(tmp_path)
    trainer = Trainer(
        TinyFinancialModel(),
        config,
        device="cpu",
        model_type="tiny",
    )
    trainer.best_val_loss = 0.125
    logged = {}

    def _log_checkpoint(path, metric=None, metric_name="loss"):
        logged["path"] = path
        logged["metric"] = metric
        logged["metric_name"] = metric_name

    trainer.logger.log_checkpoint = _log_checkpoint
    output_path = tmp_path / "tiny_final.pth"
    trainer.save_model(str(output_path))

    assert output_path.exists()
    assert logged == {
        "path": str(output_path),
        "metric": 0.125,
        "metric_name": "best_val_loss",
    }


def test_lightning_final_checkpoint_logs_save(tmp_path, monkeypatch):
    _require_lightning()

    class DummyLogger:
        def __init__(self):
            self.calls = []

        def log_checkpoint(self, path, metric=None, metric_name="loss"):
            self.calls.append((path, metric, metric_name))

    dummy_logger = DummyLogger()
    monkeypatch.setattr("src.training.lightning_module.get_training_logger", lambda log_dir="logs": dummy_logger)

    model = TinyFinancialModel()
    module = FinancialLightningModule(
        model=model,
        config=_config(),
        model_type="tiny",
    )

    class DummyTrainer:
        current_epoch = 0
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.1)]
        callback_metrics = {
            "val/loss": torch.tensor(0.25),
            "train/loss": torch.tensor(0.5),
        }

    checkpoint_path = save_final_lightning_checkpoint(
        trainer=DummyTrainer(),
        lightning_module=module,
        checkpoint_dir=str(tmp_path),
        model_type="tiny",
        checkpoint_metadata={"feature_cols": ["a", "b", "c"]},
    )

    assert Path(checkpoint_path).exists()
    assert dummy_logger.calls == [(checkpoint_path, 0.25, "val_loss")]


def test_kronos_checkpoint_save_logs_checkpoint(tmp_path, monkeypatch):
    class DummyLogger:
        def __init__(self):
            self.calls = []

        def log_checkpoint(self, path, metric=None, metric_name="loss"):
            self.calls.append((path, metric, metric_name))

    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

    dummy_logger = DummyLogger()
    monkeypatch.setattr(train_script, "get_training_logger", lambda log_dir="logs": dummy_logger)

    tokenizer = DummyModule()
    model = DummyModule()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "kronos_best.pth"

    train_script._save_kronos_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        epoch=1,
        metric=0.42,
        model_type="kronos",
        checkpoint_metadata={"num_features": 2},
        logger=None,
    )

    assert checkpoint_path.exists()
    assert dummy_logger.calls == [(str(checkpoint_path), 0.42, "score")]


def test_lightning_trainer_uses_custom_trainer_gradient_clip_config():
    _require_lightning()
    config = _config()
    config.model.training._data["GRADIENT_CLIP_VALUE"] = 0.25

    trainer = create_lightning_trainer(config=config, device="cpu")

    assert trainer.gradient_clip_val == 0.25


def test_lightning_and_custom_single_batch_training_parity(tmp_path):
    _require_lightning()
    config = _config()
    config.model.training._data["NUM_EPOCHS"] = 1
    config.model.training._data["OPTIMIZER"] = "sgd"
    config.model.training._data["LEARNING_RATE"] = 0.05
    config.model.training._data["WEIGHT_DECAY"] = 0.0
    config.model.training._data["GRADIENT_CLIP_VALUE"] = 0.0
    config.model.checkpointing._data["CHECKPOINT_DIR"] = str(tmp_path)

    torch.manual_seed(123)
    base_model = TinyFinancialModel()
    custom_model = TinyFinancialModel()
    lightning_model = TinyFinancialModel()
    custom_model.load_state_dict(base_model.state_dict())
    lightning_model.load_state_dict(base_model.state_dict())

    torch.manual_seed(456)
    batch = _batch(batch_size=4)
    custom_loader = torch.utils.data.DataLoader([_clone_batch(batch)], batch_size=None)
    lightning_loader = torch.utils.data.DataLoader([_clone_batch(batch)], batch_size=None)

    custom_trainer = Trainer(
        custom_model,
        config,
        device="cpu",
        model_type="tiny",
        experiment_tracker=None,
    )
    custom_metrics = custom_trainer.train_epoch(custom_loader)

    lightning_result = train_with_lightning(
        model=lightning_model,
        config=config,
        train_loader=lightning_loader,
        val_loader=None,
        device="cpu",
        model_type="tiny",
    )

    for name, custom_param in custom_model.state_dict().items():
        assert torch.allclose(custom_param, lightning_model.state_dict()[name], atol=1e-6), name

    assert custom_metrics["loss"] > 0
    assert lightning_result["backend"] == "lightning"


def test_lightning_custom_checkpoint_loads_through_predictor(tmp_path):
    _require_lightning()
    config = _config()
    config.model.checkpointing._data["CHECKPOINT_DIR"] = str(tmp_path)

    model = create_lstm3_attention_model(
        num_features=10,
        num_stocks=5,
        num_groups=3,
        config=config,
    )
    optimizer = torch.optim.Adam(model.parameters())
    callback = CustomFormatCheckpointCallback(
        save_dir=str(tmp_path),
        model_type="lstm3_attention",
        checkpoint_metadata={
            "num_features": 10,
            "num_stocks": 5,
            "num_groups": 3,
            "feature_cols": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ema_50",
                "rsi_14",
                "stochrsi_14",
                "macd",
                "macd_signal",
            ],
        },
    )

    class _TrainerState:
        current_epoch = 0
        optimizers = [optimizer]
        callback_metrics = {
            "val/loss": torch.tensor(0.2),
            "train/loss": torch.tensor(0.3),
        }

    class _ModuleState:
        pass

    module = _ModuleState()
    module.model = model
    callback.on_validation_epoch_end(_TrainerState(), module)

    predictor = Predictor(
        model_path=callback.best_path,
        model_config=config,
        device="cpu",
    )
    sequences = {
        "features": torch.randn(2, 30, 10).numpy().astype("float32"),
        "stock_id": torch.randint(0, 5, (2, 30)).numpy(),
        "group_id": torch.randint(0, 3, (2, 30)).numpy(),
        "day": torch.randint(1, 32, (2, 30)).numpy(),
        "month": torch.randint(1, 13, (2, 30)).numpy(),
        "dividend_flag": torch.randint(1, 3, (2, 30)).numpy(),
    }

    predictions = predictor.predict(sequences, return_raw=True)

    assert predictions.shape == (2, 1)
    assert predictor.model_metadata["metadata"]["training_backend"] == "lightning"
