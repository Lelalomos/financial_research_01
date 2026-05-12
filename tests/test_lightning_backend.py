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
    train_with_lightning,
)


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
    assert data["model"]["training_backend"]["FALLBACK"] == "custom"
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
