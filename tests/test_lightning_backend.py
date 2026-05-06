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
from src.training.lightning_module import (
    FinancialLightningModule,
    LightningDependencyError,
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


def test_model_config_accepts_lightning_default_backend():
    data = _load_raw_model_config()
    validate_config_data("model", data)
    assert data["model"]["training_backend"]["DEFAULT"] == "lightning"
    assert data["model"]["training_backend"]["FALLBACK"] == "custom"


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
