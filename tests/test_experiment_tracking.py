"""
Unit tests for optional local experiment tracking.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from src.config.config_loader import Config
from src.config.schemas import validate_config_data
from src.training import Trainer
from src.training.experiment_tracking import (
    ExperimentTrackingError,
    LocalMLflowTracker,
    NoOpTracker,
    create_experiment_tracker,
)


class TinyFinancialModel(torch.nn.Module):
    def __init__(self, num_features=3):
        super().__init__()
        self.fc = torch.nn.Linear(num_features, 1)

    def forward(self, features, stock_id, group_id, day, month, dividend_flag):
        return self.fc(features.mean(dim=1))


class RecordingTracker:
    def __init__(self):
        self.run_names = []
        self.params = []
        self.metrics = []
        self.statuses = []

    def start_run(self, run_name=None):
        self.run_names.append(run_name)

    def log_params(self, params):
        self.params.append(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def end_run(self, status="FINISHED"):
        self.statuses.append(status)


def _load_raw_model_config():
    path = Path(__file__).parent.parent / "config" / "model.json"
    with open(path, "r") as f:
        return json.load(f)


def _fresh_model_config():
    return Config(copy.deepcopy(_load_raw_model_config()))


def _batch(batch_size=2, seq_len=4, num_features=3):
    return {
        "features": torch.randn(batch_size, seq_len, num_features),
        "stock_id": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "group_id": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "day": torch.ones(batch_size, seq_len, dtype=torch.long),
        "month": torch.ones(batch_size, seq_len, dtype=torch.long),
        "dividend_flag": torch.ones(batch_size, seq_len, dtype=torch.long),
        "target": torch.randn(batch_size, 1),
    }


def test_disabled_experiment_tracking_returns_noop_tracker():
    config = _fresh_model_config()
    config.model.experiment_tracking._data["ENABLED"] = False

    tracker = create_experiment_tracker(config)

    assert isinstance(tracker, NoOpTracker)
    tracker.start_run("ignored")
    tracker.log_params({"a": 1})
    tracker.log_metrics({"loss": 1.0}, step=1)
    tracker.end_run()


def test_model_config_accepts_local_mlflow_tracking():
    data = _load_raw_model_config()
    valid = copy.deepcopy(data)
    valid["model"]["experiment_tracking"] = {
        "ENABLED": True,
        "BACKEND": "mlflow",
        "MLFLOW_TRACKING_URI": "file:./mlruns",
        "EXPERIMENT_NAME": "local-test",
        "LOG_PARAMS": True,
        "LOG_METRICS": True,
        "LOG_ARTIFACTS": False,
    }

    validate_config_data("model", valid)


def test_model_config_rejects_remote_mlflow_tracking_uri():
    data = _load_raw_model_config()
    invalid = copy.deepcopy(data)
    invalid["model"]["experiment_tracking"]["ENABLED"] = True
    invalid["model"]["experiment_tracking"]["MLFLOW_TRACKING_URI"] = "https://mlflow.example.com"

    with pytest.raises(ValidationError, match="local"):
        validate_config_data("model", invalid)


def test_local_mlflow_tracker_rejects_remote_uri_before_importing_mlflow():
    with pytest.raises(ExperimentTrackingError, match="Only local MLflow"):
        LocalMLflowTracker(tracking_uri="https://mlflow.example.com")


def test_trainer_logs_metrics_to_injected_tracker(tmp_path):
    config = _fresh_model_config()
    config.model.training._data["NUM_EPOCHS"] = 1
    config.model.training._data["USE_MIXED_PRECISION"] = False
    config.model.training._data["SCHEDULER"] = None
    config.model.logging._data["TENSORBOARD_DIR"] = None
    config.model.checkpointing._data["CHECKPOINT_DIR"] = str(tmp_path)

    tracker = RecordingTracker()
    trainer = Trainer(
        TinyFinancialModel(),
        config,
        device="cpu",
        model_type="tiny",
        experiment_tracker=tracker,
    )

    train_loader = DataLoader([_batch(), _batch()], batch_size=None)
    history = trainer.train(train_loader=train_loader, val_loader=None, num_epochs=1)

    assert len(history["train_loss"]) == 1
    assert tracker.run_names == ["tiny"]
    assert tracker.params[0]["model_type"] == "tiny"
    assert any("train/loss" in metrics for metrics, _step in tracker.metrics)
    assert tracker.statuses == ["FINISHED"]
