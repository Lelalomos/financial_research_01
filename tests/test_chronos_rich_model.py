import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.config.config_loader import Config
from src.data.dataset import FinancialDataset
from src.evaluation import evaluate_model
from src.evaluation.backtester import Backtester
from src.models import create_model
from src.training import Trainer


def _make_sequences(num_samples=12, seq_len=30, num_features=8, horizon=5):
    rng = np.random.default_rng(123)
    features = rng.normal(size=(num_samples, seq_len, num_features)).astype(np.float32)
    features[:, :, 0] = np.linspace(90.0, 100.0, seq_len, dtype=np.float32)
    future_ohlcv = rng.normal(size=(num_samples, horizon, 5)).astype(np.float32)
    future_ohlcv[:, :, 3] = np.linspace(101.0, 105.0, horizon, dtype=np.float32)
    last_close = features[:, -1, 0:1]
    future_return_path = ((future_ohlcv[:, :, 3] - last_close) / last_close) * 100.0
    target = future_return_path[:, -1].astype(np.float32)
    return {
        "features": features,
        "stock_id": rng.integers(0, 4, size=(num_samples, seq_len), dtype=np.int64),
        "group_id": rng.integers(0, 3, size=(num_samples, seq_len), dtype=np.int64),
        "day": rng.integers(1, 28, size=(num_samples, seq_len), dtype=np.int32),
        "month": rng.integers(1, 13, size=(num_samples, seq_len), dtype=np.int32),
        "dividend_flag": rng.integers(1, 3, size=(num_samples, seq_len), dtype=np.int32),
        "target": target,
        "future_ohlcv": future_ohlcv,
        "future_return_path": future_return_path.astype(np.float32),
        "future_regime": rng.integers(0, 3, size=(num_samples,), dtype=np.int64),
    }


def _small_model_config():
    config = Config(copy.deepcopy(load_config("model").to_dict()))
    config.model.experiment_tracking.ENABLED = False
    config.model.training.NUM_EPOCHS = 1
    config.model.training.BATCH_SIZE = 4
    config.model.training.EARLY_STOPPING_PATIENCE = 2
    config.model.training.LEARNING_RATE = 1e-4
    config.model.device.NUM_WORKERS = 0
    config.model.models.chronos_rich.D_MODEL = 32
    config.model.models.chronos_rich.D_KV = 8
    config.model.models.chronos_rich.D_FF = 64
    config.model.models.chronos_rich.NUM_LAYERS = 2
    config.model.models.chronos_rich.NUM_HEADS = 4
    config.model.models.chronos_rich.INPUT_PATCH_SIZE = 5
    config.model.models.chronos_rich.INPUT_PATCH_STRIDE = 5
    config.model.models.chronos_rich.HEAD_HIDDEN_SIZES = [32]
    config.model.models.chronos_rich.HEAD_DROPOUT = 0.0
    return config


def test_chronos_rich_forward_shapes():
    config = _small_model_config()
    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    batch = FinancialDataset(_make_sequences(num_samples=3), config)[0]
    output = model(
        batch["features"].unsqueeze(0),
        batch["stock_id"].unsqueeze(0),
        batch["group_id"].unsqueeze(0),
        batch["day"].unsqueeze(0),
        batch["month"].unsqueeze(0),
        batch["dividend_flag"].unsqueeze(0),
    )

    assert output["prediction"].shape == (1, 1)
    assert output["future_ohlcv"].shape == (1, 5, 5)
    assert output["future_return_path"].shape == (1, 5)
    assert output["future_regime_logits"].shape == (1, 3)
    assert output["future_regime"].shape == (1,)


def test_chronos_rich_train_epoch_runs():
    config = _small_model_config()
    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    trainer = Trainer(model, config, device="cpu", model_type="chronos_rich")
    dataset = FinancialDataset(_make_sequences(num_samples=8), config)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    metrics = trainer.train_epoch(loader)

    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


def test_chronos_rich_evaluate_and_backtest_use_scalar_prediction():
    config = _small_model_config()
    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    dataset = FinancialDataset(_make_sequences(num_samples=10), config)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    metrics = evaluate_model(model, loader, device="cpu")
    assert "mse" in metrics
    assert "directional_accuracy" in metrics

    backtester = Backtester(model, config, device="cpu")
    results = backtester.run_backtest(loader, prediction_threshold=0.0, initial_capital=1000.0)

    assert "final_capital" in results
    assert len(results["predictions"]) == len(results["targets"])


def test_chronos_rich_group_attention_changes_output_when_groups_change():
    config = _small_model_config()
    config.model.models.chronos_rich.DROPOUT_RATE = 0.0
    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    model.eval()
    sequences = _make_sequences(num_samples=2)
    batch = FinancialDataset(sequences, config)

    features = torch.stack([batch[0]["features"], batch[1]["features"]], dim=0)
    stock_id = torch.stack([batch[0]["stock_id"], batch[1]["stock_id"]], dim=0)
    day = torch.stack([batch[0]["day"], batch[1]["day"]], dim=0)
    month = torch.stack([batch[0]["month"], batch[1]["month"]], dim=0)
    dividend_flag = torch.stack([batch[0]["dividend_flag"], batch[1]["dividend_flag"]], dim=0)

    same_group = torch.zeros((2, features.shape[1]), dtype=torch.long)
    different_group = torch.tensor(
        [
            [0] * features.shape[1],
            [1] * features.shape[1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        same_output = model(features, stock_id, same_group, day, month, dividend_flag)
        different_output = model(features, stock_id, different_group, day, month, dividend_flag)

    assert not torch.allclose(same_output["prediction"], different_output["prediction"])
