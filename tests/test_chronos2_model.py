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


def _make_sequences(num_samples=12, seq_len=30, num_features=8):
    rng = np.random.default_rng(42)
    base = rng.normal(size=(num_samples, seq_len, num_features)).astype(np.float32)
    base[:, :, 0] = np.linspace(-1.0, 1.0, seq_len, dtype=np.float32)
    return {
        "features": base,
        "stock_id": rng.integers(0, 4, size=(num_samples, seq_len), dtype=np.int64),
        "group_id": rng.integers(0, 3, size=(num_samples, seq_len), dtype=np.int64),
        "day": rng.integers(1, 28, size=(num_samples, seq_len), dtype=np.int32),
        "month": rng.integers(1, 13, size=(num_samples, seq_len), dtype=np.int32),
        "dividend_flag": rng.integers(1, 3, size=(num_samples, seq_len), dtype=np.int32),
        "target": rng.normal(size=(num_samples,), loc=0.0, scale=0.3).astype(np.float32),
    }


def _small_model_config():
    config = Config(copy.deepcopy(load_config("model").to_dict()))
    config.model.experiment_tracking.ENABLED = False
    config.model.training.NUM_EPOCHS = 1
    config.model.training.BATCH_SIZE = 4
    config.model.training.EARLY_STOPPING_PATIENCE = 2
    config.model.training.LEARNING_RATE = 1e-4
    config.model.device.NUM_WORKERS = 0
    config.model.models.chronos2.D_MODEL = 32
    config.model.models.chronos2.D_KV = 8
    config.model.models.chronos2.D_FF = 64
    config.model.models.chronos2.NUM_LAYERS = 2
    config.model.models.chronos2.NUM_HEADS = 4
    config.model.models.chronos2.INPUT_PATCH_SIZE = 5
    config.model.models.chronos2.INPUT_PATCH_STRIDE = 5
    config.model.models.chronos2.HEAD_HIDDEN_SIZES = [32]
    config.model.models.chronos2.HEAD_DROPOUT = 0.0
    return config


def test_chronos2_forward_shape():
    config = _small_model_config()
    model = create_model(
        model_type="chronos2",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    sequences = _make_sequences(num_samples=3, num_features=8)
    batch = FinancialDataset(sequences, config)[0]
    output = model(
        batch["features"].unsqueeze(0),
        batch["stock_id"].unsqueeze(0),
        batch["group_id"].unsqueeze(0),
        batch["day"].unsqueeze(0),
        batch["month"].unsqueeze(0),
        batch["dividend_flag"].unsqueeze(0),
    )
    assert output.shape == (1, 1)
    assert torch.isfinite(output).all()


def test_chronos2_uses_model_specific_embedding_dims():
    config = _small_model_config()
    config.model.models.chronos2.STOCK_EMB_DIM = 11
    config.model.models.chronos2.GROUP_EMB_DIM = 7
    config.model.models.chronos2.DAY_EMB_DIM = 5
    config.model.models.chronos2.MONTH_EMB_DIM = 4
    config.model.models.chronos2.DIVIDEND_FLAG_EMB_DIM = 3
    model = create_model(
        model_type="chronos2",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )

    assert model.embeddings.stock_embedding.embedding_dim == 11
    assert model.embeddings.group_embedding.embedding_dim == 7
    assert model.embeddings.day_embedding.embedding_dim == 5
    assert model.embeddings.month_embedding.embedding_dim == 4
    assert model.embeddings.dividend_flag_embedding.embedding_dim == 3
    assert model.embeddings.output_dim == 30


def test_chronos2_can_disable_stock_and_group_embeddings():
    config = _small_model_config()
    config.model.models.chronos2.USE_STOCK_EMBEDDING = False
    config.model.models.chronos2.USE_GROUP_EMBEDDING = False
    model = create_model(
        model_type="chronos2",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )

    assert model.embeddings.stock_embedding is None
    assert model.embeddings.group_embedding is None
    expected_dim = (
        config.model.models.chronos2.DAY_EMB_DIM
        + config.model.models.chronos2.MONTH_EMB_DIM
        + config.model.models.chronos2.DIVIDEND_FLAG_EMB_DIM
    )
    assert model.embeddings.output_dim == expected_dim


def test_chronos2_train_epoch_runs():
    config = _small_model_config()
    model = create_model(
        model_type="chronos2",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    trainer = Trainer(model, config, device="cpu", model_type="chronos2")
    dataset = FinancialDataset(_make_sequences(num_samples=8, num_features=8), config)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    metrics = trainer.train_epoch(loader)

    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


def test_chronos2_evaluate_and_backtest_run():
    config = _small_model_config()
    model = create_model(
        model_type="chronos2",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    dataset = FinancialDataset(_make_sequences(num_samples=10, num_features=8), config)
    loader = DataLoader(dataset, batch_size=5, shuffle=False)

    metrics = evaluate_model(model, loader, device="cpu")
    assert "mse" in metrics
    assert "directional_accuracy" in metrics

    backtester = Backtester(model, config, device="cpu")
    results = backtester.run_backtest(loader, prediction_threshold=0.0, initial_capital=1000.0)

    assert "final_capital" in results
    assert "predictions" in results
    assert len(results["predictions"]) == len(results["targets"])
