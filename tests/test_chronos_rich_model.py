import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from src.config import load_config
from src.config.config_loader import Config
from src.data.dataset import FinancialDataset
from src.evaluation import evaluate_model, evaluate_model_with_report
from src.evaluation.backtester import Backtester
from src.models import create_model
from src.models.kronos_module import RMSNorm
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


def test_chronos_rich_uses_configured_component_losses():
    config = _small_model_config()
    config.model.models.chronos_rich.SCALAR_LOSS_TYPE = "mae"
    config.model.models.chronos_rich.SCALAR_LOSS_WEIGHT = 2.0
    config.model.models.chronos_rich.OHLCV_LOSS_TYPE = "huber"
    config.model.models.chronos_rich.OHLCV_HUBER_DELTA = 0.5
    config.model.models.chronos_rich.OHLCV_LOSS_WEIGHT = 3.0
    config.model.models.chronos_rich.RETURN_PATH_LOSS_TYPE = "pinball_loss"
    config.model.models.chronos_rich.RETURN_PATH_QUANTILE = 0.8
    config.model.models.chronos_rich.RETURN_PATH_LOSS_WEIGHT = 4.0
    config.model.models.chronos_rich.REGIME_LOSS_TYPE = "cross_entropy"
    config.model.models.chronos_rich.REGIME_LABEL_SMOOTHING = 0.1
    config.model.models.chronos_rich.REGIME_LOSS_WEIGHT = 5.0

    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    output = {
        "prediction": torch.tensor([[1.0]], dtype=torch.float32),
        "future_ohlcv": torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.float32),
        "future_return_path": torch.tensor([[0.5, 1.5]], dtype=torch.float32),
        "future_regime_logits": torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32),
    }
    batch = {
        "target": torch.tensor([[0.0]], dtype=torch.float32),
        "future_ohlcv": torch.zeros((1, 1, 5), dtype=torch.float32),
        "future_return_path": torch.zeros((1, 2), dtype=torch.float32),
        "future_regime": torch.tensor([0], dtype=torch.long),
    }

    loss = model.compute_loss(output, batch, criterion=None)
    expected = (
        2.0 * F.l1_loss(output["prediction"], batch["target"])
        + 3.0 * F.huber_loss(output["future_ohlcv"], batch["future_ohlcv"], delta=0.5)
        + 4.0 * torch.maximum(
            0.8 * (batch["future_return_path"] - output["future_return_path"]),
            (0.8 - 1.0) * (batch["future_return_path"] - output["future_return_path"]),
        ).mean()
        + 5.0 * F.cross_entropy(output["future_regime_logits"], batch["future_regime"], label_smoothing=0.1)
    )

    assert torch.isclose(loss, expected)


def test_chronos_rich_uses_configured_activation():
    config = _small_model_config()
    config.model.models.chronos_rich.ACTIVATION = "gelu"

    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )

    assert isinstance(model.encoder[0].feed_forward.ff.activation, nn.GELU)
    assert isinstance(model.forecast_head.activation, nn.GELU)
    assert isinstance(model.shared_head[0].activation, nn.GELU)


def test_chronos_rich_uses_configured_geglu_activation():
    config = _small_model_config()
    config.model.models.chronos_rich.ACTIVATION = "geglu"

    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )

    assert model.encoder[0].feed_forward.ff.is_gated is True
    assert isinstance(model.encoder[0].feed_forward.ff.activation, nn.GELU)
    assert model.encoder[0].feed_forward.ff.input_projection.out_features == 128
    assert model.forecast_head.is_gated is True
    assert model.forecast_head.input_projection.out_features == 128


def test_chronos_rich_uses_configured_swiglu_activation():
    config = _small_model_config()
    config.model.models.chronos_rich.ACTIVATION = "swiglu"

    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )

    assert model.encoder[0].feed_forward.ff.is_gated is True
    assert isinstance(model.encoder[0].feed_forward.ff.activation, nn.SiLU)


def test_chronos_rich_uses_configured_rmsnorm():
    config = _small_model_config()
    config.model.models.chronos_rich.NORM_TYPE = "rmsnorm"

    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )

    assert isinstance(model.encoder[0].time_attention.norm, RMSNorm)
    assert isinstance(model.encoder[0].group_attention.norm, RMSNorm)
    assert isinstance(model.encoder[0].feed_forward.norm, RMSNorm)
    assert isinstance(model.encoder_norm, RMSNorm)


def test_chronos_rich_uses_configured_bias_flag():
    config = _small_model_config()
    config.model.models.chronos_rich.USE_BIAS = False

    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )

    assert model.patch_projection.bias is None
    assert model.encoder[0].feed_forward.ff.input_projection.bias is None
    assert model.encoder[0].feed_forward.ff.output_projection.bias is None
    assert model.forecast_head.input_projection.bias is None
    assert model.future_ohlcv_head.bias is None
    assert model.future_regime_head.bias is None
    assert model.encoder[0].time_attention.attn.in_proj_bias is None


def test_chronos_rich_rejects_unknown_activation():
    config = _small_model_config()
    config.model.models.chronos_rich.ACTIVATION = "unknown_activation"

    with torch.no_grad():
        with pytest.raises(ValueError, match="Unsupported ChronosRich activation"):
            create_model(
                model_type="chronos_rich",
                num_features=8,
                num_stocks=4,
                num_groups=3,
                config=config,
                feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
            )


def test_chronos_rich_rejects_unknown_norm_type():
    config = _small_model_config()
    config.model.models.chronos_rich.NORM_TYPE = "bad_norm"

    with pytest.raises(ValueError, match="Unsupported ChronosRich norm type"):
        create_model(
            model_type="chronos_rich",
            num_features=8,
            num_stocks=4,
            num_groups=3,
            config=config,
            feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
        )


def test_chronos_rich_report_includes_rich_outputs(tmp_path: Path):
    config = _small_model_config()
    model = create_model(
        model_type="chronos_rich",
        num_features=8,
        num_stocks=4,
        num_groups=3,
        config=config,
        feature_cols=["close", "high", "low", "open", "volume", "feat1", "feat2", "feat3"],
    )
    dataset = FinancialDataset(_make_sequences(num_samples=6), config)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    report_path = tmp_path / "chronos_rich_report.xlsx"

    metrics, report_df, sector_stats = evaluate_model_with_report(
        model,
        loader,
        device="cpu",
        stock_id_to_ticker={idx: f"STOCK{idx}" for idx in range(4)},
        group_id_to_sector={idx: f"SECTOR{idx}" for idx in range(3)},
        output_path=str(report_path),
    )

    assert report_path.exists()
    assert "mse" in metrics
    assert sector_stats
    expected_columns = [
        "pred_future_return_path_t1",
        "pred_future_return_path_t5",
        "pred_future_regime",
        "pred_future_regime_logit_t1",
        "pred_future_regime_prob_t1",
        "pred_future_ohlcv_open_t1",
        "pred_future_ohlcv_close_t5",
        "real_future_return_path_t1",
        "real_future_regime",
        "real_future_ohlcv_open_t1",
        "real_future_ohlcv_close_t5",
    ]
    for column in expected_columns:
        assert column in report_df.columns


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
