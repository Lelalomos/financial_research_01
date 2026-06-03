"""
Unit tests for financial loss functions.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models import create_model
from src.training import (
    DirectionalHuberLoss,
    DirectionalLoss,
    DirectionalMSELoss,
    MultiPartRichLoss,
    PinballLoss,
    QuantileLoss,
    SharpeRatioLoss,
    Trainer,
)
from src.training.losses import directional_loss, pinball_loss, quantile_loss, sharpe_ratio_loss


def test_directional_loss_penalizes_wrong_direction():
    target = torch.tensor([[1.0], [-1.0]])
    correct = torch.tensor([[0.5], [-0.5]])
    wrong = torch.tensor([[-0.5], [0.5]])

    assert directional_loss(wrong, target, alpha=1.0) > directional_loss(correct, target, alpha=1.0)


def test_sharpe_ratio_loss_is_finite_and_differentiable():
    pred = torch.tensor([[0.2], [0.4], [-0.1], [0.3]], requires_grad=True)
    target = torch.tensor([[0.1], [0.2], [-0.05], [0.1]])

    loss = sharpe_ratio_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_quantile_and_pinball_loss_match():
    pred = torch.tensor([[0.0], [0.4], [0.8]])
    target = torch.tensor([[1.0], [0.1], [0.5]])

    q_loss = quantile_loss(pred, target, quantile=0.2)
    p_loss = pinball_loss(pred, target, quantile=0.2)

    assert torch.isclose(q_loss, p_loss)


def test_quantile_loss_penalizes_under_prediction_more_for_high_quantile():
    pred = torch.tensor([[0.0]])
    target = torch.tensor([[1.0]])

    low_q = quantile_loss(pred, target, quantile=0.1)
    high_q = quantile_loss(pred, target, quantile=0.9)

    assert high_q > low_q


def test_loss_modules_return_scalar_values():
    pred = torch.tensor([[0.2], [-0.1]])
    target = torch.tensor([[0.1], [0.2]])

    for loss_fn in [
        DirectionalLoss(),
        DirectionalMSELoss(alpha=0.5),
        DirectionalHuberLoss(alpha=0.5, delta=1.0),
        SharpeRatioLoss(),
        QuantileLoss(quantile=0.2),
        PinballLoss(quantile=0.2),
    ]:
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


def test_trainer_creates_directional_mse_loss():
    config = load_config("model")
    original_loss = config.model.loss._data["LOSS_TYPE"]
    config.model.loss._data["LOSS_TYPE"] = "directional_mse"
    config.model.loss._data["DIRECTIONAL_ALPHA"] = 0.25

    try:
        model = create_model(
            model_type="rnn",
            num_features=5,
            num_stocks=3,
            num_groups=2,
            config=config,
        )
        trainer = Trainer(model, config, device="cpu")
        assert isinstance(trainer.criterion, DirectionalMSELoss)
        assert trainer.criterion.alpha == 0.25
    finally:
        config.model.loss._data["LOSS_TYPE"] = original_loss


def test_trainer_creates_directional_huber_loss():
    config = load_config("model")
    original_loss = config.model.loss._data["LOSS_TYPE"]
    original_alpha = config.model.loss._data["DIRECTIONAL_ALPHA"]
    original_delta = config.model.loss._data["HUBER_DELTA"]
    config.model.loss._data["LOSS_TYPE"] = "directional_huber"
    config.model.loss._data["DIRECTIONAL_ALPHA"] = 0.4
    config.model.loss._data["HUBER_DELTA"] = 1.5

    try:
        model = create_model(
            model_type="rnn",
            num_features=5,
            num_stocks=3,
            num_groups=2,
            config=config,
        )
        trainer = Trainer(model, config, device="cpu")
        assert isinstance(trainer.criterion, DirectionalHuberLoss)
        assert trainer.criterion.alpha == 0.4
        assert trainer.criterion.delta == 1.5
    finally:
        config.model.loss._data["LOSS_TYPE"] = original_loss
        config.model.loss._data["DIRECTIONAL_ALPHA"] = original_alpha
        config.model.loss._data["HUBER_DELTA"] = original_delta


def test_trainer_creates_quantile_loss():
    config = load_config("model")
    original_loss = config.model.loss._data["LOSS_TYPE"]
    original_quantile = config.model.loss._data.get("QUANTILE", 0.5)
    config.model.loss._data["LOSS_TYPE"] = "quantile_loss"
    config.model.loss._data["QUANTILE"] = 0.2

    try:
        model = create_model(
            model_type="rnn",
            num_features=5,
            num_stocks=3,
            num_groups=2,
            config=config,
        )
        trainer = Trainer(model, config, device="cpu")
        assert isinstance(trainer.criterion, QuantileLoss)
        assert trainer.criterion.quantile == 0.2
    finally:
        config.model.loss._data["LOSS_TYPE"] = original_loss
        config.model.loss._data["QUANTILE"] = original_quantile


def test_trainer_creates_pinball_loss():
    config = load_config("model")
    original_loss = config.model.loss._data["LOSS_TYPE"]
    original_quantile = config.model.loss._data.get("QUANTILE", 0.5)
    config.model.loss._data["LOSS_TYPE"] = "pinball_loss"
    config.model.loss._data["QUANTILE"] = 0.8

    try:
        model = create_model(
            model_type="rnn",
            num_features=5,
            num_stocks=3,
            num_groups=2,
            config=config,
        )
        trainer = Trainer(model, config, device="cpu")
        assert isinstance(trainer.criterion, PinballLoss)
        assert trainer.criterion.quantile == 0.8
    finally:
        config.model.loss._data["LOSS_TYPE"] = original_loss
        config.model.loss._data["QUANTILE"] = original_quantile


def test_multi_part_rich_loss_aggregates_weighted_components():
    loss_fn = MultiPartRichLoss(
        scalar_loss_weight=2.0,
        ohlcv_loss_weight=3.0,
        return_path_loss_weight=4.0,
        regime_loss_weight=5.0,
    )
    output = {
        "prediction": torch.tensor([[1.0]]),
        "future_ohlcv": torch.tensor([[[1.0, 2.0]]]),
        "future_return_path": torch.tensor([[0.5, 1.5]]),
        "future_regime_logits": torch.tensor([[2.0, 0.0, -1.0]]),
    }
    batch = {
        "target": torch.tensor([[0.0]]),
        "future_ohlcv": torch.tensor([[[0.0, 0.0]]]),
        "future_return_path": torch.tensor([[0.0, 0.0]]),
        "future_regime": torch.tensor([0]),
    }

    loss = loss_fn(output, batch)
    expected = (
        2.0 * torch.nn.functional.mse_loss(output["prediction"], batch["target"])
        + 3.0 * torch.nn.functional.mse_loss(output["future_ohlcv"], batch["future_ohlcv"])
        + 4.0 * torch.nn.functional.mse_loss(output["future_return_path"], batch["future_return_path"])
        + 5.0 * torch.nn.functional.cross_entropy(output["future_regime_logits"], batch["future_regime"])
    )

    assert torch.isclose(loss, expected)


def test_trainer_creates_multi_part_rich_loss():
    config = load_config("model")
    original_loss = config.model.loss._data["LOSS_TYPE"]
    config.model.loss._data["LOSS_TYPE"] = "multi_part_rich_loss"

    try:
        model = create_model(
            model_type="chronos_rich",
            num_features=5,
            num_stocks=3,
            num_groups=2,
            config=config,
            feature_cols=["open", "high", "low", "close", "volume"],
        )
        trainer = Trainer(model, config, device="cpu")
        assert isinstance(trainer.criterion, MultiPartRichLoss)
    finally:
        config.model.loss._data["LOSS_TYPE"] = original_loss
