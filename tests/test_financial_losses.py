"""
Unit tests for financial loss functions.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models import create_model
from src.training import DirectionalLoss, DirectionalMSELoss, SharpeRatioLoss, Trainer
from src.training.losses import directional_loss, sharpe_ratio_loss


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


def test_loss_modules_return_scalar_values():
    pred = torch.tensor([[0.2], [-0.1]])
    target = torch.tensor([[0.1], [0.2]])

    for loss_fn in [DirectionalLoss(), DirectionalMSELoss(alpha=0.5), SharpeRatioLoss()]:
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
