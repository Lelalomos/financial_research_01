"""
Financially oriented loss functions.

These losses complement standard regression losses by optimizing properties
that matter for trading: direction correctness, distribution asymmetry,
multi-target supervision, and risk-adjusted returns.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def directional_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.1,
    base_loss: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Penalize wrong return direction in addition to regression error.

    Args:
        pred: Predicted returns.
        target: Realized returns.
        alpha: Weight for wrong-direction penalty.
        base_loss: Optional regression loss. Defaults to MSE.
    """
    if base_loss is None:
        base_loss = nn.MSELoss()

    regression = base_loss(pred, target)
    
    # Differentiable directional penalty:
    # If target > 0, we want pred > 0. Penalty = relu(-pred)
    # If target < 0, we want pred < 0. Penalty = relu(pred)
    # Combined: relu(-pred * sign(target))
    penalty = F.relu(-pred * torch.sign(target))
    
    return regression + alpha * penalty.mean()


def sharpe_ratio_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Differentiable negative Sharpe-style loss.

    Uses predicted position direction times realized target as strategy return.
    The returned value is negative because optimizers minimize loss.
    """
    strategy_returns = torch.tanh(pred) * target
    mean_return = strategy_returns.mean()
    return_std = strategy_returns.std(unbiased=False)
    sharpe = mean_return / (return_std + epsilon)
    return -sharpe


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    """
    Quantile regression loss.

    This is also known as pinball loss.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")

    error = target - pred
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.5,
) -> torch.Tensor:
    """Alias of quantile loss."""
    return quantile_loss(pred=pred, target=target, quantile=quantile)


def transaction_cost_adjusted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    transaction_cost: float = 0.05,
    turnover_weight: float = 0.1,
    return_weight: float = 1.0,
    base_loss: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Regression loss plus differentiable trading cost penalty.

    Args:
        pred: Predicted returns, in the same units as target.
        target: Realized returns.
        transaction_cost: Cost per unit position change, in target-return units.
        turnover_weight: Weight on the transaction cost penalty.
        return_weight: Weight on the net-return reward term.
        base_loss: Optional regression loss. Defaults to MSE.

    Notes:
        Positions are approximated with tanh(pred), keeping the loss
        differentiable and bounded. Turnover is the absolute change in position
        between adjacent batch rows. For shuffled batches this is only a
        regularizer, not a portfolio backtest.
    """
    if transaction_cost < 0.0:
        raise ValueError("transaction_cost must be non-negative")
    if turnover_weight < 0.0:
        raise ValueError("turnover_weight must be non-negative")
    if return_weight < 0.0:
        raise ValueError("return_weight must be non-negative")
    if base_loss is None:
        base_loss = nn.MSELoss()

    regression = base_loss(pred, target)
    position = torch.tanh(pred)
    strategy_return = position * target

    if position.shape[0] > 1:
        turnover = torch.abs(position[1:] - position[:-1]).mean()
    else:
        turnover = torch.zeros((), device=pred.device, dtype=pred.dtype)

    cost_penalty = transaction_cost * turnover
    net_return = strategy_return.mean() - cost_penalty
    return regression + turnover_weight * cost_penalty - return_weight * net_return


class DirectionalMSELoss(nn.Module):
    """MSE plus wrong-direction penalty."""

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.base_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return directional_loss(pred, target, alpha=self.alpha, base_loss=self.base_loss)


class DirectionalHuberLoss(nn.Module):
    """Huber loss plus wrong-direction penalty."""

    def __init__(self, alpha: float = 0.1, delta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.base_loss = nn.HuberLoss(delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return directional_loss(pred, target, alpha=self.alpha, base_loss=self.base_loss)


class SharpeRatioLoss(nn.Module):
    """Negative differentiable Sharpe-style loss."""

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sharpe_ratio_loss(pred, target, epsilon=self.epsilon)


class CrossEntropyLossModule(nn.Module):
    """Cross-entropy classification loss wrapper."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, target)


class QuantileLoss(nn.Module):
    """Quantile regression loss."""

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1")
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return quantile_loss(pred=pred, target=target, quantile=self.quantile)


class PinballLoss(nn.Module):
    """Pinball loss, equivalent to quantile regression loss."""

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1")
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pinball_loss(pred=pred, target=target, quantile=self.quantile)


class DirectionalLoss(nn.Module):
    """Wrong-direction penalty without a regression term."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.relu(-torch.sign(target) * pred).mean()


class MultiPartRichLoss(nn.Module):
    """
    Weighted rich-output loss for structured future-market targets.

    Expects:
    - output["prediction"], batch["target"]
    - optional future OHLCV tensors
    - optional future return path tensors
    - optional future regime labels/logits
    """

    expects_structured_output = True

    def __init__(
        self,
        scalar_loss_weight: float = 1.0,
        ohlcv_loss_weight: float = 1.0,
        return_path_loss_weight: float = 1.0,
        regime_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.scalar_loss_weight = scalar_loss_weight
        self.ohlcv_loss_weight = ohlcv_loss_weight
        self.return_path_loss_weight = return_path_loss_weight
        self.regime_loss_weight = regime_loss_weight
        self.scalar_loss = nn.MSELoss()
        self.ohlcv_loss = nn.MSELoss()
        self.return_path_loss = nn.MSELoss()
        self.regime_loss = nn.CrossEntropyLoss()

    def forward(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = self.scalar_loss_weight * self.scalar_loss(output["prediction"], batch["target"])

        if "future_ohlcv" in output and "future_ohlcv" in batch:
            loss = loss + self.ohlcv_loss_weight * self.ohlcv_loss(
                output["future_ohlcv"],
                batch["future_ohlcv"],
            )

        if "future_return_path" in output and "future_return_path" in batch:
            loss = loss + self.return_path_loss_weight * self.return_path_loss(
                output["future_return_path"],
                batch["future_return_path"],
            )

        if "future_regime_logits" in output and "future_regime" in batch:
            loss = loss + self.regime_loss_weight * self.regime_loss(
                output["future_regime_logits"],
                batch["future_regime"],
            )

        return loss


class TransactionCostAdjustedLoss(nn.Module):
    """Regression loss adjusted for approximate net trading return and turnover cost."""

    def __init__(
        self,
        transaction_cost: float = 0.05,
        turnover_weight: float = 0.1,
        return_weight: float = 1.0,
    ):
        super().__init__()
        self.transaction_cost = transaction_cost
        self.turnover_weight = turnover_weight
        self.return_weight = return_weight
        self.base_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return transaction_cost_adjusted_loss(
            pred=pred,
            target=target,
            transaction_cost=self.transaction_cost,
            turnover_weight=self.turnover_weight,
            return_weight=self.return_weight,
            base_loss=self.base_loss,
        )


def create_loss_module(
    loss_type: str,
    *,
    huber_delta: float = 1.0,
    directional_alpha: float = 0.1,
    sharpe_epsilon: float = 1e-6,
    quantile: float = 0.5,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Create a loss module from a simple typed spec."""
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_type == "huber":
        return nn.HuberLoss(delta=huber_delta)
    if loss_type == "directional":
        return DirectionalLoss()
    if loss_type == "directional_mse":
        return DirectionalMSELoss(alpha=directional_alpha)
    if loss_type == "directional_huber":
        return DirectionalHuberLoss(alpha=directional_alpha, delta=huber_delta)
    if loss_type == "sharpe":
        return SharpeRatioLoss(epsilon=sharpe_epsilon)
    if loss_type == "quantile_loss":
        return QuantileLoss(quantile=quantile)
    if loss_type == "pinball_loss":
        return PinballLoss(quantile=quantile)
    if loss_type == "cross_entropy":
        return CrossEntropyLossModule(label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss type: {loss_type}")
