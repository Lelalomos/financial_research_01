"""
Financially oriented loss functions.

These losses complement standard regression losses by optimizing properties
that matter for trading: direction correctness and risk-adjusted returns.
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


class SharpeRatioLoss(nn.Module):
    """Negative differentiable Sharpe-style loss."""

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sharpe_ratio_loss(pred, target, epsilon=self.epsilon)


class DirectionalLoss(nn.Module):
    """Wrong-direction penalty without a regression term."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.relu(-torch.sign(target) * pred).mean()


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
