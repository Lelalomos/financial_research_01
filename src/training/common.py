"""
Shared training helpers used by both custom and Lightning backends.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Optional

from .losses import (
    create_loss_module,
    DirectionalHuberLoss,
    DirectionalLoss,
    DirectionalMSELoss,
    MultiPartRichLoss,
    PinballLoss,
    QuantileLoss,
    SharpeRatioLoss,
)


def create_optimizer_for_params(
    params: Iterable[torch.nn.Parameter],
    config,
) -> optim.Optimizer:
    """Create optimizer for an arbitrary parameter iterable based on config."""
    training = config.model.training
    optimizer_name = training.OPTIMIZER
    lr = training.LEARNING_RATE
    wd = training.WEIGHT_DECAY

    if optimizer_name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if optimizer_name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if optimizer_name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if optimizer_name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_optimizer(model: nn.Module, config) -> optim.Optimizer:
    """Create optimizer based on config. Shared by Trainer and Lightning backends."""
    return create_optimizer_for_params(model.parameters(), config)


def create_scheduler(
    optimizer: optim.Optimizer,
    config,
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """Create LR scheduler based on config. Shared by Trainer and Lightning backends."""
    scheduler_name = config.model.training.SCHEDULER
    if scheduler_name is None:
        return None

    params = config.get_scheduler_params()

    if scheduler_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **params)
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def create_loss_function(config) -> nn.Module:
    """Create the configured loss function."""
    loss_type = config.model.loss.LOSS_TYPE
    if loss_type == 'quantile_loss':
        quantile = config.model.loss.get('QUANTILE', 0.5)
        return QuantileLoss(quantile=quantile)
    if loss_type == 'pinball_loss':
        quantile = config.model.loss.get('QUANTILE', 0.5)
        return PinballLoss(quantile=quantile)
    if loss_type == 'multi_part_rich_loss':
        rich_cfg = config.model.models.chronos_rich
        return MultiPartRichLoss(
            scalar_loss_weight=float(getattr(rich_cfg, 'SCALAR_LOSS_WEIGHT', 1.0)),
            ohlcv_loss_weight=float(getattr(rich_cfg, 'OHLCV_LOSS_WEIGHT', 1.0)),
            return_path_loss_weight=float(getattr(rich_cfg, 'RETURN_PATH_LOSS_WEIGHT', 1.0)),
            regime_loss_weight=float(getattr(rich_cfg, 'REGIME_LOSS_WEIGHT', 0.5)),
        )
    return create_loss_module(
        loss_type,
        huber_delta=config.model.loss.get('HUBER_DELTA', 1.0),
        directional_alpha=config.model.loss.get('DIRECTIONAL_ALPHA', 0.1),
        sharpe_epsilon=config.model.loss.get('SHARPE_EPSILON', 1e-6),
        quantile=config.model.loss.get('QUANTILE', 0.5),
        label_smoothing=config.model.loss.get('LABEL_SMOOTHING', 0.0),
    )


def calculate_prediction_health(predictions, targets=None) -> dict:
    """
    Compute simple health diagnostics for model outputs.
    """
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    targets_arr = None if targets is None else np.asarray(targets, dtype=float).reshape(-1)

    if predictions.size == 0:
        return {
            "pred_positive_rate": 0.0,
            "pred_negative_rate": 0.0,
            "pred_zero_rate": 0.0,
            "pred_std": 0.0,
            "pred_mean": 0.0,
            "target_positive_rate": None,
            "pred_target_corr": None,
        }

    stats = {
        "pred_positive_rate": float(np.mean(predictions > 0)),
        "pred_negative_rate": float(np.mean(predictions < 0)),
        "pred_zero_rate": float(np.mean(predictions == 0)),
        "pred_std": float(np.std(predictions)),
        "pred_mean": float(np.mean(predictions)),
        "target_positive_rate": None,
        "pred_target_corr": None,
    }

    if targets_arr is not None and targets_arr.size == predictions.size:
        stats["target_positive_rate"] = float(np.mean(targets_arr > 0))
        if predictions.size > 1 and np.std(predictions) > 0 and np.std(targets_arr) > 0:
            stats["pred_target_corr"] = float(np.corrcoef(predictions, targets_arr)[0, 1])

    return stats


def collapse_penalty_from_health(
    health: dict,
    positive_rate_threshold: float = 0.999,
    std_threshold: float = 1e-5,
) -> tuple[float, bool]:
    """
    Return a large penalty when predictions collapse to one side or near-zero variance.
    """
    pred_pos_rate = float(health.get("pred_positive_rate", 0.0))
    pred_neg_rate = float(health.get("pred_negative_rate", 0.0))
    pred_std = float(health.get("pred_std", 0.0))

    is_one_sided = pred_pos_rate >= positive_rate_threshold or pred_neg_rate >= positive_rate_threshold
    is_low_variance = pred_std <= std_threshold
    is_collapsed = is_one_sided or is_low_variance

    penalty = 0.0
    if is_collapsed:
        penalty += 1000.0
        penalty += abs(pred_pos_rate - 0.5) * 100.0
        if pred_std < std_threshold:
            penalty += (std_threshold - pred_std) * 100000.0

    return penalty, is_collapsed
