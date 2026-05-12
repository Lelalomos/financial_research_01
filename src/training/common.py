"""
Shared training helpers used by both custom and Lightning backends.
"""

import numpy as np
import torch.nn as nn

from .losses import DirectionalLoss, DirectionalMSELoss, SharpeRatioLoss


def create_loss_function(config) -> nn.Module:
    """Create the configured loss function."""
    loss_type = config.model.loss.LOSS_TYPE
    if loss_type == 'mse':
        return nn.MSELoss()
    if loss_type == 'mae':
        return nn.L1Loss()
    if loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    if loss_type == 'huber':
        return nn.HuberLoss(delta=config.model.loss.HUBER_DELTA)
    if loss_type == 'directional':
        return DirectionalLoss()
    if loss_type == 'directional_mse':
        alpha = config.model.loss.get('DIRECTIONAL_ALPHA', 0.1)
        return DirectionalMSELoss(alpha=alpha)
    if loss_type == 'sharpe':
        epsilon = config.model.loss.get('SHARPE_EPSILON', 1e-6)
        return SharpeRatioLoss(epsilon=epsilon)
    raise ValueError(f"Unknown loss type: {loss_type}")


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
