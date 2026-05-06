"""
Shared training helpers used by both custom and Lightning backends.
"""

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
