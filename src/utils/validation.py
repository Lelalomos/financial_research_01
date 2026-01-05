"""
Validation utilities for NaN/Inf detection and handling in tensors.

This module provides functions to:
- Detect NaN and Inf values in tensors
- Sanitize tensors by replacing invalid values
- Validate batches of training data
- Check model parameter state
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def check_tensor_for_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor"
) -> Tuple[bool, str]:
    """
    Check if tensor contains NaN or Inf values.

    Args:
        tensor: PyTorch tensor to check
        name: Name of the tensor for logging

    Returns:
        Tuple of (has_issues, message):
        - has_issues: True if NaN or Inf found
        - message: Description of issues found
    """
    if torch.isnan(tensor).any():
        count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        return True, f"{name} contains {count}/{total} NaN values"

    if torch.isinf(tensor).any():
        pos_inf_mask = (tensor > 0) & torch.isinf(tensor)
        neg_inf_mask = (tensor < 0) & torch.isinf(tensor)
        pos_inf_count = pos_inf_mask.sum().item()
        neg_inf_count = neg_inf_mask.sum().item()
        total = tensor.numel()
        return True, f"{name} contains {pos_inf_count + neg_inf_count}/{total} Inf values ({neg_inf_count} negative, {pos_inf_count} positive)"

    return False, ""


def sanitize_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    replace_value: float = 0.0
) -> torch.Tensor:
    """
    Replace NaN/Inf values in tensor with specified value.

    Args:
        tensor: PyTorch tensor to sanitize
        name: Name of the tensor for logging
        replace_value: Value to use as replacement

    Returns:
        Sanitized tensor with NaN/Inf replaced
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
        inf_count = torch.isinf(tensor).sum().item() if has_inf else 0

        logger.warning(
            f"Sanitizing {name}: replacing {nan_count} NaN and {inf_count} Inf values with {replace_value}"
        )

        sanitized = torch.where(
            torch.isnan(tensor) | torch.isinf(tensor),
            torch.tensor(replace_value, device=tensor.device, dtype=tensor.dtype),
            tensor
        )
        return sanitized

    return tensor


def check_batch_for_invalid(
    batch: Dict[str, torch.Tensor]
) -> Tuple[bool, str]:
    """
    Check entire batch for NaN/Inf values.

    Args:
        batch: Dictionary of tensor names to tensors

    Returns:
        Tuple of (has_issues, message):
        - has_issues: True if any tensor has NaN/Inf
        - message: Description of first issue found
    """
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            has_issues, message = check_tensor_for_nan_inf(tensor, key)
            if has_issues:
                return True, message
    return False, ""


def sanitize_batch(
    batch: Dict[str, torch.Tensor],
    replace_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Sanitize entire batch by replacing NaN/Inf with specified value.

    Args:
        batch: Dictionary of tensor names to tensors
        replace_value: Value to use as replacement

    Returns:
        Sanitized batch dictionary
    """
    sanitized = {}
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            sanitized[key] = sanitize_tensor(tensor, key, replace_value)
        else:
            sanitized[key] = tensor
    return sanitized


def check_model_parameters(
    model: torch.nn.Module
) -> Tuple[bool, List[str]]:
    """
    Check all model parameters for NaN/Inf values.

    Args:
        model: PyTorch model to check

    Returns:
        Tuple of (is_valid, issues):
        - is_valid: True if no NaN/Inf found
        - issues: List of issue descriptions
    """
    issues = []

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            issues.append(f"{name}: {nan_count} NaN values")

        if torch.isinf(param).any():
            inf_count = torch.isinf(param).sum().item()
            issues.append(f"{name}: {inf_count} Inf values")

    return len(issues) == 0, issues


def check_gradients(
    model: torch.nn.Module
) -> Tuple[bool, List[str], Optional[float]]:
    """
    Check all model gradients for NaN/Inf and extreme values.

    Args:
        model: PyTorch model with gradients to check

    Returns:
        Tuple of (is_valid, issues, max_grad_norm):
        - is_valid: True if no NaN/Inf found and gradients reasonable
        - issues: List of issue descriptions
        - max_grad_norm: Maximum gradient norm across all parameters
    """
    issues = []
    max_grad_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_grad_norm = max(max_grad_norm, grad_norm)

            # Check for NaN/Inf
            if torch.isnan(param.grad).any():
                nan_count = torch.isnan(param.grad).sum().item()
                issues.append(f"{name}.grad: {nan_count} NaN values")

            if torch.isinf(param.grad).any():
                inf_count = torch.isinf(param.grad).sum().item()
                issues.append(f"{name}.grad: {inf_count} Inf values")

            # Check for extreme values (potential explosion)
            if grad_norm > 1000.0:
                issues.append(f"{name}.grad: Very large gradient norm {grad_norm:.2f}")

    return len(issues) == 0, issues, max_grad_norm


def validate_input_batch(
    batch: Dict[str, torch.Tensor],
    sanitize: bool = True,
    replace_value: float = 0.0
) -> Tuple[bool, Dict[str, torch.Tensor]]:
    """
    Validate and optionally sanitize an input batch.

    Args:
        batch: Input batch dictionary
        sanitize: Whether to sanitize invalid values
        replace_value: Value to use for sanitization

    Returns:
        Tuple of (is_valid, batch):
        - is_valid: True if no invalid values found (or if sanitized)
        - batch: Validated/sanitized batch
    """
    has_issues, message = check_batch_for_invalid(batch)

    if has_issues:
        if sanitize:
            logger.warning(f"Invalid values detected in batch: {message}. Sanitizing...")
            batch = sanitize_batch(batch, replace_value)
            return True, batch
        else:
            logger.error(f"Invalid values detected in batch: {message}")
            return False, batch

    return True, batch


def create_safe_tensor(
    data: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Create a tensor from numpy array, replacing NaN/Inf with zeros.

    Args:
        data: Input numpy array
        dtype: Tensor data type
        device: Tensor device

    Returns:
        Safe tensor with no NaN/Inf values
    """
    # Replace NaN/Inf in numpy array first
    safe_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Create tensor and check again (in case of conversion issues)
    tensor = torch.tensor(safe_data, dtype=dtype, device=device)

    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        tensor = torch.where(
            torch.isnan(tensor) | torch.isinf(tensor),
            torch.zeros_like(tensor),
            tensor
        )

    return tensor
