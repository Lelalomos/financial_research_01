"""
Evaluation metrics for CRNN Financial Prediction Model.

This module provides:
- Regression metrics (MSE, MAE, RMSE, R², MAPE)
- Directional accuracy
- Sharpe ratio
- Maximum drawdown
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary with metrics
    """
    # Ensure numpy arrays
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    metrics = {}

    # Basic regression metrics
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(targets, predictions)

    # R² score
    try:
        metrics['r2'] = r2_score(targets, predictions)
    except:
        metrics['r2'] = 0.0

    # MAPE (Mean Absolute Percentage Error)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((targets - predictions) / np.abs(targets))) * 100
        metrics['mape'] = np.nanmean(mape)

    # Directional accuracy
    pred_direction = np.sign(predictions)
    true_direction = np.sign(targets)
    metrics['directional_accuracy'] = np.mean(pred_direction == true_direction)

    # Hit rate (correct direction prediction)
    metrics['hit_rate'] = metrics['directional_accuracy']

    return metrics


def directional_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate directional accuracy.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Directional accuracy (0-1)
    """
    pred_direction = np.sign(predictions)
    true_direction = np.sign(targets)
    return np.mean(pred_direction == true_direction)


def calculate_returns(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Calculate trading returns based on predictions.

    Args:
        predictions: Model predictions (percent change)
        targets: Actual returns (percent change)
        threshold: Minimum prediction magnitude to take position

    Returns:
        Array of returns
    """
    # Long if prediction > threshold, short if < -threshold, else cash
    positions = np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))
    returns = positions * targets
    return returns


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Array of returns

    Returns:
        Maximum drawdown
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns / 100)  # Convert percent to decimal
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess_returns)

    # Downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0

    downside_deviation = np.std(downside_returns)

    if downside_deviation == 0:
        return 0.0

    sortino = mean_excess / downside_deviation * np.sqrt(periods_per_year)
    return sortino


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate model on data loader.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use

    Returns:
        Dictionary with metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            stock_id = batch['stock_id'].to(device)
            group_id = batch['group_id'].to(device)
            day = batch['day'].to(device)
            month = batch['month'].to(device)
            target = batch['target'].to(device)

            output = model(features, stock_id, group_id, day, month)

            all_predictions.extend(output.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    metrics = calculate_metrics(predictions, targets)

    # Additional trading metrics
    returns = calculate_returns(predictions, targets)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    metrics['max_drawdown'] = calculate_max_drawdown(returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
    metrics['total_return'] = np.sum(returns)

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in formatted way.

    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names
    """
    print("=" * 60)
    print(f"{prefix}EVALUATION METRICS")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("=" * 60)
