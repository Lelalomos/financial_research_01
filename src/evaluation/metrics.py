"""
Evaluation metrics for CRNN Financial Prediction Model.

This module provides:
- Regression metrics (MSE, MAE, RMSE, R², MAPE)
- Directional accuracy
- Sharpe ratio
- Maximum drawdown
- Excel report generation with direction scores by sector
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import pandas as pd


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
        mape = np.abs((targets - predictions) / np.abs(targets)) * 100
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

            # Get dividend_flag if available, otherwise use ones (has dividend)
            if 'dividend_flag' in batch:
                dividend_flag = batch['dividend_flag'].to(device)
            else:
                dividend_flag = torch.ones(features.shape[0], features.shape[1],
                                              dtype=torch.long, device=device)

            output = model(features, stock_id, group_id, day, month, dividend_flag)

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


def evaluate_model_with_report(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda',
    stock_id_to_ticker: Optional[Dict[int, str]] = None,
    group_id_to_sector: Optional[Dict[int, str]] = None,
    output_path: Optional[str] = None
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Evaluate model and generate detailed Excel report with direction scores by sector.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        stock_id_to_ticker: Optional mapping from stock_id to ticker name
        group_id_to_sector: Optional mapping from group_id to sector name
        output_path: Optional path to save Excel report

    Returns:
        Tuple of (metrics, report_df, sector_stats):
        - metrics: Dictionary with overall metrics
        - report_df: DataFrame with detailed predictions
        - sector_stats: Dictionary with per-sector statistics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_stock_ids = []
    all_group_ids = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            stock_id = batch['stock_id'].to(device)
            group_id = batch['group_id'].to(device)
            day = batch['day'].to(device)
            month = batch['month'].to(device)
            target = batch['target'].to(device)

            # Get dividend_flag if available, otherwise use ones (has dividend)
            if 'dividend_flag' in batch:
                dividend_flag = batch['dividend_flag'].to(device)
            else:
                dividend_flag = torch.ones(features.shape[0], features.shape[1],
                                              dtype=torch.long, device=device)

            output = model(features, stock_id, group_id, day, month, dividend_flag)

            all_predictions.extend(output.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

            # Get the first stock_id and group_id from the sequence (representative)
            all_stock_ids.extend(stock_id[:, 0].cpu().numpy().flatten())
            all_group_ids.extend(group_id[:, 0].cpu().numpy().flatten())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    stock_ids = np.array(all_stock_ids)
    group_ids = np.array(all_group_ids)

    # Calculate overall metrics
    metrics = calculate_metrics(predictions, targets)
    returns = calculate_returns(predictions, targets)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
    metrics['max_drawdown'] = calculate_max_drawdown(returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns)
    metrics['total_return'] = np.sum(returns)

    # Create detailed report DataFrame
    report_data = {
        'stock_id': stock_ids,
        'group_id': group_ids,
        'real_target': targets,
        'predict_target': predictions,
    }

    # Add ticker and sector names if mappings provided
    if stock_id_to_ticker:
        report_data['ticker'] = [stock_id_to_ticker.get(sid, f"stock_{sid}") for sid in stock_ids]
    else:
        report_data['ticker'] = [f"stock_{sid}" for sid in stock_ids]

    if group_id_to_sector:
        report_data['sector'] = [group_id_to_sector.get(gid, f"sector_{gid}") for gid in group_ids]
    else:
        report_data['sector'] = [f"sector_{gid}" for gid in group_ids]

    report_df = pd.DataFrame(report_data)

    # Calculate distance (predict - real)
    report_df['distance'] = report_df['predict_target'] - report_df['real_target']

    # Calculate standard deviations
    report_df['std_real'] = report_df.groupby('ticker')['real_target'].transform(lambda x: x.std())
    report_df['std_predict'] = report_df.groupby('ticker')['predict_target'].transform(lambda x: x.std())

    # Calculate direction score (1 if same sign, 0 if different)
    real_direction = np.sign(report_df['real_target'].values)
    predict_direction = np.sign(report_df['predict_target'].values)
    report_df['direction_score'] = (real_direction == predict_direction).astype(int)

    # Calculate sector statistics
    sector_stats = {}
    for sector in report_df['sector'].unique():
        sector_df = report_df[report_df['sector'] == sector]
        total_count = len(sector_df)
        correct_count = sector_df['direction_score'].sum()
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        sector_stats[sector] = {
            'total': total_count,
            'correct': correct_count,
            'accuracy': accuracy,
            'mean_real': sector_df['real_target'].mean(),
            'mean_predict': sector_df['predict_target'].mean(),
            'std_real': sector_df['real_target'].std(),
            'std_predict': sector_df['predict_target'].std(),
        }

    # Save to Excel if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main report
            report_df.to_excel(writer, sheet_name='Predictions', index=False)

            # Sector statistics
            sector_df = pd.DataFrame({
                sector: {
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': stats['accuracy'],
                    'mean_real': stats['mean_real'],
                    'mean_predict': stats['mean_predict'],
                    'std_real': stats['std_real'],
                    'std_predict': stats['std_predict'],
                }
                for sector, stats in sector_stats.items()
            }).T

            sector_df.to_excel(writer, sheet_name='Sector Stats')

            # Overall metrics
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)

    return metrics, report_df, sector_stats


def print_sector_stats(sector_stats: Dict[str, Dict[str, float]]):
    """
    Print sector statistics in formatted way.

    Args:
        sector_stats: Dictionary with per-sector statistics
    """
    print("\n" + "=" * 70)
    print("DIRECTION ACCURACY BY SECTOR")
    print("=" * 70)
    print(f"{'Sector':<30} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<12}")
    print("-" * 70)

    for sector, stats in sorted(sector_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        correct = stats['correct']
        total = stats['total']
        accuracy = stats['accuracy']
        print(f"{sector:<30} | {correct:<10} | {total:<10} | {accuracy:<12.2%}")

    print("-" * 70)
    print("=" * 70)
