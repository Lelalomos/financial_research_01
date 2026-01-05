"""
Evaluation module for CRNN Financial Prediction Model.
"""

from .metrics import (
    calculate_metrics,
    directional_accuracy,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    evaluate_model,
    print_metrics,
    evaluate_model_with_report,
    print_sector_stats
)
from .validator import Validator
from .backtester import Backtester

__all__ = [
    'calculate_metrics',
    'directional_accuracy',
    'calculate_returns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_sortino_ratio',
    'evaluate_model',
    'print_metrics',
    'evaluate_model_with_report',
    'print_sector_stats',
    'Validator',
    'Backtester',
]
