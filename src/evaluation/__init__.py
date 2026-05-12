"""
Evaluation module for Multi-Model Financial Forecasting.
"""

from .metrics import (
    calculate_metrics,
    directional_accuracy,
    calculate_returns,
    calculate_turnover,
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
from .meta_labeling import (
    create_meta_labels,
    prepare_meta_label_dataset,
    validate_prediction_target_alignment,
)
from .interpretability import (
    aggregate_attention_by_position,
    attention_to_numpy,
    get_attention_report,
    summarize_attention,
)
from .feature_attribution import (
    AttributionDependencyError,
    aggregate_feature_attributions,
    attribute_batch,
    compute_feature_attribution_report,
)
from .feature_pruning import create_feature_pruning_report

__all__ = [
    'calculate_metrics',
    'directional_accuracy',
    'calculate_returns',
    'calculate_turnover',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_sortino_ratio',
    'evaluate_model',
    'print_metrics',
    'evaluate_model_with_report',
    'print_sector_stats',
    'Validator',
    'Backtester',
    'create_meta_labels',
    'prepare_meta_label_dataset',
    'validate_prediction_target_alignment',
    'aggregate_attention_by_position',
    'attention_to_numpy',
    'get_attention_report',
    'summarize_attention',
    'AttributionDependencyError',
    'aggregate_feature_attributions',
    'attribute_batch',
    'compute_feature_attribution_report',
    'create_feature_pruning_report',
]
