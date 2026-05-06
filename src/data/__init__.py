"""
Data module for Multi-Model Financial Forecasting.
"""

from .downloader import DataDownloader
from .feature_engineering import FeatureEngineer
from .preprocessing import DataPreprocessor
from .dataset import FinancialDataset, SequenceDataset, create_data_loaders
from .sampling import sample_stocks_by_group, get_sampling_stats
from .validation import DatasetValidator, validate_dataset, check_feature_consistency
from .time_series_split import TimeSeriesFold, purged_time_series_split, walk_forward_split
from .regime import MarketRegimeDetector

__all__ = [
    'DataDownloader',
    'FeatureEngineer',
    'DataPreprocessor',
    'FinancialDataset',
    'SequenceDataset',
    'create_data_loaders',
    'sample_stocks_by_group',
    'get_sampling_stats',
    'DatasetValidator',
    'validate_dataset',
    'check_feature_consistency',
    'TimeSeriesFold',
    'purged_time_series_split',
    'walk_forward_split',
    'MarketRegimeDetector',
]
