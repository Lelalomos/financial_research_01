"""
Data module for Multi-Model Financial Forecasting.
"""

from importlib import import_module

__all__ = [
    "DataDownloader",
    "FeatureEngineer",
    "DataPreprocessor",
    "FinancialDataset",
    "LazyFinancialDataset",
    "SequenceDataset",
    "create_data_loaders",
    "create_lazy_data_loaders",
    "sample_stocks_by_group",
    "get_sampling_stats",
    "DatasetValidator",
    "validate_dataset",
    "check_feature_consistency",
    "TimeSeriesFold",
    "purged_time_series_split",
    "walk_forward_split",
    "MarketRegimeDetector",
]

_EXPORTS = {
    "DataDownloader": (".downloader", "DataDownloader"),
    "FeatureEngineer": (".feature_engineering", "FeatureEngineer"),
    "DataPreprocessor": (".preprocessing", "DataPreprocessor"),
    "FinancialDataset": (".dataset", "FinancialDataset"),
    "LazyFinancialDataset": (".dataset", "LazyFinancialDataset"),
    "SequenceDataset": (".dataset", "SequenceDataset"),
    "create_data_loaders": (".dataset", "create_data_loaders"),
    "create_lazy_data_loaders": (".dataset", "create_lazy_data_loaders"),
    "sample_stocks_by_group": (".sampling", "sample_stocks_by_group"),
    "get_sampling_stats": (".sampling", "get_sampling_stats"),
    "DatasetValidator": (".validation", "DatasetValidator"),
    "validate_dataset": (".validation", "validate_dataset"),
    "check_feature_consistency": (".validation", "check_feature_consistency"),
    "TimeSeriesFold": (".time_series_split", "TimeSeriesFold"),
    "purged_time_series_split": (".time_series_split", "purged_time_series_split"),
    "walk_forward_split": (".time_series_split", "walk_forward_split"),
    "MarketRegimeDetector": (".regime", "MarketRegimeDetector"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
