"""
PyTorch Dataset for Multi-Model Financial Forecasting.

This module provides:
- FinancialDataset class for PyTorch DataLoader
- Proper handling of sequences, embeddings, and targets
- Batch generation for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from src.config import load_config
from src.utils.logger import get_logger


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy arrays to a sensible torch tensor dtype."""
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        return torch.as_tensor(array, dtype=torch.float32)
    if np.issubdtype(array.dtype, np.integer):
        return torch.as_tensor(array, dtype=torch.long)
    if np.issubdtype(array.dtype, np.bool_):
        return torch.as_tensor(array, dtype=torch.bool)
    raise TypeError(f"Unsupported optional sequence dtype: {array.dtype}")


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial time series data.

    Returns samples containing:
    - features: (seq_len, num_features)
    - stock_id: (seq_len,)
    - group_id: (seq_len,)
    - day: (seq_len,)
    - month: (seq_len,)
    - dividend_flag: (seq_len,) - 1=has dividend, 2=no dividend
    - target: (1,) - percent change prediction
    """

    def __init__(
        self,
        sequences: Dict[str, np.ndarray],
        config=None
    ):
        """
        Initialize dataset.

        Args:
            sequences: Dictionary with keys: features, stock_id, group_id, day, month, dividend_flag, target
            config: Configuration object (defaults to load_config('model') if None)
        """
        self.config = config or load_config('model')
        self.logger = get_logger("dataset", log_dir="logs")

        # Validate sequences
        required_keys = ['features', 'stock_id', 'day', 'month', 'target']
        for key in required_keys:
            if key not in sequences:
                raise ValueError(f"Missing required key: {key}")
            if len(sequences[key]) == 0:
                raise ValueError(f"Empty sequences for key: {key}")

        # Use group_id if available, otherwise use zeros
        if 'group_id' not in sequences or len(sequences['group_id']) == 0:
            self.logger.warning("No group_id provided, using zeros")
            sequences['group_id'] = np.zeros_like(sequences['stock_id'])

        # Use dividend_flag if available, otherwise use ones (default: has dividend)
        if 'dividend_flag' not in sequences or len(sequences['dividend_flag']) == 0:
            self.logger.warning("No dividend_flag provided, using ones (has dividend)")
            sequences['dividend_flag'] = np.ones_like(sequences['stock_id'], dtype=np.int32)

        self.features = torch.FloatTensor(sequences['features'])
        self.stock_id = torch.LongTensor(sequences['stock_id'])
        self.group_id = torch.LongTensor(sequences['group_id'])
        self.day = torch.LongTensor(sequences['day'])
        self.month = torch.LongTensor(sequences['month'])
        dividend_flag = np.asarray(sequences['dividend_flag'])
        if dividend_flag.ndim == 3 and dividend_flag.shape[-1] == 1:
            dividend_flag = np.squeeze(dividend_flag, axis=-1)
        if dividend_flag.shape != sequences['stock_id'].shape:
            raise ValueError(
                f"dividend_flag shape {dividend_flag.shape} must match stock_id shape {sequences['stock_id'].shape}"
            )
        if not np.isin(dividend_flag, [0, 1, 2]).all():
            raise ValueError("dividend_flag values must be 0, 1, or 2")
        self.dividend_flag = torch.LongTensor(dividend_flag)
        self.target = torch.FloatTensor(sequences['target']).unsqueeze(-1)  # (n_samples,) -> (n_samples, 1)
        self.num_samples = len(self.target)
        base_keys = {'features', 'stock_id', 'group_id', 'day', 'month', 'dividend_flag', 'target'}
        self.optional_tensors: Dict[str, torch.Tensor] = {}
        for key, value in sequences.items():
            if key in base_keys:
                continue
            if len(value) != self.num_samples:
                raise ValueError(f"Optional sequence key '{key}' must match sample count {self.num_samples}")
            self.optional_tensors[key] = _to_tensor(value)

        self.seq_len = self.features.shape[1]
        self.num_features = self.features.shape[2]

        self.logger.info(f"Dataset initialized with {self.num_samples} samples")
        self.logger.info(f"  Sequence length: {self.seq_len}")
        self.logger.info(f"  Num features: {self.num_features}")

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with features, embeddings, and target
        """
        sample = {
            'features': self.features[idx],      # (seq_len, num_features)
            'stock_id': self.stock_id[idx],     # (seq_len,)
            'group_id': self.group_id[idx],     # (seq_len,)
            'day': self.day[idx],               # (seq_len,)
            'month': self.month[idx],           # (seq_len,)
            'dividend_flag': self.dividend_flag[idx],  # (seq_len,)
            'target': self.target[idx]          # (1,)
        }
        for key, tensor in self.optional_tensors.items():
            sample[key] = tensor[idx]
        return sample

    def get_embedding_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for embeddings.

        Returns:
            Dictionary with max values for each categorical variable
        """
        return {
            'num_stocks': int(self.stock_id.max()) + 1,
            'num_groups': int(self.group_id.max()) + 1,
            'num_days': 32,   # Days 1-31 + padding
            'num_months': 13,  # Months 1-12 + padding
            'num_dividend_flags': 3  # 0=padding, 1=has dividend, 2=no dividend
        }

    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched dictionary with properly shaped tensors
        """
        # Stack all items in batch
        result = {}
        for key in batch[0].keys():
            if key == 'target':
                # Target should be (batch_size, 1)
                result[key] = torch.stack([item[key] for item in batch])
            else:
                # Other tensors are already shaped correctly
                result[key] = torch.stack([item[key] for item in batch])

        return result


class LazyFinancialDataset(Dataset):
    """
    Dataset that creates one sequence window at access time from normalized split rows.

    This avoids materializing the full `(n_samples, seq_len, n_features)` tensor in
    RAM before training starts.
    """

    def __init__(
        self,
        split_df,
        feature_cols: List[str],
        data_config=None,
    ):
        self.data_config = data_config or load_config('main')
        self.logger = get_logger("lazy_dataset", log_dir="logs")
        self.feature_cols = feature_cols
        self.sequence_length = self.data_config.data.sequences.SEQUENCE_LENGTH
        self.prediction_horizon = self.data_config.data.sequences.PREDICTION_HORIZON
        self.stride = self.data_config.data.sequences.STRIDE

        if split_df is None or split_df.empty:
            raise ValueError("split_df must contain normalized rows for lazy dataset creation")
        if not feature_cols:
            raise ValueError("feature_cols cannot be empty for lazy dataset creation")

        stock_col = 'tic_id' if 'tic_id' in split_df.columns else 'tic'
        self.stock_groups = []
        self.sample_index = []
        self.max_stock_id = 0
        self.max_group_id = 0

        for _, stock_df in split_df.groupby(stock_col, sort=True):
            ordered = stock_df.sort_values('date').reset_index(drop=True)
            features = ordered[feature_cols].to_numpy(dtype=np.float32, copy=True)
            stock_id = ordered['tic_id'].to_numpy(dtype=np.int64, copy=True)
            group_id = (
                ordered['group_id'].to_numpy(dtype=np.int64, copy=True)
                if 'group_id' in ordered.columns
                else np.zeros(len(ordered), dtype=np.int64)
            )
            day = ordered['day'].to_numpy(dtype=np.int32, copy=True)
            month = ordered['month'].to_numpy(dtype=np.int32, copy=True)
            dividend_flag = (
                ordered['dividend_flag'].to_numpy(dtype=np.int32, copy=True)
                if 'dividend_flag' in ordered.columns
                else np.ones(len(ordered), dtype=np.int32)
            )
            target = ordered['target'].to_numpy(dtype=np.float32, copy=True)

            group_idx = len(self.stock_groups)
            self.stock_groups.append(
                {
                    'features': features,
                    'stock_id': stock_id,
                    'group_id': group_id,
                    'day': day,
                    'month': month,
                    'dividend_flag': dividend_flag,
                    'target': target,
                }
            )

            self.max_stock_id = max(self.max_stock_id, int(stock_id.max()))
            self.max_group_id = max(self.max_group_id, int(group_id.max()))

            valid_window_count = len(ordered) - self.sequence_length - self.prediction_horizon + 1
            if valid_window_count <= 0:
                continue

            nan_feature_rows = np.isnan(features).any(axis=1).astype(np.int32)
            nan_feature_prefix = np.concatenate(([0], np.cumsum(nan_feature_rows)))

            for start in range(0, valid_window_count, self.stride):
                end = start + self.sequence_length
                target_idx = end + self.prediction_horizon - 1
                if np.isnan(target[target_idx]):
                    continue
                if nan_feature_prefix[end] - nan_feature_prefix[start] > 0:
                    continue
                self.sample_index.append((group_idx, start, target_idx))

        if not self.sample_index:
            raise ValueError("No valid lazy sequence windows could be created from split_df")

        self.num_samples = len(self.sample_index)
        self.num_features = len(self.feature_cols)

        self.logger.info(f"Lazy dataset initialized with {self.num_samples} samples")
        self.logger.info(f"  Sequence length: {self.sequence_length}")
        self.logger.info(f"  Num features: {self.num_features}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        group_idx, start, target_idx = self.sample_index[idx]
        group = self.stock_groups[group_idx]
        end = start + self.sequence_length

        return {
            'features': torch.from_numpy(group['features'][start:end]),
            'stock_id': torch.from_numpy(group['stock_id'][start:end]),
            'group_id': torch.from_numpy(group['group_id'][start:end]),
            'day': torch.from_numpy(group['day'][start:end]),
            'month': torch.from_numpy(group['month'][start:end]),
            'dividend_flag': torch.from_numpy(group['dividend_flag'][start:end]),
            'target': torch.tensor([group['target'][target_idx]], dtype=torch.float32),
        }

    def get_embedding_sizes(self) -> Dict[str, int]:
        return {
            'num_stocks': self.max_stock_id + 1,
            'num_groups': self.max_group_id + 1,
            'num_days': 32,
            'num_months': 13,
            'num_dividend_flags': 3,
        }

    @staticmethod
    def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
        return FinancialDataset.collate_fn(batch)


def create_data_loaders(
    train_sequences: Dict[str, np.ndarray],
    val_sequences: Optional[Dict[str, np.ndarray]] = None,
    test_sequences: Optional[Dict[str, np.ndarray]] = None,
    config=None
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Args:
        train_sequences: Training sequences dictionary
        val_sequences: Validation sequences dictionary (optional)
        test_sequences: Test sequences dictionary (optional)
        config: Configuration object (defaults to load_config('model') if None)

    Returns:
        Dictionary with DataLoader instances
    """
    config = config or load_config('model')

    # Create datasets
    datasets = {
        'train': FinancialDataset(train_sequences, config)
    }

    if val_sequences is not None and len(val_sequences.get('target', [])) > 0:
        datasets['val'] = FinancialDataset(val_sequences, config)

    if test_sequences is not None and len(test_sequences.get('target', [])) > 0:
        datasets['test'] = FinancialDataset(test_sequences, config)

    # Create data loaders
    loaders = {}

    for split_name, dataset in datasets.items():
        batch_size = config.model.validation.VAL_BATCH_SIZE if split_name != 'train' else config.model.training.BATCH_SIZE
        # Use same batch size for validation if not specified
        if batch_size is None:
            batch_size = config.model.training.BATCH_SIZE
        shuffle = (split_name == 'train')
        drop_last = shuffle and len(dataset) >= batch_size

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.model.device.NUM_WORKERS,
            pin_memory=config.model.device.PIN_MEMORY,
            prefetch_factor=config.model.device.PREFETCH_FACTOR if config.model.device.NUM_WORKERS > 0 else None,
            collate_fn=FinancialDataset.collate_fn,
            drop_last=drop_last
        )

        loaders[split_name] = loader

    return loaders


def create_lazy_data_loaders(
    train_df,
    feature_cols: List[str],
    data_config,
    model_config=None,
    val_df=None,
    test_df=None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders backed by lazy sequence datasets from normalized split rows.
    """
    model_config = model_config or load_config('model')
    datasets = {
        'train': LazyFinancialDataset(train_df, feature_cols=feature_cols, data_config=data_config)
    }

    if val_df is not None and not val_df.empty:
        datasets['val'] = LazyFinancialDataset(val_df, feature_cols=feature_cols, data_config=data_config)

    if test_df is not None and not test_df.empty:
        datasets['test'] = LazyFinancialDataset(test_df, feature_cols=feature_cols, data_config=data_config)

    loaders = {}
    for split_name, dataset in datasets.items():
        batch_size = (
            model_config.model.validation.VAL_BATCH_SIZE
            if split_name != 'train'
            else model_config.model.training.BATCH_SIZE
        )
        if batch_size is None:
            batch_size = model_config.model.training.BATCH_SIZE
        shuffle = split_name == 'train'
        drop_last = shuffle and len(dataset) >= batch_size

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=model_config.model.device.NUM_WORKERS,
            pin_memory=model_config.model.device.PIN_MEMORY,
            prefetch_factor=model_config.model.device.PREFETCH_FACTOR if model_config.model.device.NUM_WORKERS > 0 else None,
            collate_fn=LazyFinancialDataset.collate_fn,
            drop_last=drop_last,
        )

    return loaders


class SequenceDataset(Dataset):
    """
    Alternative dataset that returns data without embeddings.
    Useful for models that don't use embeddings.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.

        Args:
            features: (n_samples, seq_len, num_features)
            targets: (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (features, target)
        """
        return self.features[idx], self.targets[idx]
