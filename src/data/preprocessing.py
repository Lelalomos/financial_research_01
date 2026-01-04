"""
Preprocessing for CRNN Financial Prediction Model.

This module handles:
- Log transform normalization
- Time-based train/validation/test splitting
- Sequence creation for RNN models
- Categorical encoding (stock_id, group_id)
- Dataset column validation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import gc

from src.config import load_config
from src.utils.logger import get_logger
from .validation import validate_dataset, check_feature_consistency


class DataPreprocessor:
    """
    Preprocessor for financial time series data.

    Handles:
    - Normalization (log transform, standard, minmax, robust)
    - Time-based splitting (train 70%, val 10%, test 20%)
    - Sequence creation for RNN
    - Categorical encoding
    """

    def __init__(self, config=None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration object (defaults to load_config('main') if None)
        """
        if config is None:
            config = load_config('main')
        self.config = config
        self.logger = get_logger("preprocessing", log_dir="logs")

        # Scalers and encoders (fit on training data)
        self.feature_scaler = None
        self.target_scaler = None
        self.stock_encoder = LabelEncoder()
        self.group_encoder = LabelEncoder()

        # Store normalization parameters for inverse transform
        self.normalization_params = {}

    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize features using configured method.

        Args:
            df: DataFrame with features to normalize
            fit: Whether to fit the scaler (True for training data)
            feature_cols: List of feature columns to normalize

        Returns:
            DataFrame with normalized features
        """
        self.logger.info(f"Normalizing features using {self.config.data.normalization.NORMALIZATION_METHOD}...")

        result = df.copy()

        if feature_cols is None:
            # Exclude non-feature columns
            exclude = {'date', 'tic', 'tic_id', 'group', 'group_id', 'target', 'split'}
            feature_cols = [c for c in result.columns if c not in exclude]

        # Fill NaN/None/empty values with 0 before normalization
        # Apply to all numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        nan_count_before = result[numeric_cols].isna().sum().sum()
        if nan_count_before > 0:
            self.logger.info(f"Filling {nan_count_before} NaN values with 0...")
            result[numeric_cols] = result[numeric_cols].fillna(0)

        # Handle each feature based on normalization method
        for col in feature_cols:
            if col in ['day', 'month', 'dayofweek', 'dividend_flag']:
                # Don't normalize categorical features (they use embeddings)
                continue

            result[col] = self._normalize_column(
                result[col].values,
                col,
                fit=fit
            )

        self.logger.info(f"Normalized {len(feature_cols)} features")
        return result

    def _normalize_column(
        self,
        data: np.ndarray,
        col_name: str,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize a single column.

        Args:
            data: Column data
            col_name: Column name
            fit: Whether to fit scaler

        Returns:
            Normalized data
        """
        method = self.config.data.normalization.NORMALIZATION_METHOD

        # Handle NaN values
        mask = ~np.isnan(data)
        if not np.any(mask):
            return data  # All NaN, return as is

        valid_data = data[mask]

        if method == 'log_transform':
            # Log transform: log1p(x - min(x)) to handle negative values
            if fit:
                min_val = np.min(valid_data)
                self.normalization_params[col_name] = {'min': min_val}

            min_val = self.normalization_params[col_name]['min']
            shifted = valid_data - min_val + self.config.data.normalization.LOG_TRANSFORM_OFFSET
            normalized = np.log1p(np.maximum(shifted, 0))

        elif method == 'standard':
            # StandardScaler: (x - mean) / std
            if fit or self.feature_scaler is None:
                if self.feature_scaler is None:
                    self.feature_scaler = {}
                scaler = StandardScaler()
                normalized = scaler.fit_transform(valid_data.reshape(-1, 1)).flatten()
                self.feature_scaler[col_name] = scaler
            else:
                scaler = self.feature_scaler[col_name]
                normalized = scaler.transform(valid_data.reshape(-1, 1)).flatten()

        elif method == 'minmax':
            # MinMaxScaler: (x - min) / (max - min)
            if fit or self.feature_scaler is None:
                if self.feature_scaler is None:
                    self.feature_scaler = {}
                scaler = MinMaxScaler()
                normalized = scaler.fit_transform(valid_data.reshape(-1, 1)).flatten()
                self.feature_scaler[col_name] = scaler
            else:
                scaler = self.feature_scaler[col_name]
                normalized = scaler.transform(valid_data.reshape(-1, 1)).flatten()

        elif method == 'robust':
            # RobustScaler: (x - median) / IQR
            if fit or self.feature_scaler is None:
                if self.feature_scaler is None:
                    self.feature_scaler = {}
                scaler = RobustScaler()
                normalized = scaler.fit_transform(valid_data.reshape(-1, 1)).flatten()
                self.feature_scaler[col_name] = scaler
            else:
                scaler = self.feature_scaler[col_name]
                normalized = scaler.transform(valid_data.reshape(-1, 1)).flatten()

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Reconstruct full array with NaN values
        result = np.full_like(data, np.nan, dtype=float)
        result[mask] = normalized

        return result

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables (tic, group).

        Args:
            df: DataFrame with 'tic' column
            fit: Whether to fit encoders

        Returns:
            DataFrame with encoded columns
        """
        self.logger.info("Encoding categorical variables...")

        result = df.copy()

        # Encode stock ticker
        if 'tic' in result.columns:
            if fit:
                result['tic_id'] = self.stock_encoder.fit_transform(result['tic'])
            else:
                # Handle unseen tickers
                known_tickers = set(self.stock_encoder.classes_)
                result['tic_id'] = result['tic'].apply(
                    lambda x: self.stock_encoder.transform([x])[0] if x in known_tickers else -1
                )

            self.logger.info(f"Encoded {len(self.stock_encoder.classes_)} unique tickers")

        # Encode group (if exists)
        if 'group' in result.columns:
            if fit:
                result['group_id'] = self.group_encoder.fit_transform(result['group'])
            else:
                known_groups = set(self.group_encoder.classes_)
                result['group_id'] = result['group'].apply(
                    lambda x: self.group_encoder.transform([x])[0] if x in known_groups else -1
                )

            self.logger.info(f"Encoded {len(self.group_encoder.classes_)} unique groups")

        return result

    def time_based_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data by time (train 70%, val 10%, test 20%).

        Split is done GLOBALLY by date to avoid overlaps between splits.
        All stocks share the same date ranges for each split.

        Args:
            df: DataFrame with date column
            date_col: Name of date column

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        self.logger.info("=" * 60)
        self.logger.info("TIME-BASED DATA SPLIT (GLOBAL DATE RANGES)")
        self.logger.info("=" * 60)

        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Get all unique dates across ALL stocks, sorted
        all_dates = sorted(df[date_col].unique())
        n_dates = len(all_dates)

        self.logger.info(f"Total unique dates in dataset: {n_dates}")
        self.logger.info(f"Date range: {all_dates[0]} to {all_dates[-1]}")

        # Calculate split points based on percentage of dates
        train_end_idx = int(n_dates * self.config.data.splits.TRAIN_RATIO)
        val_end_idx = int(n_dates * (self.config.data.splits.TRAIN_RATIO + self.config.data.splits.VAL_RATIO))

        # Get date boundaries for each split
        # Train: oldest dates (first 70%)
        # Test: middle dates (next 10%) - swapped with val
        # Val: newest dates (last 20%) - swapped with test
        train_dates = set(all_dates[:train_end_idx])
        test_dates = set(all_dates[train_end_idx:val_end_idx])  # Middle -> test
        val_dates = set(all_dates[val_end_idx:])  # Newest -> val

        self.logger.info(f"Train dates: {all_dates[0]} to {all_dates[train_end_idx-1]} ({len(train_dates)} dates)")
        self.logger.info(f"Test dates: {all_dates[train_end_idx]} to {all_dates[val_end_idx-1]} ({len(test_dates)} dates)")
        self.logger.info(f"Val dates: {all_dates[val_end_idx]} to {all_dates[-1]} ({len(val_dates)} dates)")

        # Assign split labels based on global date ranges
        df['split'] = 'val'
        df.loc[df[date_col].isin(train_dates), 'split'] = 'train'
        df.loc[df[date_col].isin(test_dates), 'split'] = 'test'

        # Create splits and ensure each is sorted by tic_id, then date (old to new)
        splits = {
            'train': df[df['split'] == 'train'].drop(columns=['split']).sort_values(['tic_id', date_col]),
            'val': df[df['split'] == 'val'].drop(columns=['split']).sort_values(['tic_id', date_col]),
            'test': df[df['split'] == 'test'].drop(columns=['split']).sort_values(['tic_id', date_col])
        }

        # Log split statistics
        total_rows = sum(len(splits[s]) for s in ['train', 'val', 'test'])
        for split_name in ['train', 'val', 'test']:
            count = len(splits[split_name])
            pct = count / total_rows * 100 if total_rows > 0 else 0
            date_range = ""
            if not splits[split_name].empty:
                min_date = splits[split_name][date_col].min()
                max_date = splits[split_name][date_col].max()
                date_range = f" ({min_date} to {max_date})"

            self.logger.info(f"  {split_name}: {count:,} rows ({pct:.1f}%){date_range}")

        # Verify no date overlap
        self._verify_split_integrity(splits, date_col)

        self.logger.info("=" * 60)

        return splits

    def _verify_split_integrity(self, splits: Dict[str, pd.DataFrame], date_col: str):
        """Verify that splits don't have overlapping dates."""
        train_dates = set(splits['train'][date_col].dt.date) if not splits['train'].empty else set()
        val_dates = set(splits['val'][date_col].dt.date) if not splits['val'].empty else set()
        test_dates = set(splits['test'][date_col].dt.date) if not splits['test'].empty else set()

        # Check for overlaps
        train_val_overlap = train_dates & val_dates
        train_test_overlap = train_dates & test_dates
        val_test_overlap = val_dates & test_dates

        if train_val_overlap:
            self.logger.warning(f"Train/Val date overlap: {len(train_val_overlap)} dates")
        if train_test_overlap:
            self.logger.warning(f"Train/Test date overlap: {len(train_test_overlap)} dates")
        if val_test_overlap:
            self.logger.warning(f"Val/Test date overlap: {len(val_test_overlap)} dates")

        if not (train_val_overlap or train_test_overlap or val_test_overlap):
            self.logger.info("Split integrity verified: No date overlaps")

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sequence_length: Optional[int] = None,
        chunk_size: int = 10000,
        output_dir: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create sequences for RNN training using memory-efficient approach.

        Creates sliding window sequences of length sequence_length.
        Each sequence becomes one training sample.
        Uses memory-mapped files to avoid RAM spikes during concatenation.

        Args:
            df: DataFrame with features and identifiers
            feature_cols: List of feature column names
            sequence_length: Sequence length (default from config)
            chunk_size: Number of sequences per chunk before writing to disk
            output_dir: Directory to store temporary memmap files (default: temp/memmap)

        Returns:
            Dictionary with arrays: features, stock_id, group_id, day, month, target
        """
        import tempfile
        import shutil

        if sequence_length is None:
            sequence_length = self.config.data.sequences.SEQUENCE_LENGTH

        self.logger.info(f"Creating sequences (length={sequence_length}) with memory-efficient approach...")

        # Create temporary directory for memmap files
        if output_dir is None:
            memmap_dir = Path(tempfile.mkdtemp(prefix="sequences_memmap_"))
        else:
            memmap_dir = Path(output_dir)
            memmap_dir.mkdir(parents=True, exist_ok=True)

        # Process each stock separately (sorted by tic_id to ensure consistent order)
        if 'tic_id' in df.columns:
            stocks = sorted(df['tic_id'].unique())
            stock_col = 'tic_id'
        else:
            stocks = sorted(df['tic'].unique())
            stock_col = 'tic'

        # First pass: count sequences and determine shapes
        self.logger.info("  Pass 1: Counting sequences...")
        total_sequences = 0
        num_features = len(feature_cols)

        for stock in stocks:
            stock_df = df[df[stock_col] == stock]
            if len(stock_df) >= sequence_length + self.config.data.sequences.PREDICTION_HORIZON:
                valid_count = len(stock_df) - sequence_length - self.config.data.sequences.PREDICTION_HORIZON + 1
                total_sequences += valid_count

        self.logger.info(f"  Total sequences to create: {total_sequences:,}")

        if total_sequences == 0:
            self.logger.warning("No valid sequences can be created!")
            return {
                'features': np.array([]),
                'stock_id': np.array([]),
                'group_id': np.array([]),
                'day': np.array([]),
                'month': np.array([]),
                'target': np.array([])
            }

        # Create memory-mapped arrays
        self.logger.info("  Pass 2: Creating memory-mapped arrays...")
        memmap_files = {}

        memmap_files['features'] = (
            memmap_dir / 'features.dat',
            (total_sequences, sequence_length, num_features),
            np.float32
        )
        memmap_files['stock_id'] = (
            memmap_dir / 'stock_id.dat',
            (total_sequences, sequence_length),
            np.int64
        )
        memmap_files['group_id'] = (
            memmap_dir / 'group_id.dat',
            (total_sequences, sequence_length),
            np.int64
        )
        memmap_files['day'] = (
            memmap_dir / 'day.dat',
            (total_sequences, sequence_length),
            np.int32
        )
        memmap_files['month'] = (
            memmap_dir / 'month.dat',
            (total_sequences, sequence_length),
            np.int32
        )
        memmap_files['dividend_flag'] = (
            memmap_dir / 'dividend_flag.dat',
            (total_sequences, sequence_length),
            np.int32
        )
        memmap_files['target'] = (
            memmap_dir / 'target.dat',
            (total_sequences,),
            np.float32
        )

        # Initialize memmap arrays
        memmaps = {}
        for key, (filepath, shape, dtype) in memmap_files.items():
            memmaps[key] = np.memmap(
                filepath,
                dtype=dtype,
                mode='w+',
                shape=shape
            )

        # Second pass: write sequences directly to memmap
        self.logger.info("  Pass 3: Writing sequences to memory-mapped files...")
        current_idx = 0
        chunk_count = 0
        temp_features = []
        temp_stock_id = []
        temp_group_id = []
        temp_day = []
        temp_month = []
        temp_dividend_flag = []
        temp_target = []

        for stock_idx, stock in enumerate(stocks):
            stock_df = df[df[stock_col] == stock].sort_values('date').copy()

            # Check if we have enough data
            if len(stock_df) < sequence_length + self.config.data.sequences.PREDICTION_HORIZON:
                del stock_df
                continue

            # Get feature matrix
            feature_matrix = stock_df[feature_cols].values.astype(np.float32)

            # Create sequences
            for i in range(len(stock_df) - sequence_length - self.config.data.sequences.PREDICTION_HORIZON + 1):
                # Sequence features
                seq_features = feature_matrix[i:i + sequence_length]

                # Check for NaN in sequence
                if np.isnan(seq_features).any():
                    continue

                # Get target (at end of horizon)
                target_idx = i + sequence_length + self.config.data.sequences.PREDICTION_HORIZON - 1
                target = stock_df.iloc[target_idx]['target']

                if np.isnan(target):
                    continue

                # Get categorical features for the entire sequence
                seq_start = i
                seq_end = i + sequence_length

                # Write to memmap when chunk is full
                if len(temp_target) >= chunk_size:
                    # Flush chunk to memmap
                    start_idx = current_idx
                    end_idx = current_idx + len(temp_target)

                    memmaps['features'][start_idx:end_idx] = np.array(temp_features)
                    memmaps['stock_id'][start_idx:end_idx] = np.array(temp_stock_id)
                    memmaps['group_id'][start_idx:end_idx] = np.array(temp_group_id)
                    memmaps['day'][start_idx:end_idx] = np.array(temp_day)
                    memmaps['month'][start_idx:end_idx] = np.array(temp_month)
                    memmaps['dividend_flag'][start_idx:end_idx] = np.array(temp_dividend_flag)
                    memmaps['target'][start_idx:end_idx] = np.array(temp_target)

                    current_idx = end_idx
                    chunk_count += 1

                    # Clear temporary lists
                    temp_features = []
                    temp_stock_id = []
                    temp_group_id = []
                    temp_day = []
                    temp_month = []
                    temp_dividend_flag = []
                    temp_target = []

                    if chunk_count % 10 == 0:
                        self.logger.info(f"  Processed {current_idx:,} / {total_sequences:,} sequences ({100*current_idx/total_sequences:.1f}%)")

                # Accumulate in temporary lists
                temp_features.append(seq_features)
                temp_stock_id.append(stock_df.iloc[seq_start:seq_end]['tic_id'].values)
                temp_day.append(stock_df.iloc[seq_start:seq_end]['day'].values)
                temp_month.append(stock_df.iloc[seq_start:seq_end]['month'].values)

                if 'group_id' in stock_df.columns:
                    temp_group_id.append(stock_df.iloc[seq_start:seq_end]['group_id'].values)
                else:
                    temp_group_id.append(np.zeros(sequence_length, dtype=np.int64))

                if 'dividend_flag' in stock_df.columns:
                    temp_dividend_flag.append(stock_df.iloc[seq_start:seq_end]['dividend_flag'].values.astype(np.int32))
                else:
                    temp_dividend_flag.append(np.ones(sequence_length, dtype=np.int32))  # Default to flag 1

                temp_target.append(target)

            # Clean up stock data
            del stock_df, feature_matrix
            gc.collect()

        # Write remaining sequences
        if temp_target:
            start_idx = current_idx
            end_idx = current_idx + len(temp_target)

            memmaps['features'][start_idx:end_idx] = np.array(temp_features)
            memmaps['stock_id'][start_idx:end_idx] = np.array(temp_stock_id)
            memmaps['group_id'][start_idx:end_idx] = np.array(temp_group_id)
            memmaps['day'][start_idx:end_idx] = np.array(temp_day)
            memmaps['month'][start_idx:end_idx] = np.array(temp_month)
            memmaps['dividend_flag'][start_idx:end_idx] = np.array(temp_dividend_flag)
            memmaps['target'][start_idx:end_idx] = np.array(temp_target)

            current_idx = end_idx

        self.logger.info(f"  Wrote {current_idx:,} sequences to memory-mapped files")

        # Load back into regular numpy arrays (this is the peak memory, but much lower than before)
        self.logger.info("  Pass 4: Loading sequences from memory-mapped files...")
        sequences = {}

        for key in memmaps.keys():
            # Load in read-only mode
            sequences[key] = np.array(memmaps[key][:])
            self.logger.info(f"    Loaded {key}: {sequences[key].shape}")

        # Flush and close memmaps
        for key, mmap in memmaps.items():
            mmap.flush()
            del mmap

        # Clean up temporary files
        try:
            shutil.rmtree(memmap_dir)
        except Exception as e:
            self.logger.warning(f"Could not remove temporary memmap directory: {e}")

        self.logger.info(f"Created {len(sequences['target'])} sequences")
        if len(sequences['features']) > 0:
            self.logger.info(f"  Features shape: {sequences['features'].shape}, dtype: {sequences['features'].dtype}")

        return sequences

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        feature_cols: Optional[List[str]] = None,
        export_pre_normalize: Optional[str] = None,
        export_normalized: Optional[str] = None,
        validate_columns: bool = None
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict]]:
        """
        Run full preprocessing pipeline.

        Args:
            df: Raw DataFrame with features
            fit: Whether to fit scalers/encoders (True for training data)
            feature_cols: List of feature columns
            export_pre_normalize: Path to export pre-normalization data (parquet format)
            export_normalized: Path to export normalized data (parquet format)
            validate_columns: Whether to validate columns (default from config if None)

        Returns:
            Tuple of (preprocessed_df, splits_dict, sequences_dict, info_dict)
        """
        self.logger.info("=" * 60)
        self.logger.info("PREPROCESSING PIPELINE")
        self.logger.info("=" * 60)

        # Check if validation should be performed
        if validate_columns is None:
            validate_columns = getattr(
                self.config.data.validation,
                'VALIDATE_ON_LOAD',
                True
            )

        # Validate columns before processing
        if validate_columns:
            validate_dataset(df, "input_data", config=self.config, raise_on_missing=True)

        result = df.copy()

        # 1. Encode categorical
        result = self.encode_categorical(result, fit=fit)

        # 2. Export pre-normalization data if requested
        if export_pre_normalize is not None:
            self.logger.info(f"Exporting pre-normalization data to {export_pre_normalize}...")
            Path(export_pre_normalize).parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(export_pre_normalize, index=False)
            self.logger.info(f"Pre-normalization data exported successfully")

        # 3. Normalize features
        result = self.normalize_features(result, fit=fit, feature_cols=feature_cols)

        # 3.3. Fill NaN values in target column with 0 (if any remain after feature engineering)
        if 'target' in result.columns:
            target_nan_count = result['target'].isna().sum()
            if target_nan_count > 0:
                self.logger.info(f"Filling {target_nan_count} NaN values in target column with 0...")
                result['target'] = result['target'].fillna(0)

        # 3.5. Export normalized data if requested
        if export_normalized is not None:
            self.logger.info(f"Exporting normalized data to {export_normalized}...")
            Path(export_normalized).parent.mkdir(parents=True, exist_ok=True)

            # Drop unused columns before export (keep only numeric columns)
            columns_to_drop = ['date', 'tic', 'group', 'split']
            export_df = result.drop(columns=[col for col in columns_to_drop if col in result.columns])

            # Verify all columns are numeric
            non_numeric = export_df.select_dtypes(exclude=['number']).columns
            if len(non_numeric) > 0:
                self.logger.warning(f"Non-numeric columns found: {list(non_numeric)}")
            else:
                self.logger.info(f"All columns are numeric: {list(export_df.columns)}")

            export_df.to_parquet(export_normalized, index=False)
            self.logger.info(f"Normalized data exported successfully")

        # 4. Time-based split
        splits = self.time_based_split(result)

        # 5. Create sequences for each split
        if feature_cols is None:
            exclude = {'date', 'tic', 'tic_id', 'group', 'group_id', 'target', 'split',
                      'day', 'month', 'dayofweek', 'dividend_flag'}
            feature_cols = [c for c in result.columns if c not in exclude]

        sequences = {}
        for split_name, split_df in splits.items():
            if not split_df.empty:
                self.logger.info(f"Creating sequences for {split_name} split...")
                sequences[split_name] = self.create_sequences(
                    split_df,
                    feature_cols=feature_cols
                )
                # Free memory
                del split_df
                gc.collect()

        # Info
        num_groups = len(self.group_encoder.classes_) if hasattr(self.group_encoder, 'classes_') else 0
        # Ensure at least 1 group for embedding layer (group 0 is default)
        num_groups = max(num_groups, 1)

        info = {
            'num_stocks': len(self.stock_encoder.classes_) if fit else len(result['tic'].unique()),
            'num_groups': num_groups,
            'feature_cols': feature_cols,
            'num_features': len(feature_cols),
            'sequence_length': self.config.data.sequences.SEQUENCE_LENGTH,
            'prediction_horizon': self.config.data.sequences.PREDICTION_HORIZON,
        }

        self.logger.info("=" * 60)
        self.logger.info("PREPROCESSING COMPLETE")
        for split_name in ['train', 'val', 'test']:
            if split_name in sequences and 'target' in sequences[split_name]:
                count = len(sequences[split_name]['target'])
                self.logger.info(f"  {split_name}: {count:,} sequences")
        self.logger.info("=" * 60)

        return result, splits, sequences, info
