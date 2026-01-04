"""
Data preparation module for prediction/inference.

This module handles:
- Single row data preparation for prediction
- Batch data preparation for prediction
- Feature engineering for new data
- Normalization using fitted scalers
- Creating prediction sequences
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings

from src.config import load_config
from src.config import load_config
from src.data.feature_engineering import FeatureEngineer
from src.utils.logger import get_logger

# Optional financial metrics loader
try:
    from src.data.financial_metrics_loader import FinancialMetricsLoader
    FINANCIAL_METRICS_AVAILABLE = True
except ImportError:
    FINANCIAL_METRICS_AVAILABLE = False


class PredictionPreparator:
    """
    Prepare data for prediction/inference.

    This class handles preparing new data (single row or batch) for prediction
    using a trained model. It applies the same transformations as training:
    - Feature engineering (technical indicators, etc.)
    - Normalization using fitted scalers
    - Sequence creation for RNN models
    - Categorical encoding
    """

    def __init__(
        self,
        data_config,
        model_config,
        scaler_path: Optional[str] = None,
        encoders_path: Optional[str] = None
    ):
        """
        Initialize prediction preparator.

        Args:
            data_config instance
            model_config instance
            scaler_path: Path to saved scaler parameters (optional)
            encoders_path: Path to saved encoders (optional)
        """
        self.data_config = data_config
        self.model_config = model_config
        self.logger = get_logger("prediction_prep", log_dir="logs")

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(data_config)

        # Initialize financial metrics loader (if available)
        if FINANCIAL_METRICS_AVAILABLE:
            self.financial_loader = FinancialMetricsLoader(data_config)
        else:
            self.financial_loader = None
            self.logger.warning("FinancialMetricsLoader not available, financial metrics will use defaults")

        # Load scalers and encoders if paths provided
        self.feature_scaler_params = {}
        self.stock_encoder_mapping = {}
        self.group_encoder_mapping = {}

        if scaler_path:
            self._load_scaler(scaler_path)

        if encoders_path:
            self._load_encoders(encoders_path)

        # Feature columns
        self.feature_cols = []
        self.num_features = 0

    def _load_scaler(self, path: str):
        """Load saved scaler parameters."""
        try:
            import joblib
            params = joblib.load(path)
            self.feature_scaler_params = params.get('feature_scaler', {})
            self.logger.info(f"Loaded scaler parameters from {path}")
        except Exception as e:
            self.logger.warning(f"Could not load scaler from {path}: {e}")

    def _load_encoders(self, path: str):
        """Load saved encoders."""
        try:
            import joblib
            encoders = joblib.load(path)
            self.stock_encoder_mapping = encoders.get('stock_encoder', {})
            self.group_encoder_mapping = encoders.get('group_encoder', {})
            self.logger.info(f"Loaded encoders from {path}")
        except Exception as e:
            self.logger.warning(f"Could not load encoders from {path}: {e}")

    def set_feature_columns(self, feature_cols: List[str]):
        """
        Set the feature columns to use.

        Args:
            feature_cols: List of feature column names
        """
        self.feature_cols = feature_cols
        self.num_features = len(feature_cols)
        self.logger.info(f"Set feature columns: {len(feature_cols)} features")

    def prepare_single_row(
        self,
        data: Dict[str, Union[float, int, str]],
        stock_ticker: str,
        date: Union[str, pd.Timestamp],
        group: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare a single row of data for prediction.

        Args:
            data: Dictionary with feature values. Must include:
                - open, high, low, close, volume (OHLCV)
                - Other optional features will be computed if missing
            stock_ticker: Stock ticker symbol
            date: Date of the data point
            group: Sector/group (optional)

        Returns:
            DataFrame with prepared features (single row)
        """
        self.logger.info(f"Preparing single row for {stock_ticker} on {date}")

        # Create base DataFrame
        df = pd.DataFrame([data])

        # Add metadata columns
        df['tic'] = stock_ticker
        df['date'] = pd.to_datetime(date)

        # Calculate target (placeholder, not used for prediction)
        df['target'] = 0.0

        # Apply feature engineering
        df = self._apply_feature_engineering(df)

        # Add group if provided
        if group is not None:
            df['group'] = group
        else:
            df['group'] = 'Unknown'

        # Encode categoricals
        df = self._encode_categoricals(df)

        return df

    def prepare_batch(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Union[float, int, str]]]],
        calculate_technical_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Prepare a batch of data for prediction.

        Args:
            data: Either a DataFrame or list of dictionaries. If DataFrame,
                  must have columns: date, tic, and at least OHLCV data.
            calculate_technical_indicators: Whether to calculate technical indicators

        Returns:
            DataFrame with prepared features
        """
        self.logger.info(f"Preparing batch data...")

        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Validate required columns
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Add placeholder target
        if 'target' not in df.columns:
            df['target'] = 0.0

        # Apply feature engineering
        if calculate_technical_indicators:
            df = self._apply_feature_engineering(df)

        # Add group if missing
        if 'group' not in df.columns:
            df['group'] = 'Unknown'

        # Encode categoricals
        df = self._encode_categoricals(df)

        self.logger.info(f"Prepared batch: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        # Sort by ticker and date for proper indicator calculation
        df = df.sort_values(['tic', 'date']).reset_index(drop=True)

        # Apply technical indicators
        df = self.feature_engineer.add_technical_indicators(df)

        # Add time features
        df = self.feature_engineer.add_time_features(df)

        # Add financial metrics if available
        # Note: financial_loader.load_metrics_for_ticker() requires ticker-by-ticker loading
        # For prediction, we add default columns that can be overridden if data is available
        financial_cols = ['pe_ratio', 'peg_ratio', 'eps', 'dividend_flag',
                        'roe', 'roi', 'debt_to_equity', 'debt_to_asset',
                        'current_ratio']
        for col in financial_cols:
            if col not in df.columns:
                if col == 'dividend_flag':
                    df[col] = 2  # Default: no dividend
                else:
                    df[col] = 0.0

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using fitted encoders.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categoricals
        """
        result = df.copy()

        # Encode stock ticker
        if 'tic' in result.columns:
            if self.stock_encoder_mapping:
                # Use mapping from saved encoder
                result['tic_id'] = result['tic'].map(self.stock_encoder_mapping)
                unknown_tickers = result[result['tic_id'].isna()]['tic'].unique()
                if len(unknown_tickers) > 0:
                    self.logger.warning(f"Unknown tickers (using new ID): {list(unknown_tickers)}")
                    # Assign new IDs to unknown tickers
                    max_id = max(self.stock_encoder_mapping.values()) if self.stock_encoder_mapping else -1
                    for ticker in unknown_tickers:
                        max_id += 1
                        result.loc[result['tic'] == ticker, 'tic_id'] = max_id
                        self.stock_encoder_mapping[ticker] = max_id

                result['tic_id'] = result['tic_id'].astype(np.int64)
            else:
                # Create new encoding
                unique_tickers = result['tic'].unique()
                ticker_mapping = {t: i for i, t in enumerate(unique_tickers)}
                result['tic_id'] = result['tic'].map(ticker_mapping).astype(np.int64)
                self.stock_encoder_mapping.update(ticker_mapping)

        # Encode group
        if 'group' in result.columns:
            if self.group_encoder_mapping:
                result['group_id'] = result['group'].map(self.group_encoder_mapping)
                unknown_groups = result[result['group_id'].isna()]['group'].unique()
                if len(unknown_groups) > 0:
                    self.logger.warning(f"Unknown groups (using new ID): {list(unknown_groups)}")
                    max_id = max(self.group_encoder_mapping.values()) if self.group_encoder_mapping else -1
                    for group in unknown_groups:
                        max_id += 1
                        result.loc[result['group'] == group, 'group_id'] = max_id
                        self.group_encoder_mapping[group] = max_id

                result['group_id'] = result['group_id'].astype(np.int64)
            else:
                unique_groups = result['group'].unique()
                group_mapping = {g: i for i, g in enumerate(unique_groups)}
                result['group_id'] = result['group'].map(group_mapping).astype(np.int64)
                self.group_encoder_mapping.update(group_mapping)

        return result

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize features using fitted scaler parameters.

        Args:
            df: DataFrame with features to normalize
            feature_cols: List of feature columns to normalize

        Returns:
            DataFrame with normalized features
        """
        result = df.copy()

        if feature_cols is None:
            feature_cols = self.feature_cols

        if not feature_cols:
            self.logger.warning("No feature columns specified, skipping normalization")
            return result

        # Fill NaN values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)

        # Apply normalization using saved parameters
        for col in feature_cols:
            if col not in result.columns:
                self.logger.warning(f"Column {col} not in DataFrame, skipping")
                continue

            if col in ['day', 'month', 'dayofweek', 'dividend_flag']:
                # Don't normalize categorical features
                continue

            if col in self.feature_scaler_params:
                params = self.feature_scaler_params[col]
                method = params.get('method', self.data_config.NORMALIZATION_METHOD)

                if method == 'log_transform':
                    # Log transform: log(x + offset - min) + 1
                    offset = params.get('offset', 1.0)
                    min_val = params.get('min', 0)
                    result[col] = np.log1p(np.maximum(0, result[col] - min_val + offset))

                elif method == 'standard':
                    # Standard scaling: (x - mean) / std
                    mean = params.get('mean', 0)
                    std = params.get('std', 1)
                    result[col] = (result[col] - mean) / (std + 1e-8)

                elif method == 'minmax':
                    # Min-max scaling: (x - min) / (max - min)
                    min_val = params.get('min', 0)
                    max_val = params.get('max', 1)
                    result[col] = (result[col] - min_val) / (max_val - min_val + 1e-8)

                elif method == 'robust':
                    # Robust scaling: (x - median) / IQR
                    median = params.get('median', 0)
                    iqr = params.get('iqr', 1)
                    result[col] = (result[col] - median) / (iqr + 1e-8)

        return result

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create sequences from prepared DataFrame.

        Args:
            df: Prepared DataFrame with normalized features
            sequence_length: Length of sequences (default from config)

        Returns:
            Dictionary with sequences (features, stock_id, group_id, day, month, dividend_flag)
        """
        if sequence_length is None:
            sequence_length = self.data_config.data.sequences.SEQUENCE_LENGTH

        self.logger.info(f"Creating sequences with length {sequence_length}...")

        # Determine feature columns
        exclude = {'date', 'tic', 'tic_id', 'group', 'group_id', 'target', 'split'}
        feature_cols = [c for c in df.columns if c not in exclude]

        if not feature_cols:
            raise ValueError("No feature columns available for sequence creation")

        # Group by stock
        sequences_list = []

        for ticker in df['tic'].unique():
            stock_df = df[df['tic'] == ticker].sort_values('date').copy()

            # Need at least sequence_length rows
            if len(stock_df) < sequence_length:
                self.logger.warning(f"Stock {ticker} has only {len(stock_df)} rows, need {sequence_length}")
                continue

            # Get feature matrix
            feature_matrix = stock_df[feature_cols].values.astype(np.float32)

            # Create rolling sequences
            for i in range(len(stock_df) - sequence_length + 1):
                seq_features = feature_matrix[i:i + sequence_length]

                # Check for NaN
                if np.isnan(seq_features).any():
                    continue

                # Get categorical features
                seq_start = i
                seq_end = i + sequence_length

                seq_data = {
                    'features': seq_features,
                    'stock_id': stock_df.iloc[seq_start:seq_end]['tic_id'].values,
                    'group_id': stock_df.iloc[seq_start:seq_end]['group_id'].values,
                    'day': stock_df.iloc[seq_start:seq_end]['day'].values,
                    'month': stock_df.iloc[seq_start:seq_end]['month'].values,
                }

                if 'dividend_flag' in stock_df.columns:
                    seq_data['dividend_flag'] = stock_df.iloc[seq_start:seq_end]['dividend_flag'].values.astype(np.int32)
                else:
                    seq_data['dividend_flag'] = np.ones(sequence_length, dtype=np.int32)

                sequences_list.append(seq_data)

        if not sequences_list:
            raise ValueError("No sequences could be created from the data")

        # Combine all sequences
        result = {
            'features': np.stack([s['features'] for s in sequences_list]),
            'stock_id': np.stack([s['stock_id'] for s in sequences_list]),
            'group_id': np.stack([s['group_id'] for s in sequences_list]),
            'day': np.stack([s['day'] for s in sequences_list]),
            'month': np.stack([s['month'] for s in sequences_list]),
            'dividend_flag': np.stack([s['dividend_flag'] for s in sequences_list]),
        }

        self.logger.info(f"Created {len(result['features'])} sequences")
        return result

    def prepare_for_prediction(
        self,
        data: Union[pd.DataFrame, List[Dict], Dict],
        normalize: bool = True,
        create_seqs: bool = True,
        stock_ticker: Optional[str] = None,
        date: Optional[Union[str, pd.Timestamp]] = None,
        group: Optional[str] = None
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Complete preparation pipeline for prediction.

        Args:
            data: Input data (DataFrame, list of dicts, or single dict)
            normalize: Whether to normalize features
            create_seqs: Whether to create sequences
            stock_ticker: Stock ticker (required for single dict input)
            date: Date (required for single dict input)
            group: Group/sector (optional)

        Returns:
            Prepared DataFrame or sequences dictionary
        """
        # Handle different input types
        if isinstance(data, dict):
            if stock_ticker is None or date is None:
                raise ValueError("stock_ticker and date required for single dict input")
            df = self.prepare_single_row(data, stock_ticker, date, group)
        elif isinstance(data, list) or isinstance(data, pd.DataFrame):
            df = self.prepare_batch(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Normalize features
        if normalize and self.feature_scaler_params:
            df = self.normalize_features(df)

        # Create sequences
        if create_seqs:
            return self.create_sequences(df)

        return df

    def save_encoders(self, path: str):
        """
        Save encoders for later use.

        Args:
            path: Path to save encoders
        """
        import joblib
        encoders = {
            'stock_encoder': self.stock_encoder_mapping,
            'group_encoder': self.group_encoder_mapping,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, path)
        self.logger.info(f"Saved encoders to {path}")

    def load_from_preprocessor(self, preprocessor_path: str):
        """
        Load scaler and encoders from a saved preprocessor.

        Args:
            preprocessor_path: Path to saved preprocessor state
        """
        import joblib
        state = joblib.load(preprocessor_path)

        # Load scaler parameters
        if 'feature_scaler' in state:
            self.feature_scaler_params = state['feature_scaler']
            self.logger.info("Loaded feature scaler parameters")

        # Load encoders
        if 'stock_encoder_classes' in state:
            classes = state['stock_encoder_classes']
            self.stock_encoder_mapping = {cls: i for i, cls in enumerate(classes)}
            self.logger.info(f"Loaded stock encoder with {len(classes)} classes")

        if 'group_encoder_classes' in state:
            classes = state['group_encoder_classes']
            self.group_encoder_mapping = {cls: i for i, cls in enumerate(classes)}
            self.logger.info(f"Loaded group encoder with {len(classes)} classes")

        # Load feature columns if available
        if 'feature_cols' in state:
            self.set_feature_columns(state['feature_cols'])


def create_prediction_preparator(
    data_config = None,
    model_config = None,
    preprocessor_path: Optional[str] = None
) -> PredictionPreparator:
    """
    Create a PredictionPreparator instance.

    Args:
        data_config instance
        model_config instance
        preprocessor_path: Path to saved preprocessor state

    Returns:
        PredictionPreparator instance
    """
    if data_config is None:
        from src.config import load_config
        data_config = load_config('main')

    if model_config is None:
        from src.config import load_config
        model_config = load_config('model')

    preparator = PredictionPreparator(data_config, model_config)

    # Load from preprocessor if path provided
    if preprocessor_path:
        preparator.load_from_preprocessor(preprocessor_path)

    return preparator
