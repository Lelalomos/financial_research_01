"""
Predictor module for financial prediction model.

This module handles:
- Loading trained models
- Batch prediction
- Single row prediction
- Output formatting and post-processing
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings

from src.data.prediction_prep import PredictionPreparator, create_prediction_preparator
from src.models.model_output import get_output_components, get_prediction_tensor
from src.utils.logger import get_logger


class Predictor:
    """
    Predictor for financial time series forecasting.

    Handles loading trained models and making predictions on new data.
    Supports both single row and batch predictions.
    """

    def __init__(
        self,
        model_path: str,
        model_config = None,
        data_config = None,
        preprocessor_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            model_config instance
            data_config instance
            preprocessor_path: Path to saved preprocessor state
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        if model_config is None:
            from src.config import load_config
            model_config = load_config('model')
        self.model_config = model_config
        if data_config is None:
            from src.config import load_config
            data_config = load_config('main')
        self.data_config = data_config
        self.logger = get_logger("predictor", log_dir="logs")

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.logger.info(f"Using device: {self.device}")

        # Initialize prediction preparator
        self.preparator = create_prediction_preparator(
            data_config=self.data_config,
            model_config=self.model_config,
            preprocessor_path=preprocessor_path
        )

        # Load model
        self.model = None
        self.model_metadata = {}
        self._load_model()

    def _load_model(self):
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from {self.model_path}...")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)

        # Load model metadata - store all checkpoint info for access
        self.model_metadata = {
            'metadata': checkpoint.get('metadata', {}),
            'num_features': checkpoint.get('num_features'),
            'num_stocks': checkpoint.get('num_stocks'),
            'num_groups': checkpoint.get('num_groups'),
            'feature_cols': checkpoint.get('feature_cols'),
            'target_normalization': checkpoint.get('target_normalization'),
            'normalize_target': checkpoint.get('normalize_target'),
            'target_threshold': checkpoint.get('target_threshold'),
            'regime_params': checkpoint.get('regime_params'),
        }

        # Get model parameters
        num_features = checkpoint.get('num_features')
        num_stocks = checkpoint.get('num_stocks')
        num_groups = checkpoint.get('num_groups')

        if num_features is None or num_stocks is None:
            raise ValueError(f"Checkpoint missing required parameters: num_features={num_features}, num_stocks={num_stocks}")

        # Set feature columns
        if 'feature_cols' in checkpoint:
            self.preparator.set_feature_columns(checkpoint['feature_cols'])

        regime_params = checkpoint.get('regime_params') or checkpoint.get('metadata', {}).get('regime_params')
        if regime_params:
            self.preparator.regime_params = regime_params

        # Create model based on type
        model_type = self.model_metadata.get('metadata', {}).get('model_type', 'lstm3_attention')
        self.model = self._create_model(model_type, num_features, num_stocks, num_groups)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load scaler parameters if available
        if 'feature_scaler' in checkpoint:
            self.preparator.feature_scaler_params = checkpoint['feature_scaler']

        # Load encoders if available
        if 'stock_encoder_classes' in checkpoint:
            classes = checkpoint['stock_encoder_classes']
            self.preparator.stock_encoder_mapping = {cls: i for i, cls in enumerate(classes)}

        if 'group_encoder_classes' in checkpoint:
            classes = checkpoint['group_encoder_classes']
            self.preparator.group_encoder_mapping = {cls: i for i, cls in enumerate(classes)}

        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"  Type: {model_type}")
        self.logger.info(f"  Features: {num_features}")
        self.logger.info(f"  Stocks: {num_stocks}")
        self.logger.info(f"  Groups: {num_groups}")

    def _create_model(self, model_type: str, num_features: int, num_stocks: int, num_groups: int):
        """Create model instance using the centralized model registry."""
        from src.models import create_model
        feature_cols = self.model_metadata.get('feature_cols')
        return create_model(
            model_type=model_type,
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
            config=self.model_config,
            feature_cols=feature_cols,
        )

    def predict(
        self,
        data: Union[pd.DataFrame, List[Dict], Dict[str, np.ndarray]],
        return_raw: bool = False,
        return_components: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, Dict]]:
        """
        Make predictions on prepared data.

        Args:
            data: Either prepared sequences dict, DataFrame, or list of dicts
            return_raw: If True, return raw predictions without inverse transform

        Returns:
            Predictions array (percent change)
        """
        # If already sequences, use directly
        if isinstance(data, dict) and 'features' in data:
            sequences = data
        else:
            # Prepare data first
            sequences = self.preparator.prepare_for_prediction(
                data,
                normalize=True,
                create_seqs=True
            )

        # Convert to tensors
        features = torch.FloatTensor(sequences['features']).to(self.device)
        stock_id = torch.LongTensor(sequences['stock_id']).to(self.device)
        group_id = torch.LongTensor(sequences['group_id']).to(self.device)
        day = torch.LongTensor(sequences['day']).to(self.device)
        month = torch.LongTensor(sequences['month']).to(self.device)
        dividend_flag = torch.LongTensor(sequences['dividend_flag']).to(self.device)

        # Make predictions
        with torch.no_grad():
            output = self.model(
                features=features,
                stock_id=stock_id,
                group_id=group_id,
                day=day,
                month=month,
                dividend_flag=dividend_flag
            )

        components = {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in get_output_components(output).items()
        }
        predictions = get_prediction_tensor(output).detach().cpu().numpy()

        # Apply inverse tanh transform if target was normalized
        if not return_raw and self.data_config.data.sequences.NORMALIZE_TARGET:
            # The target was normalized using tanh: target = tanh(x / threshold)
            # Inverse: x = threshold * atanh(target)
            threshold = self.data_config.data.sequences.TARGET_THRESHOLD
            # Clamp to valid range for atanh
            predictions = np.clip(predictions, -0.99, 0.99)
            predictions = threshold * np.arctanh(predictions)
            components["prediction"] = predictions

        if return_components:
            return components

        return predictions

    def predict_single(
        self,
        data: Dict[str, Union[float, int, str]],
        stock_ticker: str,
        date: Union[str, pd.Timestamp],
        group: Optional[str] = None,
        return_raw: bool = False
    ) -> Dict[str, Union[float, Dict]]:
        """
        Make prediction for a single data point.

        Args:
            data: Dictionary with feature values (OHLCV required)
            stock_ticker: Stock ticker symbol
            date: Date of the data point
            group: Sector/group (optional)
            return_raw: If True, return raw predictions

        Returns:
            Dictionary with prediction and metadata
        """
        self.logger.info(f"Predicting for {stock_ticker} on {date}...")

        # Prepare data
        df = self.preparator.prepare_single_row(data, stock_ticker, date, group)

        # Normalize
        if self.preparator.feature_scaler_params:
            df = self.preparator.normalize_features(df)

        # Create sequences
        try:
            sequences = self.preparator.create_sequences(df)
        except ValueError as e:
            return {
                'error': str(e),
                'stock_ticker': stock_ticker,
                'date': str(date),
                'prediction': None
            }

        # Make prediction
        components = self.predict(sequences, return_raw=return_raw, return_components=True)
        predictions = components["prediction"]
        result = {
            'stock_ticker': stock_ticker,
            'date': str(date),
            'prediction': float(predictions[0][0]) if len(predictions) > 0 else None,
            'raw_prediction': float(predictions[0][0]) if return_raw and len(predictions) > 0 else None,
            'num_sequences': len(predictions),
            'metadata': self.model_metadata
        }
        if "future_return_path" in components:
            result["future_return_path"] = components["future_return_path"][0].tolist()
        if "future_regime" in components:
            result["future_regime"] = int(components["future_regime"][0])
        if "future_ohlcv" in components:
            result["future_ohlcv"] = components["future_ohlcv"][0].tolist()
        return result

    def predict_batch(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        return_raw: bool = False,
        return_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame, Dict]:
        """
        Make predictions on a batch of data.

        Args:
            data: DataFrame or list of dicts with features
            return_raw: If True, return raw predictions
            return_dataframe: If True, return results as DataFrame

        Returns:
            Predictions array or DataFrame with predictions
        """
        self.logger.info(f"Predicting batch...")

        # Prepare data
        df = self.preparator.prepare_batch(data)
        original_df = df.copy()

        # Normalize
        if self.preparator.feature_scaler_params:
            df = self.preparator.normalize_features(df)

        # Create sequences
        try:
            sequences = self.preparator.create_sequences(df)
        except ValueError as e:
            return {
                'error': str(e),
                'predictions': None
            }

        # Make predictions
        components = self.predict(sequences, return_raw=return_raw, return_components=True)
        predictions = components["prediction"]

        if return_dataframe:
            # Create result DataFrame
            results = []

            # Map sequences back to original data
            seq_idx = 0
            for ticker in original_df['tic'].unique():
                stock_df = original_df[original_df['tic'] == ticker].sort_values('date')

                # Count valid sequences for this stock
                seq_len = self.data_config.data.sequences.SEQUENCE_LENGTH
                num_possible = max(0, len(stock_df) - seq_len + 1)

                for i in range(num_possible):
                    if seq_idx >= len(predictions):
                        break

                    date = stock_df.iloc[i + seq_len - 1]['date']

                    results.append({
                        'stock_ticker': ticker,
                        'date': str(date),
                        'prediction': float(predictions[seq_idx][0])
                    })
                    if "future_regime" in components:
                        results[-1]["future_regime"] = int(components["future_regime"][seq_idx])
                    if "future_return_path" in components:
                        results[-1]["future_return_path"] = components["future_return_path"][seq_idx].tolist()
                    if "future_ohlcv" in components:
                        results[-1]["future_ohlcv"] = components["future_ohlcv"][seq_idx].tolist()
                    seq_idx += 1

            return pd.DataFrame(results)

        if return_raw and len(components) > 1:
            return components
        return predictions

    def predict_from_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        file_format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Make predictions from an input file.

        Args:
            input_path: Path to input file (csv, parquet, excel)
            output_path: Path to save results (optional)
            file_format: Format of input file ('csv', 'parquet', 'excel')

        Returns:
            DataFrame with predictions
        """
        self.logger.info(f"Loading data from {input_path}...")

        # Load input file
        if file_format == 'csv':
            df = pd.read_csv(input_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(input_path)
        elif file_format == 'excel':
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Make predictions
        result_df = self.predict_batch(df, return_dataframe=True)

        # Save results if path provided
        if output_path:
            self.logger.info(f"Saving results to {output_path}...")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if output_path.endswith('.csv'):
                result_df.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                result_df.to_parquet(output_path, index=False)
            else:
                result_df.to_csv(output_path, index=False)

        return result_df

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        metadata = self.model_metadata.get('metadata', {})
        return {
            'model_path': str(self.model_path),
            'model_type': metadata.get('model_type', 'lstm3_attention'),
            'num_features': self.model_metadata.get('num_features'),
            'num_stocks': self.model_metadata.get('num_stocks'),
            'num_groups': self.model_metadata.get('num_groups'),
            'device': str(self.device),
            'training_epochs': metadata.get('epoch'),
            'best_val_loss': metadata.get('best_val_loss'),
            'feature_cols': self.preparator.feature_cols,
            'supports_rich_output': metadata.get('model_type') == 'chronos_rich',
        }


def create_predictor(
    model_path: str,
    model_config = None,
    data_config = None,
    preprocessor_path: Optional[str] = None,
    device: Optional[str] = None
) -> Predictor:
    """
    Create a Predictor instance.

    Args:
        model_path: Path to trained model checkpoint
        model_config instance
        data_config instance
        preprocessor_path: Path to saved preprocessor state
        device: Device to use ('cuda', 'cpu', or None for auto)

    Returns:
        Predictor instance
    """
    return Predictor(
        model_path=model_path,
        model_config=model_config,
        data_config=data_config,
        preprocessor_path=preprocessor_path,
        device=device
    )
