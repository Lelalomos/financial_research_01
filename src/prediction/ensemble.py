"""
Model ensembling utilities for prediction.

The ensemble path is opt-in and wraps existing Predictor instances so the
single-model prediction behavior remains unchanged.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.config import load_config
from src.prediction.predictor import Predictor
from src.utils.logger import get_logger


class EnsembleCompatibilityError(ValueError):
    """Raised when checkpoints cannot be safely ensembled."""


class EnsemblePredictor(Predictor):
    """
    Predictor that averages outputs from multiple compatible checkpoints.

    The public prediction methods mirror Predictor. Data preparation is shared
    through the first predictor after all checkpoint metadata has been validated.
    """

    def __init__(
        self,
        model_paths: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        model_config=None,
        data_config=None,
        preprocessor_path: Optional[str] = None,
        device: Optional[str] = None,
        require_matching_features: bool = True,
        require_matching_target_normalization: bool = True,
    ):
        if len(model_paths) < 2:
            raise ValueError("EnsemblePredictor requires at least two model paths")

        self.model_paths = [Path(path) for path in model_paths]
        self.model_config = model_config or load_config('model')
        self.data_config = data_config or load_config('main')
        self.logger = get_logger("ensemble_predictor", log_dir="logs")
        self.require_matching_features = require_matching_features
        self.require_matching_target_normalization = require_matching_target_normalization

        self.weights = self._normalize_weights(weights, len(self.model_paths))
        self.predictors = [
            Predictor(
                model_path=str(path),
                model_config=self.model_config,
                data_config=self.data_config,
                preprocessor_path=preprocessor_path,
                device=device,
            )
            for path in self.model_paths
        ]

        self._validate_compatible_predictors()

        first = self.predictors[0]
        self.model_path = first.model_path
        self.device = first.device
        self.preparator = first.preparator
        self.model = first.model
        self.model_metadata = {
            'ensemble': True,
            'members': [predictor.model_metadata for predictor in self.predictors],
        }

    @staticmethod
    def _normalize_weights(
        weights: Optional[Sequence[float]],
        expected_length: int,
    ) -> np.ndarray:
        if weights is None:
            return np.full(expected_length, 1.0 / expected_length, dtype=np.float64)

        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (expected_length,):
            raise ValueError("Ensemble weights length must match model paths length")
        if np.any(weight_array < 0):
            raise ValueError("Ensemble weights must be non-negative")

        total = weight_array.sum()
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value")
        return weight_array / total

    def _feature_cols_for(self, predictor: Predictor) -> List[str]:
        feature_cols = predictor.model_metadata.get('feature_cols')
        if feature_cols is None:
            feature_cols = predictor.preparator.feature_cols
        if not feature_cols:
            raise EnsembleCompatibilityError(
                f"Checkpoint {predictor.model_path} is missing feature_cols metadata"
            )
        return list(feature_cols)

    def _target_normalization_for(self, predictor: Predictor) -> Tuple[object, object]:
        metadata = predictor.model_metadata.get('metadata', {})
        normalize_target = predictor.model_metadata.get('normalize_target')
        target_threshold = predictor.model_metadata.get('target_threshold')
        target_normalization = predictor.model_metadata.get('target_normalization')

        if target_normalization is None:
            target_normalization = metadata.get('target_normalization')

        if isinstance(target_normalization, dict):
            normalize_target = target_normalization.get('NORMALIZE_TARGET', normalize_target)
            target_threshold = target_normalization.get('TARGET_THRESHOLD', target_threshold)

        if normalize_target is None:
            normalize_target = metadata.get('NORMALIZE_TARGET', metadata.get('normalize_target'))
        if target_threshold is None:
            target_threshold = metadata.get('TARGET_THRESHOLD', metadata.get('target_threshold'))

        if normalize_target is None or target_threshold is None:
            raise EnsembleCompatibilityError(
                f"Checkpoint {predictor.model_path} is missing target normalization metadata"
            )

        return normalize_target, float(target_threshold)

    def _validate_compatible_predictors(self) -> None:
        first = self.predictors[0]
        reference_num_features = first.model_metadata.get('num_features')
        reference_feature_cols = (
            self._feature_cols_for(first)
            if self.require_matching_features
            else None
        )
        reference_target = (
            self._target_normalization_for(first)
            if self.require_matching_target_normalization
            else None
        )

        for predictor in self.predictors[1:]:
            num_features = predictor.model_metadata.get('num_features')
            if num_features != reference_num_features:
                raise EnsembleCompatibilityError(
                    "Ensemble checkpoints have different num_features: "
                    f"{reference_num_features} != {num_features}"
                )

            if self.require_matching_features:
                feature_cols = self._feature_cols_for(predictor)
                if feature_cols != reference_feature_cols:
                    raise EnsembleCompatibilityError(
                        "Ensemble checkpoints have different feature_cols order or values"
                    )

            if self.require_matching_target_normalization:
                target_signature = self._target_normalization_for(predictor)
                if target_signature != reference_target:
                    raise EnsembleCompatibilityError(
                        "Ensemble checkpoints have different target normalization metadata"
                    )

    def predict(
        self,
        data: Union[pd.DataFrame, List[Dict], Dict[str, np.ndarray]],
        return_raw: bool = False,
    ) -> np.ndarray:
        predictions = [
            predictor.predict(data, return_raw=return_raw)
            for predictor in self.predictors
        ]

        reference_shape = predictions[0].shape
        for prediction in predictions[1:]:
            if prediction.shape != reference_shape:
                raise EnsembleCompatibilityError(
                    "Ensemble member predictions returned different shapes"
                )

        stacked = np.stack(predictions, axis=0)
        return np.tensordot(self.weights, stacked, axes=(0, 0))

    def get_model_info(self) -> Dict:
        first_info = self.predictors[0].get_model_info()
        return {
            'model_path': [str(path) for path in self.model_paths],
            'model_type': 'ensemble',
            'member_model_types': [
                predictor.get_model_info()['model_type']
                for predictor in self.predictors
            ],
            'weights': self.weights.tolist(),
            'num_features': first_info['num_features'],
            'num_stocks': first_info['num_stocks'],
            'num_groups': first_info['num_groups'],
            'device': first_info['device'],
            'training_epochs': None,
            'best_val_loss': None,
            'feature_cols': first_info['feature_cols'],
        }


def create_ensemble_predictor(
    model_paths: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    model_config=None,
    data_config=None,
    preprocessor_path: Optional[str] = None,
    device: Optional[str] = None,
    require_matching_features: bool = True,
    require_matching_target_normalization: bool = True,
) -> EnsemblePredictor:
    """Create an EnsemblePredictor instance."""
    return EnsemblePredictor(
        model_paths=model_paths,
        weights=weights,
        model_config=model_config,
        data_config=data_config,
        preprocessor_path=preprocessor_path,
        device=device,
        require_matching_features=require_matching_features,
        require_matching_target_normalization=require_matching_target_normalization,
    )


def create_ensemble_predictor_from_config(
    model_config=None,
    data_config=None,
    preprocessor_path: Optional[str] = None,
    device: Optional[str] = None,
) -> EnsemblePredictor:
    """Create an EnsemblePredictor from config/model.json settings."""
    model_config = model_config or load_config('model')
    ensemble_config = model_config.model.ensemble

    if not ensemble_config.ENABLED:
        raise ValueError("model.ensemble.ENABLED must be true to create an ensemble predictor")

    return create_ensemble_predictor(
        model_paths=ensemble_config.CHECKPOINT_PATHS,
        weights=ensemble_config.WEIGHTS,
        model_config=model_config,
        data_config=data_config,
        preprocessor_path=preprocessor_path,
        device=device,
        require_matching_features=ensemble_config.REQUIRE_MATCHING_FEATURES,
        require_matching_target_normalization=ensemble_config.REQUIRE_MATCHING_TARGET_NORMALIZATION,
    )
