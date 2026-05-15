"""
Typed configuration schemas.

The project still uses JSON files and the lightweight Config wrapper for
backward-compatible attribute access. These schemas validate loaded JSON early
so bad types or impossible values fail near startup instead of deep in training.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
    """Base schema that preserves unknown keys for backward compatibility."""

    model_config = ConfigDict(extra='allow', strict=True)


class SplitConfig(StrictBaseModel):
    TRAIN_RATIO: float = Field(gt=0.0, lt=1.0)
    TEST_RATIO: float = Field(gt=0.0, lt=1.0)
    VAL_RATIO: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode='after')
    def ratios_sum_to_one(self):
        total = self.TRAIN_RATIO + self.TEST_RATIO + self.VAL_RATIO
        if abs(total - 1.0) > 1e-6:
            raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")
        return self


class SequenceConfig(StrictBaseModel):
    SEQUENCE_LENGTH: int = Field(gt=0)
    PREDICTION_HORIZON: int = Field(gt=0)
    TARGET_THRESHOLD: float = Field(gt=0.0)
    NORMALIZE_TARGET: bool
    STRIDE: int = Field(gt=0)


class NormalizationConfig(StrictBaseModel):
    NORMALIZATION_METHOD: Literal['log_transform', 'standard', 'minmax', 'robust']
    LOG_TRANSFORM_OFFSET: float = Field(gt=0.0)


class DownloadConfig(StrictBaseModel):
    DOWNLOAD_RETRY_ATTEMPTS: int = Field(ge=1)
    DOWNLOAD_RETRY_DELAY: int = Field(ge=0)


class FeatureConfig(StrictBaseModel):
    FEATURE_FLAGS: Dict[str, bool]


class DatasetModeConfig(StrictBaseModel):
    MODE: Literal['precomputed_sequences', 'on_the_fly_sequences'] = 'precomputed_sequences'


class CandlestickConfig(StrictBaseModel):
    USE_CANDLESTICK_PATTERNS: bool = True
    EXCLUDE_PATTERNS: List[str] = Field(default_factory=list)


class GeometricConfig(StrictBaseModel):
    CHANNEL_WINDOW: int = Field(default=20, ge=2)
    SWING_WINDOW: int = Field(default=20, ge=2)
    TRENDLINE_WINDOW: int = Field(default=30, ge=2)
    TRENDLINE_TOLERANCE: float = Field(default=1e-4, gt=0.0)
    TRENDLINE_MAX_ITERATIONS: int = Field(default=100, ge=1)
    ENABLE_ATR_FEATURE: bool = True
    ENABLE_ROC_FEATURE: bool = True
    ENABLE_BB_WIDTH_FEATURE: bool = True
    ENABLE_SLOPE_FEATURES: bool = True
    ENABLE_CHANNEL_COMPRESSION: bool = False
    ENABLE_CHANNEL_POSITION: bool = False
    ENABLE_SWING_DISTANCE: bool = False
    ENABLE_SWING_TIME_DISTANCE: bool = False
    ENABLE_OPTIMIZED_TRENDLINES: bool = False
    ENABLE_OPTIMIZED_CHANNEL_WIDTH: bool = False


class RegimeConfig(StrictBaseModel):
    ENABLED: bool = False
    METHOD: Literal['quantile'] = 'quantile'
    PROXY_COLUMN: str = 'vix'
    N_REGIMES: int = Field(default=3, ge=2, le=3)
    LOW_QUANTILE: float = Field(default=0.33, gt=0.0, lt=1.0)
    HIGH_QUANTILE: float = Field(default=0.66, gt=0.0, lt=1.0)
    DEFAULT_REGIME: int = Field(default=1, ge=0)

    @model_validator(mode='after')
    def validate_regime_settings(self):
        if self.N_REGIMES == 3 and self.LOW_QUANTILE >= self.HIGH_QUANTILE:
            raise ValueError("LOW_QUANTILE must be less than HIGH_QUANTILE")
        if self.DEFAULT_REGIME >= self.N_REGIMES:
            raise ValueError("DEFAULT_REGIME must be less than N_REGIMES")
        return self


class SourceConfig(StrictBaseModel):
    START_DATE: str
    END_DATE: Optional[str]
    SP500_TICKER_SOURCE: str
    USE_YFINANCE_LIVE: bool
    INDEX_FILE: str
    RAW_DATA_INDEX_PATH: str
    VIX_SYMBOL: str
    COMMODITIES: Dict[str, str]
    TREASURY_YIELDS: List[str]


class MainDataConfig(StrictBaseModel):
    sources: SourceConfig
    splits: SplitConfig
    sequences: SequenceConfig
    dataset: DatasetModeConfig = Field(default_factory=DatasetModeConfig)
    candlestick: CandlestickConfig = Field(default_factory=CandlestickConfig)
    geometric: GeometricConfig = Field(default_factory=GeometricConfig)
    features: FeatureConfig
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    normalization: NormalizationConfig
    download: DownloadConfig


class MainConfigSchema(StrictBaseModel):
    data: MainDataConfig


class EmbeddingConfig(StrictBaseModel):
    EMBEDDING_DIM_STOCK: int = Field(gt=0)
    EMBEDDING_DIM_GROUP: int = Field(gt=0)
    EMBEDDING_DIM_DAY: int = Field(gt=0)
    EMBEDDING_DIM_MONTH: int = Field(gt=0)
    EMBEDDING_DIM_DIVIDEND_FLAG: int = Field(gt=0)
    DROPOUT_EMBEDDING: float = Field(ge=0.0, lt=1.0)


class TrainingConfig(StrictBaseModel):
    LEARNING_RATE: float = Field(gt=0.0)
    WEIGHT_DECAY: float = Field(ge=0.0)
    BATCH_SIZE: int = Field(gt=0)
    NUM_EPOCHS: int = Field(gt=0)
    EARLY_STOPPING_PATIENCE: int = Field(ge=0)
    GRADIENT_CLIP_VALUE: float = Field(ge=0.0)
    ACCUMULATION_STEPS: int = Field(gt=0)
    OPTIMIZER: Literal['adam', 'adamw', 'sgd', 'rmsprop']
    SCHEDULER: Optional[Literal['reduce_on_plateau', 'cosine', 'step']]
    SCHEDULER_PARAMS: Dict[str, Dict[str, Any]]
    USE_MIXED_PRECISION: bool


class TrainingBackendConfig(StrictBaseModel):
    DEFAULT: Literal['lightning', 'custom'] = 'lightning'
    FALLBACK: Literal['custom'] = 'custom'
    ALLOW_CUSTOM_FALLBACK: bool = True


class ModelSelectionConfig(StrictBaseModel):
    DEFAULT_MODEL_TYPE: str = 'crnn_attention'


class LossConfig(StrictBaseModel):
    LOSS_TYPE: Literal['mse', 'mae', 'smooth_l1', 'huber', 'directional', 'sharpe', 'directional_mse', 'directional_huber']
    HUBER_DELTA: float = Field(gt=0.0)
    DIRECTIONAL_ALPHA: float = Field(default=0.1, ge=0.0)
    SHARPE_EPSILON: float = Field(default=1e-6, gt=0.0)


class DeviceConfig(StrictBaseModel):
    DEVICE: Literal['cuda', 'cpu']
    NUM_WORKERS: int = Field(ge=0)
    PIN_MEMORY: bool
    PREFETCH_FACTOR: int = Field(ge=1)


class CheckpointConfig(StrictBaseModel):
    CHECKPOINT_DIR: str
    SAVE_BEST_ONLY: bool
    SAVE_LAST_N: int = Field(ge=1)
    CHECKPOINT_FREQUENCY: int = Field(ge=1)


class EnsembleConfig(StrictBaseModel):
    ENABLED: bool = False
    CHECKPOINT_PATHS: List[str] = Field(default_factory=list)
    WEIGHTS: Optional[List[float]] = None
    REQUIRE_MATCHING_FEATURES: bool = True
    REQUIRE_MATCHING_TARGET_NORMALIZATION: bool = True

    @model_validator(mode='after')
    def validate_ensemble_settings(self):
        if self.ENABLED and len(self.CHECKPOINT_PATHS) < 2:
            raise ValueError("ENABLED ensemble requires at least two CHECKPOINT_PATHS")
        if self.WEIGHTS is not None:
            if len(self.WEIGHTS) != len(self.CHECKPOINT_PATHS):
                raise ValueError("WEIGHTS length must match CHECKPOINT_PATHS length")
            if any(weight < 0.0 for weight in self.WEIGHTS):
                raise ValueError("WEIGHTS must be non-negative")
            if sum(self.WEIGHTS) <= 0.0:
                raise ValueError("WEIGHTS must sum to a positive value")
        return self


class LoggingConfig(StrictBaseModel):
    LOG_FREQUENCY: int = Field(gt=0)
    TENSORBOARD_DIR: Optional[str]
    WANDB_PROJECT: Optional[str] = None


class ExperimentTrackingConfig(StrictBaseModel):
    ENABLED: bool = False
    BACKEND: Literal['mlflow'] = 'mlflow'
    MLFLOW_TRACKING_URI: str = 'file:./mlruns'
    EXPERIMENT_NAME: str = 'multi-model-financial-forecasting'
    LOG_PARAMS: bool = True
    LOG_METRICS: bool = True
    LOG_ARTIFACTS: bool = False

    @model_validator(mode='after')
    def validate_local_mlflow(self):
        uri = self.MLFLOW_TRACKING_URI.lower()
        if uri.startswith(('http://', 'https://', 'databricks://')):
            raise ValueError("MLFLOW_TRACKING_URI must be local; use file:./mlruns or a local path")
        if not (uri.startswith('file:') or '://' not in uri):
            raise ValueError("MLFLOW_TRACKING_URI must be a local file URI or local path")
        if not self.EXPERIMENT_NAME.strip():
            raise ValueError("EXPERIMENT_NAME cannot be empty")
        return self


class ModelConfigSection(StrictBaseModel):
    embeddings: EmbeddingConfig
    training: TrainingConfig
    training_backend: TrainingBackendConfig = Field(default_factory=TrainingBackendConfig)
    selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    loss: LossConfig
    device: DeviceConfig
    checkpointing: CheckpointConfig
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    logging: LoggingConfig
    experiment_tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)
    models: Dict[str, Dict[str, Any]]


class ModelConfigSchema(StrictBaseModel):
    model: ModelConfigSection


SCHEMA_BY_NAME = {
    'main': MainConfigSchema,
    'model': ModelConfigSchema,
}


def validate_config_data(config_name: str, data: Dict[str, Any]) -> None:
    """
    Validate raw JSON data for known config files.

    Unknown config files intentionally pass through so deploy/test/hyperparameter
    can be migrated incrementally.
    """
    schema = SCHEMA_BY_NAME.get(config_name)
    if schema is not None:
        schema.model_validate(data)
