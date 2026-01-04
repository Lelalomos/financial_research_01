"""
Model configuration for CRNN Financial Prediction Model.

This module defines all model-related configuration including:
- Model architecture parameters
- Training hyperparameters
- Optimization settings
- Checkpointing and logging options
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    """
    Configuration for model architecture and training.

    Attributes:
        # Model selection
        MODEL_TYPE: Type of model to use ('crnn', 'rnn', 'rnn_attention', 'crnn_attention', 'transformer', 'lstm3', 'lstm3_attention', 'bilstm4_attention')

        # Embedding dimensions
        EMBEDDING_DIM_STOCK: Dimension for stock ticker embedding
        EMBEDDING_DIM_GROUP: Dimension for sector/group embedding
        EMBEDDING_DIM_DAY: Dimension for day of month embedding
        EMBEDDING_DIM_MONTH: Dimension for month embedding
        EMBEDDING_DIM_DIVIDEND_FLAG: Dimension for dividend flag embedding (1=has dividend, 2=no dividend)

        # CNN architecture (for CRNN models)
        CNN_CHANNELS: Tuple of output channels for CNN layers
        CNN_KERNEL_SIZE: Kernel size for CNN layers
        CNN_POOL_SIZE: Pooling size for max pooling
        CNN_USE_BATCH_NORM: Whether to use batch normalization in CNN

        # RNN architecture
        RNN_HIDDEN_SIZE: Hidden size for LSTM layers
        RNN_NUM_LAYERS: Number of LSTM layers
        RNN_DROPOUT: Dropout rate for LSTM layers
        USE_BIDIRECTIONAL: Whether to use bidirectional LSTM

        # Attention
        USE_ATTENTION: Whether to use attention mechanism
        ATTENTION_HEADS: Number of attention heads (for MultiheadAttention)
        ATTENTION_HIDDEN_SIZE: Hidden size for attention layer (None = use RNN hidden size)
        ATTENTION_DROPOUT: Dropout rate for attention layer

        # Transformer (alternative model)
        TRANSFORMER_NUM_LAYERS: Number of transformer encoder layers
        TRANSFORMER_NUM_HEADS: Number of attention heads in transformer
        TRANSFORMER_D_MODEL: Dimension of transformer model
        TRANSFORMER_DIM_FEEDFORWARD: Dimension of feedforward network in transformer
        TRANSFORMER_DROPOUT: Dropout rate for transformer

        # Fully connected layers
        FC_HIDDEN_SIZES: Tuple of hidden sizes for FC layers
        FC_DROPOUT: Dropout rate for FC layers
        FC_USE_BATCH_NORM: Whether to use batch normalization in FC layers

        # Training
        LEARNING_RATE: Learning rate for optimizer
        WEIGHT_DECAY: Weight decay (L2 regularization)
        BATCH_SIZE: Batch size for training
        NUM_EPOCHS: Maximum number of training epochs
        EARLY_STOPPING_PATIENCE: Patience for early stopping
        GRADIENT_CLIP_VALUE: Value for gradient clipping (0 to disable)

        # Loss function
        LOSS_TYPE: Type of loss function ('huber', 'mse', 'mae', 'smooth_l1')
        HUBER_DELTA: Delta parameter for Huber loss

        # Optimization
        OPTIMIZER: Optimizer type ('adam', 'adamw', 'sgd', 'rmsprop')
        SCHEDULER: Learning rate scheduler type (None, 'reduce_on_plateau', 'cosine', 'step')
        SCHEDULER_PARAMS: Parameters for learning rate scheduler
        ACCUMULATION_STEPS: Gradient accumulation steps (default 1 = no accumulation)

        # Regularization
        USE_LAYER_NORM: Whether to use layer normalization
        USE_BATCH_NORM: Whether to use batch normalization
        DROPOUT_EMBEDDING: Dropout rate for embedding layer

        # Mixed precision training
        USE_MIXED_PRECISION: Whether to use automatic mixed precision (AMP)

        # Device and performance
        DEVICE: Device to use ('cuda' or 'cpu')
        NUM_WORKERS: Number of workers for DataLoader
        PIN_MEMORY: Whether to pin memory for faster GPU transfer
        PREFETCH_FACTOR: Prefetch factor for DataLoader

        # Checkpointing
        CHECKPOINT_DIR: Directory to save checkpoints
        SAVE_BEST_ONLY: Whether to save only the best model
        SAVE_LAST_N: Save last N checkpoints
        CHECKPOINT_FREQUENCY: Save checkpoint every N epochs

        # Logging
        LOG_FREQUENCY: Log metrics every N batches
        TENSORBOARD_DIR: Directory for TensorBoard logs (None to disable)
        WANDB_PROJECT: Weights & Biases project name (None to disable)

        # Validation
        VAL_FREQUENCY: Validate every N epochs
        VAL_BATCH_SIZE: Batch size for validation (None = use training batch size)

        # Reproducibility
        RANDOM_SEED: Random seed for reproducibility
        DETERMINISTIC: Whether to use deterministic algorithms (slower but reproducible)
    """

    # Model selection
    MODEL_TYPE: str = "lstm3_attention"  # 'crnn', 'rnn', 'rnn_attention', 'crnn_attention', 'transformer'

    # Embedding dimensions
    EMBEDDING_DIM_STOCK: int = 64
    EMBEDDING_DIM_GROUP: int = 32
    EMBEDDING_DIM_DAY: int = 16
    EMBEDDING_DIM_MONTH: int = 16
    EMBEDDING_DIM_DIVIDEND_FLAG: int = 8  # Small embedding for binary flag (1=has dividend, 2=no dividend)

    # CNN architecture (for CRNN models)
    CNN_CHANNELS: Tuple[int, ...] = (64, 128)
    CNN_KERNEL_SIZE: int = 3
    CNN_POOL_SIZE: int = 2
    CNN_USE_BATCH_NORM: bool = False

    # RNN architecture
    RNN_HIDDEN_SIZE: int = 128
    RNN_NUM_LAYERS: int = 2
    RNN_DROPOUT: float = 0.2
    USE_BIDIRECTIONAL: bool = True

    # Attention
    USE_ATTENTION: bool = True
    ATTENTION_HEADS: int = 4
    ATTENTION_HIDDEN_SIZE: Optional[int] = None  # None means use RNN_HIDDEN_SIZE * (2 if bidirectional else 1)
    ATTENTION_DROPOUT: float = 0.1

    # Transformer (alternative model)
    TRANSFORMER_NUM_LAYERS: int = 4
    TRANSFORMER_NUM_HEADS: int = 8
    TRANSFORMER_D_MODEL: int = 256
    TRANSFORMER_DIM_FEEDFORWARD: int = 512
    TRANSFORMER_DROPOUT: float = 0.1

    # LSTM3 architecture (3-layer BiLSTM variants)
    LSTM3_HIDDEN_SIZE: int = 256
    LSTM3_NUM_LAYERS: int = 3  # Fixed at 3 for LSTM3 models
    LSTM3_DROPOUT: float = 0.2
    LSTM3_USE_LAYER_NORM: bool = True

    # LSTM3 + Attention
    LSTM3_ATTENTION_HIDDEN_SIZE: Optional[int] = None  # None means use LSTM3_HIDDEN_SIZE * 2
    LSTM3_ATTENTION_HEADS: int = 8
    LSTM3_ATTENTION_DROPOUT: float = 0.1

    # LSTM4 architecture (4-layer BiLSTM with variable hidden sizes)
    LSTM4_HIDDEN_SIZES: Tuple[int, ...] = (128, 256, 512, 256)  # Per-layer hidden sizes
    LSTM4_DROPOUT: float = 0.4

    # LSTM4 + Attention
    LSTM4_ATTENTION_HEADS: int = 4
    LSTM4_ATTENTION_DROPOUT: float = 0.4

    # Fully connected layers
    FC_HIDDEN_SIZES: Tuple[int, ...] = (256, 128)
    FC_DROPOUT: float = 0.3
    FC_USE_BATCH_NORM: bool = False

    # Training
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 200
    EARLY_STOPPING_PATIENCE: int = 15
    GRADIENT_CLIP_VALUE: float = 1.0

    # Loss function
    LOSS_TYPE: str = "huber"  # 'huber', 'mse', 'mae', 'smooth_l1'
    HUBER_DELTA: float = 0.1

    # Optimization
    OPTIMIZER: str = "adam"  # 'adam', 'adamw', 'sgd', 'rmsprop'
    SCHEDULER: str = "reduce_on_plateau"  # None, 'reduce_on_plateau', 'cosine', 'step'
    SCHEDULER_PARAMS: Optional[dict] = None
    ACCUMULATION_STEPS: int = 1

    # Regularization
    USE_LAYER_NORM: bool = True
    USE_BATCH_NORM: bool = False
    DROPOUT_EMBEDDING: float = 0.1

    # Mixed precision training
    USE_MIXED_PRECISION: bool = False

    # Device and performance
    DEVICE: str = "cuda"  # 'cuda' or 'cpu'
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 2

    # Checkpointing
    CHECKPOINT_DIR: str = "models/checkpoints"
    SAVE_BEST_ONLY: bool = True
    SAVE_LAST_N: int = 3
    CHECKPOINT_FREQUENCY: int = 1  # Save every N epochs

    # Logging
    LOG_FREQUENCY: int = 10  # Log every N batches
    TENSORBOARD_DIR: Optional[str] = "logs/tensorboard"
    WANDB_PROJECT: Optional[str] = None  # Set to project name to enable W&B

    # Validation
    VAL_FREQUENCY: int = 1  # Validate every N epochs
    VAL_BATCH_SIZE: Optional[int] = None  # None = use training batch size

    # Reproducibility
    RANDOM_SEED: int = 42
    DETERMINISTIC: bool = False

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Validate model type
        valid_models = ['crnn', 'rnn', 'rnn_attention', 'crnn_attention', 'transformer', 'lstm3', 'lstm3_attention', 'bilstm4_attention']
        if self.MODEL_TYPE not in valid_models:
            raise ValueError(f"MODEL_TYPE must be one of {valid_models}, got {self.MODEL_TYPE}")

        # Validate loss type
        valid_losses = ['huber', 'mse', 'mae', 'smooth_l1']
        if self.LOSS_TYPE not in valid_losses:
            raise ValueError(f"LOSS_TYPE must be one of {valid_losses}, got {self.LOSS_TYPE}")

        # Validate optimizer
        valid_optimizers = ['adam', 'adamw', 'sgd', 'rmsprop']
        if self.OPTIMIZER not in valid_optimizers:
            raise ValueError(f"OPTIMIZER must be one of {valid_optimizers}, got {self.OPTIMIZER}")

        # Validate scheduler
        valid_schedulers = [None, 'reduce_on_plateau', 'cosine', 'step']
        if self.SCHEDULER not in valid_schedulers:
            raise ValueError(f"SCHEDULER must be one of {valid_schedulers}, got {self.SCHEDULER}")

        # Set default scheduler params if not specified
        if self.SCHEDULER_PARAMS is None:
            self.SCHEDULER_PARAMS = {
                'reduce_on_plateau': {
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 10
                },
                'cosine': {
                    'T_max': self.NUM_EPOCHS,
                    'eta_min': 1e-6
                },
                'step': {
                    'step_size': 30,
                    'gamma': 0.1
                }
            }.get(self.SCHEDULER, {})

        # Set attention hidden size if not specified
        if self.ATTENTION_HIDDEN_SIZE is None:
            self.ATTENTION_HIDDEN_SIZE = self.RNN_HIDDEN_SIZE * (2 if self.USE_BIDIRECTIONAL else 1)

        # Set validation batch size
        if self.VAL_BATCH_SIZE is None:
            self.VAL_BATCH_SIZE = self.BATCH_SIZE

    @property
    def total_embedding_dim(self) -> int:
        """Total dimension of all embeddings concatenated."""
        return (
            self.EMBEDDING_DIM_STOCK +
            self.EMBEDDING_DIM_GROUP +
            self.EMBEDDING_DIM_DAY +
            self.EMBEDDING_DIM_MONTH +
            self.EMBEDDING_DIM_DIVIDEND_FLAG
        )

    @property
    def lstm_output_size(self) -> int:
        """Output size of LSTM layer (accounting for bidirectional)."""
        return self.RNN_HIDDEN_SIZE * (2 if self.USE_BIDIRECTIONAL else 1)

    @property
    def cnn_output_channels(self) -> int:
        """Output channels of final CNN layer."""
        return self.CNN_CHANNELS[-1] if self.CNN_CHANNELS else 0

    def get_scheduler_params(self) -> dict:
        """Get scheduler parameters for the configured scheduler type."""
        if self.SCHEDULER == 'reduce_on_plateau':
            return self.SCHEDULER_PARAMS or {
                'mode': 'min',
                'factor': 0.5,
                'patience': 10
            }
        elif self.SCHEDULER == 'cosine':
            return self.SCHEDULER_PARAMS or {
                'T_max': self.NUM_EPOCHS,
                'eta_min': 1e-6
            }
        elif self.SCHEDULER == 'step':
            return self.SCHEDULER_PARAMS or {
                'step_size': 30,
                'gamma': 0.1
            }
        return {}


# Default configuration instance
default_model_config = ModelConfig()


# Preset configurations for different model types
def get_config_for_model(model_type: str) -> ModelConfig:
    """
    Get a preset configuration for a specific model type.

    Args:
        model_type: Type of model ('crnn', 'rnn', 'rnn_attention', 'crnn_attention', 'transformer', 'lstm3', 'lstm3_attention')

    Returns:
        ModelConfig instance with preset parameters
    """
    base_config = ModelConfig(MODEL_TYPE=model_type)

    if model_type == 'transformer':
        # Transformer-specific settings
        base_config.TRANSFORMER_NUM_LAYERS = 6
        base_config.TRANSFORMER_NUM_HEADS = 8
        base_config.TRANSFORMER_D_MODEL = 256
        base_config.LEARNING_RATE = 5e-5  # Transformers often need lower LR
    elif model_type == 'rnn':
        # Simple RNN settings
        base_config.RNN_HIDDEN_SIZE = 128
        base_config.RNN_NUM_LAYERS = 2
    elif model_type in ['rnn_attention', 'crnn_attention']:
        # Attention models benefit from larger hidden size
        base_config.RNN_HIDDEN_SIZE = 256
        base_config.ATTENTION_HEADS = 8
    elif model_type == 'lstm3':
        # LSTM3-specific settings (deeper model)
        base_config.LSTM3_HIDDEN_SIZE = 256
        base_config.LSTM3_NUM_LAYERS = 3
    elif model_type == 'lstm3_attention':
        # LSTM3 + Attention settings
        base_config.LSTM3_HIDDEN_SIZE = 256
        base_config.LSTM3_NUM_LAYERS = 3
        base_config.LSTM3_ATTENTION_HEADS = 8
    elif model_type == 'bilstm4_attention':
        # BiLSTM4 + Attention settings
        base_config.LSTM4_HIDDEN_SIZES = (128, 256, 512, 256)
        base_config.LSTM4_DROPOUT = 0.4
        base_config.LSTM4_ATTENTION_HEADS = 4
        base_config.LSTM4_ATTENTION_DROPOUT = 0.4

    return base_config


@dataclass
class HyperparameterSearchConfig:
    """
    Configuration for hyperparameter search with Optuna.

    Attributes:
        # Search settings
        N_TRIALS: Number of Optuna trials to run
        TIMEOUT: Maximum time for search (None = no timeout)
        N_JOBS: Number of parallel trials

        # Model type to tune (configurable)
        MODEL_TYPE: Default model type to tune

        # Small dataset settings
        HPARAM_STOCKS: Number of stocks to sample (ensuring all groups)
        HPARAM_YEARS: Number of years to use (None = ALL years)
        HPARAM_ALL_YEARS: If True, use all available years
        HPARAM_START_DATE: Auto-calculated when HPARAM_ALL_YEARS=True

        # Early stopping for faster trials
        HPARAM_MAX_EPOCHS: Max epochs per trial
        HPARAM_ES_PATIENCE: Early stopping patience per trial

        # Hyperparameter search ranges
        LEARNING_RATE_RANGE: (min, max) for learning rate (log scale)
        LSTM_HIDDEN_SIZE_RANGE: (min, max) for LSTM hidden size
        LSTM_NUM_LAYERS_RANGE: (min, max) for number of LSTM layers
        DROPOUT_RANGE: (min, max) for dropout
        WEIGHT_DECAY_RANGE: (min, max) for weight decay (log scale)
        SEQUENCE_LENGTH_CHOICES: Tuple of sequence length options

        # Output
        BEST_PARAMS_PATH: Path to save best hyperparameters
    """

    # Search settings
    N_TRIALS: int = 50
    TIMEOUT: Optional[int] = None  # None = no timeout
    N_JOBS: int = 1  # Number of parallel trials

    # Model type to tune (configurable)
    MODEL_TYPE: str = "bilstm4_attention"  # Can be changed via command line

    # Small dataset settings
    HPARAM_STOCKS: int = 20
    HPARAM_YEARS: Optional[int] = None  # None = use ALL years (default)
    HPARAM_ALL_YEARS: bool = True  # Default: use all available years
    HPARAM_START_DATE: Optional[str] = None  # Uses full date range when HPARAM_ALL_YEARS=True

    # Early stopping for faster trials
    HPARAM_MAX_EPOCHS: int = 50
    HPARAM_ES_PATIENCE: int = 10

    # Hyperparameter search ranges
    LEARNING_RATE_RANGE: tuple = (1e-5, 1e-3)
    LSTM_HIDDEN_SIZE_RANGE: tuple = (64, 512)
    LSTM_NUM_LAYERS_RANGE: tuple = (1, 4)
    DROPOUT_RANGE: tuple = (0.1, 0.5)
    WEIGHT_DECAY_RANGE: tuple = (1e-6, 1e-3)
    SEQUENCE_LENGTH_CHOICES: tuple = (20, 30, 60, 90)
    BATCH_SIZE_CHOICES: tuple = (32, 64, 128)

    # Output
    BEST_PARAMS_PATH: str = "best_hyperparameters.json"


# Default hyperparameter search configuration
default_hparam_config = HyperparameterSearchConfig()
