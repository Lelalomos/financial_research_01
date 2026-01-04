"""
Optuna optimizer for hyperparameter tuning.

This module provides the Optuna-based objective function and optimizer
for finding best hyperparameters for model training.
"""

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, Tuple
import json
from pathlib import Path
from datetime import datetime

from src.config import load_config, Config
from src.models import create_model
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger("optuna")


def create_objective_function(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_features: int,
    num_stocks: int,
    num_groups: int,
    model_type: str,
    model_config: Config,
    hparam_config: Config,
    device: torch.device
) -> Callable[[optuna.Trial], float]:
    """
    Create Optuna objective function for hyperparameter tuning.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_features: Number of input features
        num_stocks: Number of unique stocks
        num_groups: Number of unique groups
        model_type: Type of model to train
        model_config: Model configuration (for model settings)
        hparam_config: Hyperparameter configuration (from hyperparameter.json)
        device: Device to train on

    Returns:
        Objective function that takes a trial and returns validation loss
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Suggests hyperparameters and returns validation loss.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss (to minimize)
        """
        # Suggest hyperparameters based on model type
        if model_type in ['bilstm4_attention']:
            # Suggest number of layers (1-4)
            num_layers = trial.suggest_int('num_layers', 1, 4)

            # Suggest hidden sizes for each layer
            lstm_hidden_sizes = []
            for i in range(num_layers):
                hidden_size = trial.suggest_int(f'lstm_layer_{i}_hidden',
                                                 hparam_config.hyperparameter.LSTM_HIDDEN_SIZE_RANGE[0],
                                                 hparam_config.hyperparameter.LSTM_HIDDEN_SIZE_RANGE[1])
                lstm_hidden_sizes.append(hidden_size)

            # Ensure bidirectional output sizes are consistent
            # For simplicity, use the tuple directly
        else:
            # For other LSTM models
            lstm_hidden_size = trial.suggest_int('lstm_hidden_size',
                                                  hparam_config.hyperparameter.LSTM_HIDDEN_SIZE_RANGE[0],
                                                  hparam_config.hyperparameter.LSTM_HIDDEN_SIZE_RANGE[1])
            num_layers = trial.suggest_int('num_layers',
                                           hparam_config.hyperparameter.LSTM_NUM_LAYERS_RANGE[0],
                                           hparam_config.hyperparameter.LSTM_NUM_LAYERS_RANGE[1])
            lstm_hidden_sizes = tuple([lstm_hidden_size] * num_layers)

        # Common hyperparameters
        learning_rate = trial.suggest_float('learning_rate',
                                            hparam_config.hyperparameter.LEARNING_RATE_RANGE[0],
                                            hparam_config.hyperparameter.LEARNING_RATE_RANGE[1],
                                            log=True)
        dropout = trial.suggest_float('dropout',
                                       hparam_config.hyperparameter.DROPOUT_RANGE[0],
                                       hparam_config.hyperparameter.DROPOUT_RANGE[1])
        weight_decay = trial.suggest_float('weight_decay',
                                           hparam_config.hyperparameter.WEIGHT_DECAY_RANGE[0],
                                           hparam_config.hyperparameter.WEIGHT_DECAY_RANGE[1],
                                           log=True)
        batch_size = trial.suggest_categorical('batch_size',
                                               hparam_config.hyperparameter.BATCH_SIZE_CHOICES)

        # Create config with suggested hyperparameters
        config = load_config('model')
        config.model.architecture.MODEL_TYPE = model_type
        config.model.training.LEARNING_RATE = learning_rate
        config.model.training.WEIGHT_DECAY = weight_decay
        config.model.training.BATCH_SIZE = batch_size
        config.model.training.NUM_EPOCHS = hparam_config.hyperparameter.HPARAM_MAX_EPOCHS
        config.model.training.EARLY_STOPPING_PATIENCE = hparam_config.hyperparameter.HPARAM_ES_PATIENCE

        # Set model-specific parameters
        if model_type == 'bilstm4_attention':
            config.model.architecture.LSTM4_HIDDEN_SIZES = tuple(lstm_hidden_sizes)
            config.model.architecture.LSTM4_DROPOUT = dropout
        elif model_type in ['lstm3', 'lstm3_attention']:
            config.model.architecture.LSTM3_HIDDEN_SIZE = lstm_hidden_sizes[0]
            config.model.architecture.LSTM3_NUM_LAYERS = num_layers
            config.model.architecture.LSTM3_DROPOUT = dropout
        elif model_type in ['crnn', 'rnn', 'rnn_attention', 'crnn_attention']:
            config.model.architecture.RNN_HIDDEN_SIZE = lstm_hidden_sizes[0]
            config.model.architecture.RNN_NUM_LAYERS = num_layers
            config.model.architecture.RNN_DROPOUT = dropout

        # Update data loaders with new batch size
        train_loader.dataset.batch_size = batch_size
        val_loader.dataset.batch_size = batch_size

        # Create model
        try:
            model = create_model(
                model_type=model_type,
                num_features=num_features,
                num_stocks=num_stocks,
                num_groups=num_groups,
                config=config
            )
            model = model.to(device)
        except Exception as e:
            logger.warning(f"Failed to create model: {e}")
            return float('inf')

        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Train model
        try:
            history = trainer.train()
            val_loss = min(history['val_loss'])
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            return float('inf')

        # Report intermediate value for pruning
        trial.report(val_loss, step=config.model.training.NUM_EPOCHS)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_loss

    return objective


class OptunaOptimizer:
    """
    Optuna optimizer for hyperparameter tuning.

    Manages the Optuna study and handles hyperparameter search.
    """

    def __init__(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        model_config: Optional[Config] = None,
        hparam_config: Optional[Config] = None
    ):
        """
        Initialize Optuna optimizer.

        Args:
            study_name: Name of the Optuna study
            storage: Optuna storage URL (None = in-memory)
            model_config: Model configuration (for model settings)
            hparam_config: Hyperparameter configuration (from hyperparameter.json)
        """
        self.model_config = model_config or load_config('model')
        self.hparam_config = hparam_config or load_config('hyperparameter')
        model_type = self.hparam_config.hyperparameter.MODEL_TYPE if hasattr(self.hparam_config, 'hyperparameter') else 'bilstm4_attention'
        self.study_name = study_name or f"{model_type}_hparam_search"
        self.storage = storage

        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            load_if_exists=True,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )

    def optimize(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_features: int,
        num_stocks: int,
        num_groups: int,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_features: Number of input features
            num_stocks: Number of unique stocks
            num_groups: Number of unique groups
            device: Device to train on

        Returns:
            Dictionary with best hyperparameters and study info
        """
        model_type = self.hparam_config.hyperparameter.MODEL_TYPE if hasattr(self.hparam_config, 'hyperparameter') else 'bilstm4_attention'
        logger.info(f"Starting hyperparameter search for {model_type}")
        logger.info(f"Number of trials: {self.hparam_config.hyperparameter.N_TRIALS}")

        # Create objective function
        objective = create_objective_function(
            train_loader=train_loader,
            val_loader=val_loader,
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
            model_type=model_type,
            model_config=self.model_config,
            hparam_config=self.hparam_config,
            device=device
        )

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.hparam_config.hyperparameter.N_TRIALS,
            timeout=self.hparam_config.hyperparameter.TIMEOUT,
            n_jobs=self.hparam_config.hyperparameter.N_JOBS,
            show_progress_bar=True
        )

        # Get results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        logger.info(f"Best trial value: {best_value:.6f}")
        logger.info(f"Best hyperparameters: {best_params}")

        # Prepare result
        result = {
            "model_type": model_type,
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(self.study.trials),
            "study_name": self.study_name,
            "datetime": datetime.now().isoformat()
        }

        # Save results
        self._save_results(result)

        return result

    def _save_results(self, result: Dict[str, Any]) -> None:
        """
        Save optimization results to JSON file.

        Args:
            result: Result dictionary to save
        """
        # Add model type to filename
        model_type = result["model_type"]
        output_path = Path(self.hparam_config.hyperparameter.BEST_PARAMS_PATH)

        # Update path to include model type
        if output_path.name == "best_hyperparameters.json":
            output_path = output_path.parent / f"best_hyperparameters_{model_type}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters from study."""
        return self.study.best_params

    def get_trials_df(self):
        """Get trials as pandas DataFrame."""
        return self.study.trials_dataframe()
