"""
Training loop for Multi-Model Financial Forecasting.

This module provides:
- Training and validation loops
- Loss function handling
- Gradient clipping
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.utils.logger import TrainingLogger
from src.utils.validation import (
    check_tensor_for_nan_inf,
    sanitize_tensor,
    check_batch_for_invalid,
    sanitize_batch,
    check_model_parameters,
    check_gradients
)
from .early_stopping import EarlyStopping, ModelCheckpoint, make_weights_only_safe
from .experiment_tracking import create_experiment_tracker, training_params
from .common import create_loss_function, create_optimizer, create_scheduler


class Trainer:
    """
    Trainer for financial prediction models.

    Handles:
    - Training loop
    - Validation loop
    - Checkpointing
    - Early stopping
    - TensorBoard logging
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        device: str = 'cuda',
        model_type: Optional[str] = None,
        checkpoint_metadata: Optional[Dict] = None,
        experiment_tracker: Optional[object] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            config instance
            device: Device to use ('cuda' or 'cpu')
            model_type: Model type name for checkpoint filenames (e.g., 'crnn_attention')
            checkpoint_metadata: Additional weights-only-safe metadata to save in checkpoints
            experiment_tracker: Optional injected tracker for tests or custom tracking
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model_type = model_type or self._infer_model_type()
        self.checkpoint_metadata = checkpoint_metadata or {}

        self.logger = TrainingLogger(log_dir="logs")

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup loss function
        self.criterion = self._create_criterion()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.model.training.EARLY_STOPPING_PATIENCE,
            mode='min',
            verbose=True
        )

        # Setup checkpointing with model_type
        self.checkpoint = ModelCheckpoint(
            save_dir=config.model.checkpointing.CHECKPOINT_DIR,
            mode='min',
            save_best_only=config.model.checkpointing.SAVE_BEST_ONLY,
            save_last_n=config.model.checkpointing.SAVE_LAST_N,
            checkpoint_frequency=config.model.checkpointing.CHECKPOINT_FREQUENCY,
            model_type=self.model_type
        )

        # Setup TensorBoard
        self.writer = None
        if config.model.logging.TENSORBOARD_DIR:
            self.writer = SummaryWriter(config.model.logging.TENSORBOARD_DIR)

        self.experiment_tracker = experiment_tracker or create_experiment_tracker(config)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Setup mixed precision scaler in __init__ to avoid AttributeError
        # if train_epoch() is called before train()
        self.scaler = torch.amp.GradScaler('cuda') if config.model.training.USE_MIXED_PRECISION else None

        # Gradient accumulation steps
        self.accumulation_steps = max(1, int(getattr(config.model.training, 'ACCUMULATION_STEPS', 1)))

    def _infer_model_type(self) -> str:
        """Infer model type from model class name."""
        class_name = self.model.__class__.__name__
        # Convert class name to model_type format
        mapping = {
            'CRNNModel': 'crnn',
            'RNNModel': 'rnn',
            'RNNAttentionModel': 'rnn_attention',
            'CRNNAttentionModel': 'crnn_attention',
            'TransformerModel': 'transformer',
            'LSTM3Model': 'lstm3',
            'LSTM3AttentionModel': 'lstm3_attention',
            'BiLSTM4AttentionModel': 'bilstm4_attention',
            'MultiBranchBiLSTMModel': 'multi_branch_bilstm',
        }
        return mapping.get(class_name, class_name.lower())

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        return create_optimizer(self.model, self.config)

    def _create_criterion(self) -> nn.Module:
        """Create loss function based on config."""
        return create_loss_function(self.config)

    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler based on config."""
        return create_scheduler(self.optimizer, self.config)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            features = batch['features'].to(self.device)
            stock_id = batch['stock_id'].to(self.device)
            group_id = batch['group_id'].to(self.device)
            day = batch['day'].to(self.device)
            month = batch['month'].to(self.device)
            dividend_flag = batch['dividend_flag'].to(self.device)
            target = batch['target'].to(self.device)

            # Validate inputs for NaN/Inf (if enabled in config)
            if hasattr(self.config.model, 'nan_handling') and self.config.model.nan_handling.CHECK_INPUTS:
                replace_val = self.config.model.nan_handling.REPLACE_VALUE

                # Check and sanitize features
                if torch.isnan(features).any() or torch.isinf(features).any():
                    self.logger.warning(f"NaN/Inf in features at batch {batch_idx}, replacing with {replace_val}")
                    features = torch.where(
                        torch.isnan(features) | torch.isinf(features),
                        torch.tensor(replace_val, device=self.device),
                        features
                    )

                # Check and sanitize targets
                if torch.isnan(target).any() or torch.isinf(target).any():
                    self.logger.warning(f"NaN/Inf in target at batch {batch_idx}, replacing with {replace_val}")
                    target = torch.where(
                        torch.isnan(target) | torch.isinf(target),
                        torch.tensor(replace_val, device=self.device),
                        target
                    )

            # Forward pass — zero_grad only at accumulation boundaries
            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.zero_grad()

            if self.config.model.training.USE_MIXED_PRECISION:
                with torch.amp.autocast('cuda'):
                    output = self.model(features, stock_id, group_id, day, month, dividend_flag)
                    loss = self.criterion(output, target)

                # Backward pass with gradient scaling (accumulation-aware)
                self.scaler.scale(loss / self.accumulation_steps).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.config.model.training.GRADIENT_CLIP_VALUE > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.model.training.GRADIENT_CLIP_VALUE
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(features, stock_id, group_id, day, month, dividend_flag)
                loss = self.criterion(output, target)

                # Backward pass (accumulation-aware)
                (loss / self.accumulation_steps).backward()

                # Gradient clipping
                if self.config.model.training.GRADIENT_CLIP_VALUE > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.model.training.GRADIENT_CLIP_VALUE
                    )
                else:
                    grad_norm = 0.0

                # Clamp individual gradient values to prevent explosions
                if hasattr(self.config.model, 'nan_handling'):
                    max_grad_val = self.config.model.nan_handling.MAX_GRAD_VALUE
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-max_grad_val, max_grad_val)

                # Step optimizer only at accumulation boundaries
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"NaN/Inf loss detected at epoch {self.current_epoch}, batch {batch_idx}")
                    self.logger.error(f"Loss value: {loss.item()}")

                    # Check gradients
                    if hasattr(self.config.model, 'nan_handling') and self.config.model.nan_handling.LOG_NAN_DETAILS:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    self.logger.error(f"NaN/Inf gradient in {name}")

                        # Check model weights
                        for name, param in self.model.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                self.logger.error(f"NaN/Inf weight in {name}")

                        # Check batch inputs
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                if torch.isnan(value).any() or torch.isinf(value).any():
                                    self.logger.error(f"NaN/Inf in batch[{key}]")

                    # Raise exception to stop training if configured
                    if hasattr(self.config.model, 'nan_handling') and self.config.model.nan_handling.STOP_ON_NAN:
                        raise RuntimeError(f"NaN/Inf loss at epoch {self.current_epoch}, batch {batch_idx}")

            # Accumulate metrics
            batch_size = len(target)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.extend(output.detach().cpu().numpy().flatten())
            all_targets.extend(target.detach().cpu().numpy().flatten())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / total_samples
            })

            # Log to TensorBoard
            if self.writer and batch_idx % self.config.model.logging.LOG_FREQUENCY == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)

        # Calculate epoch metrics
        if total_samples == 0:
            raise ValueError(
                "Training loader produced zero batches. Reduce BATCH_SIZE, provide more samples, or disable drop_last."
            )

        metrics = {
            'loss': total_loss / total_samples,
            'mse': np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2),
            'mae': np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                features = batch['features'].to(self.device)
                stock_id = batch['stock_id'].to(self.device)
                group_id = batch['group_id'].to(self.device)
                day = batch['day'].to(self.device)
                month = batch['month'].to(self.device)
                dividend_flag = batch['dividend_flag'].to(self.device)
                target = batch['target'].to(self.device)

                # Forward pass
                output = self.model(features, stock_id, group_id, day, month, dividend_flag)
                loss = self.criterion(output, target)

                # Check for NaN/Inf validation loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"NaN/Inf validation loss detected!")
                    # Return NaN metrics to signal failure
                    return {
                        'loss': float('nan'),
                        'mse': float('nan'),
                        'mae': float('nan'),
                        'rmse': float('nan'),
                    }

                # Accumulate metrics
                batch_size = len(target)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        # Calculate metrics
        if total_samples == 0:
            raise ValueError("Validation loader produced zero batches.")

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        metrics = {
            'loss': total_loss / total_samples,
            'mse': np.mean((predictions - targets) ** 2),
            'mae': np.mean(np.abs(predictions - targets)),
            'rmse': np.sqrt(np.mean((predictions - targets) ** 2)),
        }

        # Directional accuracy
        if len(predictions) > 0:
            direction_pred = np.sign(predictions)
            direction_true = np.sign(targets)
            metrics['directional_accuracy'] = np.mean(direction_pred == direction_true)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs (default from config)

        Returns:
            Dictionary with training history
        """
        if num_epochs is None:
            num_epochs = self.config.model.training.NUM_EPOCHS

        # Scaler is already initialized in __init__, no need to create here

        history = {
            'train_loss': [],
            'val_loss': []
        }

        self.logger.log_config(self.config.to_dict())
        self.logger.log_model_summary(self.model)
        self.experiment_tracker.start_run(run_name=self.model_type)
        self.experiment_tracker.log_params(training_params(self.config, self.model_type))

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                self.logger.log_epoch_start(epoch + 1, num_epochs)

                # Train
                train_metrics = self.train_epoch(train_loader)
                history['train_loss'].append(train_metrics['loss'])

                self.logger.log_epoch_end(epoch + 1, train_metrics)
                self.experiment_tracker.log_metrics(
                    {f"train/{key}": value for key, value in train_metrics.items()},
                    step=epoch + 1,
                )

                # Validate
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    history['val_loss'].append(val_metrics['loss'])

                    self.logger.log_validation(val_metrics, step=epoch + 1)
                    self.experiment_tracker.log_metrics(
                        {f"val/{key}": value for key, value in val_metrics.items()},
                        step=epoch + 1,
                    )

                    # Log to TensorBoard
                    if self.writer:
                        for key, value in val_metrics.items():
                            self.writer.add_scalar(f'val/{key}', value, epoch)

                    # Learning rate scheduling
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_metrics['loss'])
                        else:
                            self.scheduler.step()

                    # Early stopping check
                    should_stop = self.early_stopping(val_metrics['loss'], epoch + 1)

                    # Checkpointing
                    extra_state = {
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'epoch': epoch + 1,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        **self.checkpoint_metadata,
                        'metadata': {
                            'model_type': self.model_type,
                            **self.checkpoint_metadata,
                        },
                    }
                    self.checkpoint(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        val_metrics['loss'],
                        train_metrics['loss'],
                        extra_state=extra_state
                    )

                    if should_stop:
                        self.logger.info(f"Training stopped early at epoch {epoch + 1}")
                        break

                # Log to TensorBoard
                if self.writer:
                    for key, value in train_metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, epoch)
        except Exception:
            self.experiment_tracker.end_run(status="FAILED")
            raise

        self.logger.info("Training complete!")

        if self.writer:
            self.writer.close()
        self.experiment_tracker.end_run(status="FINISHED")

        return history

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (default: best_model.pth)
        """
        self.checkpoint.load_checkpoint(
            self.model,
            self.optimizer,
            filepath=checkpoint_path,
            device=self.device
        )

    def save_model(self, filepath: str):
        """
        Save current model and optimizer state to a checkpoint file.

        Args:
            filepath: Destination checkpoint path.
        """
        checkpoint = {
            'model_type': self.model_type,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': self.best_val_loss,
            'loss': None,
            'best_val_loss': self.best_val_loss,
            **self.checkpoint_metadata,
            'metadata': {
                'model_type': self.model_type,
                **self.checkpoint_metadata,
            },
        }
        torch.save(make_weights_only_safe(checkpoint), filepath)
        self.logger.info(f"Saved model checkpoint to {filepath}")

    def load_best_model(self):
        """Load best model checkpoint."""
        self.checkpoint.load_best(self.model, device=self.device)

    def check_model_state(self) -> Tuple[bool, list]:
        """
        Check model parameters for NaN/Inf values.

        Returns:
            Tuple of (is_valid, issues):
            - is_valid: True if no NaN/Inf found
            - issues: List of issue descriptions
        """
        is_valid, issues = check_model_parameters(self.model)

        if not is_valid:
            self.logger.error(f"Model state check failed:")
            for issue in issues:
                self.logger.error(f"  {issue}")

        return is_valid, issues
