"""
Training loop for CRNN Financial Prediction Model.

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

from config.model_config import ModelConfig
from src.utils.logger import TrainingLogger
from .early_stopping import EarlyStopping, ModelCheckpoint


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
        config: ModelConfig,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            config: ModelConfig instance
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.logger = TrainingLogger(log_dir="logs")

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup loss function
        self.criterion = self._create_criterion()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='min',
            verbose=True
        )

        # Setup checkpointing
        self.checkpoint = ModelCheckpoint(
            save_dir=config.CHECKPOINT_DIR,
            mode='min',
            save_best_only=config.SAVE_BEST_ONLY,
            save_last_n=config.SAVE_LAST_N
        )

        # Setup TensorBoard
        self.writer = None
        if config.TENSORBOARD_DIR:
            self.writer = SummaryWriter(config.TENSORBOARD_DIR)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.OPTIMIZER == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")

    def _create_criterion(self) -> nn.Module:
        """Create loss function based on config."""
        if self.config.LOSS_TYPE == 'mse':
            return nn.MSELoss()
        elif self.config.LOSS_TYPE == 'mae':
            return nn.L1Loss()
        elif self.config.LOSS_TYPE == 'smooth_l1':
            return nn.SmoothL1Loss()
        elif self.config.LOSS_TYPE == 'huber':
            return nn.HuberLoss(delta=self.config.HUBER_DELTA)
        else:
            raise ValueError(f"Unknown loss type: {self.config.LOSS_TYPE}")

    def _create_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler based on config."""
        if self.config.SCHEDULER is None:
            return None
        elif self.config.SCHEDULER == 'reduce_on_plateau':
            params = self.config.get_scheduler_params()
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **params
            )
        elif self.config.SCHEDULER == 'cosine':
            params = self.config.get_scheduler_params()
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **params
            )
        elif self.config.SCHEDULER == 'step':
            params = self.config.get_scheduler_params()
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                **params
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.SCHEDULER}")

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

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    output = self.model(features, stock_id, group_id, day, month, dividend_flag)
                    loss = self.criterion(output, target)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                if self.config.GRADIENT_CLIP_VALUE > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRADIENT_CLIP_VALUE
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(features, stock_id, group_id, day, month, dividend_flag)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.GRADIENT_CLIP_VALUE > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRADIENT_CLIP_VALUE
                    )
                else:
                    grad_norm = 0.0

                self.optimizer.step()

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
            if self.writer and batch_idx % self.config.LOG_FREQUENCY == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)

        # Calculate epoch metrics
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

                # Accumulate metrics
                batch_size = len(target)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        # Calculate metrics
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
            num_epochs = self.config.NUM_EPOCHS

        # Setup mixed precision scaler
        if self.config.USE_MIXED_PRECISION:
            self.scaler = torch.cuda.amp.GradScaler()

        history = {
            'train_loss': [],
            'val_loss': []
        }

        self.logger.log_config(self.config.__dict__)
        self.logger.log_model_summary(self.model)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.logger.log_epoch_start(epoch + 1, num_epochs)

            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])

            self.logger.log_epoch_end(epoch + 1, train_metrics)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss'])

                self.logger.log_validation(val_metrics, step=epoch + 1)

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

        self.logger.info("Training complete!")

        if self.writer:
            self.writer.close()

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

    def load_best_model(self):
        """Load best model checkpoint."""
        self.checkpoint.load_best(self.model, device=self.device)
