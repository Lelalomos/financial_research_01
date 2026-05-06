"""
Logging utilities for Multi-Model Financial Forecasting.

This module provides comprehensive logging functionality including:
- File logging
- Console logging
- TensorBoard logging
- Structured logging with metadata
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """Colored console output formatter."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class StructuredLogger:
    """
    Structured logger with file and console output.

    Provides logging functionality with:
    - Colored console output
    - File logging with rotation
    - Structured JSON logging option
    - Metadata tracking
    """

    def __init__(
        self,
        name: str,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console_output: bool = True,
        json_output: bool = False,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            log_dir: Directory to store log files (default: logs/)
            log_file: Specific log file name (default: {name}_{timestamp}.log)
            level: Logging level
            console_output: Whether to output to console
            json_output: Whether to use JSON format for file logging
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Clear existing handlers

        # Set up log directory
        if log_dir is None:
            log_dir = "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up log file
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"
        self.log_file = self.log_dir / log_file

        # Create formatter
        if json_output:
            file_formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s}'
            )
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, kwargs)

    def _log(self, level: int, message: str, metadata: Dict[str, Any]):
        """Internal logging method with metadata support."""
        if metadata:
            message = f"{message} | Metadata: {json.dumps(metadata)}"
        self.logger.log(level, message)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """
        Log metrics in a structured format.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
            prefix: Optional prefix for metric names
        """
        message_parts = []
        if step is not None:
            message_parts.append(f"Step: {step}")

        metric_strs = [f"{prefix}{k}: {v:.6f}" for k, v in metrics.items()]
        message_parts.append("Metrics: " + ", ".join(metric_strs))

        self.info(" | ".join(message_parts))

    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration parameters.

        Args:
            config: Configuration dictionary
        """
        self.info("Configuration:")
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            self.info(f"  {key}: {value}")

    def log_model_summary(self, model):
        """
        Log model architecture summary.

        Args:
            model: PyTorch model
        """
        self.info("Model Architecture:")
        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            self.info(f"  {name}: {list(param.shape)} ({num_params:,} params)")

        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
        self.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

    def log_dataset_info(self, dataset_name: str, info: Dict[str, Any]):
        """
        Log dataset information.

        Args:
            dataset_name: Name of the dataset
            info: Dictionary with dataset information
        """
        self.info(f"{dataset_name} Information:")
        for key, value in info.items():
            self.info(f"  {key}: {value}")


class TrainingLogger(StructuredLogger):
    """
    Specialized logger for training process.

    Provides additional functionality for:
    - Training progress tracking
    - Epoch metrics
    - Model checkpoints
    - GPU/memory usage
    """

    def __init__(self, log_dir: str = "logs"):
        super().__init__("training", log_dir=log_dir)
        self.epoch = 0

    def log_epoch_start(self, epoch: int, num_epochs: int):
        """Log start of epoch."""
        self.epoch = epoch
        self.info(f"Epoch {epoch}/{num_epochs} started")

    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log end of epoch with metrics."""
        self.log_metrics(metrics, step=epoch, prefix="train_")

    def log_validation(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log validation metrics."""
        self.log_metrics(metrics, step=step, prefix="val_")

    def log_checkpoint(self, checkpoint_path: str, metric: float, metric_name: str = "loss"):
        """Log model checkpoint save."""
        self.info(f"Checkpoint saved: {checkpoint_path} ({metric_name}: {metric:.6f})")

    def log_early_stop(self, epoch: int, best_metric: float):
        """Log early stopping."""
        self.info(f"Early stopping at epoch {epoch}. Best {metric_name}: {best_metric:.6f}")

    def log_gradient_norm(self, grad_norm: float):
        """Log gradient norm."""
        self.debug(f"Gradient norm: {grad_norm:.6f}")

    def log_lr(self, lr: float):
        """Log learning rate."""
        self.debug(f"Learning rate: {lr:.8f}")


class EvaluationLogger(StructuredLogger):
    """
    Specialized logger for evaluation process.

    Provides additional functionality for:
    - Test metrics
    - Backtesting results
    - Performance statistics
    """

    def __init__(self, log_dir: str = "logs"):
        super().__init__("evaluation", log_dir=log_dir)

    def log_test_results(self, metrics: Dict[str, float]):
        """Log test results."""
        self.info("=" * 60)
        self.info("TEST RESULTS")
        self.info("=" * 60)
        for key, value in metrics.items():
            self.info(f"  {key}: {value:.6f}")
        self.info("=" * 60)

    def log_backtest_summary(self, summary: Dict[str, Any]):
        """Log backtesting summary."""
        self.info("=" * 60)
        self.info("BACKTEST SUMMARY")
        self.info("=" * 60)
        for key, value in summary.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        self.info("=" * 60)


def get_logger(
    name: str = "root",
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> StructuredLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, log_dir=log_dir, level=level)


def get_training_logger(log_dir: str = "logs") -> TrainingLogger:
    """Get or create training logger."""
    return TrainingLogger(log_dir=log_dir)


def get_evaluation_logger(log_dir: str = "logs") -> EvaluationLogger:
    """Get or create evaluation logger."""
    return EvaluationLogger(log_dir=log_dir)
