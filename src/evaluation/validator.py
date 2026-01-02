"""
Validator for CRNN Financial Prediction Model.

This module provides:
- Model validation
- Metrics logging
- Comparison between models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import json

from config.model_config import ModelConfig
from src.utils.logger import EvaluationLogger
from .metrics import evaluate_model, print_metrics, calculate_metrics


class Validator:
    """
    Model validator.

    Handles validation of models and logging of results.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: str = 'cuda'
    ):
        """
        Initialize validator.

        Args:
            model: PyTorch model
            config: ModelConfig instance
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = EvaluationLogger(log_dir="logs")

    def validate(
        self,
        val_loader: DataLoader,
        log_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            log_file: Optional file to save results

        Returns:
            Dictionary with validation metrics
        """
        self.logger.info("Running validation...")

        metrics = evaluate_model(self.model, val_loader, self.device)

        # Print metrics
        print_metrics(metrics, prefix="VALIDATION - ")

        # Log metrics
        self.logger.log_validation(metrics)

        # Save to file if specified
        if log_file:
            self._save_metrics(metrics, log_file)

        return metrics

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        val_loader: DataLoader,
        save_file: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models.

        Args:
            models: Dictionary of model_name -> model
            val_loader: Validation data loader
            save_file: Optional file to save comparison

        Returns:
            Dictionary with results for each model
        """
        self.logger.info(f"Comparing {len(models)} models...")

        results = {}

        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name}...")

            validator = Validator(model, self.config, self.device)
            metrics = validator.validate(val_loader)
            results[model_name] = metrics

        # Print comparison
        self._print_comparison(results)

        # Save comparison
        if save_file:
            self._save_comparison(results, save_file)

        return results

    def _print_comparison(self, results: Dict[str, Dict[str, float]]):
        """Print model comparison."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        # Get all metric names
        metric_names = set()
        for metrics in results.values():
            metric_names.update(metrics.keys())
        metric_names = sorted(metric_names)

        # Print header
        header = f"{'Model':<20}"
        for metric in metric_names:
            header += f"{metric:>12}"
        print(header)
        print("-" * 80)

        # Print each model's results
        for model_name, metrics in results.items():
            row = f"{model_name:<20}"
            for metric in metric_names:
                value = metrics.get(metric, 0.0)
                row += f"{value:>12.6f}"
            print(row)

        print("=" * 80)

        # Find best model for each metric
        print("\nBEST MODELS BY METRIC:")
        for metric in metric_names:
            # Lower is better for loss metrics, higher for others
            lower_is_better = metric in ['loss', 'mse', 'rmse', 'mae', 'mape', 'max_drawdown']

            best_model = min(results.items(), key=lambda x: x[1].get(metric, float('inf')))[0] if lower_is_better else \
                        max(results.items(), key=lambda x: x[1].get(metric, float('-inf')))[0]

            best_value = results[best_model].get(metric, 0.0)
            print(f"  {metric}: {best_model} ({best_value:.6f})")

    def _save_metrics(self, metrics: Dict[str, float], filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved metrics to {filepath}")

    def _save_comparison(self, results: Dict[str, Dict[str, float]], filepath: str):
        """Save comparison to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved comparison to {filepath}")
