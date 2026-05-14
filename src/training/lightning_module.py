"""
PyTorch Lightning backend for financial models.

Lightning is the default training backend when installed. The custom Trainer is
kept as the explicit backup path.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .common import (
    calculate_prediction_health,
    collapse_penalty_from_health,
    create_loss_function,
)
from .early_stopping import make_weights_only_safe, atomic_torch_save
from .experiment_tracking import create_experiment_tracker, training_params


class LightningDependencyError(ImportError):
    """Raised when Lightning is requested but not installed."""


def _require_lightning():
    try:
        import lightning as L
    except ImportError:
        try:
            import pytorch_lightning as L
        except ImportError as exc:
            raise LightningDependencyError(
                "PyTorch Lightning is required for the lightning training backend. "
                "Install it with `pip install lightning`, or run with `--backend custom`."
            ) from exc
    return L


try:
    _LIGHTNING = _require_lightning()
    _LIGHTNING_AVAILABLE = True
    _LIGHTNING_MODULE_BASE = _LIGHTNING.LightningModule
    _LIGHTNING_CALLBACK_BASE = _LIGHTNING.Callback
except LightningDependencyError:
    _LIGHTNING = None
    _LIGHTNING_AVAILABLE = False
    _LIGHTNING_MODULE_BASE = torch.nn.Module
    _LIGHTNING_CALLBACK_BASE = object


def _batch_forward(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["features"],
        batch["stock_id"],
        batch["group_id"],
        batch["day"],
        batch["month"],
        batch["dividend_flag"],
    )


class FinancialLightningModule(_LIGHTNING_MODULE_BASE):
    """
    LightningModule wrapper around existing financial prediction models.
    """

    def __init__(self, model, config, model_type: str = "model"):
        if not _LIGHTNING_AVAILABLE:
            _require_lightning()
        super().__init__()
        self.model = model
        self.config_obj = config
        self.model_type = model_type
        self.criterion = create_loss_function(config)
        self._val_epoch_predictions = []
        self._val_epoch_targets = []

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return _batch_forward(self.model, batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self(batch)
        target = batch["target"]
        loss = self.criterion(output, target)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        output = self(batch)
        target = batch["target"]
        loss = self.criterion(output, target)
        mse = torch.mean((output - target) ** 2)
        mae = torch.mean(torch.abs(output - target))
        rmse = torch.sqrt(mse)
        directional_accuracy = (torch.sign(output) == torch.sign(target)).float().mean()

        metrics = {
            "val/loss": loss,
            "val/mse": mse,
            "val/mae": mae,
            "val/rmse": rmse,
            "val/directional_accuracy": directional_accuracy,
        }
        self._val_epoch_predictions.extend(output.detach().cpu().numpy().flatten().tolist())
        self._val_epoch_targets.extend(target.detach().cpu().numpy().flatten().tolist())
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False)
        return metrics

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_predictions = []
        self._val_epoch_targets = []

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            return
        health = calculate_prediction_health(self._val_epoch_predictions, self._val_epoch_targets)
        collapse_penalty, is_collapsed = collapse_penalty_from_health(health)
        metrics = {
            "val/pred_positive_rate": health["pred_positive_rate"],
            "val/pred_negative_rate": health["pred_negative_rate"],
            "val/pred_std": health["pred_std"],
            "val/pred_mean": health["pred_mean"],
            "val/collapse_penalty": collapse_penalty,
            "val/is_collapsed": float(is_collapsed),
        }
        if health["target_positive_rate"] is not None:
            metrics["val/target_positive_rate"] = health["target_positive_rate"]
        if health["pred_target_corr"] is not None:
            metrics["val/pred_target_corr"] = health["pred_target_corr"]
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        if self.config_obj.model.training.SCHEDULER == "reduce_on_plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        from .common import create_optimizer
        return create_optimizer(self.model, self.config_obj)

    def _create_scheduler(self, optimizer):
        from .common import create_scheduler
        return create_scheduler(optimizer, self.config_obj)


class CustomFormatCheckpointCallback(_LIGHTNING_CALLBACK_BASE):
    """
    Save custom-compatible `.pth` checkpoints from Lightning training.
    """

    def __init__(
        self,
        save_dir: str,
        model_type: str,
        checkpoint_metadata: Optional[Dict] = None,
        save_best_only: bool = True,
        save_last_n: int = 3,
        checkpoint_frequency: int = 1,
    ):
        if not _LIGHTNING_AVAILABLE:
            _require_lightning()
        self.save_dir = Path(save_dir)
        self.model_type = model_type
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.save_best_only = save_best_only
        self.save_last_n = save_last_n
        self.checkpoint_frequency = max(1, int(checkpoint_frequency))
        self.best_score = None
        self.best_selection_score = None
        self.best_path = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        score = trainer.callback_metrics.get("val/loss")
        if score is None:
            return
        score_value = float(score.detach().cpu().item() if isinstance(score, torch.Tensor) else score)
        collapse_penalty = _metric_to_float(trainer.callback_metrics.get("val/collapse_penalty")) or 0.0
        is_collapsed = bool(_metric_to_float(trainer.callback_metrics.get("val/is_collapsed")) or 0.0)
        selection_score = score_value + collapse_penalty
        improved = self.best_selection_score is None or selection_score < self.best_selection_score
        epoch_number = int(trainer.current_epoch + 1)
        should_save_periodic = (epoch_number % self.checkpoint_frequency) == 0
        should_save_periodic = should_save_periodic and not self.save_best_only
        should_save = improved or should_save_periodic
        if not should_save:
            return
        if improved:
            self.best_selection_score = selection_score
            self.best_score = score_value

        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        train_loss = trainer.callback_metrics.get("train/loss")
        train_loss_value = _metric_to_float(train_loss)
        checkpoint = {
            "model_type": self.model_type,
            "epoch": epoch_number,
            "model_state_dict": pl_module.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "score": score_value,
            "selection_score": selection_score,
            "loss": train_loss_value,
            "best_val_loss": score_value,
            "val_metrics": {
                key.replace("val/", ""): value
                for key, value in {
                    "val/loss": score_value,
                    "val/mse": _metric_to_float(trainer.callback_metrics.get("val/mse")),
                    "val/mae": _metric_to_float(trainer.callback_metrics.get("val/mae")),
                    "val/rmse": _metric_to_float(trainer.callback_metrics.get("val/rmse")),
                    "val/directional_accuracy": _metric_to_float(trainer.callback_metrics.get("val/directional_accuracy")),
                    "val/pred_positive_rate": _metric_to_float(trainer.callback_metrics.get("val/pred_positive_rate")),
                    "val/pred_negative_rate": _metric_to_float(trainer.callback_metrics.get("val/pred_negative_rate")),
                    "val/pred_std": _metric_to_float(trainer.callback_metrics.get("val/pred_std")),
                    "val/pred_mean": _metric_to_float(trainer.callback_metrics.get("val/pred_mean")),
                    "val/target_positive_rate": _metric_to_float(trainer.callback_metrics.get("val/target_positive_rate")),
                    "val/pred_target_corr": _metric_to_float(trainer.callback_metrics.get("val/pred_target_corr")),
                    "val/collapse_penalty": collapse_penalty,
                    "val/is_collapsed": float(is_collapsed),
                }.items()
                if value is not None
            },
            **self.checkpoint_metadata,
            "metadata": {
                "model_type": self.model_type,
                "training_backend": "lightning",
                "is_collapsed": is_collapsed,
                **self.checkpoint_metadata,
            },
        }
        checkpoint = make_weights_only_safe(checkpoint)

        if improved:
            path = self.save_dir / f"{self.model_type}_best_lightning.pth"
            atomic_torch_save(checkpoint, str(path))
            self.best_path = str(path)

        if should_save_periodic:
            periodic_path = self.save_dir / f"{self.model_type}_latest_periodic_lightning.pth"
            atomic_torch_save(checkpoint, str(periodic_path))


class ExperimentTrackingCallback(_LIGHTNING_CALLBACK_BASE):
    """Log Lightning training metrics through the existing experiment tracker."""

    def __init__(self, config, model_type: str):
        if not _LIGHTNING_AVAILABLE:
            _require_lightning()
        self.config = config
        self.model_type = model_type
        self.tracker = create_experiment_tracker(config)
        self.run_active = False

    def on_fit_start(self, trainer, pl_module):
        self.tracker.start_run(run_name=self.model_type)
        self.tracker.log_params(training_params(self.config, self.model_type))
        self.run_active = True

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = self._extract_metrics(trainer, prefix="train/")
        if metrics:
            self.tracker.log_metrics(metrics, step=trainer.current_epoch + 1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        metrics = self._extract_metrics(trainer, prefix="val/")
        if metrics:
            self.tracker.log_metrics(metrics, step=trainer.current_epoch + 1)

    def on_fit_end(self, trainer, pl_module):
        self._end_run(status="FINISHED")

    def on_exception(self, trainer, pl_module, exception):
        self._end_run(status="FAILED")

    def _extract_metrics(self, trainer, prefix: str) -> Dict[str, float]:
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if not str(key).startswith(prefix):
                continue
            metric_value = _metric_to_float(value)
            if metric_value is not None:
                metrics[str(key)] = metric_value
        return metrics

    def _end_run(self, status: str) -> None:
        if not self.run_active:
            return
        self.tracker.end_run(status=status)
        self.run_active = False


def _metric_to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _create_tensorboard_logger(config):
    """Create a TensorBoard logger when a log directory is configured."""
    log_dir = getattr(config.model.logging, "TENSORBOARD_DIR", None)
    if not log_dir:
        return False

    L = _require_lightning()
    log_path = Path(log_dir)

    if hasattr(L, "pytorch"):
        logger_cls = L.pytorch.loggers.TensorBoardLogger
    else:
        logger_cls = L.loggers.TensorBoardLogger

    return logger_cls(
        save_dir=str(log_path.parent),
        name=log_path.name,
        default_hp_metric=False,
    )


def create_lightning_trainer(config, device: str, callbacks: Optional[list] = None):
    """Create a Lightning trainer from existing config."""
    L = _require_lightning()
    accelerator = "gpu" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if config.model.training.USE_MIXED_PRECISION and accelerator == "gpu" else "32-true"
    gradient_clip_value = config.model.training.GRADIENT_CLIP_VALUE
    return L.Trainer(
        max_epochs=config.model.training.NUM_EPOCHS,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=gradient_clip_value if gradient_clip_value > 0 else None,
        num_sanity_val_steps=0,
        log_every_n_steps=max(1, int(config.model.logging.LOG_FREQUENCY)),
        callbacks=callbacks or [],
        logger=_create_tensorboard_logger(config),
        enable_checkpointing=False,
        enable_model_summary=False,
    )


def train_with_lightning(
    model,
    config,
    train_loader,
    val_loader=None,
    device: str = "cpu",
    model_type: str = "model",
    checkpoint_metadata: Optional[Dict] = None,
) -> Dict[str, object]:
    """
    Train a model using Lightning and write custom-compatible checkpoints.
    """
    checkpoint_callback = CustomFormatCheckpointCallback(
        save_dir=config.model.checkpointing.CHECKPOINT_DIR,
        model_type=model_type,
        checkpoint_metadata=checkpoint_metadata,
        save_best_only=config.model.checkpointing.SAVE_BEST_ONLY,
        save_last_n=config.model.checkpointing.SAVE_LAST_N,
        checkpoint_frequency=config.model.checkpointing.CHECKPOINT_FREQUENCY,
    )
    experiment_callback = ExperimentTrackingCallback(
        config=config,
        model_type=model_type,
    )
    lightning_module = FinancialLightningModule(model=model, config=config, model_type=model_type)
    trainer = create_lightning_trainer(
        config=config,
        device=device,
        callbacks=[checkpoint_callback, experiment_callback],
    )
    trainer.fit(lightning_module, train_loader, val_loader)
    return {
        "backend": "lightning",
        "trainer": trainer,
        "module": lightning_module,
        "best_model_path": checkpoint_callback.best_path,
        "best_score": checkpoint_callback.best_score,
    }


def save_final_lightning_checkpoint(
    trainer,
    lightning_module,
    checkpoint_dir: str,
    model_type: str,
    checkpoint_metadata: Optional[Dict] = None,
) -> str:
    """
    Save the final trained Lightning model in the project's custom-compatible format.

    This is used to guarantee at least one checkpoint exists even when no
    validation loader is available, which means no best-checkpoint callback can
    trigger from `val/loss`.
    """
    checkpoint_metadata = checkpoint_metadata or {}
    save_dir = Path(checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = trainer.optimizers[0] if trainer.optimizers else None
    epoch_number = int(trainer.current_epoch + 1)

    callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
    checkpoint = {
        "model_type": model_type,
        "epoch": epoch_number,
        "model_state_dict": lightning_module.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "score": _metric_to_float(callback_metrics.get("val/loss")),
        "selection_score": _metric_to_float(callback_metrics.get("val/loss")),
        "loss": _metric_to_float(callback_metrics.get("train/loss")),
        "best_val_loss": _metric_to_float(callback_metrics.get("val/loss")),
        "val_metrics": {
            str(key).replace("val/", ""): value
            for key, value in (
                (key, _metric_to_float(metric_value))
                for key, metric_value in callback_metrics.items()
                if str(key).startswith("val/")
            )
            if value is not None
        },
        **checkpoint_metadata,
        "metadata": {
            "model_type": model_type,
            "training_backend": "lightning",
            **checkpoint_metadata,
        },
    }
    checkpoint = make_weights_only_safe(checkpoint)

    final_path = save_dir / f"{model_type}_final_lightning.pth"
    atomic_torch_save(checkpoint, str(final_path))
    return str(final_path)
