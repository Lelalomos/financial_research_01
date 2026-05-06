"""
PyTorch Lightning backend for financial models.

Lightning is the default training backend when installed. The custom Trainer is
kept as the explicit backup path.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .common import create_loss_function
from .early_stopping import make_weights_only_safe


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
        self.log_dict(metrics, prog_bar=False, on_epoch=True, on_step=False)
        return metrics

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
        training = self.config_obj.model.training
        if training.OPTIMIZER == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=training.LEARNING_RATE,
                weight_decay=training.WEIGHT_DECAY,
            )
        if training.OPTIMIZER == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=training.LEARNING_RATE,
                weight_decay=training.WEIGHT_DECAY,
            )
        if training.OPTIMIZER == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=training.LEARNING_RATE,
                momentum=0.9,
                weight_decay=training.WEIGHT_DECAY,
            )
        if training.OPTIMIZER == "rmsprop":
            return torch.optim.RMSprop(
                self.model.parameters(),
                lr=training.LEARNING_RATE,
                weight_decay=training.WEIGHT_DECAY,
            )
        raise ValueError(f"Unknown optimizer: {training.OPTIMIZER}")

    def _create_scheduler(self, optimizer):
        scheduler_name = self.config_obj.model.training.SCHEDULER
        if scheduler_name is None:
            return None
        params = self.config_obj.get_scheduler_params()
        if scheduler_name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        if scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, **params)
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


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
    ):
        if not _LIGHTNING_AVAILABLE:
            _require_lightning()
        self.save_dir = Path(save_dir)
        self.model_type = model_type
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.save_best_only = save_best_only
        self.best_score = None
        self.best_path = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        score = trainer.callback_metrics.get("val/loss")
        if score is None:
            return
        score_value = float(score.detach().cpu().item() if isinstance(score, torch.Tensor) else score)
        improved = self.best_score is None or score_value < self.best_score
        if self.save_best_only and not improved:
            return
        if improved:
            self.best_score = score_value

        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        train_loss = trainer.callback_metrics.get("train/loss")
        train_loss_value = _metric_to_float(train_loss)
        path = self.save_dir / f"{self.model_type}_best_lightning.pth"
        checkpoint = {
            "model_type": self.model_type,
            "epoch": int(trainer.current_epoch + 1),
            "model_state_dict": pl_module.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "score": score_value,
            "loss": train_loss_value,
            "best_val_loss": score_value,
            **self.checkpoint_metadata,
            "metadata": {
                "model_type": self.model_type,
                "training_backend": "lightning",
                **self.checkpoint_metadata,
            },
        }
        torch.save(make_weights_only_safe(checkpoint), path)
        self.best_path = str(path)


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


def create_lightning_trainer(config, device: str, callbacks: Optional[list] = None):
    """Create a Lightning trainer from existing config."""
    L = _require_lightning()
    accelerator = "gpu" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if config.model.training.USE_MIXED_PRECISION and accelerator == "gpu" else "32-true"
    return L.Trainer(
        max_epochs=config.model.training.NUM_EPOCHS,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=callbacks or [],
        logger=False,
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
    )
    lightning_module = FinancialLightningModule(model=model, config=config, model_type=model_type)
    trainer = create_lightning_trainer(
        config=config,
        device=device,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(lightning_module, train_loader, val_loader)
    return {
        "backend": "lightning",
        "trainer": trainer,
        "module": lightning_module,
        "best_model_path": checkpoint_callback.best_path,
        "best_score": checkpoint_callback.best_score,
    }
