"""
Early stopping utility for training.

Monitors validation loss and stops training when no improvement.
"""

import torch
import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' (minimize or maximize the monitored metric)
        verbose: Print messages
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current validation score
            epoch: Current epoch number

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  -> Score improved to {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  -> No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  -> Early stopping triggered at epoch {epoch}")
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def get_best_score(self) -> Optional[float]:
        """Get best score seen so far."""
        return self.best_score

    def get_best_epoch(self) -> int:
        """Get epoch of best score."""
        return self.best_epoch


class ModelCheckpoint:
    """
    Model checkpointing utility.

    Saves best model checkpoints during training.
    """

    def __init__(
        self,
        save_dir: str,
        mode: str = 'min',
        save_best_only: bool = True,
        save_last_n: int = 3,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.

        Args:
            save_dir: Directory to save checkpoints
            mode: 'min' or 'max' (minimize or maximize the monitored metric)
            save_best_only: Only save best model
            save_last_n: Save last N checkpoints
            verbose: Print messages
        """
        self.save_dir = save_dir
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last_n = save_last_n
        self.verbose = verbose

        self.best_score = None
        self.checkpoint_history = []

        import os
        os.makedirs(save_dir, exist_ok=True)

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        loss: float,
        extra_state: Optional[dict] = None
    ) -> str:
        """
        Save checkpoint if score improved.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            score: Validation score
            loss: Training loss
            extra_state: Extra state to save (lr, scheduler state, etc.)

        Returns:
            Path to saved checkpoint
        """
        improved = False

        if self.best_score is None:
            improved = True
        elif self.mode == 'min' and score < self.best_score:
            improved = True
        elif self.mode == 'max' and score > self.best_score:
            improved = True

        if improved or not self.save_best_only:
            import os
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'loss': loss,
            }

            if extra_state:
                checkpoint.update(extra_state)

            # Generate filename
            if improved:
                filename = 'best_model.pth'
                self.best_score = score
                if self.verbose:
                    print(f"  -> Saving best model (score: {score:.6f})")
            else:
                filename = f'checkpoint_epoch_{epoch}.pth'
                if self.verbose:
                    print(f"  -> Saving checkpoint (score: {score:.6f})")

            filepath = os.path.join(self.save_dir, filename)
            torch.save(checkpoint, filepath)

            # Manage checkpoint history
            self.checkpoint_history.append(filepath)

            # Keep only last N checkpoints
            if not self.save_best_only:
                while len(self.checkpoint_history) > self.save_last_n:
                    old_checkpoint = self.checkpoint_history.pop(0)
                    if old_checkpoint != filepath:
                        try:
                            os.remove(old_checkpoint)
                        except:
                            pass

            return filepath

        return ""

    def load_best(self, model: torch.nn.Module, device: str = 'cuda') -> dict:
        """
        Load best checkpoint.

        Args:
            model: Model to load weights into
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        import os
        filepath = os.path.join(self.save_dir, 'best_model.pth')

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if self.verbose:
            print(f"Loaded best model from epoch {checkpoint['epoch']} (score: {checkpoint['score']:.6f})")

        return checkpoint

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filepath: Optional[str] = None,
        device: str = 'cuda'
    ) -> dict:
        """
        Load specific checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            filepath: Checkpoint file path (default: best_model.pth)
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        import os
        if filepath is None:
            filepath = os.path.join(self.save_dir, 'best_model.pth')

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.verbose:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (score: {checkpoint['score']:.6f})")

        return checkpoint

    @property
    def has_checkpoint(self) -> bool:
        """Check if best checkpoint exists."""
        import os
        return os.path.exists(os.path.join(self.save_dir, 'best_model.pth'))
