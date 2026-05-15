"""
Early stopping utility for training.

Monitors validation loss and stops training when no improvement.
"""

import torch
import numpy as np
from typing import Optional
from datetime import datetime
import os
import glob
import re
import tempfile

from src.utils.logger import get_training_logger


def make_weights_only_safe(value):
    """
    Convert checkpoint metadata to PyTorch weights_only-compatible types.

    Safe checkpoint loading accepts tensors, primitive Python types, lists,
    tuples, and dicts. Metrics often contain NumPy scalar values, so convert
    those before saving.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: make_weights_only_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_weights_only_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(make_weights_only_safe(item) for item in value)
    return value


def atomic_torch_save(payload, filepath: str) -> None:
    """Save a checkpoint atomically to avoid partial-file corruption."""
    directory = os.path.dirname(filepath) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_checkpoint_", suffix=".pth", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _checkpoint_matches_requirements(
    filepath: str,
    model_type: Optional[str] = None,
    num_features: Optional[int] = None,
    num_stocks: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> bool:
    """
    Check whether a checkpoint is compatible with the current dataset/model shape.

    Compatibility rules:
    - model_type matches when checkpoint metadata provides it
    - num_features uses explicit metadata when available
    - num_stocks / num_groups use explicit metadata when available, otherwise
      fall back to embedding weight shapes
    """
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    except Exception:
        return False

    checkpoint_model_type = checkpoint.get('model_type')
    if model_type and checkpoint_model_type and checkpoint_model_type != model_type:
        return False

    checkpoint_num_features = checkpoint.get('num_features')
    if num_features is not None and checkpoint_num_features is not None:
        if checkpoint_num_features != num_features:
            return False

    checkpoint_num_stocks = checkpoint.get('num_stocks')
    if num_stocks is not None and checkpoint_num_stocks is not None:
        if checkpoint_num_stocks != num_stocks:
            return False

    checkpoint_num_groups = checkpoint.get('num_groups')
    if num_groups is not None and checkpoint_num_groups is not None:
        if checkpoint_num_groups != num_groups:
            return False

    state_dict = checkpoint.get('model_state_dict')
    if not isinstance(state_dict, dict):
        return True

    stock_shape = state_dict.get('embeddings.stock_embedding.weight')
    if num_stocks is not None and stock_shape is not None and stock_shape.shape[0] != num_stocks:
        return False

    group_shape = state_dict.get('embeddings.group_embedding.weight')
    if num_groups is not None and group_shape is not None and group_shape.shape[0] != num_groups:
        return False

    return True


def _choose_compatible_checkpoint(
    files: list,
    model_type: Optional[str] = None,
    num_features: Optional[int] = None,
    num_stocks: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> str:
    """
    Choose the newest checkpoint compatible with the current dataset.

    Falls back to the newest checkpoint overall when no compatibility filters
    are provided or when no candidate matches.
    """
    if not files:
        raise FileNotFoundError("No checkpoint candidates provided")

    files = sorted(files, key=os.path.getmtime, reverse=True)

    if num_features is None and num_stocks is None and num_groups is None and model_type is None:
        return files[0]

    compatible = [
        path for path in files
        if _checkpoint_matches_requirements(
            path,
            model_type=model_type,
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
        )
    ]
    if compatible:
        return compatible[0]

    return files[0]


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

    Saves best model checkpoints during training with timestamp and model name.
    """

    def __init__(
        self,
        save_dir: str,
        mode: str = 'min',
        save_best_only: bool = True,
        save_last_n: int = 3,
        checkpoint_frequency: int = 1,
        verbose: bool = True,
        model_type: str = 'model'
    ):
        """
        Initialize model checkpoint.

        Args:
            save_dir: Directory to save checkpoints
            mode: 'min' or 'max' (minimize or maximize the monitored metric)
            save_best_only: Only save best model
            save_last_n: Save last N checkpoints
            checkpoint_frequency: Save periodic epoch checkpoints every N epochs
            verbose: Print messages
            model_type: Model type name for filename (e.g., 'crnn_attention')
        """
        self.save_dir = save_dir
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last_n = save_last_n
        self.checkpoint_frequency = max(1, int(checkpoint_frequency))
        self.verbose = verbose
        self.model_type = model_type

        self.best_score = None
        self.periodic_checkpoint_history = []
        self.logger = get_training_logger(log_dir="logs")

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

        should_save_periodic = (epoch % self.checkpoint_frequency) == 0
        should_save_periodic = should_save_periodic and not self.save_best_only
        should_save = improved or should_save_periodic

        if should_save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint = {
                'model_type': self.model_type,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'loss': loss,
            }

            if extra_state:
                checkpoint.update(extra_state)
            checkpoint = make_weights_only_safe(checkpoint)

            # Generate filename with model name and timestamp
            best_filepath = ""
            periodic_filepath = ""
            if improved:
                filename = f'{self.model_type}_best.pth'
                self.best_score = score
                if self.verbose:
                    print(f"  -> Saving best model (score: {score:.6f})")
                best_filepath = os.path.join(self.save_dir, filename)
                atomic_torch_save(checkpoint, best_filepath)
                self.logger.log_checkpoint(best_filepath, score, metric_name="score")

            if should_save_periodic:
                filename = f'{self.model_type}_latest_periodic.pth'
                if self.verbose:
                    print(f"  -> Saving periodic checkpoint (score: {score:.6f})")
                periodic_filepath = os.path.join(self.save_dir, filename)
                atomic_torch_save(checkpoint, periodic_filepath)
                self.logger.log_checkpoint(periodic_filepath, score, metric_name="score")

            return best_filepath or periodic_filepath

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
        import glob
        # Find best model file by pattern (supports both new and old naming)
        exact_path = os.path.join(self.save_dir, f'{self.model_type}_best.pth')
        if os.path.exists(exact_path):
            filepath = exact_path
        else:
            pattern = os.path.join(self.save_dir, f'{self.model_type}_best_*.pth')
            files = glob.glob(pattern)

            # Fallback to old naming convention for backward compatibility
            if not files:
                old_pattern = os.path.join(self.save_dir, 'best_model.pth')
                if os.path.exists(old_pattern):
                    filepath = old_pattern
                else:
                    raise FileNotFoundError(f"No checkpoint found matching pattern {pattern} or {old_pattern}")
            else:
                # Get the most recent best model
                filepath = max(files, key=os.path.getmtime)

        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        if self.verbose:
            model_type = checkpoint.get('model_type', 'unknown')
            print(f"Loaded best {model_type} model from {os.path.basename(filepath)} (epoch: {checkpoint['epoch']}, score: {checkpoint['score']:.6f})")

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
            filepath: Checkpoint file path (default: finds best model for model_type)
            device: Device to load to

        Returns:
            Checkpoint dictionary
        """
        import glob
        if filepath is None:
            exact_path = os.path.join(self.save_dir, f'{self.model_type}_best.pth')
            if os.path.exists(exact_path):
                filepath = exact_path
            else:
                # Try legacy pattern next
                pattern = os.path.join(self.save_dir, f'{self.model_type}_best_*.pth')
                files = glob.glob(pattern)

                if files:
                    filepath = max(files, key=os.path.getmtime)
                else:
                    # Fallback to old naming convention
                    old_path = os.path.join(self.save_dir, 'best_model.pth')
                    if os.path.exists(old_path):
                        filepath = old_path
                    else:
                        raise FileNotFoundError(f"No checkpoint found matching {exact_path}, {pattern}, or {old_path}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.verbose:
            model_type = checkpoint.get('model_type', 'unknown')
            print(f"Loaded {model_type} checkpoint from {os.path.basename(filepath)} (epoch: {checkpoint['epoch']}, score: {checkpoint['score']:.6f})")

        return checkpoint

    @property
    def has_checkpoint(self) -> bool:
        """Check if best checkpoint exists."""
        exact_path = os.path.join(self.save_dir, f'{self.model_type}_best.pth')
        if os.path.exists(exact_path):
            return True
        pattern = os.path.join(self.save_dir, f'{self.model_type}_best_*.pth')
        if glob.glob(pattern):
            return True
        # Fallback to old naming
        return os.path.exists(os.path.join(self.save_dir, 'best_model.pth'))


def find_checkpoint_path(
    model_input: str,
    checkpoint_dir: str = 'models/checkpoints',
    model_type: Optional[str] = None,
    num_features: Optional[int] = None,
    num_stocks: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> str:
    """
    Find checkpoint path from various input formats.

    Supports:
    - Direct file path: /path/to/model.pth
    - "best": Find latest best model for model_type (or any model if model_type not specified)
    - "final": Find latest final-trained checkpoint (any model or specific model_type)
    - "{model_type}": Find best model for specific model type
    - Pattern: "crnn_attention", "*transformer*", etc.

    Args:
        model_input: Model identifier (path, "best", "final", or pattern)
        checkpoint_dir: Directory to search for checkpoints
        model_type: Preferred model type for "best" or "latest" searches

    Returns:
        Full path to checkpoint file

    Raises:
        FileNotFoundError: If no matching checkpoint found
    """
    # If it's an existing file path, return it
    if os.path.isfile(model_input):
        return model_input

    # If it's a relative path that exists
    full_path = os.path.join(checkpoint_dir, model_input)
    if os.path.isfile(full_path):
        return full_path

    # Handle special keywords
    if model_input.lower() == 'best':
        return _find_best_checkpoint(
            checkpoint_dir,
            model_type,
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
        )
    elif model_input.lower() == 'final':
        return _find_final_checkpoint(
            checkpoint_dir,
            model_type,
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
        )
    elif model_input.lower() == 'latest':
        raise FileNotFoundError(
            "Checkpoint alias 'latest' is no longer supported. "
            "Use 'best', 'final', or provide an exact checkpoint path."
        )
    else:
        # Treat as pattern/model_type and find best match
        return _find_best_checkpoint(
            checkpoint_dir,
            model_input,
            num_features=num_features,
            num_stocks=num_stocks,
            num_groups=num_groups,
        )


def _find_best_checkpoint(
    checkpoint_dir: str,
    model_type: Optional[str] = None,
    num_features: Optional[int] = None,
    num_stocks: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> str:
    """
    Find best checkpoint for a model type.

    Args:
        checkpoint_dir: Directory to search
        model_type: Model type to search for (e.g., 'crnn_attention')

    Returns:
        Path to best checkpoint

    Raises:
        FileNotFoundError: If no checkpoint found
    """
    if model_type:
        files = []
        stable_best = os.path.join(checkpoint_dir, f'{model_type}_best.pth')
        if os.path.exists(stable_best):
            files.append(stable_best)
        pattern = os.path.join(checkpoint_dir, f'{model_type}_best_*.pth')
        files.extend(glob.glob(pattern))

        if files:
            return _choose_compatible_checkpoint(
                files,
                model_type=model_type,
                num_features=num_features,
                num_stocks=num_stocks,
                num_groups=num_groups,
            )

        # Search for old pattern: best_model.pth
        old_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(old_path):
            return old_path

        raise FileNotFoundError(
            f"No best checkpoint found for model_type '{model_type}' in {checkpoint_dir}. "
            f"Searched for: {stable_best}, {pattern} and {old_path}"
        )
    else:
        files = glob.glob(os.path.join(checkpoint_dir, '*_best.pth'))
        files.extend(glob.glob(os.path.join(checkpoint_dir, '*_best_*.pth')))

        if files:
            return _choose_compatible_checkpoint(
                files,
                model_type=model_type,
                num_features=num_features,
                num_stocks=num_stocks,
                num_groups=num_groups,
            )

        # Fallback to old pattern
        old_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(old_path):
            return old_path

        raise FileNotFoundError(
            f"No best checkpoint found in {checkpoint_dir}. "
            f"Please specify model_type or provide a direct path."
        )


def _find_final_checkpoint(
    checkpoint_dir: str,
    model_type: Optional[str] = None,
    num_features: Optional[int] = None,
    num_stocks: Optional[int] = None,
    num_groups: Optional[int] = None,
) -> str:
    """
    Find most recent final-trained checkpoint.

    Args:
        checkpoint_dir: Directory to search
        model_type: Model type to search for

    Returns:
        Path to most recent checkpoint

    Raises:
        FileNotFoundError: If no checkpoint found
    """
    if model_type:
        pattern = os.path.join(checkpoint_dir, f'{model_type}_final*.pth')
        files = glob.glob(pattern)
    else:
        pattern = os.path.join(checkpoint_dir, '*_final*.pth')
        files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No final checkpoint found in {checkpoint_dir} "
            f"(searched: {pattern})"
        )

    return _choose_compatible_checkpoint(
        files,
        model_type=model_type,
        num_features=num_features,
        num_stocks=num_stocks,
        num_groups=num_groups,
    )


def list_checkpoints(checkpoint_dir: str = 'models/checkpoints', model_type: Optional[str] = None) -> list:
    """
    List all available checkpoints with metadata.

    Args:
        checkpoint_dir: Directory to search
        model_type: Filter by model type (optional)

    Returns:
        List of dicts with checkpoint info (path, model_type, timestamp, size)
    """
    pattern = os.path.join(checkpoint_dir, '*.pth')
    files = glob.glob(pattern)

    checkpoints = []
    for filepath in files:
        filename = os.path.basename(filepath)

        # Extract model_type from filename
        # Pattern: {model_type}_best_{timestamp}.pth or {model_type}_epoch{N}_{timestamp}.pth
        # We need to find where _best_ or _epoch occurs to determine the model_type
        model_match = re.match(r'(.+?)_(?:best|epoch\d+)_(\d{8}_\d{6})\.pth$', filename)
        if model_match:
            file_model_type = model_match.group(1)
            timestamp_str = model_match.group(2)
        elif filename.endswith('_best.pth'):
            file_model_type = filename[:-len('_best.pth')]
            timestamp_str = None
        elif filename.endswith('_latest_periodic.pth'):
            file_model_type = filename[:-len('_latest_periodic.pth')]
            timestamp_str = None
        elif filename == 'best_model.pth':
            file_model_type = 'unknown'
            timestamp_str = None
        else:
            file_model_type = 'unknown'
            timestamp_str = None

        # Filter by model_type if specified
        if model_type and file_model_type != model_type:
            continue

        checkpoints.append({
            'path': filepath,
            'filename': filename,
            'model_type': file_model_type,
            'timestamp': timestamp_str,
            'size_mb': os.path.getsize(filepath) / (1024 * 1024),
            'mtime': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
        })

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
    return checkpoints
