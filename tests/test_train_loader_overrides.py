from pathlib import Path
from types import SimpleNamespace
import sys


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import apply_limited_batch_loader_overrides
from src.config import load_config


def test_limited_batch_runs_disable_loader_workers():
    config = load_config("model")
    config.model.device.NUM_WORKERS = 4
    args = SimpleNamespace(max_train_batches=1, max_val_batches=None)

    apply_limited_batch_loader_overrides(config, args)

    assert config.model.device.NUM_WORKERS == 0


def test_full_runs_keep_configured_loader_workers():
    config = load_config("model")
    config.model.device.NUM_WORKERS = 4
    args = SimpleNamespace(max_train_batches=None, max_val_batches=None)

    apply_limited_batch_loader_overrides(config, args)

    assert config.model.device.NUM_WORKERS == 4
