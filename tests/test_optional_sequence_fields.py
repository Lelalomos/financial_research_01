import importlib.util
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.dataset import FinancialDataset


REPO_ROOT = Path(__file__).parent.parent


def _load_script(script_name: str, module_name: str):
    script_path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_sequences():
    return {
        "features": np.ones((2, 3, 2), dtype=np.float32),
        "stock_id": np.zeros((2, 3), dtype=np.int64),
        "group_id": np.zeros((2, 3), dtype=np.int64),
        "day": np.ones((2, 3), dtype=np.int32),
        "month": np.ones((2, 3), dtype=np.int32),
        "dividend_flag": np.ones((2, 3), dtype=np.int32),
        "target": np.array([0.1, 0.2], dtype=np.float32),
        "future_ohlcv": np.ones((2, 2, 5), dtype=np.float32),
        "future_return_path": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "future_regime": np.array([1, 2], dtype=np.int64),
    }


def test_financial_dataset_returns_optional_sequence_fields():
    dataset = FinancialDataset(_make_sequences(), load_config("model"))
    sample = dataset[0]

    assert "future_ohlcv" in sample
    assert "future_return_path" in sample
    assert "future_regime" in sample
    assert sample["future_ohlcv"].shape == (2, 5)
    assert sample["future_regime"].dtype == torch.long


def test_financial_dataset_collate_fn_stacks_optional_sequence_fields():
    dataset = FinancialDataset(_make_sequences(), load_config("model"))
    batch = FinancialDataset.collate_fn([dataset[0], dataset[1]])

    assert batch["future_ohlcv"].shape == (2, 2, 5)
    assert batch["future_return_path"].shape == (2, 2)
    assert batch["future_regime"].shape == (2,)


def test_train_load_sequences_reads_optional_arrays(tmp_path):
    module = _load_script("train.py", "train_script_optional_sequences")
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    for key, value in _make_sequences().items():
        np.save(split_dir / f"{key}.npy", value, allow_pickle=False)

    sequences = module.load_sequences(tmp_path, "train")

    assert "future_ohlcv" in sequences
    assert "future_return_path" in sequences
    assert sequences["future_regime"].shape == (2,)


def test_eval_load_sequences_reads_optional_arrays(tmp_path):
    module = _load_script("test.py", "test_script_optional_sequences")
    split_dir = tmp_path / "test"
    split_dir.mkdir(parents=True)
    for key, value in _make_sequences().items():
        np.save(split_dir / f"{key}.npy", value, allow_pickle=False)

    sequences = module.load_sequences(tmp_path, "test")

    assert "future_ohlcv" in sequences
    assert "future_return_path" in sequences
