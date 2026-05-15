"""
Unit tests for typed config validation.
"""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import load_config
from src.config.schemas import validate_config_data


def _load_raw_config(name: str):
    path = Path(__file__).parent.parent / "config" / f"{name}.json"
    with open(path, "r") as f:
        return json.load(f)


def test_main_config_schema_accepts_current_config():
    data = _load_raw_config("main")
    validate_config_data("main", data)


def test_main_config_accepts_enabled_market_regime():
    data = _load_raw_config("main")
    valid = copy.deepcopy(data)
    valid["data"]["regime"] = {
        "ENABLED": True,
        "METHOD": "quantile",
        "PROXY_COLUMN": "vix",
        "N_REGIMES": 3,
        "LOW_QUANTILE": 0.25,
        "HIGH_QUANTILE": 0.75,
        "DEFAULT_REGIME": 1,
    }
    valid["data"]["features"]["FEATURE_FLAGS"]["market_regime"] = True

    validate_config_data("main", valid)


def test_main_config_accepts_on_the_fly_sequence_mode():
    data = _load_raw_config("main")
    valid = copy.deepcopy(data)
    valid["data"]["dataset"] = {"MODE": "on_the_fly_sequences"}

    validate_config_data("main", valid)


def test_main_config_rejects_invalid_regime_quantiles():
    data = _load_raw_config("main")
    invalid = copy.deepcopy(data)
    invalid["data"]["regime"] = {
        "ENABLED": True,
        "METHOD": "quantile",
        "PROXY_COLUMN": "vix",
        "N_REGIMES": 3,
        "LOW_QUANTILE": 0.8,
        "HIGH_QUANTILE": 0.2,
        "DEFAULT_REGIME": 1,
    }

    with pytest.raises(ValidationError, match="LOW_QUANTILE"):
        validate_config_data("main", invalid)


def test_model_config_schema_accepts_current_config():
    data = _load_raw_config("model")
    validate_config_data("model", data)


def test_model_config_accepts_directional_huber_loss():
    data = _load_raw_config("model")
    valid = copy.deepcopy(data)
    valid["model"]["loss"]["LOSS_TYPE"] = "directional_huber"
    valid["model"]["loss"]["HUBER_DELTA"] = 1.5
    valid["model"]["loss"]["DIRECTIONAL_ALPHA"] = 0.4

    validate_config_data("model", valid)


def test_model_config_accepts_enabled_ensemble():
    data = _load_raw_config("model")
    valid = copy.deepcopy(data)
    valid["model"]["ensemble"] = {
        "ENABLED": True,
        "CHECKPOINT_PATHS": ["models/checkpoints/a.pth", "models/checkpoints/b.pth"],
        "WEIGHTS": [0.25, 0.75],
        "REQUIRE_MATCHING_FEATURES": True,
        "REQUIRE_MATCHING_TARGET_NORMALIZATION": True,
    }

    validate_config_data("model", valid)


def test_model_config_rejects_enabled_ensemble_with_one_checkpoint():
    data = _load_raw_config("model")
    invalid = copy.deepcopy(data)
    invalid["model"]["ensemble"] = {
        "ENABLED": True,
        "CHECKPOINT_PATHS": ["models/checkpoints/a.pth"],
        "WEIGHTS": None,
        "REQUIRE_MATCHING_FEATURES": True,
        "REQUIRE_MATCHING_TARGET_NORMALIZATION": True,
    }

    with pytest.raises(ValidationError, match="at least two"):
        validate_config_data("model", invalid)


def test_model_config_rejects_ensemble_weight_length_mismatch():
    data = _load_raw_config("model")
    invalid = copy.deepcopy(data)
    invalid["model"]["ensemble"] = {
        "ENABLED": True,
        "CHECKPOINT_PATHS": ["models/checkpoints/a.pth", "models/checkpoints/b.pth"],
        "WEIGHTS": [1.0],
        "REQUIRE_MATCHING_FEATURES": True,
        "REQUIRE_MATCHING_TARGET_NORMALIZATION": True,
    }

    with pytest.raises(ValidationError, match="WEIGHTS length"):
        validate_config_data("model", invalid)


def test_model_config_rejects_string_batch_size():
    data = _load_raw_config("model")
    invalid = copy.deepcopy(data)
    invalid["model"]["training"]["BATCH_SIZE"] = "128"

    with pytest.raises(ValidationError):
        validate_config_data("model", invalid)


def test_main_config_rejects_invalid_split_sum():
    data = _load_raw_config("main")
    invalid = copy.deepcopy(data)
    invalid["data"]["splits"]["TRAIN_RATIO"] = 0.8

    with pytest.raises(ValidationError, match="must equal 1.0"):
        validate_config_data("main", invalid)


def test_load_config_still_returns_attribute_config():
    config = load_config("model")
    assert config.model.training.BATCH_SIZE > 0
