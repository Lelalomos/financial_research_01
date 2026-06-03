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


def test_main_config_includes_chronos2_preparation_settings():
    data = _load_raw_config("main")
    chronos2_prep = data["data"]["chronos2_preparation"]
    assert chronos2_prep["OUTPUT_DIR"]
    assert chronos2_prep["TARGET_COLUMN"] == "close"
    assert chronos2_prep["INCLUDE_SCALAR_TARGET"] is True
    assert chronos2_prep["TARGET_MODE"] == "trend_extension"
    assert chronos2_prep["TREND_LOOKBACK"] == 7
    assert chronos2_prep["TREND_METHOD"] == "mean_gap"


def test_main_config_includes_kronos_rich_preparation_settings():
    data = _load_raw_config("main")
    prep = data["data"]["kronos_rich_preparation"]
    assert prep["OUTPUT_DIR"] == "data/processed_kronos_rich"
    assert prep["OHLCV_COLUMNS"] == ["open", "high", "low", "close", "volume"]
    assert prep["INCLUDE_RETURN_PATH"] is True
    assert prep["INCLUDE_REGIME_LABEL"] is True


def test_main_config_includes_chronos_rich_preparation_settings():
    data = _load_raw_config("main")
    prep = data["data"]["chronos_rich_preparation"]
    assert prep["OUTPUT_DIR"] == "data/processed_chronos_rich"
    assert prep["OHLCV_COLUMNS"] == ["open", "high", "low", "close", "volume"]
    assert prep["INCLUDE_RETURN_PATH"] is True
    assert prep["INCLUDE_REGIME_LABEL"] is True


def test_main_config_includes_cointegration_feature_settings():
    data = _load_raw_config("main")
    flags = data["data"]["features"]["FEATURE_FLAGS"]
    cointegration = data["data"]["cointegration"]

    assert isinstance(flags["cointegration_features"], bool)
    assert cointegration["ROLLING_WINDOW"] == 252
    assert cointegration["NORMALIZATION_WINDOW"] == 252
    assert cointegration["JOHANSEN_DET_ORDER"] == 0
    assert cointegration["JOHANSEN_K_AR_DIFF"] == 1


def test_main_config_includes_sampling_settings():
    data = _load_raw_config("main")
    sampling = data["data"]["sampling"]

    assert sampling["STOCK_SELECTION_MODE"] == "sorted"
    assert sampling["MARKET_CAP_METADATA_DIR"] == "raw_data/ticket_data/us"


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


def test_model_config_accepts_quantile_loss():
    data = _load_raw_config("model")
    valid = copy.deepcopy(data)
    valid["model"]["loss"]["LOSS_TYPE"] = "quantile_loss"
    valid["model"]["loss"]["QUANTILE"] = 0.2

    validate_config_data("model", valid)


def test_model_config_accepts_pinball_loss():
    data = _load_raw_config("model")
    valid = copy.deepcopy(data)
    valid["model"]["loss"]["LOSS_TYPE"] = "pinball_loss"
    valid["model"]["loss"]["QUANTILE"] = 0.8

    validate_config_data("model", valid)


def test_model_config_accepts_multi_part_rich_loss():
    data = _load_raw_config("model")
    valid = copy.deepcopy(data)
    valid["model"]["loss"]["LOSS_TYPE"] = "multi_part_rich_loss"

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


def test_model_config_default_model_type_is_current_default():
    data = _load_raw_config("model")
    assert data["model"]["selection"]["DEFAULT_MODEL_TYPE"] == "chronos_rich"
    assert "chronos_rich" in data["model"]["models"]


def test_model_config_includes_chronos2_embedding_fields():
    data = _load_raw_config("model")
    chronos2 = data["model"]["models"]["chronos2"]
    assert chronos2["USE_STOCK_EMBEDDING"] is True
    assert chronos2["USE_GROUP_EMBEDDING"] is True
    assert chronos2["STOCK_EMB_DIM"] > 0
    assert chronos2["GROUP_EMB_DIM"] > 0
    assert chronos2["DAY_EMB_DIM"] > 0
    assert chronos2["MONTH_EMB_DIM"] > 0
    assert chronos2["DIVIDEND_FLAG_EMB_DIM"] > 0


def test_model_config_includes_separate_kronos_rich_block():
    data = _load_raw_config("model")
    kronos = data["model"]["models"]["kronos"]
    kronos_rich = data["model"]["models"]["kronos_rich"]

    assert kronos_rich["tokenizer"]["D_MODEL"] == kronos["tokenizer"]["D_MODEL"]
    assert kronos_rich["network"]["D_MODEL"] == kronos["network"]["D_MODEL"]
    assert kronos_rich["predictor"]["MAX_CONTEXT"] == kronos["predictor"]["MAX_CONTEXT"]
    assert kronos["tokenizer"]["ACTIVATION"] == "silu"
    assert kronos["network"]["ACTIVATION"] == "silu"
    assert kronos_rich["tokenizer"]["ACTIVATION"] == "silu"
    assert kronos_rich["network"]["ACTIVATION"] == "silu"
    assert kronos_rich["network"]["USE_STOCK_EMBEDDING"] is True
    assert kronos_rich["network"]["USE_GROUP_EMBEDDING"] is True
    assert kronos_rich["RECON_LOSS_TYPE"] == "mse"
    assert kronos_rich["PRE_LOSS_TYPE"] == "mse"
    assert kronos_rich["TOKEN_LOSS_TYPE"] == "cross_entropy"


def test_model_config_includes_chronos_rich_component_loss_settings():
    data = _load_raw_config("model")
    chronos_rich = data["model"]["models"]["chronos_rich"]

    assert chronos_rich["ACTIVATION"] == "relu"
    assert chronos_rich["SCALAR_LOSS_TYPE"] == "quantile_loss"
    assert chronos_rich["OHLCV_LOSS_TYPE"] == "quantile_loss"
    assert chronos_rich["RETURN_PATH_LOSS_TYPE"] == "quantile_loss"
    assert chronos_rich["REGIME_LOSS_TYPE"] == "cross_entropy"
