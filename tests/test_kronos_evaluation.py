import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

from src.evaluation.kronos import (
    _infer_feature_inverse_transform,
    _inverse_feature_values,
    build_kronos_sequence_metadata,
    build_kronos_report,
    compute_kronos_backtest_results,
    is_kronos_family,
    load_kronos_checkpoint,
    resolve_kronos_embedding_sizes,
)
from src.models.kronos_model import create_kronos_model, create_kronos_rich_model, create_kronos_tokenizer
from src.models.kronos_module import RMSNorm
from src.config.config_loader import Config


def test_build_kronos_sequence_metadata_aligns_windows(tmp_path):
    processed_dir = tmp_path / "processed"
    cache_dir = processed_dir / ".cache" / "normalized_splits"
    cache_dir.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    raw_close = np.array([100.0, 101.0, 102.0, 104.0, 103.0, 105.0], dtype=float)
    target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float)

    raw_df = pd.DataFrame(
        {
            "date": dates,
            "tic": ["AAA"] * len(dates),
            "tic_id": [0] * len(dates),
            "group": ["Tech"] * len(dates),
            "group_id": [1] * len(dates),
            "close": raw_close,
            "volume": np.arange(len(dates)) + 1,
            "day": dates.day,
            "month": dates.month,
            "target": target,
        }
    )
    split_df = raw_df.copy()
    split_df["close"] = (split_df["close"] - split_df["close"].mean()) / split_df["close"].std()

    raw_df.to_parquet(tmp_path / "pre_normalized.parquet", index=False)
    split_df.to_parquet(cache_dir / "test.parquet", index=False)

    metadata = build_kronos_sequence_metadata(
        data_dir=processed_dir,
        split="test",
        feature_cols=["close", "volume"],
        sequence_length=2,
        prediction_horizon=2,
        normalize_target=False,
        target_threshold=2.0,
    )

    assert metadata["x_dates"].shape == (3, 2)
    assert metadata["y_dates"].shape == (3, 2)
    assert np.isclose(metadata["last_close"][0], 101.0)
    assert np.isclose(metadata["future_close"][0], 104.0)
    expected_return = ((104.0 - 101.0) / 101.0) * 100.0
    assert np.isclose(metadata["targets"][0], expected_return)
    assert np.isclose(metadata["raw_targets"][0], expected_return)


def test_infer_feature_inverse_transform_recovers_affine_close_scale(tmp_path):
    processed_dir = tmp_path / "processed"
    cache_dir = processed_dir / ".cache" / "normalized_splits"
    cache_dir.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    normalized_close = np.array([-1.0, 0.0, 0.5, 2.0], dtype=float)
    raw_close = 20.0 + 5.0 * normalized_close

    pd.DataFrame(
        {
            "date": dates,
            "tic_id": [0] * len(dates),
            "close": normalized_close,
        }
    ).to_parquet(cache_dir / "train.parquet", index=False)

    pd.DataFrame(
        {
            "date": dates,
            "tic_id": [0] * len(dates),
            "close": raw_close,
        }
    ).to_parquet(tmp_path / "pre_normalized.parquet", index=False)

    transform = _infer_feature_inverse_transform(str(processed_dir), "close")
    restored = _inverse_feature_values(np.array([1.5, -0.5], dtype=np.float32), transform)

    assert transform["kind"] == "affine"
    assert np.allclose(restored, np.array([27.5, 17.5], dtype=np.float32))


def test_build_kronos_report_includes_percent_columns():
    predictions = np.array([0.5, -0.25], dtype=np.float32)
    targets = np.array([0.4, -0.1], dtype=np.float32)
    raw_predictions = np.array([1.1, -0.7], dtype=np.float32)
    raw_targets = np.array([0.8, -0.3], dtype=np.float32)
    stock_ids = np.array([0, 1], dtype=np.int64)
    group_ids = np.array([3, 4], dtype=np.int64)

    report_df, sector_stats = build_kronos_report(
        predictions=predictions,
        targets=targets,
        stock_ids=stock_ids,
        group_ids=group_ids,
        raw_predictions=raw_predictions,
        raw_targets=raw_targets,
        stock_id_to_ticker={0: "AAA", 1: "BBB"},
        group_id_to_sector={3: "Tech", 4: "Health"},
    )

    assert list(report_df["ticker"]) == ["AAA", "BBB"]
    assert list(report_df["sector"]) == ["Tech", "Health"]
    assert np.allclose(report_df["predict_target_percent"], raw_predictions)
    assert np.allclose(report_df["real_target_percent"], raw_targets)
    assert np.allclose(report_df["distance_percent"], raw_predictions - raw_targets)
    assert sector_stats["Tech"]["accuracy"] == 1.0


def test_resolve_kronos_embedding_sizes_prefers_full_metadata():
    num_stocks, num_groups = resolve_kronos_embedding_sizes(
        info={"num_stocks": 150, "num_groups": 11},
        fallback_sizes={"num_stocks": 1, "num_groups": 7},
    )

    assert num_stocks == 150
    assert num_groups == 11


def test_compute_kronos_backtest_results_returns_expected_shapes():
    predictions = np.array([0.5, -0.25, 0.1], dtype=np.float32)
    targets = np.array([0.4, -0.1, -0.2], dtype=np.float32)
    stock_ids = np.array([0, 1, 0], dtype=np.int64)
    group_ids = np.array([3, 3, 4], dtype=np.int64)

    results = compute_kronos_backtest_results(
        predictions=predictions,
        targets=targets,
        stock_ids=stock_ids,
        group_ids=group_ids,
        prediction_threshold=0.0,
        initial_capital=1000.0,
    )

    assert results["num_trades"] == 3
    assert results["predictions"].shape == (3,)
    assert results["targets"].shape == (3,)
    assert "sector_stats" in results
    assert results["final_capital"] > 0


@pytest.mark.parametrize("model_type", ["kronos", "kronos_rich"])
def test_is_kronos_family_accepts_both_model_types(model_type):
    assert is_kronos_family(model_type) is True


def test_is_kronos_family_rejects_non_kronos_model():
    assert is_kronos_family("chronos2") is False


def test_load_kronos_checkpoint_uses_requested_model_type(tmp_path, monkeypatch):
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {"D_IN": 1},
                        "network": {},
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    },
                    "kronos_rich": {
                        "tokenizer": {"D_IN": 1},
                        "network": {},
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    },
                }
            }
        }
    )

    class DummyModule:
        def __init__(self, label):
            self.label = label
            self.loaded_state = None
            self.eval_called = False

        def to(self, _device):
            return self

        def load_state_dict(self, state):
            self.loaded_state = state

        def eval(self):
            self.eval_called = True
            return self

    created = {}

    def fake_tokenizer_factory(config):
        created["tokenizer"] = DummyModule("tokenizer")
        return created["tokenizer"]

    def fake_model_factory(config, num_stocks=None, num_groups=None):
        created["model"] = DummyModule("model")
        created["num_stocks"] = num_stocks
        created["num_groups"] = num_groups
        return created["model"]

    monkeypatch.setattr("src.evaluation.kronos._KRONOS_CREATORS", {
        "kronos_rich": (fake_tokenizer_factory, fake_model_factory),
    })
    monkeypatch.setattr(
        "torch.load",
        lambda *args, **kwargs: {
            "tokenizer_state_dict": {"tok": 1},
            "model_state_dict": {"mdl": 2},
            "epoch": 3,
        },
    )

    tokenizer, model, checkpoint = load_kronos_checkpoint(
        checkpoint_path=str(tmp_path / "fake.pth"),
        config=model_config,
        num_features=9,
        num_stocks=4,
        num_groups=2,
        device="cpu",
        model_type="kronos_rich",
    )

    assert model_config.model.models.kronos_rich.tokenizer.D_IN == 9
    assert created["num_stocks"] == 4
    assert created["num_groups"] == 2
    assert tokenizer.loaded_state == {"tok": 1}
    assert model.loaded_state == {"mdl": 2}
    assert checkpoint["epoch"] == 3


def test_create_kronos_rich_model_uses_upstream_style_without_repo_embeddings():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos_rich": {
                        "tokenizer": {"D_IN": 6},
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": True,
                            "USE_GROUP_EMBEDDING": True,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    model = create_kronos_rich_model(config=model_config, num_stocks=10, num_groups=3)

    assert model.use_stock_embedding is False
    assert model.use_group_embedding is False
    assert model.stock_embedding is None
    assert model.group_embedding is None


def test_create_kronos_model_keeps_repo_embeddings_when_enabled():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {"D_IN": 6, "ACTIVATION": "silu"},
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": True,
                            "USE_GROUP_EMBEDDING": True,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    model = create_kronos_model(config=model_config, num_stocks=10, num_groups=3, model_key="kronos")

    assert model.use_stock_embedding is True
    assert model.use_group_embedding is True
    assert model.stock_embedding is not None
    assert model.group_embedding is not None


def test_create_kronos_uses_configured_activations():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {
                            "D_IN": 6,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "N_ENC_LAYERS": 2,
                            "N_DEC_LAYERS": 2,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "gelu",
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "BETA": 0.25,
                            "GAMMA0": 1.0,
                            "GAMMA": 1.0,
                            "ZETA": 1.0,
                            "GROUP_SIZE": 2,
                        },
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "relu",
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": False,
                            "USE_GROUP_EMBEDDING": False,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    tokenizer = create_kronos_tokenizer(config=model_config, model_key="kronos")
    model = create_kronos_model(config=model_config, num_stocks=10, num_groups=3, model_key="kronos")

    assert isinstance(tokenizer.encoder[0].ffn.activation, nn.GELU)
    assert isinstance(model.transformer[0].ffn.activation, nn.ReLU)


def test_create_kronos_uses_configured_geglu_activation():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {
                            "D_IN": 6,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "N_ENC_LAYERS": 2,
                            "N_DEC_LAYERS": 2,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "geglu",
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "BETA": 0.25,
                            "GAMMA0": 1.0,
                            "GAMMA": 1.0,
                            "ZETA": 1.0,
                            "GROUP_SIZE": 2,
                        },
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "swiglu",
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": False,
                            "USE_GROUP_EMBEDDING": False,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    tokenizer = create_kronos_tokenizer(config=model_config, model_key="kronos")
    model = create_kronos_model(config=model_config, num_stocks=10, num_groups=3, model_key="kronos")

    assert tokenizer.encoder[0].ffn.is_gated is True
    assert isinstance(tokenizer.encoder[0].ffn.activation, nn.GELU)
    assert isinstance(model.transformer[0].ffn.activation, nn.SiLU)


def test_create_kronos_uses_configured_layernorm():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {
                            "D_IN": 6,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "N_ENC_LAYERS": 2,
                            "N_DEC_LAYERS": 2,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "NORM_TYPE": "layernorm",
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "BETA": 0.25,
                            "GAMMA0": 1.0,
                            "GAMMA": 1.0,
                            "ZETA": 1.0,
                            "GROUP_SIZE": 2,
                        },
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "NORM_TYPE": "layernorm",
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": False,
                            "USE_GROUP_EMBEDDING": False,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    tokenizer = create_kronos_tokenizer(config=model_config, model_key="kronos")
    model = create_kronos_model(config=model_config, num_stocks=10, num_groups=3, model_key="kronos")

    assert isinstance(tokenizer.encoder[0].norm1, nn.LayerNorm)
    assert isinstance(tokenizer.encoder[0].norm2, nn.LayerNorm)
    assert isinstance(model.transformer[0].norm1, nn.LayerNorm)
    assert isinstance(model.transformer[0].norm2, nn.LayerNorm)
    assert isinstance(model.dep_layer.norm, nn.LayerNorm)
    assert isinstance(model.norm, nn.LayerNorm)
    assert not isinstance(model.norm, RMSNorm)


def test_create_kronos_uses_configured_bias_flag():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {
                            "D_IN": 6,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "N_ENC_LAYERS": 2,
                            "N_DEC_LAYERS": 2,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "NORM_TYPE": "rmsnorm",
                            "USE_BIAS": False,
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "BETA": 0.25,
                            "GAMMA0": 1.0,
                            "GAMMA": 1.0,
                            "ZETA": 1.0,
                            "GROUP_SIZE": 2,
                        },
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "NORM_TYPE": "rmsnorm",
                            "USE_BIAS": False,
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": True,
                            "USE_GROUP_EMBEDDING": True,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    tokenizer = create_kronos_tokenizer(config=model_config, model_key="kronos")
    model = create_kronos_model(config=model_config, num_stocks=10, num_groups=3, model_key="kronos")

    assert tokenizer.embed.bias is None
    assert tokenizer.quant_embed.bias is None
    assert tokenizer.encoder[0].ffn.w1.bias is None
    assert tokenizer.encoder[0].self_attn.q_proj.bias is None
    assert model.transformer[0].ffn.w1.bias is None
    assert model.transformer[0].self_attn.q_proj.bias is None
    assert model.head.proj_s1.bias is None
    assert model.stock_projection.bias is None
    assert model.group_projection.bias is None


def test_create_kronos_rejects_unknown_activation():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {
                            "D_IN": 6,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "N_ENC_LAYERS": 2,
                            "N_DEC_LAYERS": 2,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "bad_activation",
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "BETA": 0.25,
                            "GAMMA0": 1.0,
                            "GAMMA": 1.0,
                            "ZETA": 1.0,
                            "GROUP_SIZE": 2,
                        },
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": False,
                            "USE_GROUP_EMBEDDING": False,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported Kronos activation"):
        create_kronos_tokenizer(config=model_config, model_key="kronos")


def test_create_kronos_rejects_unknown_norm_type():
    model_config = Config(
        {
            "model": {
                "models": {
                    "kronos": {
                        "tokenizer": {
                            "D_IN": 6,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "N_ENC_LAYERS": 2,
                            "N_DEC_LAYERS": 2,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "NORM_TYPE": "bad_norm",
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "BETA": 0.25,
                            "GAMMA0": 1.0,
                            "GAMMA": 1.0,
                            "ZETA": 1.0,
                            "GROUP_SIZE": 2,
                        },
                        "network": {
                            "S1_BITS": 2,
                            "S2_BITS": 2,
                            "N_LAYERS": 1,
                            "D_MODEL": 16,
                            "N_HEADS": 4,
                            "FF_DIM": 32,
                            "FFN_DROPOUT_P": 0.0,
                            "ATTN_DROPOUT_P": 0.0,
                            "RESID_DROPOUT_P": 0.0,
                            "ACTIVATION": "silu",
                            "NORM_TYPE": "rmsnorm",
                            "TOKEN_DROPOUT_P": 0.0,
                            "LEARN_TE": True,
                            "USE_STOCK_EMBEDDING": False,
                            "USE_GROUP_EMBEDDING": False,
                            "STOCK_EMB_DIM": 8,
                            "GROUP_EMB_DIM": 4,
                        },
                        "predictor": {"MAX_CONTEXT": 16, "CLIP": 5.0},
                    }
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported Kronos norm type"):
        create_kronos_tokenizer(config=model_config, model_key="kronos")
