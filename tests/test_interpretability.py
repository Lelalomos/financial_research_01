"""
Unit tests for attention interpretability utilities.
"""

import numpy as np
import pytest
import torch

from src.config import load_config
from src.evaluation.interpretability import (
    aggregate_attention_by_position,
    attention_to_numpy,
    get_attention_report,
    summarize_attention,
)
from src.models import create_model


def _sample_inputs(batch_size=2, seq_len=12, num_features=6):
    return {
        "features": torch.randn(batch_size, seq_len, num_features),
        "stock_id": torch.randint(0, 5, (batch_size, seq_len)),
        "group_id": torch.randint(0, 3, (batch_size, seq_len)),
        "day": torch.randint(1, 32, (batch_size, seq_len)),
        "month": torch.randint(1, 13, (batch_size, seq_len)),
        "dividend_flag": torch.randint(1, 3, (batch_size, seq_len)),
    }


@pytest.mark.parametrize(
    "model_type",
    ["rnn_attention", "lstm3_attention", "bilstm4_attention", "crnn_attention"],
)
def test_forward_with_attention_shapes(model_type):
    config = load_config("model")
    num_features = 6
    model = create_model(
        model_type=model_type,
        num_features=num_features,
        num_stocks=5,
        num_groups=3,
        config=config,
    )
    model.eval()
    inputs = _sample_inputs(num_features=num_features)

    with torch.no_grad():
        regular_output = model(**inputs)
        output, attention_weights = model.forward_with_attention(**inputs)

    assert regular_output.shape == (2, 1)
    assert output.shape == (2, 1)
    assert attention_weights.ndim == 4
    assert attention_weights.shape[0] == 2
    assert attention_weights.shape[2] == attention_weights.shape[3]


def test_attention_to_numpy_accepts_three_dimensional_weights():
    weights = np.ones((2, 4, 4), dtype=np.float32)

    result = attention_to_numpy(weights)

    assert result.shape == (2, 1, 4, 4)


def test_aggregate_attention_by_position_returns_expected_columns():
    weights = np.zeros((2, 2, 3, 3), dtype=np.float32)
    weights[:, :, :, 1] = 1.0

    summary = aggregate_attention_by_position(weights, positions=["old", "mid", "new"])

    assert summary.columns.tolist() == ["position", "attention_score"]
    assert summary.loc[summary["attention_score"].idxmax(), "position"] == "mid"


def test_aggregate_attention_by_head():
    weights = np.ones((1, 2, 3, 3), dtype=np.float32)

    summary = aggregate_attention_by_position(weights, average_heads=False)

    assert summary.columns.tolist() == ["head", "position", "attention_score"]
    assert len(summary) == 6


def test_summarize_attention_top_k():
    weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
    weights[:, :, :, 2] = 0.8
    weights[:, :, :, 0] = 0.2

    summary = summarize_attention(weights, top_k=1)

    assert len(summary) == 1
    assert summary.iloc[0]["position"] == 2


def test_get_attention_report_rejects_non_attention_model():
    model = create_model(
        model_type="rnn",
        num_features=6,
        num_stocks=5,
        num_groups=3,
        config=load_config("model"),
    )

    with pytest.raises(ValueError, match="forward_with_attention"):
        get_attention_report(model, _sample_inputs())


def test_get_attention_report_returns_summary():
    model = create_model(
        model_type="lstm3_attention",
        num_features=6,
        num_stocks=5,
        num_groups=3,
        config=load_config("model"),
    )

    report = get_attention_report(model, _sample_inputs())

    assert report["predictions"].shape == (2, 1)
    assert report["attention_weights"].ndim == 4
    assert {"position", "attention_score"}.issubset(report["summary"].columns)
