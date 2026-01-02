"""
Unit tests for LSTM3Model and LSTM3AttentionModel.
"""

import pytest
import torch

from config.model_config import ModelConfig
from src.models.lstm3_model import LSTM3Model, create_model as create_lstm3
from src.models.lstm3_attn_model import LSTM3AttentionModel, create_model as create_lstm3_attention


class TestLSTM3Model:
    """Test cases for LSTM3Model."""

    def test_model_initialization(self):
        """Test that LSTM3Model can be initialized."""
        config = ModelConfig(MODEL_TYPE='lstm3')
        model = LSTM3Model(
            num_features=10,
            num_stocks=100,
            num_groups=10,
            config=config
        )
        assert model is not None
        assert isinstance(model, LSTM3Model)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = ModelConfig(MODEL_TYPE='lstm3')
        model = LSTM3Model(
            num_features=10,
            num_stocks=100,
            num_groups=10,
            config=config
        )

        batch_size, seq_len = 4, 60
        features = torch.randn(batch_size, seq_len, 10)
        stock_id = torch.randint(0, 100, (batch_size, seq_len))
        group_id = torch.randint(0, 10, (batch_size, seq_len))
        day = torch.randint(1, 32, (batch_size, seq_len))
        month = torch.randint(1, 13, (batch_size, seq_len))

        output = model(features, stock_id, group_id, day, month)

        assert output.shape == (batch_size, 1)

    def test_create_model_factory(self):
        """Test that create_model factory function works."""
        model = create_lstm3(num_features=10, num_stocks=100, num_groups=10)
        assert isinstance(model, LSTM3Model)

    def test_lstm3_num_layers(self):
        """Test that LSTM3Model uses 3 layers as configured."""
        config = ModelConfig(MODEL_TYPE='lstm3')
        assert config.LSTM3_NUM_LAYERS == 3

        model = LSTM3Model(
            num_features=10,
            num_stocks=100,
            num_groups=10,
            config=config
        )

        # Check that the LSTM block has the correct number of layers
        assert model.lstm.lstm.num_layers == 3


class TestLSTM3AttentionModel:
    """Test cases for LSTM3AttentionModel."""

    def test_model_initialization(self):
        """Test that LSTM3AttentionModel can be initialized."""
        config = ModelConfig(MODEL_TYPE='lstm3_attention')
        model = LSTM3AttentionModel(
            num_features=10,
            num_stocks=100,
            num_groups=10,
            config=config
        )
        assert model is not None
        assert isinstance(model, LSTM3AttentionModel)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = ModelConfig(MODEL_TYPE='lstm3_attention')
        model = LSTM3AttentionModel(
            num_features=10,
            num_stocks=100,
            num_groups=10,
            config=config
        )

        batch_size, seq_len = 4, 60
        features = torch.randn(batch_size, seq_len, 10)
        stock_id = torch.randint(0, 100, (batch_size, seq_len))
        group_id = torch.randint(0, 10, (batch_size, seq_len))
        day = torch.randint(1, 32, (batch_size, seq_len))
        month = torch.randint(1, 13, (batch_size, seq_len))

        output = model(features, stock_id, group_id, day, month)

        assert output.shape == (batch_size, 1)

    def test_create_model_factory(self):
        """Test that create_model factory function works."""
        model = create_lstm3_attention(num_features=10, num_stocks=100, num_groups=10)
        assert isinstance(model, LSTM3AttentionModel)

    def test_lstm3_attention_num_layers(self):
        """Test that LSTM3AttentionModel uses 3 layers as configured."""
        config = ModelConfig(MODEL_TYPE='lstm3_attention')
        assert config.LSTM3_NUM_LAYERS == 3

        model = LSTM3AttentionModel(
            num_features=10,
            num_stocks=100,
            num_groups=10,
            config=config
        )

        # Check that the LSTM block has the correct number of layers
        assert model.lstm.lstm.num_layers == 3

    def test_attention_mechanism_exists(self):
        """Test that attention mechanism is properly initialized."""
        config = ModelConfig(MODEL_TYPE='lstm3_attention')
        model = LSTM3AttentionModel(
            num_features=10,
            num_stocks=100,
            num_groups=10,
            config=config
        )

        assert hasattr(model, 'attention')
        assert isinstance(model.attention, torch.nn.MultiheadAttention)
        assert model.attention.num_heads == config.LSTM3_ATTENTION_HEADS


class TestModelRegistry:
    """Test cases for model registry integration."""

    def test_lstm3_in_registry(self):
        """Test that lstm3 is in the model registry."""
        from src.models import _MODEL_REGISTRY, list_available_models

        assert 'lstm3' in _MODEL_REGISTRY
        assert 'lstm3' in list_available_models()

    def test_lstm3_attention_in_registry(self):
        """Test that lstm3_attention is in the model registry."""
        from src.models import _MODEL_REGISTRY, list_available_models

        assert 'lstm3_attention' in _MODEL_REGISTRY
        assert 'lstm3_attention' in list_available_models()

    def test_create_model_via_registry(self):
        """Test creating models through the registry."""
        from src.models import create_model

        # Test lstm3
        model1 = create_model('lstm3', num_features=10, num_stocks=100, num_groups=10)
        assert isinstance(model1, LSTM3Model)

        # Test lstm3_attention
        model2 = create_model('lstm3_attention', num_features=10, num_stocks=100, num_groups=10)
        assert isinstance(model2, LSTM3AttentionModel)
