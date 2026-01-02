"""
Unit tests for all model components.

Tests all 7 model variants:
- CRNNModel
- RNNModel
- RNNAttentionModel
- CRNNAttentionModel
- TransformerModel
- LSTM3Model
- LSTM3AttentionModel
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from src.models import (
    create_model,
    list_available_models,
    CRNNModel,
    RNNModel,
    RNNAttentionModel,
    CRNNAttentionModel,
    TransformerModel,
    LSTM3Model,
    LSTM3AttentionModel,
)


# All model types in the registry
ALL_MODEL_TYPES = [
    'crnn',
    'rnn',
    'rnn_attention',
    'crnn_attention',
    'transformer',
    'lstm3',
    'lstm3_attention'
]


@pytest.fixture
def model_config():
    """Create a default model config."""
    return ModelConfig()


@pytest.fixture
def sample_inputs():
    """Create sample input data for testing."""
    config = ModelConfig()
    num_features = 50
    num_stocks = 100
    num_groups = 10
    batch_size = 8
    seq_len = 30

    return {
        'features': torch.randn(batch_size, seq_len, num_features),
        'stock_id': torch.randint(0, num_stocks, (batch_size, seq_len)),
        'group_id': torch.randint(0, num_groups, (batch_size, seq_len)),
        'day': torch.randint(1, 32, (batch_size, seq_len)),
        'month': torch.randint(1, 13, (batch_size, seq_len)),
        'batch_size': batch_size,
        'num_features': num_features,
        'num_stocks': num_stocks,
        'num_groups': num_groups,
    }


class TestCRNNModel:
    """Test CRNN model."""

    def test_initialization(self, sample_inputs, model_config):
        """Test that CRNNModel can be initialized."""
        model = CRNNModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )
        assert model is not None
        assert isinstance(model, CRNNModel)

    def test_forward_pass(self, sample_inputs, model_config):
        """Test forward pass produces correct output shape."""
        model = CRNNModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)

    def test_gradient_flow(self, sample_inputs, model_config):
        """Test gradient flows through model."""
        model = CRNNModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestRNNModel:
    """Test RNN model."""

    def test_initialization(self, sample_inputs, model_config):
        """Test that RNNModel can be initialized."""
        model = RNNModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )
        assert model is not None
        assert isinstance(model, RNNModel)

    def test_forward_pass(self, sample_inputs, model_config):
        """Test forward pass produces correct output shape."""
        model = RNNModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)


class TestRNNAttentionModel:
    """Test RNN + Attention model."""

    def test_initialization(self, sample_inputs, model_config):
        """Test that RNNAttentionModel can be initialized."""
        model = RNNAttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )
        assert model is not None
        assert isinstance(model, RNNAttentionModel)

    def test_forward_pass(self, sample_inputs, model_config):
        """Test forward pass produces correct output shape."""
        model = RNNAttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)


class TestCRNNAttentionModel:
    """Test CRNN + Attention model."""

    def test_initialization(self, sample_inputs, model_config):
        """Test that CRNNAttentionModel can be initialized."""
        model = CRNNAttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )
        assert model is not None
        assert isinstance(model, CRNNAttentionModel)

    def test_forward_pass(self, sample_inputs, model_config):
        """Test forward pass produces correct output shape."""
        model = CRNNAttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)


class TestTransformerModel:
    """Test Transformer model."""

    def test_initialization(self, sample_inputs, model_config):
        """Test that TransformerModel can be initialized."""
        model = TransformerModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )
        assert model is not None
        assert isinstance(model, TransformerModel)

    def test_forward_pass(self, sample_inputs, model_config):
        """Test forward pass produces correct output shape."""
        model = TransformerModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)


class TestLSTM3Model:
    """Test LSTM3 model."""

    def test_initialization(self, sample_inputs, model_config):
        """Test that LSTM3Model can be initialized."""
        model = LSTM3Model(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )
        assert model is not None
        assert isinstance(model, LSTM3Model)

    def test_forward_pass(self, sample_inputs, model_config):
        """Test forward pass produces correct output shape."""
        model = LSTM3Model(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)

    def test_lstm3_num_layers(self, sample_inputs, model_config):
        """Test that LSTM3Model uses 3 layers as configured."""
        assert model_config.LSTM3_NUM_LAYERS == 3

        model = LSTM3Model(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        assert model.lstm.lstm.num_layers == 3


class TestLSTM3AttentionModel:
    """Test LSTM3 + Attention model."""

    def test_initialization(self, sample_inputs, model_config):
        """Test that LSTM3AttentionModel can be initialized."""
        model = LSTM3AttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )
        assert model is not None
        assert isinstance(model, LSTM3AttentionModel)

    def test_forward_pass(self, sample_inputs, model_config):
        """Test forward pass produces correct output shape."""
        model = LSTM3AttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)

    def test_lstm3_attention_num_layers(self, sample_inputs, model_config):
        """Test that LSTM3AttentionModel uses 3 layers as configured."""
        assert model_config.LSTM3_NUM_LAYERS == 3

        model = LSTM3AttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        assert model.lstm.lstm.num_layers == 3

    def test_attention_mechanism_exists(self, sample_inputs, model_config):
        """Test that attention mechanism is properly initialized."""
        model = LSTM3AttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        assert hasattr(model, 'attention')
        assert isinstance(model.attention, torch.nn.MultiheadAttention)
        assert model.attention.num_heads == model_config.LSTM3_ATTENTION_HEADS


class TestModelRegistry:
    """Test model registry functionality."""

    def test_list_available_models(self):
        """Test that all 7 models are listed."""
        models = list_available_models()
        assert len(models) == 7
        expected = {'crnn', 'rnn', 'rnn_attention', 'crnn_attention',
                   'transformer', 'lstm3', 'lstm3_attention'}
        assert set(models) == expected

    @pytest.mark.parametrize("model_type", ALL_MODEL_TYPES)
    def test_all_models_in_registry(self, model_type):
        """Test that all model types are in the registry."""
        from src.models import _MODEL_REGISTRY
        assert model_type in _MODEL_REGISTRY

    @pytest.mark.parametrize("model_type", ALL_MODEL_TYPES)
    def test_create_model_for_all_types(self, model_type, sample_inputs):
        """Test creating models through the registry."""
        model = create_model(
            model_type=model_type,
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups']
        )

        # Test forward pass
        output = model(
            sample_inputs['features'],
            sample_inputs['stock_id'],
            sample_inputs['group_id'],
            sample_inputs['day'],
            sample_inputs['month']
        )

        assert output.shape == (sample_inputs['batch_size'], 1)

    @pytest.mark.parametrize("model_type,expected_class", [
        ('crnn', CRNNModel),
        ('rnn', RNNModel),
        ('rnn_attention', RNNAttentionModel),
        ('crnn_attention', CRNNAttentionModel),
        ('transformer', TransformerModel),
        ('lstm3', LSTM3Model),
        ('lstm3_attention', LSTM3AttentionModel),
    ])
    def test_create_model_returns_correct_class(self, model_type, expected_class, sample_inputs):
        """Test that create_model returns the correct class."""
        model = create_model(
            model_type=model_type,
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups']
        )

        assert isinstance(model, expected_class)


class TestModelParameters:
    """Test model parameter-related functionality."""

    def test_model_has_parameters(self, sample_inputs, model_config):
        """Test model has parameters."""
        model = CRNNAttentionModel(
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups'],
            config=model_config
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert total_params == trainable_params

    @pytest.mark.parametrize("model_type", ALL_MODEL_TYPES)
    def test_all_models_have_parameters(self, model_type, sample_inputs):
        """Test that all models have parameters."""
        model = create_model(
            model_type=model_type,
            num_features=sample_inputs['num_features'],
            num_stocks=sample_inputs['num_stocks'],
            num_groups=sample_inputs['num_groups']
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0, f"{model_type} has no parameters"
        assert total_params == trainable_params, f"{model_type} has frozen parameters"
