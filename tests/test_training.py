"""
Unit tests for training components.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from src.models import create_model
from src.training import Trainer, EarlyStopping, ModelCheckpoint
from src.data.dataset import FinancialDataset


class TestTraining(unittest.TestCase):
    """Test training components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(NUM_EPOCHS=2, BATCH_SIZE=4, EARLY_STOPPING_PATIENCE=2)
        self.device = 'cpu'

        # Create sample data
        self.train_sequences = {
            'features': np.random.randn(100, 30, 20),
            'stock_id': np.random.randint(0, 10, (100, 30)),
            'group_id': np.random.randint(0, 5, (100, 30)),
            'day': np.random.randint(1, 32, (100, 30)),
            'month': np.random.randint(1, 13, (100, 30)),
            'target': np.random.randn(100)
        }

        self.val_sequences = {
            'features': np.random.randn(50, 30, 20),
            'stock_id': np.random.randint(0, 10, (50, 30)),
            'group_id': np.random.randint(0, 5, (50, 30)),
            'day': np.random.randint(1, 32, (50, 30)),
            'month': np.random.randint(1, 13, (50, 30)),
            'target': np.random.randn(50)
        }

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = create_model(
            model_type='crnn_attention',
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=self.config
        )

        trainer = Trainer(model, self.config, device=self.device)

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)

    def test_train_epoch(self):
        """Test training epoch."""
        model = create_model(
            model_type='crnn_attention',
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=self.config
        )

        trainer = Trainer(model, self.config, device=self.device)

        # Create data loaders
        train_dataset = FinancialDataset(self.train_sequences, self.config)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Train one epoch
        metrics = trainer.train_epoch(train_loader)

        self.assertIn('loss', metrics)
        self.assertGreater(metrics['loss'], 0)

    def test_validate(self):
        """Test validation."""
        model = create_model(
            model_type='crnn_attention',
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=self.config
        )

        trainer = Trainer(model, self.config, device=self.device)

        # Create data loader
        val_dataset = FinancialDataset(self.val_sequences, self.config)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # Validate
        metrics = trainer.validate(val_loader)

        self.assertIn('loss', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)

    def test_early_stopping(self):
        """Test early stopping."""
        early_stop = EarlyStopping(patience=2, mode='min')

        # First call - should not stop
        should_stop = early_stop(1.0, epoch=1)
        self.assertFalse(should_stop)

        # Second call - worse, should not stop
        should_stop = early_stop(1.1, epoch=2)
        self.assertFalse(should_stop)
        self.assertEqual(early_stop.counter, 1)

        # Third call - worse again, should stop
        should_stop = early_stop(1.2, epoch=3)
        self.assertTrue(should_stop)

    def test_model_checkpoint(self):
        """Test model checkpointing."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(save_dir=tmpdir, mode='min')

            model = create_model(
                model_type='crnn_attention',
                num_features=20,
                num_stocks=10,
                num_groups=5,
                config=self.config
            )

            optimizer = torch.optim.Adam(model.parameters())

            # Save checkpoint
            checkpoint(model, optimizer, epoch=1, score=1.0, loss=1.5)

            # Check file exists
            self.assertTrue(checkpoint.has_checkpoint)

            # Load checkpoint
            loaded = checkpoint.load_best(model, device='cpu')
            self.assertIn('epoch', loaded)
            self.assertEqual(loaded['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
