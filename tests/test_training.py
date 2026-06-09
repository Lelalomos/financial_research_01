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

from src.config import load_config
from src.models import create_model
from src.training import Trainer, EarlyStopping, ModelCheckpoint, find_checkpoint_path
from src.data.dataset import FinancialDataset


class TestTraining(unittest.TestCase):
    """Test training components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = load_config('model')
        self.config.model.training.NUM_EPOCHS = 2
        self.config.model.training.BATCH_SIZE = 4
        self.config.model.training.EARLY_STOPPING_PATIENCE = 2
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

    def test_move_batch_to_device_moves_target_for_loss(self):
        """Batch tensors used by loss computation should be on the trainer device."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer.__new__(Trainer)
        trainer.device = device
        batch = {
            'features': torch.randn(2, 3, 4),
            'target': torch.randn(2),
            'ticker': ['AAPL', 'MSFT'],
        }

        moved = trainer._move_batch_to_device(batch)

        self.assertEqual(moved['features'].device.type, device)
        self.assertEqual(moved['target'].device.type, device)
        self.assertEqual(moved['ticker'], ['AAPL', 'MSFT'])

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

    def test_model_checkpoint_frequency_controls_periodic_saves(self):
        """Periodic checkpoints should honor CHECKPOINT_FREQUENCY semantics."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(
                save_dir=tmpdir,
                mode='min',
                save_best_only=False,
                save_last_n=2,
                checkpoint_frequency=2,
                verbose=False,
                model_type='tiny',
            )

            model = create_model(
                model_type='crnn_attention',
                num_features=20,
                num_stocks=10,
                num_groups=5,
                config=self.config
            )
            optimizer = torch.optim.Adam(model.parameters())

            checkpoint(model, optimizer, epoch=1, score=1.0, loss=1.5)
            checkpoint(model, optimizer, epoch=2, score=1.1, loss=1.6)
            checkpoint(model, optimizer, epoch=3, score=1.2, loss=1.7)
            checkpoint(model, optimizer, epoch=4, score=1.3, loss=1.8)

            periodic_path = Path(tmpdir) / 'tiny_latest_periodic.pth'
            self.assertTrue(periodic_path.exists())
            periodic = torch.load(periodic_path, map_location='cpu', weights_only=True)
            self.assertEqual(periodic['epoch'], 4)

    def test_find_checkpoint_path_prefers_compatible_checkpoint(self):
        """Newest incompatible checkpoints should not block a compatible one."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            incompatible = Path(tmpdir) / 'bilstm4_attention_best_20260511_161034.pth'
            compatible = Path(tmpdir) / 'bilstm4_attention_best_lightning.pth'

            torch.save({
                'model_type': 'bilstm4_attention',
                'model_state_dict': {
                    'embeddings.stock_embedding.weight': torch.zeros(3, 64),
                    'embeddings.group_embedding.weight': torch.zeros(2, 32),
                },
            }, incompatible)

            torch.save({
                'model_type': 'bilstm4_attention',
                'num_features': 90,
                'num_stocks': 150,
                'num_groups': 11,
                'model_state_dict': {
                    'embeddings.stock_embedding.weight': torch.zeros(150, 64),
                    'embeddings.group_embedding.weight': torch.zeros(11, 32),
                },
            }, compatible)

            os.utime(compatible, (1, 1))
            os.utime(incompatible, (2, 2))

            path = find_checkpoint_path(
                'best',
                checkpoint_dir=tmpdir,
                model_type='bilstm4_attention',
                num_features=90,
                num_stocks=150,
                num_groups=11,
            )

            self.assertEqual(path, str(compatible))


if __name__ == '__main__':
    unittest.main()
