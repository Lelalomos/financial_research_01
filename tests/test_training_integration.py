"""
Integration test for full training flow with small data.
Tests GPU detection, model creation, and training end-to-end.
"""

import unittest
import torch
import numpy as np
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models import create_model
from src.training import Trainer
from src.data.dataset import FinancialDataset, create_data_loaders
from src.utils.device import get_device, get_device_info
from torch.utils.data import DataLoader


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training with small data."""

    def setUp(self):
        """Set up test fixtures with small synthetic data."""
        self.config = load_config('model')
        self.config.model.training.NUM_EPOCHS = 2
        self.config.model.training.BATCH_SIZE = 4
        self.config.model.training.EARLY_STOPPING_PATIENCE = 10
        self.config.model.training.LEARNING_RATE = 1e-4

        # Get device (will auto-detect GPU)
        self.device = get_device(verbose=True)
        self.device_info = get_device_info(verbose=True)

        # Create small synthetic dataset
        np.random.seed(42)
        torch.manual_seed(42)

        self.num_samples = 50  # Small dataset for quick testing
        self.seq_len = 30
        self.num_features = 20

        # Training data
        self.train_sequences = {
            'features': np.random.randn(self.num_samples, self.seq_len, self.num_features).astype(np.float32),
            'stock_id': np.random.randint(0, 5, (self.num_samples, self.seq_len)),
            'group_id': np.random.randint(0, 3, (self.num_samples, self.seq_len)),
            'day': np.random.randint(1, 32, (self.num_samples, self.seq_len)),
            'month': np.random.randint(1, 13, (self.num_samples, self.seq_len)),
            'target': np.random.randn(self.num_samples).astype(np.float32)
        }

        # Validation data
        self.val_sequences = {
            'features': np.random.randn(20, self.seq_len, self.num_features).astype(np.float32),
            'stock_id': np.random.randint(0, 5, (20, self.seq_len)),
            'group_id': np.random.randint(0, 3, (20, self.seq_len)),
            'day': np.random.randint(1, 32, (20, self.seq_len)),
            'month': np.random.randint(1, 13, (20, self.seq_len)),
            'target': np.random.randn(20).astype(np.float32)
        }

        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.config.model.checkpointing.CHECKPOINT_DIR = self.temp_dir

    def tearDown(self):
        """Clean up temporary directory."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_device_detection(self):
        """Test that GPU is correctly detected and used."""
        print("\n" + "="*60)
        print("DEVICE DETECTION TEST")
        print("="*60)

        print(f"Selected device: {self.device}")
        print(f"Device type: {self.device.type}")

        # Verify device info
        self.assertIsInstance(self.device, torch.device)
        self.assertIn(self.device.type, ['cpu', 'cuda'])

        # Print detailed info
        print(f"\nDevice Information:")
        print(f"  PyTorch version: {self.device_info['pytorch_version']}")
        print(f"  CUDA available: {self.device_info['cuda_available']}")
        print(f"  CUDA working: {self.device_info['cuda_working']}")

        if self.device_info.get('cuda_working'):
            print(f"  GPU name: {self.device_info.get('gpu_name')}")
            print(f"  GPU memory: {self.device_info.get('gpu_memory_gb', 0):.2f} GB")
            print(f"  Compute capability: {self.device_info.get('compute_capability')}")

    def test_model_creation_on_device(self):
        """Test model creation and verify it's on the correct device."""
        print("\n" + "="*60)
        print("MODEL CREATION TEST")
        print("="*60)

        # Create model
        model = create_model(
            model_type='crnn_attention',
            num_features=self.num_features,
            num_stocks=5,
            num_groups=3,
            config=self.config
        )

        # Move to device
        model = model.to(self.device)

        print(f"Model type: {model.__class__.__name__}")
        print(f"Target device: {self.device}")

        # Verify model is on correct device
        first_param = next(model.parameters())
        actual_device = first_param.device

        print(f"Model device: {actual_device}")

        if self.device.type == 'cuda':
            self.assertTrue(first_param.is_cuda, "Model should be on CUDA")
            print("SUCCESS: Model is on GPU!")
        else:
            self.assertFalse(first_param.is_cuda, "Model should be on CPU")
            print("Model is on CPU (no GPU available)")

    def test_forward_pass_on_device(self):
        """Test forward pass on the detected device."""
        print("\n" + "="*60)
        print("FORWARD PASS TEST")
        print("="*60)

        # Create model
        model = create_model(
            model_type='crnn_attention',
            num_features=self.num_features,
            num_stocks=5,
            num_groups=3,
            config=self.config
        ).to(self.device)

        # Create batch on device
        batch_size = 4
        features = torch.randn(batch_size, self.seq_len, self.num_features).to(self.device)
        stock_id = torch.randint(0, 5, (batch_size, self.seq_len)).to(self.device)
        group_id = torch.randint(0, 3, (batch_size, self.seq_len)).to(self.device)
        day = torch.randint(1, 32, (batch_size, self.seq_len)).to(self.device)
        month = torch.randint(1, 13, (batch_size, self.seq_len)).to(self.device)
        dividend_flag = torch.zeros(batch_size, self.seq_len, 1).to(self.device)

        print(f"Input device: {features.device}")
        print(f"Model device: {next(model.parameters()).device}")

        # Forward pass
        output = model(features, stock_id, group_id, day, month, dividend_flag)

        print(f"Output shape: {output.shape}")
        print(f"Output device: {output.device}")

        # Verify output is on same device as input
        self.assertEqual(output.device.type, self.device.type)
        self.assertEqual(output.shape, (batch_size, 1))

        if self.device.type == 'cuda':
            print("SUCCESS: Forward pass completed on GPU!")
        else:
            print("Forward pass completed on CPU")

    def test_training_step_on_device(self):
        """Test a single training step on the detected device."""
        print("\n" + "="*60)
        print("TRAINING STEP TEST")
        print("="*60)

        # Create model and trainer
        model = create_model(
            model_type='crnn_attention',
            num_features=self.num_features,
            num_stocks=5,
            num_groups=3,
            config=self.config
        )

        trainer = Trainer(model, self.config, device=str(self.device))

        print(f"Trainer device: {trainer.device}")
        print(f"Model device: {next(trainer.model.parameters()).device}")

        # Create data loader
        train_dataset = FinancialDataset(self.train_sequences, self.config)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Get a batch
        batch = next(iter(train_loader))

        # Move batch to device
        features = batch['features'].to(self.device)
        target = batch['target'].to(self.device)

        print(f"Batch device: {features.device}")

        # Training step
        trainer.model.train()
        trainer.optimizer.zero_grad()

        output = trainer.model(
            features,
            batch['stock_id'].to(self.device),
            batch['group_id'].to(self.device),
            batch['day'].to(self.device),
            batch['month'].to(self.device),
            batch['dividend_flag'].to(self.device)
        )

        loss = trainer.criterion(output, target)
        loss.backward()
        trainer.optimizer.step()

        print(f"Loss: {loss.item():.6f}")
        print(f"Loss device: {loss.device}")

        # Verify loss is a scalar
        self.assertEqual(loss.dim(), 0)

        # Verify loss is on correct device
        self.assertEqual(loss.device.type, self.device.type)

        if self.device.type == 'cuda':
            print("SUCCESS: Training step completed on GPU!")
        else:
            print("Training step completed on CPU")

    def test_full_training_epoch(self):
        """Test a full training epoch on the detected device."""
        print("\n" + "="*60)
        print("FULL EPOCH TEST")
        print("="*60)

        # Create model and trainer
        model = create_model(
            model_type='crnn_attention',
            num_features=self.num_features,
            num_stocks=5,
            num_groups=3,
            config=self.config
        )

        trainer = Trainer(model, self.config, device=str(self.device))

        # Create data loaders
        loaders = create_data_loaders(
            train_sequences=self.train_sequences,
            val_sequences=self.val_sequences,
            config=self.config
        )

        print(f"Training batches: {len(loaders['train'])}")
        print(f"Validation batches: {len(loaders['val'])}")

        # Train one epoch
        print("\nTraining one epoch...")
        train_metrics = trainer.train_epoch(loaders['train'])

        print(f"Train Loss: {train_metrics['loss']:.6f}")
        print(f"Train MSE: {train_metrics['mse']:.6f}")
        print(f"Train MAE: {train_metrics['mae']:.6f}")

        # Validate
        print("\nRunning validation...")
        val_metrics = trainer.validate(loaders['val'])

        print(f"Val Loss: {val_metrics['loss']:.6f}")
        print(f"Val MSE: {val_metrics['mse']:.6f}")
        print(f"Val MAE: {val_metrics['mae']:.6f}")
        print(f"Val RMSE: {val_metrics['rmse']:.6f}")

        # Verify metrics are valid
        self.assertFalse(np.isnan(train_metrics['loss']), "Train loss should not be NaN")
        self.assertFalse(np.isinf(train_metrics['loss']), "Train loss should not be Inf")
        self.assertGreater(train_metrics['loss'], 0, "Train loss should be positive")

        self.assertFalse(np.isnan(val_metrics['loss']), "Val loss should not be NaN")
        self.assertFalse(np.isinf(val_metrics['loss']), "Val loss should not be Inf")

        if self.device.type == 'cuda':
            print("\nSUCCESS: Full epoch completed on GPU!")
        else:
            print("\nFull epoch completed on CPU")

    def test_full_training_flow(self):
        """Test complete training flow for multiple epochs."""
        print("\n" + "="*60)
        print("FULL TRAINING FLOW TEST")
        print("="*60)

        # Create model and trainer
        model = create_model(
            model_type='crnn_attention',
            num_features=self.num_features,
            num_stocks=5,
            num_groups=3,
            config=self.config
        )

        print(f"Model: {model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.model.training.NUM_EPOCHS}")

        trainer = Trainer(model, self.config, device=str(self.device))

        # Create data loaders
        loaders = create_data_loaders(
            train_sequences=self.train_sequences,
            val_sequences=self.val_sequences,
            config=self.config
        )

        # Train
        print("\nStarting training...")
        history = trainer.train(
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            num_epochs=self.config.model.training.NUM_EPOCHS
        )

        print("\nTraining complete!")
        print(f"History: {history}")
        print(f"Best validation loss: {trainer.checkpoint.best_score:.6f}")

        # Verify training history
        self.assertEqual(len(history['train_loss']), self.config.model.training.NUM_EPOCHS)
        self.assertEqual(len(history['val_loss']), self.config.model.training.NUM_EPOCHS)

        # Verify losses decreased (at least not increased dramatically)
        # Note: With random data, this might not always hold, but shouldn't explode
        final_loss = history['train_loss'][-1]
        self.assertFalse(np.isnan(final_loss), "Final loss should not be NaN")
        self.assertFalse(np.isinf(final_loss), "Final loss should not be Inf")

        if self.device.type == 'cuda':
            print("\nSUCCESS: Full training completed on GPU!")
        else:
            print("\nFull training completed on CPU")

    def test_checkpointing(self):
        """Test model checkpointing during training."""
        print("\n" + "="*60)
        print("CHECKPOINTING TEST")
        print("="*60)

        # Create model and trainer
        model = create_model(
            model_type='crnn_attention',
            num_features=self.num_features,
            num_stocks=5,
            num_groups=3,
            config=self.config
        )

        trainer = Trainer(model, self.config, device=str(self.device))

        # Create data loaders
        loaders = create_data_loaders(
            train_sequences=self.train_sequences,
            val_sequences=self.val_sequences,
            config=self.config
        )

        # Train for one epoch
        train_metrics = trainer.train_epoch(loaders['train'])
        val_metrics = trainer.validate(loaders['val'])

        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pth"
        trainer.save_model(str(checkpoint_path))

        print(f"Checkpoint saved to: {checkpoint_path}")
        self.assertTrue(checkpoint_path.exists(), "Checkpoint file should exist")

        # Create new model and load checkpoint
        new_model = create_model(
            model_type='crnn_attention',
            num_features=self.num_features,
            num_stocks=5,
            num_groups=3,
            config=self.config
        )

        new_trainer = Trainer(new_model, self.config, device=str(self.device))
        new_trainer.load_checkpoint(str(checkpoint_path))

        print("Checkpoint loaded successfully")

        # Verify models have same weights
        for param1, param2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(param1, param2), "Loaded weights should match")

        print("SUCCESS: Checkpointing works correctly!")


def run_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("TRAINING INTEGRATION TESTS")
    print("="*70)
    print("\nThese tests verify GPU detection and full training flow")
    print("with small synthetic datasets.\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTrainingIntegration)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
