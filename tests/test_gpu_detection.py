"""
Unit tests for GPU detection and device utilities.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.device import (
    get_device,
    get_device_info,
    print_gpu_info,
    get_gpu_memory_info,
    clear_gpu_memory,
    check_gpu_compatibility,
    get_recommended_batch_size
)


class TestGPUDetection(unittest.TestCase):
    """Test GPU detection utilities."""

    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a valid torch.device."""
        device = get_device(verbose=False)

        # Should return a torch.device
        self.assertIsInstance(device, torch.device)

        # Should be either 'cpu' or 'cuda'
        self.assertIn(device.type, ['cpu', 'cuda'])

    def test_get_device_with_force_cpu(self):
        """Test that force_cpu parameter works correctly."""
        device = get_device(force_cpu=True, verbose=False)

        # Should return CPU device when forced
        self.assertIsInstance(device, torch.device)
        self.assertEqual(device.type, 'cpu')

    def test_get_device_info_returns_dict(self):
        """Test that get_device_info returns a dictionary with expected keys."""
        info = get_device_info(verbose=False)

        # Should be a dictionary
        self.assertIsInstance(info, dict)

        # Should have expected keys
        expected_keys = [
            'device',
            'cuda_available',
            'cuda_device_count',
            'pytorch_version',
            'cuda_version',
            'cudnn_version',
            'cuda_working'
        ]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_get_device_info_pytorch_version(self):
        """Test that PyTorch version is detected correctly."""
        info = get_device_info(verbose=False)

        # Should have PyTorch version
        self.assertIsNotNone(info['pytorch_version'])
        self.assertIsInstance(info['pytorch_version'], str)
        # Check it starts with a number (e.g., "2.0.0")
        self.assertRegex(info['pytorch_version'], r'^\d+\.\d+\.\d+')

    def test_cuda_available_flag(self):
        """Test that CUDA availability is detected correctly."""
        info = get_device_info(verbose=False)

        # The cuda_available flag should match torch.cuda.is_available()
        self.assertEqual(info['cuda_available'], torch.cuda.is_available())

        # cuda_version should be None if CUDA is not available
        if not info['cuda_available']:
            self.assertIsNone(info['cuda_version'])
            self.assertIsNone(info['cudnn_version'])

    def test_cuda_working_flag(self):
        """Test that CUDA working status is detected correctly."""
        info = get_device_info(verbose=False)

        # If CUDA is not available, cuda_working should be False
        if not info['cuda_available']:
            self.assertFalse(info['cuda_working'])
        # If CUDA is available but no GPUs, cuda_working should be False
        elif info['cuda_device_count'] == 0:
            self.assertFalse(info['cuda_working'])

    def test_check_gpu_compatibility_returns_tuple(self):
        """Test that check_gpu_compatibility returns a tuple."""
        is_compatible, message = check_gpu_compatibility()

        # Should return a tuple
        self.assertIsInstance(is_compatible, bool)
        self.assertIsInstance(message, str)

        # If CUDA is not available, should not be compatible
        if not torch.cuda.is_available():
            self.assertFalse(is_compatible)

    def test_get_recommended_batch_size_returns_positive_int(self):
        """Test that get_recommended_batch_size returns a positive integer."""
        # Test for CPU
        batch_size = get_recommended_batch_size(
            model_params=1_000_000,
            device=torch.device('cpu'),
            input_size=(32, 30, 20)
        )
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

        # Test for different model sizes
        batch_size_small = get_recommended_batch_size(
            model_params=500_000,
            device=torch.device('cpu'),
            input_size=(32, 30, 20)
        )
        batch_size_large = get_recommended_batch_size(
            model_params=50_000_000,
            device=torch.device('cpu'),
            input_size=(32, 30, 20)
        )

        # Larger models should get smaller batch sizes (on CPU)
        self.assertLessEqual(batch_size_large, batch_size_small)

    def test_get_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        # This should not crash even if GPU is not available
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                memory_info = get_gpu_memory_info(0)
                self.assertIsInstance(memory_info, str)
                # Should contain memory information
                self.assertIn('MB', memory_info)
            else:
                # Should not crash when GPU is not available
                # Just skip the test
                pass
        except Exception as e:
            self.fail(f"get_gpu_memory_info raised an exception: {e}")

    def test_clear_gpu_memory(self):
        """Test GPU memory clearing."""
        # This should not crash even if GPU is not available
        try:
            clear_gpu_memory()
            # Also test with explicit device
            clear_gpu_memory(torch.device('cpu'))
        except Exception as e:
            self.fail(f"clear_gpu_memory raised an exception: {e}")

    def test_print_gpu_info(self):
        """Test that print_gpu_info doesn't crash."""
        # This should not crash, just print info
        try:
            print_gpu_info()
        except Exception as e:
            self.fail(f"print_gpu_info raised an exception: {e}")

    def test_device_selection_consistency(self):
        """Test that device selection is consistent across calls."""
        device1 = get_device(verbose=False)
        device2 = get_device(verbose=False)

        # Should return the same device type (unless GPU state changed)
        self.assertEqual(device1.type, device2.type)

    def test_gpu_device_selection_when_available(self):
        """Test that GPU is selected when available."""
        device = get_device(verbose=False)

        if torch.cuda.is_available():
            # Try to verify CUDA is actually working
            try:
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()

                # If we got here, CUDA is working
                # The device should be cuda
                self.assertEqual(device.type, 'cuda')
            except RuntimeError:
                # CUDA runtime error, CPU is acceptable
                self.assertEqual(device.type, 'cpu')
        else:
            # CUDA not available, should be CPU
            self.assertEqual(device.type, 'cpu')


class TestDeviceIntegration(unittest.TestCase):
    """Integration tests for device functionality with training."""

    def test_model_to_device(self):
        """Test that models can be moved to the detected device."""
        from src.models import create_model
        from src.config import load_config

        config = load_config('model')
        device = get_device(verbose=False)

        # Create a simple model
        model = create_model(
            model_type='crnn_attention',
            num_features=20,
            num_stocks=10,
            num_groups=5,
            config=config
        )

        # Move to device
        model = model.to(device)

        # Verify model is on the correct device
        if device.type == 'cuda':
            # Check first parameter
            param = next(model.parameters())
            self.assertTrue(param.is_cuda)
        else:
            # CPU
            param = next(model.parameters())
            self.assertFalse(param.is_cuda)

    def test_tensor_operations_on_device(self):
        """Test that tensor operations work on the detected device."""
        device = get_device(verbose=False)

        # Create tensors on device
        x = torch.randn(10, 5).to(device)
        y = torch.randn(10, 5).to(device)

        # Perform operation
        z = x + y

        # Verify result is on same device
        self.assertEqual(z.device.type, device.type)

    def test_cuda_memory_allocation_when_available(self):
        """Test CUDA memory allocation when GPU is available."""
        device = get_device(verbose=False)

        if device.type == 'cuda':
            # Allocate tensor on GPU
            x = torch.randn(100, 100).cuda()

            # Check it's on GPU
            self.assertTrue(x.is_cuda)

            # Clean up
            del x
            torch.cuda.empty_cache()
        else:
            # Skip test if no GPU
            pass


if __name__ == '__main__':
    unittest.main()
