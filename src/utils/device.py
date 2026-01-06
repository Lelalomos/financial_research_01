"""
Device utilities for PyTorch.

This module provides comprehensive utilities for detecting, configuring, and
managing CUDA/GPU devices for PyTorch training and inference.

Features:
    - Automatic GPU detection with fallback to CPU
    - Comprehensive device information retrieval
    - GPU memory management utilities
    - Compute capability checking
    - Batch size recommendations based on GPU memory

Usage:
    >>> from src.utils.device import get_device, get_device_info
    >>> device = get_device()
    >>> info = get_device_info()
    >>> print(f"Using device: {device}")
"""

import torch
import sys
from typing import Tuple, Dict, Optional


def get_device(
    force_cpu: bool = False,
    verbose: bool = True
) -> torch.device:
    """
    Get the best available device for training.

    This function performs comprehensive checks to determine if CUDA is available
    and properly configured. It validates that:
        - CUDA is available in PyTorch
        - At least one GPU is present
        - CUDA runtime is properly initialized

    The function will automatically fall back to CPU if CUDA is not available
    or if there are any errors during GPU initialization.

    Args:
        force_cpu: If True, force CPU usage regardless of GPU availability.
                   This is useful for debugging or testing on CPU-only systems.
        verbose: If True, print detailed device information including GPU name,
                 memory, compute capability, and any warnings or errors.

    Returns:
        torch.device: The selected device (cuda:0 or cpu). The device is
                     guaranteed to be usable for PyTorch operations.

    Raises:
        No exceptions are raised. All errors are handled by falling back to CPU.

    Examples:
        >>> device = get_device()
        Using device: cuda:0
        >>> model = model.to(device)

        >>> device = get_device(force_cpu=True)
        Forcing CPU usage as requested

        >>> device = get_device(verbose=False)
        # No output, returns device silently

    Note:
        In Docker containers, ensure GPU access is configured via:
        - docker-compose.yml device requests
        - CUDA_VISIBLE_DEVICES environment variable
        - nvidia-docker runtime
    """
    if force_cpu:
        if verbose:
            print("Forcing CPU usage as requested")
        return torch.device('cpu')

    # Check if CUDA is available
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA is not available. Using CPU.")
            print("\nPossible reasons:")
            print("  - PyTorch was installed without CUDA support")
            print("  - No CUDA-capable GPU is detected")
            print("  - NVIDIA drivers are not installed or incompatible")
            print("  - Running in a container without GPU access")
            print("\nTo verify GPU access, run: ./scripts/test_gpu_activation.sh")
        return torch.device('cpu')

    # Check if there are any GPUs
    if torch.cuda.device_count() == 0:
        if verbose:
            print("No CUDA-capable GPU detected. Using CPU.")
            print("\nCheck that:")
            print("  - GPU is visible on host (run: nvidia-smi)")
            print("  - CUDA_VISIBLE_DEVICES is set correctly")
            print("  - Container has GPU access configured")
        return torch.device('cpu')

    # Test CUDA is actually working by allocating a tensor
    try:
        # Try to create a test tensor on GPU
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if verbose:
            print(f"CUDA runtime error: {e}")
            print("\nFalling back to CPU. This could indicate:")
            print("  - GPU memory is insufficient")
            print("  - NVIDIA driver version mismatch")
            print("  - GPU is already in use by another process")
        return torch.device('cpu')

    # CUDA is available and working
    device = torch.device('cuda:0')

    if verbose:
        print_gpu_info()

    return device


def print_gpu_info() -> None:
    """Print detailed GPU information.

    Displays comprehensive GPU information including:
        - PyTorch and CUDA versions
        - cuDNN version
        - Number of GPUs available
        - GPU name, memory, and compute capability for each device
        - Current memory usage (allocated/reserved/total)

    Note:
        This function only prints information if CUDA is available.
        Otherwise, it silently returns without output.
    """
    print(f"CUDA is available and working!")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {get_gpu_memory_info(i)}")

        # Get compute capability
        props = torch.cuda.get_device_properties(i)
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")


def get_gpu_memory_info(device_id: int = 0) -> str:
    """
    Get GPU memory usage information.

    Args:
        device_id: GPU device ID

    Returns:
        String with memory info (allocated / reserved / total)
    """
    allocated = torch.cuda.memory_allocated(device_id) / 1024**2
    reserved = torch.cuda.memory_reserved(device_id) / 1024**2
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2

    return f"{allocated:.1f}MB allocated / {reserved:.1f}MB reserved / {total:.1f}MB total"


def get_device_info(
    verbose: bool = True
) -> Dict[str, any]:
    """
    Get comprehensive device information as a dictionary.

    This function gathers detailed information about the available computing
    devices, including CUDA availability, GPU specifications, and whether
    CUDA is actually working (tested via tensor allocation).

    Args:
        verbose: If True, print a formatted summary of device information.

    Returns:
        Dictionary containing:
            - device (str): Selected device ('cpu' or 'cuda:0')
            - cuda_available (bool): Whether CUDA is available in PyTorch
            - cuda_device_count (int): Number of CUDA-capable GPUs
            - pytorch_version (str): PyTorch version
            - cuda_version (str or None): CUDA toolkit version
            - cudnn_version (str or None): cuDNN version
            - cuda_working (bool): Whether CUDA is actually functional
            - gpu_name (str or None): GPU device name
            - gpu_memory_gb (float or None): Total GPU memory in GB
            - compute_capability (str or None): GPU compute capability
            - cuda_error (str or None): Error message if CUDA is not working

    Examples:
        >>> info = get_device_info()
        >>> print(f"Device: {info['device']}")
        >>> print(f"GPU: {info['gpu_name']}")
        >>> print(f"Memory: {info['gpu_memory_gb']:.2f} GB")

        >>> info = get_device_info(verbose=False)
        >>> if info['cuda_working']:
        ...     print("GPU is ready!")
    """
    info = {
        'device': 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # Test CUDA is working by allocating a tensor
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            info['cuda_working'] = True
            info['device'] = 'cuda:0'
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            props = torch.cuda.get_device_properties(0)
            info['compute_capability'] = f"{props.major}.{props.minor}"
        except RuntimeError as e:
            info['cuda_working'] = False
            info['cuda_error'] = str(e)
    else:
        info['cuda_working'] = False

    if verbose:
        print_device_info(info)

    return info


def print_device_info(info: Dict[str, any]) -> None:
    """
    Print device information from a dictionary.

    Args:
        info: Device information dictionary
    """
    print(f"Device Information:")
    print(f"  Selected device: {info['device']}")
    print(f"  PyTorch version: {info['pytorch_version']}")

    if info['cuda_available']:
        print(f"  CUDA available: Yes")
        print(f"  CUDA version: {info['cuda_version']}")
        print(f"  cuDNN version: {info['cudnn_version']}")
        print(f"  GPU count: {info['cuda_device_count']}")

        if info.get('cuda_working'):
            print(f"  CUDA working: Yes")
            print(f"  GPU name: {info.get('gpu_name', 'Unknown')}")
            print(f"  GPU memory: {info.get('gpu_memory_gb', 0):.2f} GB")
            print(f"  Compute capability: {info.get('compute_capability', 'Unknown')}")
        elif 'cuda_error' in info:
            print(f"  CUDA working: No")
            print(f"  Error: {info['cuda_error']}")
    else:
        print(f"  CUDA available: No")


def clear_gpu_memory(device: Optional[torch.device] = None) -> None:
    """
    Clear GPU memory cache.

    Args:
        device: Device to clear (if None, clears all GPUs)
    """
    if torch.cuda.is_available():
        if device is None or device.type == 'cuda':
            torch.cuda.empty_cache()
            if device is not None:
                torch.cuda.synchronize(device)


def set_gpu_device(device_id: int) -> torch.device:
    """
    Set the current GPU device.

    Args:
        device_id: GPU device ID to use

    Returns:
        The selected device

    Raises:
        ValueError: If device_id is invalid
    """
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    if device_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid device_id {device_id}. Available: {torch.cuda.device_count()}")

    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)

    return device


def check_gpu_compatibility(
    required_compute_capability: Tuple[int, int] = (7, 0)
) -> Tuple[bool, str]:
    """
    Check if GPU meets minimum compute capability requirements.

    Args:
        required_compute_capability: Minimum required (major, minor) version

    Returns:
        Tuple of (is_compatible, message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    if torch.cuda.device_count() == 0:
        return False, "No GPU detected"

    device_id = 0
    props = torch.cuda.get_device_properties(device_id)
    actual_cc = (props.major, props.minor)

    if actual_cc < required_compute_capability:
        required_str = f"{required_compute_capability[0]}.{required_compute_capability[1]}"
        actual_str = f"{actual_cc[0]}.{actual_cc[1]}"
        return False, f"GPU compute capability {actual_str} is below required {required_str}"

    return True, f"GPU compute capability {actual_cc[0]}.{actual_cc[1]} meets requirements"


def get_recommended_batch_size(
    model_params: int,
    device: torch.device,
    input_size: Tuple[int, ...] = (32, 30, 20),
    memory_fraction: float = 0.8
) -> int:
    """
    Get recommended batch size based on GPU memory and model size.

    Args:
        model_params: Number of model parameters
        device: Device to use
        input_size: Input tensor size (batch_size, seq_len, features)
        memory_fraction: Fraction of GPU memory to use

    Returns:
        Recommended batch size
    """
    if device.type != 'cuda':
        # For CPU, use a conservative batch size
        return min(32, 256 // max(1, model_params // 1_000_000))

    # Get GPU memory
    props = torch.cuda.get_device_properties(0)
    total_memory_gb = props.total_memory / 1024**3

    # Rough estimation based on model size and memory
    # This is a heuristic and may need adjustment
    base_batch_size = 32
    memory_multiplier = max(1, int(total_memory_gb * memory_fraction))

    # Adjust for model size
    if model_params < 1_000_000:
        size_multiplier = 4
    elif model_params < 10_000_000:
        size_multiplier = 2
    else:
        size_multiplier = 1

    batch_size = base_batch_size * memory_multiplier * size_multiplier

    # Clamp to reasonable range
    return min(max(16, batch_size), 512)
