# GPU Troubleshooting Guide

This guide helps diagnose and resolve GPU-related issues when running the CRNN Financial Prediction Model.

## Quick Diagnostics

### Run GPU Activation Test

The fastest way to check GPU status is to run the automated test script:

```bash
./scripts/test_gpu_activation.sh
```

This script performs comprehensive checks and provides detailed output about GPU status.

### Manual Quick Checks

```bash
# Check host GPU
nvidia-smi

# Check container GPU access
docker exec crnn_predictor nvidia-smi

# Test PyTorch CUDA
docker exec crnn_predictor python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Common Issues and Solutions

### 1. CUDA Not Available

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- Error: "CUDA is not available"
- Model trains on CPU despite GPU presence

**Diagnosis:**

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**Solutions:**

1. **Reinstall PyTorch with CUDA support:**
   ```bash
   # Check CUDA version on host
   nvidia-smi  # Look for "CUDA Version: XX.X"

   # Install PyTorch with matching CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Verify PyTorch CUDA build:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.version.cuda)  # Should show version
   ```

3. **In Docker, rebuild image with CUDA:**
   ```bash
   docker-compose build --no-cache
   ```

### 2. No GPU Detected in Container

**Symptoms:**
- `torch.cuda.device_count()` returns `0`
- `nvidia-smi` works on host but not in container
- Error: "No CUDA-capable GPU detected"

**Diagnosis:**

```bash
# Check host GPU
nvidia-smi

# Check container GPU access
docker exec crnn_predictor nvidia-smi

# Check Docker runtime
docker inspect crnn_predictor | grep -A 10 "DeviceRequests"
```

**Solutions:**

1. **Install nvidia-docker2:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Verify docker-compose.yml has GPU configuration:**
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

3. **Set CUDA_VISIBLE_DEVICES:**
   ```yaml
   environment:
     - CUDA_VISIBLE_DEVICES=0
   ```

4. **Restart container:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### 3. CUDA Runtime Error

**Symptoms:**
- Error: "CUDA runtime error (XXX)"
- Error: "CUDA out of memory"
- Tensor operations fail after initial success

**Diagnosis:**

```python
import torch
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x + y
    print("CUDA operation successful")
except RuntimeError as e:
    print(f"Error: {e}")
```

**Solutions:**

1. **Check GPU memory usage:**
   ```bash
   nvidia-smi
   # Look at "Memory-Usage" column
   ```

2. **Reduce batch size in config/model.json:**
   ```json
   {
     "model": {
       "training": {
         "BATCH_SIZE": 64  # Reduce from 128
       }
     }
   }
   ```

3. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Kill other GPU processes:**
   ```bash
   nvidia-smi  # Find process PID
   kill -9 <PID>
   ```

5. **Check GPU compute capability:**
   ```python
   import torch
   props = torch.cuda.get_device_properties(0)
   print(f"Compute capability: {props.major}.{props.minor}")
   # Minimum required: 7.0
   ```

### 4. NVIDIA Driver Version Mismatch

**Symptoms:**
- Error: "CUDA driver version is insufficient for CUDA runtime version"
- nvidia-smi shows different CUDA version than PyTorch

**Diagnosis:**

```bash
# Check NVIDIA driver version
nvidia-smi  # Look for "Driver Version"

# Check PyTorch CUDA version
python3 -c "import torch; print(torch.version.cuda)"

# Check supported CUDA version
cat /usr/local/cuda/version.txt  # If CUDA is installed
```

**Solutions:**

1. **Update NVIDIA drivers:**
   ```bash
   sudo apt-get update
   sudo apt-get install nvidia-driver-535  # Or latest version
   sudo reboot
   ```

2. **Install PyTorch with compatible CUDA version:**
   ```bash
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Rebuild Docker image with correct CUDA:**
   ```dockerfile
   FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
   ```

### 5. GPU Not Utilized During Training

**Symptoms:**
- GPU shows 0% utilization in nvidia-smi
- Training is slow
- CPU usage is high but GPU idle

**Diagnosis:**

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Check PyTorch device
python3 -c "from src.utils.device import get_device; print(get_device())"
```

**Solutions:**

1. **Verify model is on GPU:**
   ```python
   model = model.to(device)
   print(next(model.parameters()).device)  # Should be cuda:0
   ```

2. **Verify data is on GPU:**
   ```python
   for batch in dataloader:
       batch = batch.to(device)
   ```

3. **Check device selection in training script:**
   ```python
   from src.utils.device import get_device
   device = get_device()  # Don't use force_cpu=True
   ```

4. **Enable CUDA in config:**
   ```json
   {
     "model": {
       "training": {
         "DEVICE": "cuda"
       }
     }
   }
   ```

## Advanced Diagnostics

### PyTorch CUDA Test

```python
import torch

def test_cuda():
    print("=" * 60)
    print("PyTorch CUDA Diagnostic")
    print("=" * 60)

    # Basic availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nCUDA is not available. Cannot proceed with tests.")
        return

    # Version info
    print(f"\n2. PyTorch Version: {torch.__version__}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")

    # Device count
    print(f"\n3. Device Count: {torch.cuda.device_count()}")

    # Device properties
    for i in range(torch.cuda.device_count()):
        print(f"\n4. Device {i}:")
        props = torch.cuda.get_device_properties(i)
        print(f"   Name: {torch.cuda.get_device_name(i)}")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   Multi-processors: {props.multi_processor_count}")

    # Memory test
    print(f"\n5. Memory Test:")
    try:
        # Allocate different tensor sizes
        for size in [(1000, 1000), (5000, 5000)]:
            x = torch.randn(*size).cuda()
            mem_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"   Allocated {size}: {mem_mb:.2f} MB")
            del x

        torch.cuda.empty_cache()
        print(f"   ✓ Memory test passed")
    except RuntimeError as e:
        print(f"   ✗ Memory test failed: {e}")

    # Operation test
    print(f"\n6. Operation Test:")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x + y
        print(f"   ✓ Tensor operation successful")
        del x, y, z
    except RuntimeError as e:
        print(f"   ✗ Operation failed: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_cuda()
```

### Docker GPU Check Script

```bash
#!/bin/bash
# check_docker_gpu.sh

echo "Checking Docker GPU Configuration..."
echo ""

# Check Docker runtime
echo "1. Docker NVIDIA Runtime:"
docker info 2>/dev/null | grep -i nvidia || echo "   NVIDIA runtime not found"
echo ""

# Check container configuration
echo "2. Container Device Requests:"
docker inspect crnn_predictor 2>/dev/null | grep -A 5 "DeviceRequests" || echo "   Container not running"
echo ""

# Test nvidia-smi in container
echo "3. nvidia-smi in Container:"
docker exec crnn_predictor nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "   Failed"
echo ""

# Check environment variables
echo "4. GPU Environment Variables:"
docker exec crnn_predictor env | grep -E "(CUDA|GPU)" || echo "   No GPU env vars found"
echo ""
```

## Prevention and Best Practices

### 1. Always Test GPU Before Training

```bash
./scripts/test_gpu_activation.sh
```

### 2. Monitor GPU During Training

```bash
# In another terminal
watch -n 1 nvidia-smi
```

### 3. Use Appropriate Batch Sizes

Start with smaller batch sizes and increase gradually:

```json
{
  "model": {
    "training": {
      "BATCH_SIZE": 32  # Start small
    }
  }
}
```

### 4. Clear GPU Cache Regularly

```python
import torch
torch.cuda.empty_cache()
```

### 5. Keep Drivers Updated

```bash
# Check for updates
sudo apt-get update
sudo apt-get install nvidia-driver-535
```

## Getting Help

If issues persist after trying these solutions:

1. **Run the full diagnostic script:**
   ```bash
   ./scripts/test_gpu_activation.sh > gpu_diagnostic.txt 2>&1
   ```

2. **Collect system information:**
   ```bash
   nvidia-smi > nvidia_smi.txt
   docker info > docker_info.txt
   docker version > docker_version.txt
   ```

3. **Check logs:**
   ```bash
   docker logs crnn_predictor > container_logs.txt
   ```

4. **Verify Python environment:**
   ```bash
   pip list | grep -i torch
   python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
   ```

## Additional Resources

- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [PyTorch GPU Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
