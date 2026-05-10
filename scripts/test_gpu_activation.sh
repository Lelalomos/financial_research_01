#!/bin/bash
#
# GPU Activation Test Script for Docker Container d633c5977c4f
# This script tests GPU availability, CUDA, and PyTorch GPU functionality
#

set -e

CONTAINER_ID="9987284b5cef"
CONTAINER_NAME="crnn_predictor"

echo "=========================================="
echo "GPU Activation Test for Container: $CONTAINER_ID"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check if container is running
echo "[TEST 1] Checking if container is running..."
if docker ps --filter "id=$CONTAINER_ID" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
    echo -e "${GREEN}✓ Container $CONTAINER_ID ($CONTAINER_NAME) is running${NC}"
else
    echo -e "${RED}✗ Container $CONTAINER_ID is not running${NC}"
    echo "Please start the container first with: docker start $CONTAINER_ID"
    exit 1
fi
echo ""

# Test 2: Check NVIDIA driver and nvidia-smi
echo "[TEST 2] Checking NVIDIA driver and nvidia-smi..."
if docker exec "$CONTAINER_ID" nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✓ nvidia-smi is available in container${NC}"
    docker exec "$CONTAINER_ID" nvidia-smi -L
else
    echo -e "${RED}✗ nvidia-smi not available in container${NC}"
    echo "Container may not have GPU access configured"
fi
echo ""

# Test 3: Check NVIDIA runtime
echo "[TEST 3] Checking NVIDIA runtime configuration..."
HOST_NVIDIA_RUNTIME=$(docker inspect "$CONTAINER_ID" --format='{{.HostConfig.Runtime}}' 2>/dev/null || echo "N/A")
echo "Host runtime: $HOST_NVIDIA_RUNTIME"

# Check device requests
DEVICE_COUNT=$(docker inspect "$CONTAINER_ID" --format='{{len .HostConfig.DeviceRequests}}' 2>/dev/null || echo "0")
echo "Device requests count: $DEVICE_COUNT"
echo ""

# Test 4: Check CUDA libraries in container
echo "[TEST 4] Checking CUDA libraries in container..."
CUDA_LIBS=$(docker exec "$CONTAINER_ID" find /usr/local/cuda -name "libcudart.so*" 2>/dev/null | wc -l)
if [ "$CUDA_LIBS" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $CUDA_LIBS CUDA runtime libraries${NC}"
else
    echo -e "${YELLOW}⚠ CUDA libraries not found in standard location${NC}"
fi
echo ""

# Test 5: Test PyTorch CUDA availability
echo "[TEST 5] Testing PyTorch CUDA availability..."
PYTORCH_TEST=$(docker exec "$CONTAINER_ID" python3 -c "
import sys
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'Device count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
            print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
        # Test actual CUDA operation
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = x + y
            print('✓ CUDA tensor operation successful')
            del x, y, z
            torch.cuda.empty_cache()
            print('✓ GPU memory cleared successfully')
        except Exception as e:
            print(f'✗ CUDA operation failed: {e}')
            sys.exit(1)
    else:
        print('✗ CUDA is not available in PyTorch')
        sys.exit(1)
except ImportError as e:
    print(f'✗ PyTorch not installed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
" 2>&1)

if echo "$PYTORCH_TEST" | grep -q "CUDA tensor operation successful"; then
    echo -e "${GREEN}✓ PyTorch CUDA test passed${NC}"
    echo "$PYTORCH_TEST" | sed 's/^/  /'
else
    echo -e "${RED}✗ PyTorch CUDA test failed${NC}"
    echo "$PYTORCH_TEST" | sed 's/^/  /'
fi
echo ""

# Test 6: Check environment variables
echo "[TEST 6] Checking GPU-related environment variables..."
ENV_VARS=$(docker exec "$CONTAINER_ID" env | grep -E "(CUDA|GPU|NVIDIA)" || echo "No GPU-related env vars found")
echo "$ENV_VARS" | sed 's/^/  /'
echo ""

# Test 7: Test device detection utility
echo "[TEST 7] Testing device detection utility..."
DEVICE_TEST=$(docker exec "$CONTAINER_ID" python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from src.utils.device import get_device, get_device_info, print_gpu_info

    device = get_device(verbose=False)
    print(f'Detected device: {device}')

    info = get_device_info(verbose=False)
    print(f'Device info:')
    for key, value in info.items():
        print(f'  {key}: {value}')

    if info.get('cuda_working'):
        print('✓ GPU is working correctly')
    else:
        print('✗ GPU not working')
        sys.exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

if echo "$DEVICE_TEST" | grep -q "GPU is working correctly"; then
    echo -e "${GREEN}✓ Device detection test passed${NC}"
    echo "$DEVICE_TEST" | sed 's/^/  /'
else
    echo -e "${RED}✗ Device detection test failed${NC}"
    echo "$DEVICE_TEST" | sed 's/^/  /'
fi
echo ""

# Test 8: Run actual unit tests
echo "[TEST 8] Running GPU detection unit tests..."
UNIT_TEST_RESULT=$(docker exec "$CONTAINER_ID" python3 -m pytest tests/test_gpu_detection.py -v 2>&1)
UNIT_TEST_EXIT=$?

if [ $UNIT_TEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ All unit tests passed${NC}"
    echo "$UNIT_TEST_RESULT" | tail -5
else
    echo -e "${RED}✗ Some unit tests failed${NC}"
    echo "$UNIT_TEST_RESULT" | tail -20
fi
echo ""

# Test 9: GPU Memory test
echo "[TEST 9: Testing GPU memory allocation..."
MEMORY_TEST=$(docker exec "$CONTAINER_ID" python3 -c "
import torch
if torch.cuda.is_available():
    try:
        # Allocate different tensor sizes
        sizes = [(1000, 1000), (5000, 5000), (10000, 10000)]
        for size in sizes:
            x = torch.randn(*size).cuda()
            mem_used = torch.cuda.memory_allocated() / 1024**2
            print(f'  Allocated {size}: {mem_used:.2f} MB')
            del x

        torch.cuda.empty_cache()
        print('✓ Memory allocation test passed')
    except RuntimeError as e:
        print(f'✗ Memory allocation failed: {e}')
        exit(1)
else:
    print('✗ CUDA not available')
    exit(1)
" 2>&1)

if echo "$MEMORY_TEST" | grep -q "Memory allocation test passed"; then
    echo -e "${GREEN}✓ GPU memory test passed${NC}"
    echo "$MEMORY_TEST" | sed 's/^/  /'
else
    echo -e "${RED}✗ GPU memory test failed${NC}"
    echo "$MEMORY_TEST" | sed 's/^/  /'
fi
echo ""

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
docker exec "$CONTAINER_ID" nvidia-smi -L 2>/dev/null || echo "GPU info not available"
echo ""
echo "Container: $CONTAINER_ID ($CONTAINER_NAME)"
echo "Test completed at: $(date)"
echo ""
