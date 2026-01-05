#!/bin/bash
# NaN Loss Fix Test Script
# Usage: ./scripts/test_nan_fix.sh [options]
#
# This script runs the NaN loss fix tests inside the Docker container.
#
# Options:
#   --container NAME  Container name or ID (default: a2df9be22991)
#   --verbose         Show detailed test output
#   --coverage        Run with coverage report
#   --help            Show this help message

set -e

# Default values
CONTAINER_NAME="a2df9be22991"
VERBOSE=""
COVERAGE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="-v -s"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=src --cov-report=html --cov-report=term"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --container NAME  Container name or ID (default: a2df9be22991)"
            echo "  --verbose         Show detailed test output"
            echo "  --coverage        Run with coverage report"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run NaN fix tests in container"
            echo "  $0 --verbose                    # Run with verbose output"
            echo "  $0 --coverage                   # Run with coverage report"
            echo "  $0 --container my_container     # Run in specific container"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "RUNNING NaN LOSS FIX TESTS"
echo "=========================================="
echo "Container: $CONTAINER_NAME"
echo "=========================================="
echo ""

# Check if container is running
if ! docker inspect "$CONTAINER_NAME" > /dev/null 2>&1; then
    echo "Error: Container '$CONTAINER_NAME' not found"
    echo "Please ensure the container is running"
    exit 1
fi

if ! docker inspect "$CONTAINER_NAME" | grep -q '"Status": "running"'; then
    echo "Error: Container '$CONTAINER_NAME' is not running"
    echo "Please start the container first"
    exit 1
fi

# Run the tests
echo "Running NaN loss fix tests..."
echo ""

docker exec "$CONTAINER_NAME" pytest tests/test_nan_loss_fix.py $VERBOSE $COVERAGE

echo ""
echo "=========================================="
echo "NaN LOSS FIX TESTS COMPLETE"
echo "=========================================="
echo ""

# Optionally run validation tests
echo "Running validation tests..."
echo ""

docker exec "$CONTAINER_NAME" pytest tests/test_validation.py::TestNaNHandling $VERBOSE

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="
