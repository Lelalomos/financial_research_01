#!/bin/bash
# Test model script
# Usage: ./scripts/test.sh [options]
#
# Options:
#   --model PATH       Model checkpoint path (default: models/best_model.pth)
#   --help             Show this help message

set -e

# Default values
MODEL_PATH="models/best_model.pth"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH       Model checkpoint path (default: models/best_model.pth)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python scripts/test.py --model $MODEL_PATH"

echo "=========================================="
echo "TESTING MODEL"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Command: $CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "TESTING COMPLETE"
echo "=========================================="
