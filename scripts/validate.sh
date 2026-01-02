#!/bin/bash
# Validate model script
# Usage: ./scripts/validate.sh [options]
#
# Options:
#   --model PATH       Model checkpoint path (default: models/best_model.pth)
#   --data-split SPLIT Data split to validate on: val, test (default: val)
#   --help             Show this help message

set -e

# Default values
MODEL_PATH="models/best_model.pth"
DATA_SPLIT="val"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-split)
            DATA_SPLIT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH       Model checkpoint path (default: models/best_model.pth)"
            echo "  --data-split SPLIT Data split to validate on: val, test (default: val)"
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
CMD="python scripts/validate.py --model $MODEL_PATH --data-split $DATA_SPLIT"

echo "=========================================="
echo "VALIDATING MODEL"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Data split: $DATA_SPLIT"
echo "Command: $CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "VALIDATION COMPLETE"
echo "=========================================="
