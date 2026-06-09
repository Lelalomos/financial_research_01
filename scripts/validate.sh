#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Validate model script
# Usage: ./scripts/validate.sh [options]
#
# Options:
#   --model PATH       Model checkpoint path (default: models/best_model.pth)
#   --data-split SPLIT Data split to validate on: val, test (default: val)
#   --help             Show this help message

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"

# Default values
MODEL_PATH="models/best_model.pth"
DATA_SPLIT="val"
MODEL_TYPE=""
DATA_DIR=""
DEVICE=""
FORCE_CPU=false

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
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --force-cpu)
            FORCE_CPU=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH       Model checkpoint path (default: models/best_model.pth)"
            echo "  --data-split SPLIT Data split to validate on: val, test (default: val)"
            echo "  --model-type TYPE  Override model type from config/model.json"
            echo "  --data-dir PATH    Processed data directory override"
            echo "  --device DEVICE    Device override (e.g. cuda, cuda:0, cpu)"
            echo "  --force-cpu        Force CPU usage"
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

MODEL_TYPE="$(resolve_model_type "$MODEL_TYPE")"

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$(resolve_data_dir_for_model_type "$MODEL_TYPE")"
fi

# Build command
CMD="python scripts/validate.py --model $MODEL_PATH --split $DATA_SPLIT --data-dir $DATA_DIR"

if [ -n "$MODEL_TYPE" ]; then
    CMD="$CMD --model-type $MODEL_TYPE"
fi

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ "$FORCE_CPU" = true ]; then
    CMD="$CMD --force-cpu"
fi

echo "=========================================="
echo "VALIDATING MODEL"
echo "=========================================="
echo "Model path: $MODEL_PATH"
if [ -n "$MODEL_TYPE" ]; then
    echo "Model type override: $MODEL_TYPE"
else
    echo "Model type: from config/model.json"
fi
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
