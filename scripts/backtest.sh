#!/bin/bash
# Backtest model script
# Usage: ./scripts/backtest.sh [options]
#
# Options:
#   --model PATH       Model checkpoint path (default: models/best_model.pth)
#   --output PATH      Output Excel file path (default: outputs/backtest_report.xlsx)
#   --help             Show this help message

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"

# Default values
MODEL_PATH="models/best_model.pth"
OUTPUT_PATH="outputs/backtest_report.xlsx"
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
        --output)
            OUTPUT_PATH="$2"
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
            echo "  --output PATH      Output Excel file path (default: outputs/backtest_report.xlsx)"
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
CMD="python scripts/backtest.py --model $MODEL_PATH --output $OUTPUT_PATH --data-dir $DATA_DIR"

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
echo "BACKTESTING MODEL"
echo "=========================================="
echo "Model path: $MODEL_PATH"
if [ -n "$MODEL_TYPE" ]; then
    echo "Model type override: $MODEL_TYPE"
else
    echo "Model type: from config/model.json"
fi
echo "Output path: $OUTPUT_PATH"
echo "Command: $CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "BACKTESTING COMPLETE"
echo "Report saved to: $OUTPUT_PATH"
echo "=========================================="
