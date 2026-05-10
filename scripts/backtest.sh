#!/bin/bash
# Backtest model script
# Usage: ./scripts/backtest.sh [options]
#
# Options:
#   --model PATH       Model checkpoint path (default: models/best_model.pth)
#   --output PATH      Output Excel file path (default: outputs/backtest_report.xlsx)
#   --help             Show this help message

set -e

# Default values
MODEL_PATH="models/best_model.pth"
OUTPUT_PATH="outputs/backtest_report.xlsx"

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
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH       Model checkpoint path (default: models/best_model.pth)"
            echo "  --output PATH      Output Excel file path (default: outputs/backtest_report.xlsx)"
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
CMD="python scripts/backtest.py --model latest --output $OUTPUT_PATH"

echo "=========================================="
echo "BACKTESTING MODEL"
echo "=========================================="
echo "Model path: $MODEL_PATH"
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
