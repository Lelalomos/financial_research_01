#!/bin/bash
# Test model script
# Usage: ./scripts/test.sh [options]
#
# Options:
#   --model PATH       Model checkpoint path (default: models/bilstm4_attention_best_20260105_113045.pth)
#   --excel-report PATH Custom Excel report path (default: reports/test_report_TIMESTAMP.xlsx)
#   --no-cleanup       Skip cleanup of old test files before running
#   --help             Show this help message

set -e

# Default values
MODEL_PATH="models/bilstm4_attention_best_20260105_113045.pth"
REPORTS_DIR="reports"
CLEANUP_OLD_FILES=true
MODEL_TYPE=""
DEVICE=""
FORCE_CPU=false

# Create reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

# Cleanup function to remove old test files using Python cleanup module
cleanup_old_test_files() {
    echo "Cleaning up old test files..."
    python -c "from src.utils.cleanup import cleanup_test_files; cleanup_test_files(keep_latest=3, verbose=True)"
    echo ""
}

# Generate default report path with timestamp (ALWAYS generate Excel report)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXCEL_REPORT="$REPORTS_DIR/test_report_${TIMESTAMP}.xlsx"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --excel-report)
            # Custom path
            EXCEL_REPORT="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
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
        --no-cleanup)
            CLEANUP_OLD_FILES=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH       Model checkpoint path (default: models/bilstm4_attention_best_20260105_113045.pth)"
            echo "  --excel-report PATH Custom Excel report path (default: auto-generated with timestamp)"
            echo "  --model-type TYPE  Override model type from config/model.json"
            echo "  --device DEVICE    Device override (e.g. cuda, cuda:0, cpu)"
            echo "  --force-cpu        Force CPU usage"
            echo "  --no-cleanup       Skip cleanup of old test files before running"
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

# Build command - ALWAYS include Excel report
CMD="python scripts/test.py --model $MODEL_PATH --excel-report $EXCEL_REPORT"

if [ -n "$MODEL_TYPE" ]; then
    CMD="$CMD --model-type $MODEL_TYPE"
fi

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ "$FORCE_CPU" = true ]; then
    CMD="$CMD --force-cpu"
fi

# Run cleanup before testing if enabled
if [ "$CLEANUP_OLD_FILES" = true ]; then
    cleanup_old_test_files
fi

echo "=========================================="
echo "TESTING MODEL"
echo "=========================================="
echo "Model path: $MODEL_PATH"
if [ -n "$MODEL_TYPE" ]; then
    echo "Model type override: $MODEL_TYPE"
else
    echo "Model type: from config/model.json"
fi
echo "Excel report: $EXCEL_REPORT"
echo "Command: $CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "TESTING COMPLETE"
echo "Excel report saved to: $EXCEL_REPORT"
echo "=========================================="
