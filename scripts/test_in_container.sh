#!/bin/bash
# Test model from inside the runtime container

set -e

MODEL_PATH="models/bilstm4_attention_best_20260105_113045.pth"
REPORTS_DIR="reports"
CLEANUP_OLD_FILES=true

mkdir -p "$REPORTS_DIR"

cleanup_old_test_files() {
    python -c "from src.utils.cleanup import cleanup_test_files; cleanup_test_files(keep_latest=3, verbose=True)"
}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXCEL_REPORT="$REPORTS_DIR/test_report_${TIMESTAMP}.xlsx"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --excel-report)
            EXCEL_REPORT="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP_OLD_FILES=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --model PATH"
            echo "  --excel-report PATH"
            echo "  --no-cleanup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="python scripts/test.py --model $MODEL_PATH --excel-report $EXCEL_REPORT"

if [ "$CLEANUP_OLD_FILES" = true ]; then
    cleanup_old_test_files
fi

echo "=========================================="
echo "TESTING MODEL (IN CONTAINER)"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="
echo ""

eval $CMD

