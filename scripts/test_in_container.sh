#!/bin/bash
# Test model from inside the runtime container

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"

MODEL_PATH="best"
REPORTS_DIR="outputs"
CLEANUP_OLD_FILES=true
MODEL_TYPE=""
DATA_DIR=""
DEVICE=""
FORCE_CPU=false
MAX_SAMPLES=""

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
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP_OLD_FILES=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --model PATH_OR_ALIAS   Checkpoint path or alias: best|final"
            echo "  --excel-report PATH"
            echo "  --model-type TYPE"
            echo "  --data-dir PATH"
            echo "  --device DEVICE"
            echo "  --force-cpu"
            echo "  --max-samples N"
            echo "  --no-cleanup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODEL_TYPE="$(resolve_model_type "$MODEL_TYPE")"

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$(resolve_data_dir_for_model_type "$MODEL_TYPE")"
fi

CMD="python scripts/test.py --model $MODEL_PATH --excel-report $EXCEL_REPORT --data-dir $DATA_DIR"

if [ -n "$MODEL_TYPE" ]; then
    CMD="$CMD --model-type $MODEL_TYPE"
fi

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ "$FORCE_CPU" = true ]; then
    CMD="$CMD --force-cpu"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

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
