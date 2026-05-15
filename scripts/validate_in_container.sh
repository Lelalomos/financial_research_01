#!/bin/bash
# Validate model from inside the runtime container

set -e

MODEL_PATH="best"
MODEL_TYPE=""
DATA_DIR="data/processed"
SPLIT="val"
DEVICE=""
FORCE_CPU=false
EXCEL_REPORT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
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
        --data-split|--split)
            SPLIT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --excel-report)
            EXCEL_REPORT="$2"
            shift 2
            ;;
        --force-cpu)
            FORCE_CPU=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --model PATH_OR_ALIAS   Checkpoint path or alias: best|final"
            echo "  --model-type TYPE      Override model type from config/model.json"
            echo "  --data-dir PATH"
            echo "  --data-split NAME"
            echo "  --device DEVICE       Device override (e.g. cuda, cuda:0, cpu)"
            echo "  --excel-report PATH   Write detailed Excel validation report"
            echo "  --force-cpu           Force CPU usage"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="python scripts/validate.py --model $MODEL_PATH --data-dir $DATA_DIR --split $SPLIT"

if [ -n "$MODEL_TYPE" ]; then
    CMD="$CMD --model-type $MODEL_TYPE"
fi

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$EXCEL_REPORT" ]; then
    CMD="$CMD --excel-report $EXCEL_REPORT"
fi

if [ "$FORCE_CPU" = true ]; then
    CMD="$CMD --force-cpu"
fi

echo "=========================================="
echo "VALIDATING MODEL (IN CONTAINER)"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="
echo ""

eval $CMD
