#!/bin/bash
# Backtest model from inside the runtime container

set -e

MODEL_PATH="best"
MODEL_TYPE="crnn_attention"
OUTPUT_PATH="outputs/backtest_report.xlsx"
DATA_DIR="data/processed"
SPLIT="test"
THRESHOLD=""
INITIAL_CAPITAL=""

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
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --initial-capital)
            INITIAL_CAPITAL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --model PATH_OR_ALIAS   Checkpoint path or alias (default: best)"
            echo "  --model-type TYPE       Model type (default: crnn_attention)"
            echo "  --output PATH           Output report path"
            echo "  --data-dir PATH         Processed data directory"
            echo "  --split NAME            train|val|test (default: test)"
            echo "  --threshold VALUE       Prediction threshold"
            echo "  --initial-capital NUM   Initial capital"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="python scripts/backtest.py --model $MODEL_PATH --model-type $MODEL_TYPE --data-dir $DATA_DIR --split $SPLIT --output $OUTPUT_PATH"

if [ -n "$THRESHOLD" ]; then
    CMD="$CMD --threshold $THRESHOLD"
fi

if [ -n "$INITIAL_CAPITAL" ]; then
    CMD="$CMD --initial-capital $INITIAL_CAPITAL"
fi

echo "=========================================="
echo "BACKTESTING MODEL (IN CONTAINER)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Model type: $MODEL_TYPE"
echo "Split: $SPLIT"
echo "Output: $OUTPUT_PATH"
echo "Command: $CMD"
echo "=========================================="
echo ""

eval $CMD

