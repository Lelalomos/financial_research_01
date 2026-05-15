#!/bin/bash
# Backtest model from inside the runtime container

set -e

CONTAINER_NAME="crnn_predictor"
ORIGINAL_ARGS=("$@")
MODEL_PATH="final"
MODEL_TYPE=""
OUTPUT_PATH="outputs/backtest_report.xlsx"
DATA_DIR="data/processed"
SPLIT="test"
THRESHOLD=""
INITIAL_CAPITAL=""
DEVICE=""
FORCE_CPU=false

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
            echo "  --model PATH_OR_ALIAS   Checkpoint path or alias: best|final (default: best)"
            echo "  --model-type TYPE       Override model type from config/model.json"
            echo "  --output PATH           Output report path"
            echo "  --data-dir PATH         Processed data directory"
            echo "  --split NAME            train|val|test (default: test)"
            echo "  --threshold VALUE       Prediction threshold"
            echo "  --initial-capital NUM   Initial capital"
            echo "  --device DEVICE         Device override (e.g. cuda, cuda:0, cpu)"
            echo "  --force-cpu            Force CPU usage"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ! -f "/.dockerenv" ]; then
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "Container '$CONTAINER_NAME' is not running."
        echo "Start it with: docker compose up -d"
        exit 1
    fi

    echo "Host environment detected. Re-running inside container '$CONTAINER_NAME'..."
    exec docker exec -i "$CONTAINER_NAME" bash -lc 'cd /app && bash ./scripts/backtest_in_container.sh "$@"' -- "${ORIGINAL_ARGS[@]}"
fi

CMD=(python scripts/backtest.py --model "$MODEL_PATH" --data-dir "$DATA_DIR" --split "$SPLIT" --output "$OUTPUT_PATH")

if [ -n "$MODEL_TYPE" ]; then
    CMD+=(--model-type "$MODEL_TYPE")
fi

if [ -n "$THRESHOLD" ]; then
    CMD+=(--threshold "$THRESHOLD")
fi

if [ -n "$INITIAL_CAPITAL" ]; then
    CMD+=(--initial-capital "$INITIAL_CAPITAL")
fi

if [ -n "$DEVICE" ]; then
    CMD+=(--device "$DEVICE")
fi

if [ "$FORCE_CPU" = true ]; then
    CMD+=(--force-cpu)
fi

echo "=========================================="
echo "BACKTESTING MODEL (IN CONTAINER)"
echo "=========================================="
echo "Model: $MODEL_PATH"
if [ -n "$MODEL_TYPE" ]; then
    echo "Model type override: $MODEL_TYPE"
else
    echo "Model type: from config/model.json"
fi
echo "Split: $SPLIT"
echo "Output: $OUTPUT_PATH"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo ""
echo "=========================================="
echo ""

"${CMD[@]}"
