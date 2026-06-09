#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Plot real_target vs predict_target from the latest Excel output inside the runtime container.

set -e

OUTPUT_DIR="outputs"
EXCEL_FILE=""
IMAGE_FILE=""
SPLIT_INDEX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --excel-file)
            EXCEL_FILE="$2"
            shift 2
            ;;
        --image-file)
            IMAGE_FILE="$2"
            shift 2
            ;;
        --split-index)
            SPLIT_INDEX="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --output-dir PATH     Directory containing Excel reports (default: outputs)"
            echo "  --excel-file PATH     Explicit Excel file to use"
            echo "  --image-file PATH     Explicit output image path"
            echo "  --split-index N       Chunk size for index splitting, e.g. 100 -> 0-99, 100-199"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="python scripts/plot_latest_output.py --output-dir $OUTPUT_DIR"

if [ -n "$EXCEL_FILE" ]; then
    CMD="$CMD --excel-file $EXCEL_FILE"
fi

if [ -n "$IMAGE_FILE" ]; then
    CMD="$CMD --image-file $IMAGE_FILE"
fi

if [ -n "$SPLIT_INDEX" ]; then
    CMD="$CMD --split-index $SPLIT_INDEX"
fi

echo "=========================================="
echo "PLOTTING LATEST OUTPUT (IN CONTAINER)"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="
echo ""

docker exec crnn_predictor bash -lc "$CMD"
