#!/bin/bash
# Export correlation analysis report for the active processed dataset.

set -e

CONTAINER_NAME="crnn_predictor"
ORIGINAL_ARGS=("$@")

PRE_DATA="data/pre_normalized.parquet"
NORMALIZED_DATA="data/normalized_data.parquet"
INFO_PATH="data/processed/info.json"
MAIN_CONFIG="config/main.json"
OUTPUT_DIR="outputs/correlation_analysis"
PAIR_THRESHOLD="0.95"
TOP_N="25"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pre-data)
            PRE_DATA="$2"
            shift 2
            ;;
        --normalized-data)
            NORMALIZED_DATA="$2"
            shift 2
            ;;
        --info)
            INFO_PATH="$2"
            shift 2
            ;;
        --main-config)
            MAIN_CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pair-threshold)
            PAIR_THRESHOLD="$2"
            shift 2
            ;;
        --top-n)
            TOP_N="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --pre-data PATH         Pre-normalized parquet dataset"
            echo "  --normalized-data PATH  Normalized parquet dataset"
            echo "  --info PATH             Processed dataset metadata JSON"
            echo "  --main-config PATH      Main config JSON"
            echo "  --output-dir PATH       Output directory for report bundle"
            echo "  --pair-threshold FLOAT  Absolute correlation threshold for feature pairs"
            echo "  --top-n N               Number of top target-correlated features to chart"
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
    exec docker exec -i "$CONTAINER_NAME" bash -lc 'cd /app && bash ./scripts/analyze_correlation_in_container.sh "$@"' -- "${ORIGINAL_ARGS[@]}"
fi

CMD=(
    python scripts/analyze_correlation.py
    --pre-data "$PRE_DATA"
    --normalized-data "$NORMALIZED_DATA"
    --info "$INFO_PATH"
    --main-config "$MAIN_CONFIG"
    --output-dir "$OUTPUT_DIR"
    --pair-threshold "$PAIR_THRESHOLD"
    --top-n "$TOP_N"
)

echo "=========================================="
echo "CORRELATION ANALYSIS (IN CONTAINER)"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo ""
echo "=========================================="
echo ""

"${CMD[@]}"
