#!/bin/bash
# Run small dataset test from inside the runtime container

set -e

SPECIFIC_MODEL=""
VERBOSE=""
STANDALONE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="-v -s"
            shift
            ;;
        --standalone)
            STANDALONE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --model MODEL_TYPE"
            echo "  --verbose"
            echo "  --standalone"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "RUNNING SMALL DATASET TEST (IN CONTAINER)"
echo "=========================================="

if [ "$STANDALONE" = true ]; then
    CMD="python tests/test_small_dataset.py"
elif [ -n "$SPECIFIC_MODEL" ]; then
    CMD="pytest tests/test_small_dataset.py $VERBOSE -k test_full_pipeline[$SPECIFIC_MODEL]"
else
    CMD="pytest tests/test_small_dataset.py $VERBOSE"
fi

echo "Command: $CMD"
echo ""
eval $CMD

