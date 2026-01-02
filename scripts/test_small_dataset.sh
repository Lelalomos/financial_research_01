#!/bin/bash
# Run small dataset test script
# Usage: ./scripts/test_small_dataset.sh [options]
#
# Options:
#   --model MODEL_TYPE  Test specific model only (default: all)
#   --verbose           Show detailed test output
#   --standalone        Run as standalone Python script (pytest mode by default)
#   --help              Show this help message

set -e

SPECIFIC_MODEL=""
VERBOSE=""
STANDALONE=false

# Parse arguments
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
            echo ""
            echo "Options:"
            echo "  --model MODEL_TYPE  Test specific model only (crnn, rnn, rnn_attention, ...)"
            echo "  --verbose           Show detailed test output"
            echo "  --standalone        Run as standalone Python script (pytest mode by default)"
            echo "  --help              Show this help message"
            echo ""
            echo "Available models: crnn, rnn, rnn_attention, crnn_attention, transformer, lstm3, lstm3_attention"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "RUNNING SMALL DATASET FULL FLOW TEST"
echo "=========================================="

if [ "$STANDALONE" = true ]; then
    echo "Running as standalone Python script..."
    echo ""
    python tests/test_small_dataset.py
else
    if [ -n "$SPECIFIC_MODEL" ]; then
        echo "Testing model: $SPECIFIC_MODEL"
        echo ""
        pytest tests/test_small_dataset.py $VERBOSE -k "test_full_pipeline[$SPECIFIC_MODEL]"
    else
        echo "Testing all models with pytest..."
        echo ""
        pytest tests/test_small_dataset.py $VERBOSE
    fi
fi

echo ""
echo "=========================================="
echo "SMALL DATASET TEST COMPLETE"
echo "=========================================="
