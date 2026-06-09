#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Run unit tests script
# Usage: ./scripts/test_unit.sh [options]
#
# Options:
#   --coverage          Run with coverage report
#   --verbose           Show detailed test output
#   --model MODEL_TYPE  Test specific model only (crnn, rnn, rnn_attention, ...)
#   --help              Show this help message

set -e

# Default values
COVERAGE=false
VERBOSE=""
SPECIFIC_MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE="-v -s"
            shift
            ;;
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --coverage          Run with coverage report"
            echo "  --verbose           Show detailed test output"
            echo "  --model MODEL_TYPE  Test specific model only"
            echo "  --help              Show this help message"
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
echo "RUNNING UNIT TESTS FOR ALL MODELS"
echo "=========================================="

if [ -n "$SPECIFIC_MODEL" ]; then
    echo "Testing model: $SPECIFIC_MODEL"
    CMD="pytest tests/test_models.py $VERBOSE -k \"$SPECIFIC_MODEL\""
elif [ "$COVERAGE" = true ]; then
    echo "Running with coverage report..."
    CMD="pytest tests/test_models.py $VERBOSE --cov=src/models --cov-report=html --cov-report=term"
else
    CMD="pytest tests/test_models.py $VERBOSE"
fi

echo "Command: $CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "UNIT TESTS COMPLETE"
echo "=========================================="
