#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Run unit tests from inside the runtime container

set -e

COVERAGE=false
VERBOSE=""
SPECIFIC_MODEL=""

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
            echo "  --coverage"
            echo "  --verbose"
            echo "  --model MODEL_TYPE"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -n "$SPECIFIC_MODEL" ]; then
    CMD="pytest tests/test_models.py $VERBOSE -k $SPECIFIC_MODEL"
elif [ "$COVERAGE" = true ]; then
    CMD="pytest tests/test_models.py $VERBOSE --cov=src/models --cov-report=html --cov-report=term"
else
    CMD="pytest tests/test_models.py $VERBOSE"
fi

echo "=========================================="
echo "RUNNING UNIT TESTS (IN CONTAINER)"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="
echo ""

eval $CMD

