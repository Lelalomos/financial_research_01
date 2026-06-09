#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Comprehensive test runner script
# Usage: ./scripts/test_all.sh [options]
#
# Options:
#   --unit-only         Run only unit tests
#   --full-only         Run only full flow tests
#   --model MODEL_TYPE  Test specific model type only
#   --coverage          Run with coverage report
#   --verbose           Show detailed test output
#   --help              Show this help message

set -e

# Default values
RUN_UNIT=true
RUN_FULL=true
SPECIFIC_MODEL=""
COVERAGE=false
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit-only)
            RUN_FULL=false
            shift
            ;;
        --full-only)
            RUN_UNIT=false
            shift
            ;;
        --model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE="-v -s"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --unit-only         Run only unit tests"
            echo "  --full-only         Run only full flow tests"
            echo "  --model MODEL_TYPE  Test specific model (crnn, rnn, rnn_attention, ...)"
            echo "  --coverage          Run with coverage report"
            echo "  --verbose           Show detailed test output"
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
echo "COMPREHENSIVE TEST RUNNER"
echo "=========================================="

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 1: UNIT TESTS"
    echo "=========================================="

    if [ -n "$SPECIFIC_MODEL" ]; then
        echo "Testing model: $SPECIFIC_MODEL"
        CMD="pytest tests/test_models.py $VERBOSE -k \"test_$SPECIFIC_MODEL or $SPECIFIC_MODEL\""
    else
        echo "Testing all models"
        if [ "$COVERAGE" = true ]; then
            CMD="pytest tests/test_models.py $VERBOSE --cov=src/models --cov-report=html --cov-report=term"
        else
            CMD="pytest tests/test_models.py $VERBOSE"
        fi
    fi

    echo "Command: $CMD"
    eval $CMD
fi

# Run full flow tests
if [ "$RUN_FULL" = true ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 2: FULL FLOW TESTS"
    echo "=========================================="

    if [ -n "$SPECIFIC_MODEL" ]; then
        echo "Testing full pipeline for: $SPECIFIC_MODEL"
        CMD="pytest tests/test_small_dataset.py $VERBOSE -k \"test_full_pipeline[$SPECIFIC_MODEL]\""
    else
        echo "Testing full pipeline for all models"
        CMD="pytest tests/test_small_dataset.py $VERBOSE"
    fi

    echo "Command: $CMD"
    eval $CMD
fi

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="
