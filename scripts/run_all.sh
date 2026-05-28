#!/bin/bash
# Run full pipeline script
# Usage: ./scripts/run_all.sh [options]
#
# Options:
#   --start-date DATE    Start date for data download (default: 2010-01-01)
#   --model-type TYPE    Model type (default: crnn_attention)
#   --epochs N           Number of epochs (default: 100)
#   --skip-preprocess    Skip preprocessing step
#   --skip-train         Skip training step
#   --skip-validate      Skip validation step
#   --skip-test          Skip testing step
#   --skip-backtest      Skip backtesting step
#   --help               Show this help message

set -e

# Default values
START_DATE="2000-01-01"
MODEL_TYPE="chronos2"
EPOCHS=100
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_VALIDATE=false
SKIP_TEST=false
SKIP_BACKTEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-validate)
            SKIP_VALIDATE=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-backtest)
            SKIP_BACKTEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --start-date DATE    Start date for data download (default: 2010-01-01)"
            echo "  --model-type TYPE    Model type (default: crnn_attention)"
            echo "  --epochs N           Number of epochs (default: 100)"
            echo "  --skip-preprocess    Skip preprocessing step"
            echo "  --skip-train         Skip training step"
            echo "  --skip-validate      Skip validation step"
            echo "  --skip-test          Skip testing step"
            echo "  --skip-backtest      Skip backtesting step"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

TOTAL_START=$(date +%s)

echo "=========================================="
echo "RUNNING FULL PIPELINE"
echo "=========================================="
echo "Start date: $START_DATE"
echo "Model type: $MODEL_TYPE"
echo "Epochs: $EPOCHS"
echo "=========================================="
echo ""

# Step 1: Preprocess
if [ "$SKIP_PREPROCESS" = false ]; then
    STEP_START=$(date +%s)
    echo "=========================================="
    echo "STEP 1: PREPROCESSING DATA"
    echo "=========================================="
    bash /app/scripts/preprocess.sh --start-date "$START_DATE"
    STEP_END=$(date +%s)
    echo "Preprocessing time: $((STEP_END - STEP_START))s"
    echo ""
else
    echo "Skipping preprocessing..."
    echo ""
fi

# Step 2: Train
if [ "$SKIP_TRAIN" = false ]; then
    STEP_START=$(date +%s)
    echo "=========================================="
    echo "STEP 2: TRAINING MODEL"
    echo "=========================================="
    bash /app/scripts/train.sh --model-type "$MODEL_TYPE" --epochs "$EPOCHS"
    STEP_END=$(date +%s)
    echo "Training time: $((STEP_END - STEP_START))s"
    echo ""
else
    echo "Skipping training..."
    echo ""
fi

# Step 3: Validate
if [ "$SKIP_VALIDATE" = false ]; then
    STEP_START=$(date +%s)
    echo "=========================================="
    echo "STEP 3: VALIDATING MODEL"
    echo "=========================================="
    bash /app/scripts/validate.sh
    STEP_END=$(date +%s)
    echo "Validation time: $((STEP_END - STEP_START))s"
    echo ""
else
    echo "Skipping validation..."
    echo ""
fi

# Step 4: Test
if [ "$SKIP_TEST" = false ]; then
    STEP_START=$(date +%s)
    echo "=========================================="
    echo "STEP 4: TESTING MODEL"
    echo "=========================================="
    bash /app/scripts/test.sh
    STEP_END=$(date +%s)
    echo "Testing time: $((STEP_END - STEP_START))s"
    echo ""
else
    echo "Skipping testing..."
    echo ""
fi

# Step 5: Backtest
if [ "$SKIP_BACKTEST" = false ]; then
    STEP_START=$(date +%s)
    echo "=========================================="
    echo "STEP 5: BACKTESTING MODEL"
    echo "=========================================="
    bash /app/scripts/backtest.sh
    STEP_END=$(date +%s)
    echo "Backtesting time: $((STEP_END - STEP_START))s"
    echo ""
else
    echo "Skipping backtesting..."
    echo ""
fi

TOTAL_END=$(date +%s)

echo "=========================================="
echo "FULL PIPELINE COMPLETE"
echo "=========================================="
echo "Total time: $((TOTAL_END - TOTAL_START))s"
echo "=========================================="
