#!/bin/bash
# Run full pipeline from inside the runtime container

set -e

START_DATE="2000-01-01"
MODEL_TYPE="chronos2"
EPOCHS=100
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_VALIDATE=false
SKIP_TEST=false
SKIP_BACKTEST=false

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
            echo "  --start-date DATE"
            echo "  --model-type TYPE"
            echo "  --epochs N"
            echo "  --skip-preprocess"
            echo "  --skip-train"
            echo "  --skip-validate"
            echo "  --skip-test"
            echo "  --skip-backtest"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TOTAL_START=$(date +%s)

if [ "$SKIP_PREPROCESS" = false ]; then
    bash scripts/preprocess_in_container.sh --start-date "$START_DATE"
fi

if [ "$SKIP_TRAIN" = false ]; then
    bash scripts/train_in_container.sh --model-type "$MODEL_TYPE" --epochs "$EPOCHS"
fi

if [ "$SKIP_VALIDATE" = false ]; then
    bash scripts/validate_in_container.sh
fi

if [ "$SKIP_TEST" = false ]; then
    bash scripts/test_in_container.sh
fi

if [ "$SKIP_BACKTEST" = false ]; then
    bash scripts/backtest_in_container.sh
fi

TOTAL_END=$(date +%s)
echo "=========================================="
echo "FULL PIPELINE COMPLETE (IN CONTAINER)"
echo "=========================================="
echo "Total time: $((TOTAL_END - TOTAL_START))s"
echo "=========================================="
