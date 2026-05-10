#!/bin/bash
# Optuna tuning from inside the runtime container

set -e

MODEL_TYPE="bilstm4_attention"
N_TRIALS=50
STOCKS=20
YEARS=""
MAX_EPOCHS=50
CREATE_DATASET=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --stocks)
            STOCKS="$2"
            shift 2
            ;;
        --years)
            YEARS="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --no-create-dataset)
            CREATE_DATASET=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --model-type TYPE"
            echo "  --n-trials N"
            echo "  --stocks N"
            echo "  --years N"
            echo "  --max-epochs N"
            echo "  --no-create-dataset"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="python scripts/optuna_tune.py --model-type $MODEL_TYPE --n-trials $N_TRIALS --stocks $STOCKS --max-epochs $MAX_EPOCHS"

if [ -n "$YEARS" ]; then
    CMD="$CMD --years $YEARS"
fi
if [ "$CREATE_DATASET" = true ]; then
    CMD="$CMD --create-dataset"
fi

echo "=========================================="
echo "OPTUNA TUNING (IN CONTAINER)"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="
echo ""

eval $CMD

