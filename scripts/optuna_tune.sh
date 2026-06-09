#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Optuna hyperparameter tuning shell script
#
# Usage:
#   bash scripts/optuna_tune.sh [options]
#
# Options:
#   --model-type TYPE     Model type to tune (default: bilstm4_attention)
#   --n-trials N           Number of Optuna trials (default: 50)
#   --stocks N             Number of stocks for dataset (default: 20)
#   --years N              Number of years (default: ALL)
#   --create-dataset       Create small dataset before tuning
#   --max-epochs N         Max epochs per trial (default: 50)

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"

# Default values
MODEL_TYPE=""
N_TRIALS=50
STOCKS=20
YEARS=""
MAX_EPOCHS=50
CREATE_DATASET=true

# Parse arguments
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODEL_TYPE="$(resolve_model_type "$MODEL_TYPE")"

# Build command
CMD="python scripts/optuna_tune.py"
CMD="$CMD --model-type $MODEL_TYPE"
CMD="$CMD --n-trials $N_TRIALS"
CMD="$CMD --stocks $STOCKS"
CMD="$CMD --max-epochs $MAX_EPOCHS"

if [ -n "$YEARS" ]; then
    CMD="$CMD --years $YEARS"
fi

if [ "$CREATE_DATASET" = true ]; then
    CMD="$CMD --create-dataset"
fi

echo "=========================================="
echo "OPTUNA HYPERPARAMETER TUNING"
echo "=========================================="
echo "Model type: $MODEL_TYPE"
echo "Trials: $N_TRIALS"
echo "Stocks: $STOCKS"
echo "Years: ${YEARS:-ALL}"
echo "Max epochs: $MAX_EPOCHS"
echo "Create dataset: $CREATE_DATASET"
echo "=========================================="
echo ""
echo "Running: $CMD"
echo ""

# Run command
python $CMD

exit $?
