#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Train model script
# Usage: ./scripts/train.sh [options]
#
# Options:
#   --model-type TYPE     Override model type from config/model.json
#   --epochs N            Number of epochs (default: 100)
#   --batch-size N        Batch size (default: 256)
#   --learning-rate RATE  Learning rate (default: 1e-4)
#   --backend TYPE        Training backend: lightning or custom (default: lightning)
#   --data-dir PATH       Processed data directory override
#   --device DEV          Device to use: cuda or cpu (default: auto-detect)
#   --force-cpu           Force CPU usage even if GPU is available
#   --stocks T1 T2 ...    Fine-tune on specific stocks (e.g., AAPL MSFT GOOGL)
#   --fine-tune PATH      Path to checkpoint to fine-tune from
#   --freeze-embeddings   Freeze stock/group embeddings during fine-tuning
#   --monitor             Start TensorBoard on the host
#   --mlflow              Start MLflow UI on the host
#   --monitor-all         Start both TensorBoard and MLflow UI on the host
#   --tensorboard-port N  TensorBoard port (default: 6006)
#   --mlflow-port N       MLflow UI port (default: 5000)
#   --help                Show this help message

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"

# Default values
MODEL_TYPE=""
EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
BACKEND="lightning"
DATA_DIR=""
STOCKS=""
FINE_TUNE=""
FREEZE_EMBEDDINGS=""
DEVICE=""
FORCE_CPU=""
START_TENSORBOARD=0
START_MLFLOW=0
TENSORBOARD_PORT=6006
MLFLOW_PORT=5000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --force-cpu)
            FORCE_CPU="--force-cpu"
            shift
            ;;
        --monitor)
            START_TENSORBOARD=1
            shift
            ;;
        --mlflow)
            START_MLFLOW=1
            shift
            ;;
        --monitor-all)
            START_TENSORBOARD=1
            START_MLFLOW=1
            shift
            ;;
        --tensorboard-port)
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        --mlflow-port)
            MLFLOW_PORT="$2"
            shift 2
            ;;
        --stocks)
            shift
            STOCKS="$@"
            break
            ;;
        --fine-tune)
            FINE_TUNE="$2"
            shift 2
            ;;
        --freeze-embeddings)
            FREEZE_EMBEDDINGS="--freeze-embeddings"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-type TYPE     Override model type from config/model.json"
            echo "  --epochs N            Number of epochs (default: 100)"
            echo "  --batch-size N        Batch size (default: 256)"
            echo "  --learning-rate RATE  Learning rate (default: 1e-4)"
            echo "  --backend TYPE        Training backend: lightning or custom (default: lightning)"
            echo "  --data-dir PATH       Processed data directory override"
            echo "  --device DEV          Device to use: cuda or cpu (default: auto-detect)"
            echo "  --force-cpu           Force CPU usage even if GPU is available"
            echo "  --stocks T1 T2 ...    Fine-tune on specific stocks (e.g., AAPL MSFT)"
            echo "  --fine-tune PATH      Path to checkpoint to fine-tune from"
            echo "  --freeze-embeddings   Freeze stock/group embeddings during fine-tuning"
            echo "  --monitor             Start TensorBoard on the host"
            echo "  --mlflow              Start MLflow UI on the host"
            echo "  --monitor-all         Start both TensorBoard and MLflow UI on the host"
            echo "  --tensorboard-port N  TensorBoard port (default: 6006)"
            echo "  --mlflow-port N       MLflow UI port (default: 5000)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

MODEL_TYPE="$(resolve_model_type "$MODEL_TYPE")"

if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$(resolve_data_dir_for_model_type "$MODEL_TYPE")"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

start_tensorboard() {
    if ! command -v tensorboard >/dev/null 2>&1; then
        echo "TensorBoard not found on host. Install it first to use --monitor."
        return
    fi

    if pgrep -f "tensorboard.*--logdir[= ]$REPO_ROOT/logs/tensorboard.*--port[= ]$TENSORBOARD_PORT" >/dev/null 2>&1; then
        echo "TensorBoard is already running on port $TENSORBOARD_PORT"
        return
    fi

    nohup tensorboard --logdir "$REPO_ROOT/logs/tensorboard" --host 127.0.0.1 --port "$TENSORBOARD_PORT" >/tmp/research_02_tensorboard.log 2>&1 &
    echo "TensorBoard started: http://127.0.0.1:$TENSORBOARD_PORT"
}

start_mlflow_ui() {
    if ! command -v mlflow >/dev/null 2>&1; then
        echo "MLflow CLI not found on host. Install it first to use --mlflow."
        return
    fi

    if pgrep -f "mlflow ui.*--backend-store-uri[= ]$REPO_ROOT/mlruns.*--port[= ]$MLFLOW_PORT" >/dev/null 2>&1; then
        echo "MLflow UI is already running on port $MLFLOW_PORT"
        return
    fi

    nohup mlflow ui --backend-store-uri "$REPO_ROOT/mlruns" --host 127.0.0.1 --port "$MLFLOW_PORT" >/tmp/research_02_mlflow.log 2>&1 &
    echo "MLflow UI started: http://127.0.0.1:$MLFLOW_PORT"
}

# Build command
CMD="docker exec crnn_predictor python scripts/train.py --backend $BACKEND --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LEARNING_RATE --data-dir $DATA_DIR $FORCE_CPU"

if [ -n "$MODEL_TYPE" ]; then
    CMD="$CMD --model-type $MODEL_TYPE"
fi

# Add device if specified
if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$STOCKS" ]; then
    CMD="$CMD --stocks $STOCKS"
fi

if [ -n "$FINE_TUNE" ]; then
    CMD="$CMD --fine-tune $FINE_TUNE"
fi

if [ -n "$FREEZE_EMBEDDINGS" ]; then
    CMD="$CMD $FREEZE_EMBEDDINGS"
fi

echo "=========================================="
echo "TRAINING MODEL"
echo "=========================================="
echo "Model type: $MODEL_TYPE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Backend: $BACKEND"
echo "Data dir: $DATA_DIR"
if [ -n "$DEVICE" ]; then
    echo "Device: $DEVICE"
fi
if [ -n "$FORCE_CPU" ]; then
    echo "Force CPU: yes"
fi
if [ -n "$STOCKS" ]; then
    echo "Stocks: $STOCKS"
fi
if [ -n "$FINE_TUNE" ]; then
    echo "Fine-tune from: $FINE_TUNE"
fi
if [ -n "$FREEZE_EMBEDDINGS" ]; then
    echo "Freeze embeddings: yes"
fi
if [ "$START_TENSORBOARD" -eq 1 ]; then
    echo "TensorBoard monitor: http://127.0.0.1:$TENSORBOARD_PORT"
fi
if [ "$START_MLFLOW" -eq 1 ]; then
    echo "MLflow monitor: http://127.0.0.1:$MLFLOW_PORT"
fi
echo "Command: $CMD"
echo "=========================================="
echo ""

if [ "$START_TENSORBOARD" -eq 1 ]; then
    start_tensorboard
fi

if [ "$START_MLFLOW" -eq 1 ]; then
    start_mlflow_ui
fi

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
