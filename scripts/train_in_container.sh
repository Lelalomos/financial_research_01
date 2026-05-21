#!/bin/bash
# Train model from inside the runtime container
# Usage: ./scripts/train_in_container.sh [options]
#
# Options:
#   --model-type TYPE     Override model type from config/model.json
#   --epochs N            Number of epochs (default: 30)
#   --batch-size N        Batch size (default: 32)
#   --learning-rate RATE  Learning rate (default: 1e-4)
#   --backend TYPE        Training backend: lightning or custom (default: lightning)
#   --max-train-batches N Optional limit for train batches per epoch
#   --max-val-batches N   Optional limit for validation batches per epoch
#   --device DEV          Device to use: cuda or cpu (default: auto-detect)
#   --force-cpu           Force CPU usage even if GPU is available
#   --stocks T1 T2 ...    Fine-tune on specific stocks (e.g., AAPL MSFT GOOGL)
#   --fine-tune PATH      Path to checkpoint to fine-tune from
#   --freeze-embeddings   Freeze stock/group embeddings during fine-tuning
#   --monitor             Start TensorBoard inside the container
#   --mlflow              Start MLflow UI inside the container
#   --monitor-all         Start both TensorBoard and MLflow UI inside the container
#   --tensorboard-port N  TensorBoard port (default: 6006)
#   --mlflow-port N       MLflow UI port (default: 5000)
#   --help                Show this help message

set -e
CONTAINER_NAME="crnn_predictor"
ORIGINAL_ARGS=("$@")
# bilstm4_attention_best_lightning.pth
MODEL_TYPE="kronos"
EPOCHS=10
BATCH_SIZE=128
LEARNING_RATE=0.0001
BACKEND="lightning"
STOCKS=""
FINE_TUNE=""
FREEZE_EMBEDDINGS=""
DEVICE=""
FORCE_CPU=""
MAX_TRAIN_BATCHES=""
MAX_VAL_BATCHES=""
START_TENSORBOARD=1
START_MLFLOW=1
TENSORBOARD_PORT=6006
MLFLOW_PORT=5000

export PATH="$HOME/.local/bin:$PATH"

if [ ! -f "/.dockerenv" ]; then
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "Container '$CONTAINER_NAME' is not running."
        echo "Start it with: docker compose up -d"
        exit 1
    fi

    echo "Host environment detected. Re-running inside container '$CONTAINER_NAME'..."
    exec docker exec -i "$CONTAINER_NAME" bash -lc 'cd /app && bash ./scripts/train_in_container.sh "$@"' -- "${ORIGINAL_ARGS[@]}"
fi

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
        --max-train-batches)
            MAX_TRAIN_BATCHES="$2"
            shift 2
            ;;
        --max-val-batches)
            MAX_VAL_BATCHES="$2"
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
            STOCK_VALUES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                STOCK_VALUES+=("$1")
                shift
            done
            STOCKS="${STOCK_VALUES[*]}"
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
            echo "  --epochs N            Number of epochs (default: 30)"
            echo "  --batch-size N        Batch size (default: 32)"
            echo "  --learning-rate RATE  Learning rate (default: 1e-4)"
            echo "  --backend TYPE        Training backend: lightning or custom (default: lightning)"
            echo "  --max-train-batches N Optional limit for train batches per epoch"
            echo "  --max-val-batches N   Optional limit for validation batches per epoch"
            echo "  --device DEV          Device to use: cuda or cpu (default: auto-detect)"
            echo "  --force-cpu           Force CPU usage even if GPU is available"
            echo "  --stocks T1 T2 ...    Fine-tune on specific stocks (e.g., AAPL MSFT)"
            echo "  --fine-tune PATH      Path to checkpoint to fine-tune from"
            echo "  --freeze-embeddings   Freeze stock/group embeddings during fine-tuning"
            echo "  --monitor             Start TensorBoard inside the container"
            echo "  --mlflow              Start MLflow UI inside the container"
            echo "  --monitor-all         Start both TensorBoard and MLflow UI inside the container"
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

start_tensorboard() {
    if ! command -v tensorboard >/dev/null 2>&1; then
        echo "TensorBoard is not installed inside the container."
        return
    fi

    if is_local_port_open "$TENSORBOARD_PORT"; then
        echo "TensorBoard is already running on port $TENSORBOARD_PORT"
        return
    fi

    nohup tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port "$TENSORBOARD_PORT" >/tmp/research_02_tensorboard_in_container.log 2>&1 &
    echo "TensorBoard started: http://127.0.0.1:$TENSORBOARD_PORT"
}

start_mlflow_ui() {
    ensure_mlflow_installed

    if python - <<'PY' >/dev/null 2>&1
import json
with open("config/model.json", "r") as f:
    data = json.load(f)
enabled = data.get("model", {}).get("experiment_tracking", {}).get("ENABLED", False)
raise SystemExit(0 if enabled else 1)
PY
    then
        :
    else
        echo "Warning: MLflow UI is starting, but model.experiment_tracking.ENABLED is false in config/model.json."
    fi

    if is_local_port_open "$MLFLOW_PORT"; then
        echo "MLflow UI is already running on port $MLFLOW_PORT"
        return
    fi

    nohup mlflow ui --backend-store-uri file:/app/mlruns --host 0.0.0.0 --port "$MLFLOW_PORT" >/tmp/research_02_mlflow_in_container.log 2>&1 &
    echo "MLflow UI started: http://127.0.0.1:$MLFLOW_PORT"
}

ensure_mlflow_installed() {
    if command -v mlflow >/dev/null 2>&1; then
        return
    fi

    echo "MLflow CLI not found. Installing mlflow for the current container user..."
    python -m pip install --user mlflow >/tmp/research_02_mlflow_install.log 2>&1

    if ! command -v mlflow >/dev/null 2>&1; then
        echo "MLflow installation failed. See /tmp/research_02_mlflow_install.log"
        exit 1
    fi
}

is_local_port_open() {
    python - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    sock.close()
PY
}

CMD=(python scripts/train.py --backend "$BACKEND" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LEARNING_RATE")

if [ -n "$FORCE_CPU" ]; then
    CMD+=(--force-cpu)
fi

if [ -n "$MODEL_TYPE" ]; then
    CMD+=(--model-type "$MODEL_TYPE")
fi

if [ -n "$DEVICE" ]; then
    CMD+=(--device "$DEVICE")
fi

if [ -n "$MAX_TRAIN_BATCHES" ]; then
    CMD+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi

if [ -n "$MAX_VAL_BATCHES" ]; then
    CMD+=(--max-val-batches "$MAX_VAL_BATCHES")
fi

if [ -n "$STOCKS" ]; then
    read -r -a STOCK_ARGS <<< "$STOCKS"
    CMD+=(--stocks "${STOCK_ARGS[@]}")
fi

if [ -n "$FINE_TUNE" ]; then
    CMD+=(--fine-tune "$FINE_TUNE")
fi

if [ -n "$FREEZE_EMBEDDINGS" ]; then
    CMD+=(--freeze-embeddings)
fi

echo "=========================================="
echo "TRAINING MODEL (IN CONTAINER)"
echo "=========================================="
if [ -n "$MODEL_TYPE" ]; then
    echo "Model type override: $MODEL_TYPE"
else
    echo "Model type: from config/model.json"
fi
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Backend: $BACKEND"
if [ -n "$DEVICE" ]; then
    echo "Device: $DEVICE"
fi
if [ -n "$MAX_TRAIN_BATCHES" ]; then
    echo "Max train batches: $MAX_TRAIN_BATCHES"
fi
if [ -n "$MAX_VAL_BATCHES" ]; then
    echo "Max val batches: $MAX_VAL_BATCHES"
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
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo ""
echo "=========================================="
echo ""

if [ "$START_TENSORBOARD" -eq 1 ]; then
    start_tensorboard
fi

if [ "$START_MLFLOW" -eq 1 ]; then
    start_mlflow_ui
fi

"${CMD[@]}"

echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
