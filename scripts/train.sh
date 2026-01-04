#!/bin/bash
# Train model script
# Usage: ./scripts/train.sh [options]
#
# Options:
#   --model-type TYPE     Model type: crnn, rnn, rnn_attention, crnn_attention, transformer (default: crnn_attention)
#   --epochs N            Number of epochs (default: 100)
#   --batch-size N        Batch size (default: 256)
#   --learning-rate RATE  Learning rate (default: 1e-4)
#   --stocks T1 T2 ...    Fine-tune on specific stocks (e.g., AAPL MSFT GOOGL)
#   --fine-tune PATH      Path to checkpoint to fine-tune from
#   --freeze-embeddings   Freeze stock/group embeddings during fine-tuning
#   --help                Show this help message

set -e

# Default values
MODEL_TYPE="bilstm4_attention"
EPOCHS=5
BATCH_SIZE=256
LEARNING_RATE=0.00001
STOCKS=""
FINE_TUNE=""
FREEZE_EMBEDDINGS=""

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
            echo "  --model-type TYPE     Model type (default: crnn_attention)"
            echo "  --epochs N            Number of epochs (default: 100)"
            echo "  --batch-size N        Batch size (default: 256)"
            echo "  --learning-rate RATE  Learning rate (default: 1e-4)"
            echo "  --stocks T1 T2 ...    Fine-tune on specific stocks (e.g., AAPL MSFT)"
            echo "  --fine-tune PATH      Path to checkpoint to fine-tune from"
            echo "  --freeze-embeddings   Freeze stock/group embeddings during fine-tuning"
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

# Build command
CMD="python scripts/train.py --model-type $MODEL_TYPE --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LEARNING_RATE"

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
if [ -n "$STOCKS" ]; then
    echo "Stocks: $STOCKS"
fi
if [ -n "$FINE_TUNE" ]; then
    echo "Fine-tune from: $FINE_TUNE"
fi
if [ -n "$FREEZE_EMBEDDINGS" ]; then
    echo "Freeze embeddings: yes"
fi
echo "Command: $CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
