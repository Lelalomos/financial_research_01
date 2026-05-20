#!/bin/bash
# Preprocess data from inside the runtime container

set -e

START_DATE="2000-01-01"
END_DATE=""
STOCK_LIMIT=""
STOCKS="150"
TICKERS=""
EXPORT_PRE_NORMALIZE=""
SKIP_DOWNLOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --stock-limit)
            STOCK_LIMIT="$2"
            shift 2
            ;;
        --stocks)
            STOCKS="$2"
            shift 2
            ;;
        --tickers)
            shift
            TICKERS="$@"
            break
            ;;
        --export-pre-normalize)
            EXPORT_PRE_NORMALIZE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --start-date DATE"
            echo "  --end-date DATE"
            echo "  --stock-limit N"
            echo "  --stocks N"
            echo "  --tickers T1 T2 ..."
            echo "  --export-pre-normalize PATH"
            echo "  --skip-download"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="python scripts/preprocess_data.py --start-date $START_DATE $SKIP_DOWNLOAD"

if [ -n "$END_DATE" ]; then
    CMD="$CMD --end-date $END_DATE"
fi
if [ -n "$STOCK_LIMIT" ]; then
    CMD="$CMD --stock-limit $STOCK_LIMIT"
fi
if [ -n "$STOCKS" ]; then
    CMD="$CMD --stocks $STOCKS"
fi
if [ -n "$TICKERS" ]; then
    CMD="$CMD --tickers $TICKERS"
fi
if [ -n "$EXPORT_PRE_NORMALIZE" ]; then
    CMD="$CMD --export-pre-normalize $EXPORT_PRE_NORMALIZE"
fi

echo "=========================================="
echo "PREPROCESSING DATA (IN CONTAINER)"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="
echo ""

eval $CMD

