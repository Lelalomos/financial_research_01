#!/bin/bash
# Preprocess data script
# Usage: ./scripts/preprocess.sh [options]
#
# Options:
#   --start-date DATE    Start date for data download (default: 2000-01-01)
#   --end-date DATE      End date for data download (default: current)
#   --stock-limit N      Limit number of stocks from index (e.g., 400)
#   --stocks N           Sample N stocks balanced across ALL group_ids
#   --tickers T1 T2 ...  Specific tickers to download
#   --export-pre-normalize PATH  Export pre-normalization data to this path (parquet format)
#   --help               Show this help message

set -e

# Default values
START_DATE="2000-01-01"
END_DATE=""
STOCK_LIMIT=""
STOCKS="150"
TICKERS=""
EXPORT_PRE_NORMALIZE=""

# Parse arguments
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
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --start-date DATE    Start date for data download (default: 2000-01-01)"
            echo "  --end-date DATE      End date for data download (default: current)"
            echo "  --stock-limit N      Limit number of stocks from index (e.g., 400 for first 400)"
            echo "  --stocks N           Sample N stocks balanced across ALL group_ids"
            echo "  --tickers T1 T2 ...  Specific tickers to download (e.g., AAPL MSFT GOOGL)"
            echo "  --export-pre-normalize PATH  Export pre-normalization data to this path"
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

# Build command
CMD="python scripts/preprocess_data.py --start-date $START_DATE"

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
echo "PREPROCESSING DATA"
echo "=========================================="
echo "Start date: $START_DATE"
if [ -n "$END_DATE" ]; then
    echo "End date: $END_DATE"
fi
if [ -n "$STOCK_LIMIT" ]; then
    echo "Stock limit: $STOCK_LIMIT"
fi
if [ -n "$STOCKS" ]; then
    echo "Stocks (balanced sampling): $STOCKS"
fi
if [ -n "$TICKERS" ]; then
    echo "Tickers: $TICKERS"
fi
if [ -n "$EXPORT_PRE_NORMALIZE" ]; then
    echo "Export pre-normalize: $EXPORT_PRE_NORMALIZE"
fi
echo "Command: $CMD"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "=========================================="
echo "PREPROCESSING COMPLETE"
echo "=========================================="
