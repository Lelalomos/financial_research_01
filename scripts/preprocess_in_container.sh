#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi

# Preprocess data from inside the runtime container

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"

START_DATE="2000-01-01"
END_DATE=""
MODEL_TYPE="chronos_rich"
STOCK_LIMIT=""
STOCKS="50"
TICKERS=""
EXPORT_PRE_NORMALIZE=""
EXPORT_NORMALIZED=""
SKIP_DOWNLOAD=""
NO_RESUME_CACHE=""

ensure_python_module_installed() {
    local module_name="$1"
    local package_spec="${2:-$1}"
    local install_log="/tmp/research_02_${module_name}_install.log"

    if python - "$module_name" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
    then
        return
    fi

    echo "Python module '$module_name' not found. Installing $package_spec for the current container user..."
    python -m pip install --user "$package_spec" >"$install_log" 2>&1

    if ! python - "$module_name" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
    then
        echo "Installation of Python module '$module_name' failed. See $install_log"
        exit 1
    fi
}

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
        --model-type)
            MODEL_TYPE="$2"
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
            STOCKS=""
            TICKERS="$@"
            break
            ;;
        --export-pre-normalize)
            EXPORT_PRE_NORMALIZE="$2"
            shift 2
            ;;
        --export-normalized)
            EXPORT_NORMALIZED="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --no-resume-cache)
            NO_RESUME_CACHE="--no-resume-cache"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --start-date DATE"
            echo "  --end-date DATE"
            echo "  --model-type TYPE"
            echo "  --stock-limit N"
            echo "  --stocks N"
            echo "  --tickers T1 T2 ..."
            echo "  --export-pre-normalize PATH"
            echo "  --export-normalized PATH"
            echo "  --skip-download"
            echo "  --no-resume-cache"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MODEL_TYPE="$(resolve_model_type "$MODEL_TYPE")"

if [ "$MODEL_TYPE" = "chronos2" ]; then
    CMD="python scripts/prepare_chronos2_data.py --start-date $START_DATE $SKIP_DOWNLOAD $NO_RESUME_CACHE"
elif [ "$MODEL_TYPE" = "chronos_rich" ]; then
    CMD="python scripts/prepare_chronos_rich_data.py --start-date $START_DATE $SKIP_DOWNLOAD $NO_RESUME_CACHE"
elif [ "$MODEL_TYPE" = "kronos_rich" ]; then
    CMD="python scripts/prepare_kronos_rich_data.py --start-date $START_DATE $SKIP_DOWNLOAD $NO_RESUME_CACHE"
else
    CMD="python scripts/preprocess_data.py --start-date $START_DATE $SKIP_DOWNLOAD $NO_RESUME_CACHE"
fi

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
if [ -n "$EXPORT_NORMALIZED" ]; then
    CMD="$CMD --export-normalized $EXPORT_NORMALIZED"
fi

echo "=========================================="
echo "PREPROCESSING DATA (IN CONTAINER)"
echo "=========================================="
echo "Command: $CMD"
echo "=========================================="
echo ""

ensure_python_module_installed "statsmodels" "statsmodels>=0.14.0"

eval $CMD
