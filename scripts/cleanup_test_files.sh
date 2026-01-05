#!/bin/bash
# Cleanup test-generated files
# Usage: ./scripts/cleanup_test_files.sh [--all]
#
# Options:
#   --all    Remove all test files (default: keeps latest 5 test reports)

set -e

# Parse arguments
REMOVE_ALL=false
if [ "$1" = "--all" ]; then
    REMOVE_ALL=true
fi

echo "=========================================="
echo "CLEANUP TEST FILES"
echo "=========================================="

# Remove Excel reports from tests
echo "Cleaning up test Excel reports..."
if [ "$REMOVE_ALL" = true ]; then
    # Remove all test reports
    rm -fv reports/test_report_*.xlsx 2>/dev/null || echo "  No test Excel reports found"
else
    # Keep latest 5, remove older ones
    if ls reports/test_report_*.xlsx 2>/dev/null | head -1 >/dev/null; then
        TOTAL=$(ls reports/test_report_*.xlsx 2>/dev/null | wc -l)
        if [ $TOTAL -gt 5 ]; then
            ls -t reports/test_report_*.xlsx 2>/dev/null | tail -n +6 | xargs -r rm -fv
            echo "  Removed $((TOTAL - 5)) old test reports (kept latest 5)"
        else
            echo "  Found $TOTAL test reports (keeping all, less than 5)"
        fi
    else
        echo "  No test Excel reports found"
    fi
fi

# Remove backtest reports (keep latest 3)
echo "Cleaning up backtest reports..."
if [ "$REMOVE_ALL" = true ]; then
    rm -fv outputs/backtest_report_*.xlsx 2>/dev/null || echo "  No backtest Excel reports found"
    rm -fv outputs/backtest_report_*.csv 2>/dev/null || echo "  No backtest CSV reports found"
    rm -fv outputs/backtest_report_*.json 2>/dev/null || echo "  No backtest JSON reports found"
else
    for ext in xlsx csv json; do
        if ls outputs/backtest_report_*.$ext 2>/dev/null | head -1 >/dev/null; then
            TOTAL=$(ls outputs/backtest_report_*.$ext 2>/dev/null | wc -l)
            if [ $TOTAL -gt 3 ]; then
                ls -t outputs/backtest_report_*.$ext 2>/dev/null | tail -n +4 | xargs -r rm -fv
                echo "  Removed $((TOTAL - 3)) old backtest $ext reports (kept latest 3)"
            else
                echo "  Found $TOTAL backtest $ext reports (keeping all)"
            fi
        fi
    done
fi

# Remove test JSON outputs
echo "Removing test JSON outputs..."
rm -fv outputs/test_results_*.json 2>/dev/null || echo "  No test JSON outputs found"

# Remove any checkpoint files created during tests
echo "Removing test checkpoints..."
rm -fv models/*_test_*.pth 2>/dev/null || echo "  No test checkpoints in models/"
rm -fv checkpoints/*_test_*.pth 2>/dev/null || echo "  No test checkpoints in checkpoints/"

# Remove test temp directories if empty
echo "Removing empty test directories..."
rmdir reports/ 2>/dev/null && echo "  Removed empty reports/ directory" || echo "  reports/ directory not empty or doesn't exist"
rmdir outputs/ 2>/dev/null && echo "  Removed empty outputs/ directory" || echo "  outputs/ directory not empty or doesn't exist"

echo ""
echo "=========================================="
echo "CLEANUP COMPLETE"
echo "=========================================="
