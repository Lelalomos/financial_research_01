"""
Test cleanup utilities.

This module provides utilities for cleaning up test-generated files.
"""

import os
import glob
from pathlib import Path
from typing import List, Optional
import shutil


def cleanup_test_files(
    keep_latest: int = 5,
    remove_all: bool = False,
    verbose: bool = True
) -> dict:
    """
    Clean up test-generated files.

    Args:
        keep_latest: Number of latest files to keep (default: 5)
        remove_all: If True, remove all files instead of keeping latest
        verbose: If True, print cleanup information

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        'removed_reports': 0,
        'removed_backtest_reports': 0,
        'removed_checkpoints': 0,
        'removed_json_outputs': 0,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("CLEANING UP TEST FILES")
        print("=" * 60)

    # Clean up test Excel reports
    stats['removed_reports'] += _cleanup_files_by_pattern(
        "reports/test_report_*.xlsx",
        keep_latest if not remove_all else 0,
        verbose
    )

    # Clean up backtest reports
    for pattern in ["outputs/backtest_report_*.xlsx",
                    "outputs/backtest_report_*.csv",
                    "outputs/backtest_report_*.json"]:
        stats['removed_backtest_reports'] += _cleanup_files_by_pattern(
            pattern,
            keep_latest if not remove_all else 0,
            verbose
        )

    # Clean up test JSON outputs
    stats['removed_json_outputs'] += _cleanup_files_by_pattern(
        "outputs/test_results_*.json",
        0,  # Remove all
        verbose
    )

    # Clean up test checkpoints
    for pattern in ["models/*_test_*.pth", "checkpoints/*_test_*.pth"]:
        stats['removed_checkpoints'] += _cleanup_files_by_pattern(
            pattern,
            0,  # Remove all test checkpoints
            verbose
        )

    # Remove empty directories
    _remove_empty_dirs(["reports", "outputs"], verbose)

    if verbose:
        print("\n" + "=" * 60)
        print(f"CLEANUP COMPLETE - Removed {sum(stats.values())} files")
        print("=" * 60)

    return stats


def _cleanup_files_by_pattern(
    pattern: str,
    keep_latest: int,
    verbose: bool
) -> int:
    """
    Clean up files matching a pattern.

    Args:
        pattern: Glob pattern for files to clean
        keep_latest: Number of latest files to keep
        verbose: If True, print information

    Returns:
        Number of files removed
    """
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if not files:
        return 0

    files_to_remove = files[keep_latest:] if keep_latest > 0 else files

    for filepath in files_to_remove:
        try:
            os.remove(filepath)
            if verbose:
                print(f"  Removed: {filepath}")
        except OSError as e:
            if verbose:
                print(f"  Warning: Could not remove {filepath}: {e}")

    if keep_latest > 0 and len(files) > keep_latest and verbose:
        print(f"  Kept latest {keep_latest}, removed {len(files_to_remove)} from {pattern}")

    return len(files_to_remove)


def _remove_empty_dirs(dirs: List[str], verbose: bool) -> None:
    """Remove empty directories."""
    for dir_name in dirs:
        if os.path.exists(dir_name) and not os.listdir(dir_name):
            try:
                os.rmdir(dir_name)
                if verbose:
                    print(f"  Removed empty directory: {dir_name}/")
            except OSError:
                pass


def cleanup_specific_files(files: List[str], verbose: bool = True) -> int:
    """
    Clean up specific files.

    Args:
        files: List of file paths to remove
        verbose: If True, print information

    Returns:
        Number of files removed
    """
    removed = 0
    for filepath in files:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                if verbose:
                    print(f"  Removed: {filepath}")
                removed += 1
            except OSError as e:
                if verbose:
                    print(f"  Warning: Could not remove {filepath}: {e}")
    return removed
