#!/usr/bin/env python
"""
Orchestrator script to run the full pipeline.

Runs: preprocess -> train -> validate -> test -> backtest
"""

import argparse
import sys
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run full pipeline')

    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing if data exists'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='crnn_attention',
        help='Model type'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip backtesting'
    )

    return parser.parse_args()


def run_command(cmd: list, description: str) -> int:
    """Run a command and return exit code."""
    logger = get_logger("run_all", log_dir="logs")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Failed: {description}")
    else:
        logger.info(f"Complete: {description}")

    return result.returncode


def main():
    """Main orchestrator."""
    args = parse_args()
    logger = get_logger("run_all", log_dir="logs")

    logger.info("=" * 70)
    logger.info("FULL PIPELINE ORCHESTRATOR")
    logger.info("=" * 70)

    exit_code = 0

    # Step 1: Preprocess data
    if not args.skip_preprocess:
        logger.info("Step 1: Preprocessing data...")
        cmd = [sys.executable, 'scripts/preprocess_data.py']
        exit_code = run_command(cmd, "Preprocessing")

        if exit_code != 0:
            logger.error("Preprocessing failed. Exiting.")
            return exit_code
    else:
        logger.info("Skipping preprocessing (--skip-preprocess flag)")

    # Step 2: Train model
    logger.info("Step 2: Training model...")
    cmd = [sys.executable, 'scripts/train.py', '--model-type', args.model_type]

    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])

    exit_code = run_command(cmd, "Training")

    if exit_code != 0:
        logger.error("Training failed. Exiting.")
        return exit_code

    # Step 3: Validate model
    logger.info("Step 3: Validating model...")
    cmd = [
        sys.executable, 'scripts/validate.py',
        '--model', 'best',
        '--model-type', args.model_type
    ]
    exit_code = run_command(cmd, "Validation")

    if exit_code != 0:
        logger.warning("Validation failed (non-critical). Continuing...")

    # Step 4: Test model
    logger.info("Step 4: Testing model...")
    cmd = [
        sys.executable, 'scripts/test.py',
        '--model', 'best',
        '--model-type', args.model_type
    ]
    exit_code = run_command(cmd, "Testing")

    if exit_code != 0:
        logger.warning("Testing failed (non-critical). Continuing...")

    # Step 5: Backtest
    if not args.skip_backtest:
        logger.info("Step 5: Backtesting...")
        cmd = [
            sys.executable, 'scripts/backtest.py',
            '--model', 'best',
            '--model-type', args.model_type
        ]
        exit_code = run_command(cmd, "Backtesting")

        if exit_code != 0:
            logger.warning("Backtesting failed (non-critical).")
    else:
        logger.info("Skipping backtesting (--skip-backtest flag)")

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
