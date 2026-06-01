#!/usr/bin/env python
"""
Prediction script for financial forecasting model.

This script provides CLI for making predictions using trained models.
Supports:
- Single row prediction
- Batch prediction from file
- Interactive mode
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.prediction.ensemble import create_ensemble_predictor_from_config
from src.prediction.predictor import create_predictor
from src.utils.logger import get_logger


logger = get_logger("predict", log_dir="logs")


def create_configured_predictor(args):
    """Create a single-model or ensemble predictor from CLI args and config."""
    model_config = load_config('model')
    ensemble_config = model_config.model.ensemble

    if ensemble_config.ENABLED:
        return create_ensemble_predictor_from_config(
            model_config=model_config,
            preprocessor_path=args.preprocessor,
            device=args.device,
        )

    if not args.model:
        raise ValueError("--model is required when model.ensemble.ENABLED is false")

    return create_predictor(
        model_path=args.model,
        model_config=model_config,
        preprocessor_path=args.preprocessor,
        device=args.device,
    )


def parse_single_input(input_str: str) -> Dict[str, float]:
    """
    Parse single input string into feature dictionary.

    Format: "key1=value1,key2=value2,..." or JSON string

    Args:
        input_str: Input string

    Returns:
        Dictionary with feature values
    """
    # Try JSON first
    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        pass

    # Parse key=value pairs
    result = {}
    for pair in input_str.split(','):
        pair = pair.strip()
        if '=' in pair:
            key, value = pair.split('=', 1)
            result[key.strip()] = float(value.strip())

    return result


def predict_single_row(args):
    """Make prediction for a single data point."""
    logger.info("=" * 60)
    logger.info("SINGLE ROW PREDICTION")
    logger.info("=" * 60)

    # Create predictor
    predictor = create_configured_predictor(args)

    # Parse input data
    data = parse_single_input(args.input)

    # Required fields
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [r for r in required if r not in data]
    if missing:
        logger.error(f"Missing required fields: {missing}")
        logger.info("Required fields: " + ", ".join(required))
        return 1

    # Make prediction
    result = predictor.predict_single(
        data=data,
        stock_ticker=args.ticker,
        date=args.date,
        group=args.group
    )

    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Stock Ticker: {result['stock_ticker']}")
    print(f"Date: {result['date']}")

    if result['prediction'] is not None:
        pred = result['prediction']
        print(f"Predicted Change: {pred:+.2f}%")
        if 'future_regime' in result:
            print(f"Predicted Regime: {result['future_regime']}")
        if 'future_return_path' in result:
            print(f"Future Return Path: {result['future_return_path']}")
        if 'future_ohlcv' in result:
            print(f"Future OHLCV: {result['future_ohlcv']}")

        if pred > 0:
            print(f"Signal: BUY (Positive movement expected)")
        elif pred < 0:
            print(f"Signal: SELL (Negative movement expected)")
        else:
            print(f"Signal: HOLD (No significant change expected)")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("=" * 60)

    return 0


def predict_batch_file(args):
    """Make predictions from an input file."""
    logger.info("=" * 60)
    logger.info("BATCH PREDICTION FROM FILE")
    logger.info("=" * 60)

    # Create predictor
    predictor = create_configured_predictor(args)

    # Make predictions
    result_df = predictor.predict_from_file(
        input_path=args.input,
        output_path=args.output,
        file_format=args.format
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total predictions: {len(result_df)}")

    if 'prediction' in result_df.columns:
        print(f"Mean prediction: {result_df['prediction'].mean():+.2f}%")
        print(f"Min prediction: {result_df['prediction'].min():+.2f}%")
        print(f"Max prediction: {result_df['prediction'].max():+.2f}%")

        positive = (result_df['prediction'] > 0).sum()
        negative = (result_df['prediction'] < 0).sum()
        print(f"\nPositive predictions: {positive} ({100*positive/len(result_df):.1f}%)")
        print(f"Negative predictions: {negative} ({100*negative/len(result_df):.1f}%)")

    print("\nFirst 10 predictions:")
    print(result_df.head(10).to_string(index=False))

    if args.output:
        print(f"\nResults saved to: {args.output}")

    print("=" * 60)

    return 0


def predict_interactive(args):
    """Interactive prediction mode."""
    logger.info("=" * 60)
    logger.info("INTERACTIVE PREDICTION MODE")
    logger.info("=" * 60)

    # Create predictor
    predictor = create_configured_predictor(args)

    print("\nPredictor initialized. Enter 'quit' to exit.")
    print("\nRequired fields for each prediction:")
    print("  ticker, date, open, high, low, close, volume")
    print("\nOptional fields (auto-computed if not provided):")
    print("  group, EMA, RSI, MACD, etc.")
    print("\nFormat: key=value,key2=value2,...")
    print("-" * 60)

    while True:
        print("\n" + "-" * 60)
        user_input = input("Enter prediction data (or 'quit'): ").strip()

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Exiting interactive mode.")
            break

        try:
            data = parse_single_input(user_input)

            # Check required fields
            required = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            missing = [r for r in required if r not in data]
            if missing:
                print(f"Error: Missing required fields: {missing}")
                continue

            ticker = data.pop('ticker')
            date = data.pop('date')
            group = data.pop('group', None)

            # Make prediction
            result = predictor.predict_single(
                data=data,
                stock_ticker=ticker,
                date=date,
                group=group
            )

            # Print result
            if result['prediction'] is not None:
                pred = result['prediction']
                print(f"\nPrediction for {ticker} on {date}:")
                print(f"  Predicted Change: {pred:+.2f}%")
                if 'future_regime' in result:
                    print(f"  Predicted Regime: {result['future_regime']}")
                if 'future_return_path' in result:
                    print(f"  Future Return Path: {result['future_return_path']}")
                if 'future_ohlcv' in result:
                    print(f"  Future OHLCV: {result['future_ohlcv']}")

                if pred > 1:
                    print(f"  Signal: Strong BUY")
                elif pred > 0:
                    print(f"  Signal: BUY")
                elif pred < -1:
                    print(f"  Signal: Strong SELL")
                elif pred < 0:
                    print(f"  Signal: SELL")
                else:
                    print(f"  Signal: HOLD")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Prediction error")

    return 0


def show_model_info(args):
    """Display model information."""
    logger.info("=" * 60)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 60)

    predictor = create_configured_predictor(args)

    info = predictor.get_model_info()

    print("\nModel Information:")
    print("-" * 60)
    print(f"Model Path: {info['model_path']}")
    print(f"Model Type: {info['model_type']}")
    print(f"Device: {info['device']}")
    print(f"Number of Features: {info['num_features']}")
    print(f"Number of Stocks: {info['num_stocks']}")
    print(f"Number of Groups: {info['num_groups']}")
    print(f"Supports Rich Output: {info['supports_rich_output']}")

    if info['training_epochs']:
        print(f"Training Epochs: {info['training_epochs']}")
    if info['best_val_loss']:
        print(f"Best Val Loss: {info['best_val_loss']:.6f}")

    if info['feature_cols']:
        print(f"\nFeature Columns ({len(info['feature_cols'])}):")
        for col in info['feature_cols']:
            print(f"  - {col}")

    print("=" * 60)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Make predictions using trained financial forecasting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single row prediction
  python scripts/predict.py single \\
    --model models/checkpoints/best_model.pt \\
    --ticker AAPL \\
    --date 2024-01-15 \\
    --input "open=150.0,high=152.0,low=149.0,close=151.5,volume=50000000"

  # Batch prediction from CSV file
  python scripts/predict.py batch \\
    --model models/checkpoints/best_model.pt \\
    --input data/prediction_input.csv \\
    --output data/predictions.csv \\
    --format csv

  # Interactive mode
  python scripts/predict.py interactive \\
    --model models/checkpoints/best_model.pt

  # Show model info
  python scripts/predict.py info \\
    --model models/checkpoints/best_model.pt
        """
    )

    parser.add_argument(
        'mode',
        choices=['single', 'batch', 'interactive', 'info'],
        help='Prediction mode'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model checkpoint. Optional when model.ensemble.ENABLED is true.'
    )

    parser.add_argument(
        '--preprocessor',
        type=str,
        default=None,
        help='Path to saved preprocessor state'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )

    # Single prediction arguments
    single_parser = argparse.ArgumentParser(add_help=False)
    single_parser.add_argument(
        '--ticker',
        type=str,
        help='Stock ticker symbol'
    )
    single_parser.add_argument(
        '--date',
        type=str,
        help='Date (YYYY-MM-DD format)'
    )
    single_parser.add_argument(
        '--input',
        type=str,
        help='Input data as key=value pairs or JSON'
    )
    single_parser.add_argument(
        '--group',
        type=str,
        default=None,
        help='Sector/group (optional)'
    )

    # Batch prediction arguments
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input file'
    )
    batch_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (optional)'
    )
    batch_parser.add_argument(
        '--format',
        type=str,
        default='csv',
        choices=['csv', 'parquet', 'excel'],
        help='Input file format'
    )

    # Parse arguments
    args, remaining = parser.parse_known_args()

    # Parse mode-specific arguments
    if args.mode == 'single':
        single_args = single_parser.parse_args(remaining)
        for key, value in vars(single_args).items():
            setattr(args, key, value)
        return predict_single_row(args)

    elif args.mode == 'batch':
        batch_args = batch_parser.parse_args(remaining)
        for key, value in vars(batch_args).items():
            setattr(args, key, value)
        return predict_batch_file(args)

    elif args.mode == 'interactive':
        return predict_interactive(args)

    elif args.mode == 'info':
        return show_model_info(args)


if __name__ == "__main__":
    sys.exit(main())
