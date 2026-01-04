#!/usr/bin/env python
"""
Test prediction pipeline with real trained model.
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.config import load_config
from src.prediction.predictor import create_predictor
from src.utils.logger import get_logger

logger = get_logger("test_prediction_real", log_dir="logs")


def main():
    logger.info("=" * 60)
    logger.info("TESTING PREDICTION WITH REAL TRAINED MODEL")
    logger.info("=" * 60)

    model_path = "models/checkpoints/best_model.pth"

    # First, load the checkpoint to get the necessary parameters
    logger.info(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['model_state_dict']

    # Calculate model parameters from state dict
    stock_emb = 64
    group_emb = 32
    day_emb = 16
    month_emb = 16
    dividend_emb = 8
    total_emb = stock_emb + group_emb + day_emb + month_emb + dividend_emb

    lstm_input = state_dict['lstm.lstm.weight_ih_l0'].shape[1]
    num_features = lstm_input - total_emb
    num_stocks = state_dict['embeddings.stock_embedding.weight'].shape[0]
    num_groups = state_dict['embeddings.group_embedding.weight'].shape[0]

    logger.info(f"Model parameters:")
    logger.info(f"  Num features: {num_features}")
    logger.info(f"  Num stocks: {num_stocks}")
    logger.info(f"  Num groups: {num_groups}")
    logger.info(f"  Model type: lstm3_attention")

    logger.info(f"\nTraining info:")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Val loss: {checkpoint['val_metrics']['loss']:.6f}")
    logger.info(f"  Val RMSE: {checkpoint['val_metrics']['rmse']:.6f}")
    logger.info(f"  Directional accuracy: {checkpoint['val_metrics']['directional_accuracy']:.4f}")

    # The checkpoint doesn't have the expected format for the Predictor class
    # We need to add the required keys
    logger.info("\nConverting checkpoint format...")

    # Create a new checkpoint with the expected format
    new_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict'],
        'metadata': {
            'model_type': 'lstm3_attention',
            'epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['val_metrics']['loss'],
        },
        'num_features': num_features,
        'num_stocks': num_stocks,
        'num_groups': num_groups,
    }

    # Save the converted checkpoint
    converted_path = "models/checkpoints/best_model_converted.pt"
    torch.save(new_checkpoint, converted_path)
    logger.info(f"Saved converted checkpoint to {converted_path}")

    # Now create predictor with the converted checkpoint
    logger.info("\nCreating predictor...")
    predictor = create_predictor(
        model_path=converted_path,
        device='cpu'
    )

    # Get model info
    info = predictor.get_model_info()
    logger.info("\nModel Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    # Test with sample data (simulated data for the 3 stocks in the model)
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PREDICTION WITH SAMPLE DATA")
    logger.info("=" * 60)

    # Create sample sequences that match the model's expected input
    # Since we don't know the exact feature columns, we'll create dummy data
    batch_size = 2
    seq_len = 30

    sample_sequences = {
        'features': np.random.randn(batch_size, seq_len, num_features).astype(np.float32),
        'stock_id': np.random.randint(0, num_stocks, (batch_size, seq_len)),
        'group_id': np.random.randint(0, num_groups, (batch_size, seq_len)),
        'day': np.random.randint(1, 32, (batch_size, seq_len)),
        'month': np.random.randint(1, 13, (batch_size, seq_len)),
        'dividend_flag': np.random.randint(1, 3, (batch_size, seq_len)),
    }

    logger.info(f"\nSample sequences:")
    logger.info(f"  Features shape: {sample_sequences['features'].shape}")
    logger.info(f"  Stock IDs shape: {sample_sequences['stock_id'].shape}")
    logger.info(f"  Group IDs shape: {sample_sequences['group_id'].shape}")
    logger.info(f"  Day shape: {sample_sequences['day'].shape}")
    logger.info(f"  Month shape: {sample_sequences['month'].shape}")
    logger.info(f"  Dividend flag shape: {sample_sequences['dividend_flag'].shape}")

    # Make prediction
    logger.info("\nMaking predictions...")
    predictions = predictor.predict(sample_sequences, return_raw=True)

    logger.info(f"\nPredictions (raw, normalized):")
    for i, pred in enumerate(predictions):
        logger.info(f"  Sample {i}: {pred[0]:.6f}")

    # Apply inverse transform (the model uses tanh normalization)
    threshold = 10.0  # TARGET_THRESHOLD from config
    predictions_denorm = np.clip(predictions, -0.99, 0.99)
    predictions_denorm = threshold * np.arctanh(predictions_denorm)

    logger.info(f"\nPredictions (denormalized, % change):")
    for i, pred in enumerate(predictions_denorm):
        logger.info(f"  Sample {i}: {pred[0]:+.4f}%")

    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
