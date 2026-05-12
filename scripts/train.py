#!/usr/bin/env python
"""
Training script for CRNN models.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
from typing import Optional, List
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset import FinancialDataset, create_data_loaders
from src.models import create_model
from src.training import LightningDependencyError, Trainer, train_with_lightning
from src.utils.logger import get_logger
from src.utils.device import get_device, print_gpu_info, get_device_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CRNN model')

    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        help='Model type. Defaults to config.model.selection.DEFAULT_MODEL_TYPE.'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config override JSON'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (default from config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default from config)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default from config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda or cpu). If not specified, will auto-detect the best available device.'
    )

    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/checkpoints',
        help='Checkpoint directory'
    )

    parser.add_argument(
        '--stocks',
        type=str,
        nargs='+',
        default=None,
        help='Fine-tune on specific stock names (e.g., AAPL MSFT). '
             'If provided, only these stocks will be used for training.'
    )

    parser.add_argument(
        '--fine-tune',
        type=str,
        default=None,
        help='Path to checkpoint to fine-tune from. '
             'Use this to continue training on specific stocks.'
    )

    parser.add_argument(
        '--freeze-embeddings',
        action='store_true',
        help='Freeze stock and group embeddings during fine-tuning'
    )

    parser.add_argument(
        '--backend',
        type=str,
        choices=['lightning', 'custom'],
        default=None,
        help='Training backend. Defaults to config.model.training_backend.DEFAULT.'
    )

    return parser.parse_args()


def load_sequences(data_dir: Path, split: str, stock_names: Optional[List[str]] = None,
                    stock_encoder=None, group_encoder=None):
    """
    Load sequences from directory.

    Args:
        data_dir: Data directory
        split: Split name ('train', 'val', 'test')
        stock_names: Optional list of stock names to filter by
        stock_encoder: LabelEncoder for stock names (needed if stock_names is provided)
        group_encoder: LabelEncoder for group names (needed if stock_names is provided)

    Returns:
        Dictionary of sequences or None
    """
    split_dir = data_dir / split

    if not split_dir.exists():
        return None

    sequences = {}
    for file in ['features', 'stock_id', 'group_id', 'day', 'month', 'target']:
        file_path = split_dir / f'{file}.npy'
        if file_path.exists():
            sequences[file] = np.load(file_path)

    if len(sequences) == 0:
        return None

    # Filter by specific stocks if provided
    if stock_names is not None and stock_encoder is not None:
        # Get stock IDs for the specified stock names
        try:
            stock_ids = [stock_encoder.transform([s])[0] for s in stock_names
                        if s in stock_encoder.classes_]

            if len(stock_ids) == 0:
                return None

            # Filter sequences where first stock_id in sequence matches any of our target stocks
            mask = np.isin(sequences['stock_id'][:, 0], stock_ids)

            for key in sequences:
                sequences[key] = sequences[key][mask]

            if len(sequences['target']) == 0:
                return None

        except Exception as e:
            print(f"Warning: Failed to filter by stocks: {e}")

    return sequences


def _load_model_weights(model, checkpoint_path: str, device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)


def main():
    """Main training function."""
    args = parse_args()

    logger = get_logger("train", log_dir="logs")

    logger.info("=" * 60)
    logger.info("TRAINING SCRIPT")
    logger.info("=" * 60)

    # Determine device
    if args.device:
        device = torch.device(args.device)
        logger.info(f"Using manually specified device: {device}")
    else:
        device = get_device(force_cpu=args.force_cpu, verbose=True)
        device_info = get_device_info(verbose=False)
        logger.info(f"Auto-detected device: {device}")
        logger.info(f"CUDA available: {device_info['cuda_available']}")
        if device_info.get('cuda_working'):
            logger.info(f"GPU: {device_info.get('gpu_name', 'Unknown')}")
            logger.info(f"GPU Memory: {device_info.get('gpu_memory_gb', 0):.2f} GB")

    # Load config
    config = load_config('model')
    model_type = args.model_type or config.get_default_model_type()
    available_model_types = config.get_available_model_types()
    if model_type not in available_model_types:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {available_model_types}"
        )

    # Override config from arguments
    if args.epochs:
        config.model.training.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.model.training.BATCH_SIZE = args.batch_size
    if args.lr:
        config.model.training.LEARNING_RATE = args.lr

    config.model.checkpointing.CHECKPOINT_DIR = args.checkpoint_dir
    backend = args.backend or config.model.training_backend.DEFAULT

    logger.info(
        "Effective training config: "
        f"model_type={model_type}, "
        f"epochs={config.model.training.NUM_EPOCHS}, "
        f"batch_size={config.model.training.BATCH_SIZE}, "
        f"lr={config.model.training.LEARNING_RATE}, "
        f"backend={backend}"
    )

    # Load config override if provided
    if args.config:
        with open(args.config, 'r') as f:
            override = json.load(f)
            for key, value in override.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Check if fine-tuning mode
    is_finetuning = args.stocks is not None or args.fine_tune is not None

    # Load data
    data_dir = Path(args.data_dir)

    logger.info(f"Loading data from {data_dir}...")

    preprocessing_info = {}
    info_path = data_dir / 'info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            preprocessing_info = json.load(f)
        logger.info(f"Loaded preprocessing info from {info_path}")

    # Load stock encoders if needed for stock filtering
    stock_encoder = None
    group_encoder = None

    if args.stocks is not None:
        # Need to load the pre-normalized data to get stock names
        # or we need to load from info
        try:
            # Try to load from preprocessing checkpoint
            pre_norm_path = data_dir.parent / 'pre_normalized.parquet'
            if pre_norm_path.exists():
                import pandas as pd
                df = pd.read_parquet(pre_norm_path)
                unique_stocks = df['tic'].unique()
                stock_encoder = LabelEncoder()
                stock_encoder.fit(unique_stocks)
                logger.info(f"Loaded stock encoder with {len(stock_encoder.classes_)} stocks")
        except Exception as e:
            logger.warning(f"Could not load stock encoder: {e}")

    # Load sequences
    train_sequences = load_sequences(data_dir, 'train', stock_names=args.stocks, stock_encoder=stock_encoder)
    val_sequences = load_sequences(data_dir, 'val', stock_names=args.stocks, stock_encoder=stock_encoder)

    if train_sequences is None:
        logger.error(f"No training data found in {data_dir}/train/")
        logger.error("Run preprocess_data.py first")
        return 1

    # Log if filtering by stocks
    if args.stocks is not None:
        logger.info(f"Filtered training to stocks: {args.stocks}")
        logger.info(f"Loaded {len(train_sequences['target'])} training samples (filtered)")
        if val_sequences:
            logger.info(f"Loaded {len(val_sequences['target'])} validation samples (filtered)")
    else:
        logger.info(f"Loaded {len(train_sequences['target'])} training samples")
        if val_sequences:
            logger.info(f"Loaded {len(val_sequences['target'])} validation samples")

    # Create data loaders
    logger.info("Creating data loaders...")

    loaders = create_data_loaders(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        config=config
    )

    # Get embedding sizes
    train_dataset = loaders['train'].dataset
    embedding_sizes = train_dataset.get_embedding_sizes()

    num_features = train_dataset.num_features

    logger.info(f"Creating {model_type} model...")
    logger.info(f"  Num features: {num_features}")
    logger.info(f"  Num stocks: {embedding_sizes['num_stocks']}")
    logger.info(f"  Num groups: {embedding_sizes['num_groups']}")

    # Create model
    model = create_model(
        model_type=model_type,
        num_features=num_features,
        num_stocks=embedding_sizes['num_stocks'],
        num_groups=embedding_sizes['num_groups'],
        config=config
    )

    # Create trainer
    checkpoint_metadata = {
        'feature_cols': preprocessing_info.get('feature_cols'),
        'num_features': preprocessing_info.get('num_features'),
        'num_stocks': embedding_sizes['num_stocks'],
        'num_groups': embedding_sizes['num_groups'],
        'target_normalization': {
            'NORMALIZE_TARGET': preprocessing_info.get(
                'normalize_target',
                load_config('main').data.sequences.NORMALIZE_TARGET
            ),
            'TARGET_THRESHOLD': preprocessing_info.get(
                'target_threshold',
                load_config('main').data.sequences.TARGET_THRESHOLD
            ),
        },
        'regime_params': preprocessing_info.get('regime_params'),
    }
    checkpoint_metadata = {
        key: value for key, value in checkpoint_metadata.items()
        if value is not None
    }

    # Load checkpoint for fine-tuning if specified
    if args.fine_tune:
        logger.info(f"Loading checkpoint for fine-tuning: {args.fine_tune}")
        if backend == 'custom':
            trainer = Trainer(
                model,
                config,
                device=str(device),
                model_type=model_type,
                checkpoint_metadata=checkpoint_metadata
            )
            trainer.load_checkpoint(args.fine_tune)
        else:
            _load_model_weights(model, args.fine_tune, device)
        logger.info("Checkpoint loaded successfully")

        # Freeze embeddings if requested
        if args.freeze_embeddings:
            logger.info("Freezing stock and group embeddings...")
            for name, param in model.named_parameters():
                if 'stock_embedding' in name or 'group_embedding' in name:
                    param.requires_grad = False
            logger.info("Embeddings frozen")

    # Resume from checkpoint if specified (different from fine-tune)
    elif args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        if backend == 'custom':
            trainer = Trainer(
                model,
                config,
                device=str(device),
                model_type=model_type,
                checkpoint_metadata=checkpoint_metadata
            )
            trainer.load_checkpoint(args.resume)
        else:
            logger.warning("Lightning backend loads model weights from custom checkpoints; optimizer resume is not available in Task 5.1")
            _load_model_weights(model, args.resume, device)

    # Train
    if is_finetuning:
        logger.info(f"Starting fine-tuning on stocks: {args.stocks if args.stocks else 'all'}...")
        logger.info("Note: Model will adapt to the specified stocks while retaining knowledge from previous training.")
    else:
        logger.info("Starting training...")

    logger.info(f"Training backend: {backend}")
    trainer = None
    lightning_result = None
    try:
        if backend == 'lightning':
            lightning_result = train_with_lightning(
                model=model,
                config=config,
                train_loader=loaders['train'],
                val_loader=loaders.get('val'),
                device=str(device),
                model_type=model_type,
                checkpoint_metadata=checkpoint_metadata,
            )
        else:
            trainer = Trainer(
                model,
                config,
                device=str(device),
                model_type=model_type,
                checkpoint_metadata=checkpoint_metadata
            )
            trainer.train(
                train_loader=loaders['train'],
                val_loader=loaders.get('val'),
                num_epochs=config.model.training.NUM_EPOCHS
            )
    except LightningDependencyError as exc:
        if backend != 'lightning' or not config.model.training_backend.ALLOW_CUSTOM_FALLBACK:
            raise
        logger.warning(f"{exc}")
        logger.warning("Falling back to custom Trainer because ALLOW_CUSTOM_FALLBACK=true")
        backend = 'custom'
        trainer = Trainer(
            model,
            config,
            device=str(device),
            model_type=model_type,
            checkpoint_metadata=checkpoint_metadata
        )
        trainer.train(
            train_loader=loaders['train'],
            val_loader=loaders.get('val'),
            num_epochs=config.model.training.NUM_EPOCHS
        )

    logger.info("Training complete!")
    if backend == 'lightning' and lightning_result is not None:
        best_score = lightning_result.get('best_score')
        if best_score is not None:
            logger.info(f"Best validation loss: {best_score:.6f}")
        if lightning_result.get('best_model_path'):
            logger.info(f"Saved Lightning custom-compatible checkpoint to {lightning_result['best_model_path']}")
    elif trainer is not None:
        logger.info(f"Best validation loss: {trainer.checkpoint.best_score:.6f}")

    # Save final model with suffix if fine-tuning
    if args.stocks:
        stock_suffix = "_".join(args.stocks[:3])  # Use first 3 stock names for filename
        if len(args.stocks) > 3:
            stock_suffix += f"_etc{len(args.stocks)}"
        final_path = Path(config.model.checkpointing.CHECKPOINT_DIR) / f"best_model_{stock_suffix}.pth"
        if trainer is None:
            trainer = Trainer(
                model,
                config,
                device=str(device),
                model_type=model_type,
                checkpoint_metadata=checkpoint_metadata
            )
        trainer.save_model(str(final_path))
        logger.info(f"Saved fine-tuned model to {final_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
