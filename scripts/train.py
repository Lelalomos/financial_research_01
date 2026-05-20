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
import torch.nn.functional as F
from typing import Optional, List, Dict
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import DataPreprocessor
from src.data.dataset import create_data_loaders, create_lazy_data_loaders
from src.models import create_kronos_model, create_kronos_tokenizer, create_model
from src.training import (
    LightningDependencyError,
    Trainer,
    save_final_lightning_checkpoint,
    train_with_lightning,
)
from src.training.common import create_scheduler
from src.training.early_stopping import EarlyStopping, atomic_torch_save, make_weights_only_safe
from src.utils.data_preview import load_ticker_mapping, log_sequence_preview
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

    parser.add_argument(
        '--max-train-batches',
        type=int,
        default=None,
        help='Optional limit for number of train batches per epoch. Useful for smoke testing.'
    )

    parser.add_argument(
        '--max-val-batches',
        type=int,
        default=None,
        help='Optional limit for number of validation batches per epoch. Useful for smoke testing.'
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
    for file in ['features', 'stock_id', 'group_id', 'day', 'month', 'dividend_flag', 'target']:
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


def load_normalized_split(data_dir: Path, split: str):
    split_path = data_dir / '.cache' / 'normalized_splits' / f'{split}.parquet'
    if not split_path.exists():
        return None

    import pandas as pd
    return pd.read_parquet(split_path)


def load_normalized_splits_for_training(
    data_dir: Path,
    logger,
    stock_names: Optional[List[str]] = None,
):
    splits = {}
    for split_name in ['train', 'val']:
        split_df = load_normalized_split(data_dir, split_name)
        if split_df is None or split_df.empty:
            splits[split_name] = None
            continue

        if stock_names is not None and 'tic' in split_df.columns:
            split_df = split_df[split_df['tic'].isin(stock_names)].copy()
            if split_df.empty:
                splits[split_name] = None
                continue

        logger.info(f"Loaded normalized {split_name} split with {len(split_df):,} rows")
        splits[split_name] = split_df

    return splits.get('train'), splits.get('val')


def build_sequences_from_normalized_splits(
    data_dir: Path,
    feature_cols: List[str],
    data_config,
    logger,
    stock_names: Optional[List[str]] = None,
):
    preprocessor = DataPreprocessor(data_config)
    sequences_by_split = {}

    for split_name in ['train', 'val']:
        split_df = load_normalized_split(data_dir, split_name)
        if split_df is None or split_df.empty:
            sequences_by_split[split_name] = None
            continue

        if stock_names is not None and 'tic' in split_df.columns:
            split_df = split_df[split_df['tic'].isin(stock_names)].copy()
            if split_df.empty:
                sequences_by_split[split_name] = None
                continue

        logger.info(f"Building {split_name} sequences from normalized split cache...")
        sequences_by_split[split_name] = preprocessor.create_sequences(
            split_df,
            feature_cols=feature_cols,
        )

    return sequences_by_split.get('train'), sequences_by_split.get('val')


def _load_model_weights(model, checkpoint_path: str, device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)


def _create_optimizer_for_params(params, config):
    training = config.model.training
    optimizer_name = training.OPTIMIZER
    lr = training.LEARNING_RATE
    wd = training.WEIGHT_DECAY

    if optimizer_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if optimizer_name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _save_kronos_checkpoint(
    checkpoint_path: Path,
    tokenizer,
    model,
    optimizer,
    epoch: int,
    metric: float,
    model_type: str,
    checkpoint_metadata: Optional[Dict] = None,
):
    payload = {
        'epoch': epoch,
        'model_type': model_type,
        'best_score': metric,
        'tokenizer_state_dict': tokenizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
    }
    if checkpoint_metadata:
        payload.update(make_weights_only_safe(checkpoint_metadata))
    atomic_torch_save(payload, str(checkpoint_path))


def _load_kronos_checkpoint(tokenizer, model, checkpoint_path: str, device, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    tokenizer_state_dict = checkpoint.get('tokenizer_state_dict')
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    if tokenizer_state_dict is not None:
        tokenizer.load_state_dict(tokenizer_state_dict)
    model.load_state_dict(model_state_dict)
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def _train_kronos_epoch(tokenizer, model, train_loader, optimizer, device, config, max_batches=None):
    tokenizer.train()
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        features = batch['features'].to(device)
        stock_id = batch['stock_id'].to(device)
        group_id = batch['group_id'].to(device)
        day = batch['day'].to(device)
        month = batch['month'].to(device)
        dividend_flag = batch['dividend_flag'].to(device)

        optimizer.zero_grad()

        (z_pre, z_full), bsq_loss, _, _ = tokenizer(features)
        recon_loss = F.mse_loss(z_full, features)
        pre_loss = F.mse_loss(z_pre, features)

        with torch.no_grad():
            s1_ids, s2_ids = tokenizer.encode(features, half=True)

        input_s1 = s1_ids[:, :-1]
        input_s2 = s2_ids[:, :-1]
        target_s1 = s1_ids[:, 1:]
        target_s2 = s2_ids[:, 1:]

        s1_logits, s2_logits = model(
            input_s1,
            input_s2,
            stock_id=stock_id[:, :-1],
            group_id=group_id[:, :-1],
            day=day[:, :-1],
            month=month[:, :-1],
            dividend_flag=dividend_flag[:, :-1],
            use_teacher_forcing=True,
            s1_targets=target_s1,
        )
        token_loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, target_s1, target_s2)
        loss = recon_loss + pre_loss + bsq_loss + token_loss
        loss.backward()

        if config.model.training.GRADIENT_CLIP_VALUE > 0:
            torch.nn.utils.clip_grad_norm_(
                list(tokenizer.parameters()) + list(model.parameters()),
                config.model.training.GRADIENT_CLIP_VALUE,
            )

        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("Kronos training loader produced zero batches.")
    return total_loss / total_batches


def _validate_kronos_epoch(tokenizer, model, val_loader, device, max_batches=None):
    tokenizer.eval()
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            features = batch['features'].to(device)
            stock_id = batch['stock_id'].to(device)
            group_id = batch['group_id'].to(device)
            day = batch['day'].to(device)
            month = batch['month'].to(device)
            dividend_flag = batch['dividend_flag'].to(device)

            (z_pre, z_full), bsq_loss, _, _ = tokenizer(features)
            recon_loss = F.mse_loss(z_full, features)
            pre_loss = F.mse_loss(z_pre, features)
            s1_ids, s2_ids = tokenizer.encode(features, half=True)

            input_s1 = s1_ids[:, :-1]
            input_s2 = s2_ids[:, :-1]
            target_s1 = s1_ids[:, 1:]
            target_s2 = s2_ids[:, 1:]

            s1_logits, s2_logits = model(
                input_s1,
                input_s2,
                stock_id=stock_id[:, :-1],
                group_id=group_id[:, :-1],
                day=day[:, :-1],
                month=month[:, :-1],
                dividend_flag=dividend_flag[:, :-1],
                use_teacher_forcing=True,
                s1_targets=target_s1,
            )
            token_loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, target_s1, target_s2)
            loss = recon_loss + pre_loss + bsq_loss + token_loss
            total_loss += float(loss.cpu().item())
            total_batches += 1

    if total_batches == 0:
        raise ValueError("Kronos validation loader produced zero batches.")
    return total_loss / total_batches


def train_kronos(
    loaders,
    config,
    device,
    model_type,
    checkpoint_metadata,
    logger,
    num_features,
    embedding_sizes,
    args,
):
    config.model.models.kronos.tokenizer.D_IN = num_features
    config.model.models.kronos.network.NUM_STOCKS = embedding_sizes['num_stocks']
    config.model.models.kronos.network.NUM_GROUPS = embedding_sizes['num_groups']

    tokenizer = create_kronos_tokenizer(config=config).to(device)
    model = create_kronos_model(config=config).to(device)
    optimizer = _create_optimizer_for_params(
        list(tokenizer.parameters()) + list(model.parameters()),
        config,
    )
    scheduler = create_scheduler(optimizer, config)

    if args.fine_tune:
        logger.info(f"Loading Kronos checkpoint for fine-tuning: {args.fine_tune}")
        _load_kronos_checkpoint(tokenizer, model, args.fine_tune, device, optimizer=None)
    elif args.resume:
        logger.info(f"Resuming Kronos checkpoint: {args.resume}")
        _load_kronos_checkpoint(tokenizer, model, args.resume, device, optimizer=optimizer)

    early_stopping = EarlyStopping(
        patience=config.model.training.EARLY_STOPPING_PATIENCE,
        mode='min',
        verbose=True,
    )
    checkpoint_dir = Path(config.model.checkpointing.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{model_type}_best.pth"
    final_checkpoint_path = checkpoint_dir / f"{model_type}_final.pth"

    best_metric = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config.model.training.NUM_EPOCHS):
        train_loss = _train_kronos_epoch(
            tokenizer,
            model,
            loaders['train'],
            optimizer,
            device,
            config,
            max_batches=args.max_train_batches,
        )
        history['train_loss'].append(train_loss)

        if loaders.get('val') is not None:
            val_loss = _validate_kronos_epoch(
                tokenizer,
                model,
                loaders['val'],
                device,
                max_batches=args.max_val_batches,
            )
        else:
            val_loss = train_loss
        history['val_loss'].append(val_loss)

        logger.info(
            f"Kronos epoch {epoch + 1}/{config.model.training.NUM_EPOCHS}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        if scheduler is not None:
            if config.model.training.SCHEDULER == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_metric:
            best_metric = val_loss
            _save_kronos_checkpoint(
                checkpoint_path=best_checkpoint_path,
                tokenizer=tokenizer,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric=val_loss,
                model_type=model_type,
                checkpoint_metadata=checkpoint_metadata,
            )
            logger.info(f"Saved best Kronos checkpoint to {best_checkpoint_path}")

        if early_stopping(val_loss, epoch + 1):
            break

    _save_kronos_checkpoint(
        checkpoint_path=final_checkpoint_path,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        epoch=len(history['train_loss']),
        metric=history['val_loss'][-1],
        model_type=model_type,
        checkpoint_metadata=checkpoint_metadata,
    )
    logger.info(f"Saved final Kronos checkpoint to {final_checkpoint_path}")

    return {
        'tokenizer': tokenizer,
        'model': model,
        'history': history,
        'best_score': best_metric,
        'best_model_path': str(best_checkpoint_path),
        'final_model_path': str(final_checkpoint_path),
    }


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
    data_config = load_config('main')
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
        f"backend={backend}, "
        f"data_mode={getattr(getattr(data_config.data, 'dataset', None), 'MODE', 'precomputed_sequences')}"
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

    data_mode = getattr(getattr(data_config.data, 'dataset', None), 'MODE', 'precomputed_sequences')
    if data_mode == 'precomputed_sequences':
        train_sequences = load_sequences(
            data_dir,
            'train',
            stock_names=args.stocks,
            stock_encoder=stock_encoder,
        )
        val_sequences = load_sequences(
            data_dir,
            'val',
            stock_names=args.stocks,
            stock_encoder=stock_encoder,
        )
        if train_sequences is None:
            feature_cols = preprocessing_info.get('feature_cols')
            if not feature_cols:
                logger.error("Missing feature_cols in preprocessing info; cannot build sequences before training.")
                return 1
            logger.info("Saved precomputed sequence arrays not found; building from normalized split cache...")
            train_sequences, val_sequences = build_sequences_from_normalized_splits(
                data_dir=data_dir,
                feature_cols=feature_cols,
                data_config=data_config,
                logger=logger,
                stock_names=args.stocks,
            )
        train_split_df = None
        val_split_df = None
    elif data_mode == 'on_the_fly_sequences':
        feature_cols = preprocessing_info.get('feature_cols')
        if not feature_cols:
            logger.error("Missing feature_cols in preprocessing info; cannot stream sequences lazily.")
            return 1
        train_split_df, val_split_df = load_normalized_splits_for_training(
            data_dir=data_dir,
            logger=logger,
            stock_names=args.stocks,
        )
        train_sequences = None
        val_sequences = None
    else:
        train_sequences = load_sequences(data_dir, 'train', stock_names=args.stocks, stock_encoder=stock_encoder)
        val_sequences = load_sequences(data_dir, 'val', stock_names=args.stocks, stock_encoder=stock_encoder)
        train_split_df = None
        val_split_df = None

    if data_mode == 'on_the_fly_sequences':
        if train_split_df is None:
            logger.error(f"No normalized split cache found in {data_dir}/.cache/normalized_splits/")
            logger.error("Run preprocess_data.py with normalized split output first")
            return 1
    elif train_sequences is None:
        logger.error(f"No training data found in {data_dir}/train/")
        logger.error("Run preprocess_data.py first")
        return 1

    # Log if filtering by stocks
    if args.stocks is not None:
        logger.info(f"Filtered training to stocks: {args.stocks}")
        if data_mode == 'on_the_fly_sequences':
            logger.info(f"Loaded {len(train_split_df):,} normalized training rows (filtered)")
            if val_split_df is not None:
                logger.info(f"Loaded {len(val_split_df):,} normalized validation rows (filtered)")
        else:
            logger.info(f"Loaded {len(train_sequences['target'])} training samples (filtered)")
            if val_sequences:
                logger.info(f"Loaded {len(val_sequences['target'])} validation samples (filtered)")
    else:
        if data_mode == 'on_the_fly_sequences':
            logger.info(f"Loaded {len(train_split_df):,} normalized training rows")
            if val_split_df is not None:
                logger.info(f"Loaded {len(val_split_df):,} normalized validation rows")
        else:
            logger.info(f"Loaded {len(train_sequences['target'])} training samples")
            if val_sequences:
                logger.info(f"Loaded {len(val_sequences['target'])} validation samples")

    ticker_map = load_ticker_mapping(data_dir)
    if train_sequences is not None:
        log_sequence_preview(
            logger=logger,
            sequences=train_sequences,
            feature_cols=preprocessing_info.get('feature_cols'),
            ticker_map=ticker_map,
            split_name='train',
            max_rows=10,
        )

    # Create data loaders
    logger.info("Creating data loaders...")

    if data_mode == 'on_the_fly_sequences':
        loaders = create_lazy_data_loaders(
            train_df=train_split_df,
            val_df=val_split_df,
            feature_cols=preprocessing_info.get('feature_cols'),
            data_config=data_config,
            model_config=config,
        )
    else:
        loaders = create_data_loaders(
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            config=config
        )

    if loaders.get('val') is None and backend == 'lightning':
        if config.model.training.SCHEDULER == 'reduce_on_plateau':
            logger.warning(
                "No validation loader is available. Disabling reduce_on_plateau "
                "scheduler for this run because it requires val/loss."
            )
            config.model.training.SCHEDULER = None

    # Get embedding sizes
    train_dataset = loaders['train'].dataset
    embedding_sizes = train_dataset.get_embedding_sizes()

    num_features = train_dataset.num_features

    logger.info(f"Creating {model_type} model...")
    logger.info(f"  Num features: {num_features}")
    logger.info(f"  Num stocks: {embedding_sizes['num_stocks']}")
    logger.info(f"  Num groups: {embedding_sizes['num_groups']}")

    checkpoint_metadata = {
        'feature_cols': preprocessing_info.get('feature_cols'),
        'num_features': preprocessing_info.get('num_features'),
        'num_stocks': embedding_sizes['num_stocks'],
        'num_groups': embedding_sizes['num_groups'],
        'target_normalization': {
            'NORMALIZE_TARGET': preprocessing_info.get(
                'normalize_target',
                data_config.data.sequences.NORMALIZE_TARGET
            ),
            'TARGET_THRESHOLD': preprocessing_info.get(
                'target_threshold',
                data_config.data.sequences.TARGET_THRESHOLD
            ),
        },
        'regime_params': preprocessing_info.get('regime_params'),
    }
    checkpoint_metadata = {
        key: value for key, value in checkpoint_metadata.items()
        if value is not None
    }

    if model_type == 'kronos':
        if backend == 'lightning':
            logger.warning("Kronos training is not integrated with Lightning yet. Falling back to custom backend.")
            backend = 'custom'

        kronos_result = train_kronos(
            loaders=loaders,
            config=config,
            device=device,
            model_type=model_type,
            checkpoint_metadata=checkpoint_metadata,
            logger=logger,
            num_features=num_features,
            embedding_sizes=embedding_sizes,
            args=args,
        )
        logger.info("Kronos training complete!")
        logger.info(f"Best validation loss: {kronos_result['best_score']:.6f}")
        logger.info(f"Saved best Kronos checkpoint to {kronos_result['best_model_path']}")
        logger.info(f"Saved final Kronos checkpoint to {kronos_result['final_model_path']}")
        return 0

    # Create model
    model = create_model(
        model_type=model_type,
        num_features=num_features,
        num_stocks=embedding_sizes['num_stocks'],
        num_groups=embedding_sizes['num_groups'],
        config=config,
        feature_cols=preprocessing_info.get('feature_cols'),
    )

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
    final_checkpoint_path = None
    if backend == 'lightning' and lightning_result is not None:
        best_score = lightning_result.get('best_score')
        if best_score is not None:
            logger.info(f"Best validation loss: {best_score:.6f}")
        if lightning_result.get('best_model_path'):
            logger.info(f"Saved Lightning custom-compatible checkpoint to {lightning_result['best_model_path']}")
        final_checkpoint_path = save_final_lightning_checkpoint(
            trainer=lightning_result['trainer'],
            lightning_module=lightning_result['module'],
            checkpoint_dir=config.model.checkpointing.CHECKPOINT_DIR,
            model_type=model_type,
            checkpoint_metadata=checkpoint_metadata,
        )
        logger.info(f"Saved final trained Lightning checkpoint to {final_checkpoint_path}")
    elif trainer is not None:
        if trainer.checkpoint.best_score is not None:
            logger.info(f"Best validation loss: {trainer.checkpoint.best_score:.6f}")
        else:
            logger.info("No validation-based best checkpoint was produced.")
        final_checkpoint_path = str(
            Path(config.model.checkpointing.CHECKPOINT_DIR) / f"{model_type}_final.pth"
        )
        trainer.save_model(final_checkpoint_path)
        logger.info(f"Saved final trained checkpoint to {final_checkpoint_path}")

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
