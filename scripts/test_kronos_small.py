#!/usr/bin/env python
"""
Small prepared-data smoke test for Kronos.

This script is meant for quick container-side validation with already prepared
sequence arrays. It does three things on a tiny subset:

1. loads prepared train/val sequences
2. runs a few small training steps for the Kronos tokenizer and generator
3. runs a one-step prediction decode from a validation window
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.config.config_loader import Config
from src.data.dataset import FinancialDataset
from src.models import create_kronos_model, create_kronos_tokenizer
from src.utils.logger import get_logger


REQUIRED_SEQUENCE_ARRAY_KEYS = ["features", "stock_id", "group_id", "day", "month", "target"]
OPTIONAL_SEQUENCE_ARRAY_KEYS = ["dividend_flag"]


def parse_args():
    parser = argparse.ArgumentParser(description="Small Kronos prepared-data smoke test")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Prepared data directory")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--train-samples", type=int, default=32, help="Number of train samples to use")
    parser.add_argument("--val-samples", type=int, default=8, help="Number of val samples to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of smoke-test epochs")
    parser.add_argument("--max-batches", type=int, default=2, help="Maximum batches per epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate override")
    return parser.parse_args()


def load_split_sequences(data_dir: Path, split: str) -> Optional[Dict[str, np.ndarray]]:
    split_dir = data_dir / split
    if not split_dir.exists():
        return None

    sequences = {}
    for key in REQUIRED_SEQUENCE_ARRAY_KEYS:
        file_path = split_dir / f"{key}.npy"
        if not file_path.exists():
            return None
        sequences[key] = np.load(file_path, allow_pickle=False)
    for key in OPTIONAL_SEQUENCE_ARRAY_KEYS:
        file_path = split_dir / f"{key}.npy"
        if file_path.exists():
            sequences[key] = np.load(file_path, allow_pickle=False)
    return sequences


def subset_sequences(sequences: Dict[str, np.ndarray], max_samples: int) -> Dict[str, np.ndarray]:
    count = min(max_samples, len(sequences["target"]))
    return {key: value[:count] for key, value in sequences.items()}


def build_local_model_config(train_sequences: Dict[str, np.ndarray], base_config=None) -> Config:
    source = base_config or load_config("model")
    config = Config(copy.deepcopy(source.to_dict()))

    network_cfg = config.model.models.kronos.network
    network_cfg.NUM_STOCKS = int(np.max(train_sequences["stock_id"])) + 1
    network_cfg.NUM_GROUPS = int(np.max(train_sequences["group_id"])) + 1
    config.model.models.kronos.tokenizer.D_IN = int(train_sequences["features"].shape[-1])
    return config


def make_dataloader(sequences: Dict[str, np.ndarray], batch_size: int) -> DataLoader:
    dataset = FinancialDataset(sequences, config=load_config("model"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=FinancialDataset.collate_fn,
    )


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_kronos_small(
    train_sequences: Dict[str, np.ndarray],
    val_sequences: Optional[Dict[str, np.ndarray]],
    config: Config,
    device: torch.device,
    batch_size: int,
    epochs: int,
    max_batches: int,
    learning_rate: float,
):
    if create_kronos_model is None or create_kronos_tokenizer is None:
        raise RuntimeError(
            "Kronos helpers are unavailable. Install required deps such as einops and huggingface_hub in the container."
        )

    tokenizer = create_kronos_tokenizer(config=config).to(device)
    model = create_kronos_model(config=config).to(device)

    train_loader = make_dataloader(train_sequences, batch_size=batch_size)
    val_loader = make_dataloader(val_sequences, batch_size=batch_size) if val_sequences is not None else None

    optimizer = torch.optim.Adam(
        list(tokenizer.parameters()) + list(model.parameters()),
        lr=learning_rate,
    )

    history = {"train_loss": None, "val_loss": None}

    for _epoch in range(epochs):
        tokenizer.train()
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            batch = _to_device(batch, device)
            features = batch["features"]

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
                stock_id=batch["stock_id"][:, :-1],
                group_id=batch["group_id"][:, :-1],
                day=batch["day"][:, :-1],
                month=batch["month"][:, :-1],
                dividend_flag=batch["dividend_flag"][:, :-1],
                use_teacher_forcing=True,
                s1_targets=target_s1,
            )
            token_loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, target_s1, target_s2)
            loss = recon_loss + pre_loss + bsq_loss + token_loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            batch_count += 1

        if batch_count > 0:
            history["train_loss"] = running_loss / batch_count

        if val_loader is not None:
            tokenizer.eval()
            model.eval()
            val_running = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_batches:
                        break
                    batch = _to_device(batch, device)
                    features = batch["features"]
                    (_, z_full), bsq_loss, _, _ = tokenizer(features)
                    recon_loss = F.mse_loss(z_full, features)
                    s1_ids, s2_ids = tokenizer.encode(features, half=True)
                    input_s1 = s1_ids[:, :-1]
                    input_s2 = s2_ids[:, :-1]
                    target_s1 = s1_ids[:, 1:]
                    target_s2 = s2_ids[:, 1:]
                    s1_logits, s2_logits = model(
                        input_s1,
                        input_s2,
                        stock_id=batch["stock_id"][:, :-1],
                        group_id=batch["group_id"][:, :-1],
                        day=batch["day"][:, :-1],
                        month=batch["month"][:, :-1],
                        dividend_flag=batch["dividend_flag"][:, :-1],
                        use_teacher_forcing=True,
                        s1_targets=target_s1,
                    )
                    token_loss, _, _ = model.head.compute_loss(s1_logits, s2_logits, target_s1, target_s2)
                    val_running += float((recon_loss + bsq_loss + token_loss).cpu().item())
                    val_batches += 1
            if val_batches > 0:
                history["val_loss"] = val_running / val_batches

    return tokenizer, model, history


def predict_next_step(
    tokenizer,
    model,
    sample_sequences: Dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    tokenizer.eval()

    features = torch.from_numpy(sample_sequences["features"][:1]).float().to(device)
    stock_id = torch.from_numpy(sample_sequences["stock_id"][:1]).long().to(device)
    group_id = torch.from_numpy(sample_sequences["group_id"][:1]).long().to(device)
    day = torch.from_numpy(sample_sequences["day"][:1]).long().to(device)
    month = torch.from_numpy(sample_sequences["month"][:1]).long().to(device)

    with torch.no_grad():
        s1_ids, s2_ids = tokenizer.encode(features, half=True)
        s1_logits, context = model.decode_s1(
            s1_ids,
            s2_ids,
            stock_id=stock_id,
            group_id=group_id,
            day=day,
            month=month,
        )
        next_s1 = torch.argmax(s1_logits[:, -1, :], dim=-1, keepdim=True)
        next_context = context[:, -1:, :]
        next_stock = stock_id[:, -1:].contiguous()
        next_group = group_id[:, -1:].contiguous()
        next_day = day[:, -1:].contiguous()
        next_month = month[:, -1:].contiguous()
        s2_logits = model.decode_s2(
            next_context,
            next_s1,
            padding_mask=None,
        )
        next_s2 = torch.argmax(s2_logits[:, -1, :], dim=-1, keepdim=True)
        decoded = tokenizer.decode([next_s1, next_s2], half=True)
        return decoded[:, -1, :].cpu().numpy()


def main():
    args = parse_args()
    logger = get_logger("test_kronos_small", log_dir="logs")
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("KRONOS SMALL PREPARED-DATA SMOKE TEST")
    logger.info("=" * 60)

    train_sequences = load_split_sequences(data_dir, "train")
    val_sequences = load_split_sequences(data_dir, "val")
    if train_sequences is None:
        logger.error(f"Missing prepared train sequence arrays in {data_dir / 'train'}")
        return 1

    train_sequences = subset_sequences(train_sequences, args.train_samples)
    if val_sequences is not None:
        val_sequences = subset_sequences(val_sequences, args.val_samples)

    model_config = build_local_model_config(train_sequences)
    model_config.model.training.LEARNING_RATE = args.learning_rate

    tokenizer, model, history = train_kronos_small(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        config=model_config,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_batches=args.max_batches,
        learning_rate=args.learning_rate,
    )

    prediction = predict_next_step(
        tokenizer=tokenizer,
        model=model,
        sample_sequences=val_sequences or train_sequences,
        device=device,
    )

    logger.info(f"Train loss: {history['train_loss']}")
    logger.info(f"Val loss: {history['val_loss']}")
    logger.info(f"Next-step decoded prediction shape: {prediction.shape}")
    logger.info(f"Next-step decoded prediction sample: {prediction[0].tolist()}")

    summary = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "prediction_shape": list(prediction.shape),
        "prediction_sample": prediction[0].tolist(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
