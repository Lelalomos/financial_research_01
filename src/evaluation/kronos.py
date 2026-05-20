"""
Kronos-specific evaluation helpers.

Kronos generates future feature rows instead of a direct scalar target, so the
standard evaluation path in this repo does not apply directly. These helpers
rebuild aligned split metadata, generate future rows, and convert the generated
close path into the same percent-change style signal used by the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.models import create_kronos_model, create_kronos_tokenizer
from src.models.kronos_model import auto_regressive_inference

from .metrics import (
    calculate_max_drawdown,
    calculate_metrics,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_turnover,
)


def _dates_to_stamp_tensor(date_array: np.ndarray, device) -> torch.Tensor:
    flat = pd.to_datetime(date_array.reshape(-1))
    stamp = np.stack(
        [
            flat.minute.to_numpy(dtype=np.int64),
            flat.hour.to_numpy(dtype=np.int64),
            flat.weekday.to_numpy(dtype=np.int64),
            flat.day.to_numpy(dtype=np.int64),
            flat.month.to_numpy(dtype=np.int64),
        ],
        axis=1,
    )
    stamp = stamp.reshape(date_array.shape[0], date_array.shape[1], 5)
    return torch.as_tensor(stamp, dtype=torch.float32, device=device)


def _normalize_target_values(returns: np.ndarray, normalize_target: bool, target_threshold: float) -> np.ndarray:
    if not normalize_target:
        return returns.astype(np.float32)
    return np.tanh(returns / max(float(target_threshold), 1e-8)).astype(np.float32)


def _load_split_with_raw_close(data_dir: Path, split: str) -> pd.DataFrame:
    split_path = data_dir / ".cache" / "normalized_splits" / f"{split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing normalized split cache: {split_path}")

    raw_path = data_dir.parent / "pre_normalized.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing pre-normalized parquet: {raw_path}")

    split_df = pd.read_parquet(split_path)
    raw_df = pd.read_parquet(raw_path, columns=["date", "tic_id", "close"]).rename(columns={"close": "raw_close"})
    merged = split_df.merge(raw_df, on=["date", "tic_id"], how="left")
    if merged["raw_close"].isna().any():
        missing = int(merged["raw_close"].isna().sum())
        raise ValueError(f"Could not align {missing} raw close values for Kronos evaluation.")
    return merged


def build_kronos_sequence_metadata(
    data_dir: Path,
    split: str,
    feature_cols,
    sequence_length: int,
    prediction_horizon: int,
    normalize_target: bool,
    target_threshold: float,
    expected_samples: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Rebuild sequence-aligned metadata using the same ordering logic as preprocessing.
    """
    df = _load_split_with_raw_close(data_dir, split)
    stocks = sorted(df["tic_id"].unique())

    x_dates = []
    y_dates = []
    last_close = []
    future_close = []
    targets = []

    for stock in stocks:
        stock_df = df[df["tic_id"] == stock].sort_values("date").copy()
        if len(stock_df) < sequence_length + prediction_horizon:
            continue

        feature_matrix = stock_df[feature_cols].to_numpy(dtype=np.float32)
        for i in range(len(stock_df) - sequence_length - prediction_horizon + 1):
            seq_features = feature_matrix[i : i + sequence_length]
            if np.isnan(seq_features).any():
                continue

            target_idx = i + sequence_length + prediction_horizon - 1
            target = stock_df.iloc[target_idx]["target"]
            if np.isnan(target):
                continue

            history_rows = stock_df.iloc[i : i + sequence_length]
            future_rows = stock_df.iloc[i + sequence_length : i + sequence_length + prediction_horizon]
            if len(future_rows) != prediction_horizon:
                continue
            if future_rows["raw_close"].isna().any():
                continue

            start_close = float(history_rows.iloc[-1]["raw_close"])
            end_close = float(future_rows.iloc[-1]["raw_close"])
            realized_return = ((end_close - start_close) / start_close) * 100.0 if start_close != 0 else 0.0

            x_dates.append(history_rows["date"].to_numpy(dtype="datetime64[ns]"))
            y_dates.append(future_rows["date"].to_numpy(dtype="datetime64[ns]"))
            last_close.append(start_close)
            future_close.append(end_close)
            targets.append(_normalize_target_values(np.array([realized_return]), normalize_target, target_threshold)[0])

    metadata = {
        "x_dates": np.asarray(x_dates, dtype="datetime64[ns]"),
        "y_dates": np.asarray(y_dates, dtype="datetime64[ns]"),
        "last_close": np.asarray(last_close, dtype=np.float32),
        "future_close": np.asarray(future_close, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
    }

    if max_samples is not None:
        limit = max(int(max_samples), 0)
        metadata = {key: value[:limit] for key, value in metadata.items()}

    if expected_samples is not None and len(metadata["targets"]) != int(expected_samples):
        raise ValueError(
            f"Kronos metadata/sample mismatch: expected {expected_samples}, got {len(metadata['targets'])}."
        )

    return metadata


def load_kronos_checkpoint(
    checkpoint_path: str,
    config,
    num_features: int,
    num_stocks: int,
    num_groups: int,
    device,
):
    if create_kronos_model is None or create_kronos_tokenizer is None:
        raise RuntimeError("Kronos helpers are unavailable. Install required Kronos dependencies first.")

    config.model.models.kronos.tokenizer.D_IN = int(num_features)
    config.model.models.kronos.network.NUM_STOCKS = int(max(num_stocks, 1))
    config.model.models.kronos.network.NUM_GROUPS = int(max(num_groups, 1))

    tokenizer = create_kronos_tokenizer(config=config).to(device)
    model = create_kronos_model(config=config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    tokenizer_state = checkpoint.get("tokenizer_state_dict")
    model_state = checkpoint.get("model_state_dict", checkpoint)
    if tokenizer_state is not None:
        tokenizer.load_state_dict(tokenizer_state)
    model.load_state_dict(model_state)
    tokenizer.eval()
    model.eval()
    return tokenizer, model, checkpoint


def generate_kronos_predictions(
    sequences: Dict[str, np.ndarray],
    metadata: Dict[str, np.ndarray],
    config,
    tokenizer,
    model,
    device,
    batch_size: int,
    normalize_target: bool,
    target_threshold: float,
    feature_cols,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_config = load_config("main")
    prediction_horizon = int(data_config.data.sequences.PREDICTION_HORIZON)

    predictor_cfg = config.model.models.kronos.predictor
    close_index = list(feature_cols).index("close") if "close" in feature_cols else 0
    predictions = []

    total = len(sequences["features"])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        x = torch.as_tensor(sequences["features"][start:end], dtype=torch.float32, device=device)
        x_stamp = _dates_to_stamp_tensor(metadata["x_dates"][start:end], device)
        y_stamp = _dates_to_stamp_tensor(metadata["y_dates"][start:end], device)
        stock_id = torch.as_tensor(sequences["stock_id"][start:end], dtype=torch.long, device=device)
        group_id = torch.as_tensor(sequences["group_id"][start:end], dtype=torch.long, device=device)
        day = torch.as_tensor(sequences["day"][start:end], dtype=torch.long, device=device)
        month = torch.as_tensor(sequences["month"][start:end], dtype=torch.long, device=device)
        future_day = torch.as_tensor(
            pd.to_datetime(metadata["y_dates"][start:end].reshape(-1)).day.to_numpy().reshape(end - start, prediction_horizon),
            dtype=torch.long,
            device=device,
        )
        future_month = torch.as_tensor(
            pd.to_datetime(metadata["y_dates"][start:end].reshape(-1)).month.to_numpy().reshape(end - start, prediction_horizon),
            dtype=torch.long,
            device=device,
        )

        preds = auto_regressive_inference(
            tokenizer,
            model,
            x,
            x_stamp,
            y_stamp,
            predictor_cfg.MAX_CONTEXT,
            prediction_horizon,
            clip=predictor_cfg.CLIP,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
            stock_id=stock_id,
            group_id=group_id,
            day=day,
            month=month,
            future_day=future_day,
            future_month=future_month,
        )

        predicted_close = preds[:, -1, close_index]
        base_close = metadata["last_close"][start:end]
        predicted_return = np.where(base_close != 0, ((predicted_close - base_close) / base_close) * 100.0, 0.0)
        predictions.append(_normalize_target_values(predicted_return, normalize_target, target_threshold))

    prediction_array = np.concatenate(predictions, axis=0) if predictions else np.array([], dtype=np.float32)
    return (
        prediction_array,
        metadata["targets"],
        sequences["stock_id"][:, 0],
        sequences["group_id"][:, 0],
    )


def compute_kronos_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    metrics = calculate_metrics(predictions, targets)
    returns = calculate_returns(predictions, targets)
    metrics["sharpe_ratio"] = calculate_sharpe_ratio(returns)
    metrics["max_drawdown"] = calculate_max_drawdown(returns)
    metrics["sortino_ratio"] = calculate_sortino_ratio(returns)
    metrics["total_return"] = float(np.sum(returns))
    return metrics


def build_kronos_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    stock_ids: np.ndarray,
    group_ids: np.ndarray,
    stock_id_to_ticker: Optional[Dict[int, str]] = None,
    group_id_to_sector: Optional[Dict[int, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    tickers = np.array([stock_id_to_ticker.get(int(sid), f"stock_{int(sid)}") for sid in stock_ids]) if stock_id_to_ticker else np.array([f"stock_{int(sid)}" for sid in stock_ids])
    sectors = np.array([group_id_to_sector.get(int(gid), f"sector_{int(gid)}") for gid in group_ids]) if group_id_to_sector else np.array([f"sector_{int(gid)}" for gid in group_ids])
    direction_scores = (np.sign(predictions) == np.sign(targets)).astype(int)

    report_df = pd.DataFrame(
        {
            "ticker": tickers,
            "stock_id": stock_ids,
            "sector": sectors,
            "group_id": group_ids,
            "real_target": targets,
            "predict_target": predictions,
            "distance": predictions - targets,
            "direction_score": direction_scores,
        }
    )

    sector_stats: Dict[str, Dict[str, float]] = {}
    for sector in np.unique(sectors):
        mask = sectors == sector
        total = int(mask.sum())
        correct = int(direction_scores[mask].sum())
        sector_stats[str(sector)] = {
            "total": total,
            "correct": correct,
            "accuracy": float(correct / total) if total > 0 else 0.0,
        }

    return report_df, sector_stats


def compute_kronos_backtest_results(
    predictions: np.ndarray,
    targets: np.ndarray,
    stock_ids: np.ndarray,
    group_ids: np.ndarray,
    prediction_threshold: float = 0.0,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    stock_id_to_ticker: Optional[Dict[int, str]] = None,
    group_id_to_sector: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    gross_returns = calculate_returns(predictions, targets, threshold=prediction_threshold)
    turnover = calculate_turnover(predictions, threshold=prediction_threshold)
    transaction_costs = turnover * commission * 100.0
    strategy_returns = gross_returns - transaction_costs
    cumulative_returns = np.cumprod(1 + strategy_returns / 100)
    portfolio_values = initial_capital * cumulative_returns

    tickers = np.array([stock_id_to_ticker.get(int(sid), f"stock_{int(sid)}") for sid in stock_ids]) if stock_id_to_ticker else np.array([f"stock_{int(sid)}" for sid in stock_ids])
    sectors = np.array([group_id_to_sector.get(int(gid), f"sector_{int(gid)}") for gid in group_ids]) if group_id_to_sector else np.array([f"sector_{int(gid)}" for gid in group_ids])
    direction_scores = (np.sign(predictions) == np.sign(targets)).astype(int)

    sector_stats: Dict[str, Dict[str, float]] = {}
    for sector in np.unique(sectors):
        mask = sectors == sector
        total = int(mask.sum())
        correct = int(direction_scores[mask].sum())
        sector_stats[str(sector)] = {
            "total": total,
            "correct": correct,
            "accuracy": float(correct / total) if total > 0 else 0.0,
        }

    winning_trades = strategy_returns > 0
    total_losses = abs(np.sum(strategy_returns[~winning_trades])) if np.any(~winning_trades) else 1.0

    return {
        "initial_capital": float(initial_capital),
        "final_capital": float(portfolio_values[-1]) if len(portfolio_values) else float(initial_capital),
        "total_return_pct": float(((portfolio_values[-1] / initial_capital) - 1) * 100) if len(portfolio_values) else 0.0,
        "total_return_value": float(portfolio_values[-1] - initial_capital) if len(portfolio_values) else 0.0,
        "sharpe_ratio": float(calculate_sharpe_ratio(strategy_returns)),
        "sortino_ratio": float(calculate_sortino_ratio(strategy_returns)),
        "max_drawdown_pct": float(calculate_max_drawdown(strategy_returns) * 100),
        "risk_adjusted_return": float(calculate_sharpe_ratio(strategy_returns)),
        "num_trades": int(len(strategy_returns)),
        "num_position_changes": int(np.count_nonzero(turnover)),
        "win_rate_pct": float(np.mean(winning_trades) * 100) if len(strategy_returns) > 0 else 0.0,
        "avg_win_pct": float(np.mean(strategy_returns[winning_trades])) if np.any(winning_trades) else 0.0,
        "avg_loss_pct": float(np.mean(strategy_returns[~winning_trades])) if np.any(~winning_trades) else 0.0,
        "profit_factor": float((np.sum(strategy_returns[winning_trades]) if np.any(winning_trades) else 0.0) / total_losses),
        "average_turnover": float(np.mean(turnover)) if len(turnover) > 0 else 0.0,
        "total_turnover": float(np.sum(turnover)) if len(turnover) > 0 else 0.0,
        "commission_rate": float(commission),
        "total_transaction_cost_pct": float(np.sum(transaction_costs)) if len(transaction_costs) > 0 else 0.0,
        "total_transaction_cost_value": float(initial_capital * ((np.sum(transaction_costs)) / 100.0)) if len(transaction_costs) > 0 else 0.0,
        "predictions": predictions,
        "targets": targets,
        "returns": strategy_returns,
        "gross_returns": gross_returns,
        "transaction_costs": transaction_costs,
        "turnover": turnover,
        "portfolio_values": portfolio_values,
        "tickers": tickers,
        "sectors": sectors,
        "direction_scores": direction_scores,
        "sector_stats": sector_stats,
        "stock_ids": stock_ids,
        "group_ids": group_ids,
    }
