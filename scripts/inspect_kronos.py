import os
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import json

# Add project root to sys.path
sys.path.insert(0, "/app")

from src.config import load_config
from src.data.dataset import FinancialDataset
from src.evaluation.kronos import (
    load_kronos_checkpoint,
    build_kronos_sequence_metadata,
    _infer_feature_inverse_transform,
    _inverse_feature_values,
    _normalize_target_values,
)
from src.models.kronos_model import auto_regressive_inference

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config('model')
    data_dir = Path("/app/data/processed")
    
    # Load info
    with open(data_dir / 'info.json', 'r') as f:
        info = json.load(f)
        
    sequences = {}
    for file in ['features', 'stock_id', 'group_id', 'day', 'month', 'dividend_flag', 'target']:
        file_path = data_dir / 'test' / f'{file}.npy'
        if file_path.exists():
            sequences[file] = np.load(file_path)[:10]  # Limit to 10 samples for inspection
            
    dataset = FinancialDataset(sequences, config)
    
    # Load checkpoint
    checkpoint_path = "/app/models/checkpoints/kronos_best.pth"
    tokenizer, model, checkpoint = load_kronos_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        num_features=dataset.num_features,
        num_stocks=150,  # from logs
        num_groups=11,   # from logs
        device=device,
    )
    
    # Resolve metadata
    feature_cols = info.get('feature_cols') or []
    metadata = build_kronos_sequence_metadata(
        data_dir=data_dir,
        split='test',
        feature_cols=feature_cols,
        sequence_length=info['sequence_length'],
        prediction_horizon=info['prediction_horizon'],
        normalize_target=bool(info.get('normalize_target', False)),
        target_threshold=float(info.get('target_threshold', 1.0)),
        expected_samples=len(sequences['target']),
        max_samples=10,
    )
    
    # Get inputs
    x = torch.as_tensor(sequences["features"], dtype=torch.float32, device=device)
    from src.evaluation.kronos import _dates_to_stamp_tensor
    x_stamp = _dates_to_stamp_tensor(metadata["x_dates"], device)
    y_stamp = _dates_to_stamp_tensor(metadata["y_dates"], device)
    stock_id = torch.as_tensor(sequences["stock_id"], dtype=torch.long, device=device)
    group_id = torch.as_tensor(sequences["group_id"], dtype=torch.long, device=device)
    day = torch.as_tensor(sequences["day"], dtype=torch.long, device=device)
    month = torch.as_tensor(sequences["month"], dtype=torch.long, device=device)
    
    prediction_horizon = info['prediction_horizon']
    future_day = torch.as_tensor(
        pd.to_datetime(metadata["y_dates"].reshape(-1)).day.to_numpy().reshape(10, prediction_horizon),
        dtype=torch.long,
        device=device,
    )
    future_month = torch.as_tensor(
        pd.to_datetime(metadata["y_dates"].reshape(-1)).month.to_numpy().reshape(10, prediction_horizon),
        dtype=torch.long,
        device=device,
    )
    
    predictor_cfg = config.model.models.kronos.predictor
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
    
    close_index = list(feature_cols).index("close") if "close" in feature_cols else 0
    close_inverse_transform = _infer_feature_inverse_transform(str(data_dir), "close")
    
    print("Close inverse transform info:", close_inverse_transform)
    
    for i in range(10):
        raw_pred_close_val = preds[i, -1, close_index]
        predicted_close = _inverse_feature_values(preds[i, -1, close_index], close_inverse_transform)
        base_close = metadata["last_close"][i]
        predicted_return = ((predicted_close - base_close) / base_close) * 100.0 if base_close != 0 else 0.0
        
        print(f"Sample {i}:")
        print(f"  Raw Pred Close (model output space): {raw_pred_close_val}")
        print(f"  Predicted Close (denormalized): {predicted_close}")
        print(f"  Base Close (last close): {base_close}")
        print(f"  Predicted Return %: {predicted_return:.4f}%")

if __name__ == "__main__":
    main()
