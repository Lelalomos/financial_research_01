import numpy as np
import pandas as pd

from src.evaluation.kronos import (
    build_kronos_sequence_metadata,
    compute_kronos_backtest_results,
)


def test_build_kronos_sequence_metadata_aligns_windows(tmp_path):
    processed_dir = tmp_path / "processed"
    cache_dir = processed_dir / ".cache" / "normalized_splits"
    cache_dir.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    raw_close = np.array([100.0, 101.0, 102.0, 104.0, 103.0, 105.0], dtype=float)
    target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float)

    raw_df = pd.DataFrame(
        {
            "date": dates,
            "tic": ["AAA"] * len(dates),
            "tic_id": [0] * len(dates),
            "group": ["Tech"] * len(dates),
            "group_id": [1] * len(dates),
            "close": raw_close,
            "volume": np.arange(len(dates)) + 1,
            "day": dates.day,
            "month": dates.month,
            "target": target,
        }
    )
    split_df = raw_df.copy()
    split_df["close"] = (split_df["close"] - split_df["close"].mean()) / split_df["close"].std()

    raw_df.to_parquet(tmp_path / "pre_normalized.parquet", index=False)
    split_df.to_parquet(cache_dir / "test.parquet", index=False)

    metadata = build_kronos_sequence_metadata(
        data_dir=processed_dir,
        split="test",
        feature_cols=["close", "volume"],
        sequence_length=2,
        prediction_horizon=2,
        normalize_target=False,
        target_threshold=2.0,
    )

    assert metadata["x_dates"].shape == (3, 2)
    assert metadata["y_dates"].shape == (3, 2)
    assert np.isclose(metadata["last_close"][0], 101.0)
    assert np.isclose(metadata["future_close"][0], 104.0)
    expected_return = ((104.0 - 101.0) / 101.0) * 100.0
    assert np.isclose(metadata["targets"][0], expected_return)


def test_compute_kronos_backtest_results_returns_expected_shapes():
    predictions = np.array([0.5, -0.25, 0.1], dtype=np.float32)
    targets = np.array([0.4, -0.1, -0.2], dtype=np.float32)
    stock_ids = np.array([0, 1, 0], dtype=np.int64)
    group_ids = np.array([3, 3, 4], dtype=np.int64)

    results = compute_kronos_backtest_results(
        predictions=predictions,
        targets=targets,
        stock_ids=stock_ids,
        group_ids=group_ids,
        prediction_threshold=0.0,
        initial_capital=1000.0,
    )

    assert results["num_trades"] == 3
    assert results["predictions"].shape == (3,)
    assert results["targets"].shape == (3,)
    assert "sector_stats" in results
    assert results["final_capital"] > 0
