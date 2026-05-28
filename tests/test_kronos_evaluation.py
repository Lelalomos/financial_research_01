import numpy as np
import pandas as pd

from src.evaluation.kronos import (
    _infer_feature_inverse_transform,
    _inverse_feature_values,
    build_kronos_sequence_metadata,
    build_kronos_report,
    compute_kronos_backtest_results,
    resolve_kronos_embedding_sizes,
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
    assert np.isclose(metadata["raw_targets"][0], expected_return)


def test_infer_feature_inverse_transform_recovers_affine_close_scale(tmp_path):
    processed_dir = tmp_path / "processed"
    cache_dir = processed_dir / ".cache" / "normalized_splits"
    cache_dir.mkdir(parents=True)

    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    normalized_close = np.array([-1.0, 0.0, 0.5, 2.0], dtype=float)
    raw_close = 20.0 + 5.0 * normalized_close

    pd.DataFrame(
        {
            "date": dates,
            "tic_id": [0] * len(dates),
            "close": normalized_close,
        }
    ).to_parquet(cache_dir / "train.parquet", index=False)

    pd.DataFrame(
        {
            "date": dates,
            "tic_id": [0] * len(dates),
            "close": raw_close,
        }
    ).to_parquet(tmp_path / "pre_normalized.parquet", index=False)

    transform = _infer_feature_inverse_transform(str(processed_dir), "close")
    restored = _inverse_feature_values(np.array([1.5, -0.5], dtype=np.float32), transform)

    assert transform["kind"] == "affine"
    assert np.allclose(restored, np.array([27.5, 17.5], dtype=np.float32))


def test_build_kronos_report_includes_percent_columns():
    predictions = np.array([0.5, -0.25], dtype=np.float32)
    targets = np.array([0.4, -0.1], dtype=np.float32)
    raw_predictions = np.array([1.1, -0.7], dtype=np.float32)
    raw_targets = np.array([0.8, -0.3], dtype=np.float32)
    stock_ids = np.array([0, 1], dtype=np.int64)
    group_ids = np.array([3, 4], dtype=np.int64)

    report_df, sector_stats = build_kronos_report(
        predictions=predictions,
        targets=targets,
        stock_ids=stock_ids,
        group_ids=group_ids,
        raw_predictions=raw_predictions,
        raw_targets=raw_targets,
        stock_id_to_ticker={0: "AAA", 1: "BBB"},
        group_id_to_sector={3: "Tech", 4: "Health"},
    )

    assert list(report_df["ticker"]) == ["AAA", "BBB"]
    assert list(report_df["sector"]) == ["Tech", "Health"]
    assert np.allclose(report_df["predict_target_percent"], raw_predictions)
    assert np.allclose(report_df["real_target_percent"], raw_targets)
    assert np.allclose(report_df["distance_percent"], raw_predictions - raw_targets)
    assert sector_stats["Tech"]["accuracy"] == 1.0


def test_resolve_kronos_embedding_sizes_prefers_full_metadata():
    num_stocks, num_groups = resolve_kronos_embedding_sizes(
        info={"num_stocks": 150, "num_groups": 11},
        fallback_sizes={"num_stocks": 1, "num_groups": 7},
    )

    assert num_stocks == 150
    assert num_groups == 11


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
