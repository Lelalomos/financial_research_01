"""
Market structure feature engineering utilities.

This module creates machine-learning-friendly support/resistance and market
structure features using past-only rolling calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketStructureFeatureResult:
    """Container for market structure feature outputs."""

    dataframe: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class MarketStructureFeatureSettings:
    """Resolved settings for market structure features."""

    enabled: bool
    windows: List[int]
    breakout_windows: List[int]
    count_window: int
    near_52w_threshold: float
    volume_window: int
    atr_windows: List[int]
    trend_windows: List[int]
    lags: List[int]


class MarketStructureFeatureBuilder:
    """Build vectorized market structure features for OHLCV data."""

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
    SIGNAL_COLUMNS = [
        "higher_high",
        "lower_high",
        "higher_low",
        "lower_low",
        "breakout_20d",
        "breakdown_20d",
        "trend_strength_score",
    ]

    def __init__(self, settings: MarketStructureFeatureSettings):
        self.settings = settings

    @classmethod
    def from_config(cls, geometric_config: Any) -> "MarketStructureFeatureBuilder":
        """Create a builder from the geometric config section."""
        settings = MarketStructureFeatureSettings(
            enabled=bool(getattr(geometric_config, "ENABLE_MARKET_STRUCTURE_FEATURES", True)),
            windows=sorted({int(v) for v in getattr(geometric_config, "MARKET_STRUCTURE_WINDOWS", [20, 60, 120, 252])}),
            breakout_windows=sorted({int(v) for v in getattr(geometric_config, "BREAKOUT_WINDOWS", [20, 60, 120])}),
            count_window=int(getattr(geometric_config, "MARKET_STRUCTURE_COUNT_WINDOW", 20)),
            near_52w_threshold=float(getattr(geometric_config, "NEAR_52W_THRESHOLD", 0.05)),
            volume_window=int(getattr(geometric_config, "VOLUME_CONFIRMATION_WINDOW", 20)),
            atr_windows=sorted({int(v) for v in getattr(geometric_config, "ATR_WINDOWS", [14, 20])}),
            trend_windows=sorted({int(v) for v in getattr(geometric_config, "TREND_WINDOWS", [20, 60])}),
            lags=sorted({int(v) for v in getattr(geometric_config, "MARKET_STRUCTURE_LAGS", [1, 3, 5, 10, 20])}),
        )
        return cls(settings=settings)

    def transform(self, df: pd.DataFrame) -> MarketStructureFeatureResult:
        """
        Create market structure features.

        Existing columns are preserved. Only missing columns are generated.
        """
        missing_required = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_required:
            raise ValueError(f"Missing required OHLCV columns: {sorted(missing_required)}")

        if not self.settings.enabled:
            return MarketStructureFeatureResult(
                dataframe=df,
                metadata={
                    "enabled": False,
                    "generated_features": [],
                    "preserved_features": [],
                    "signal_columns": [],
                },
            )

        ordered = df.copy()
        sort_cols = [col for col in ["tic", "date"] if col in ordered.columns]
        if sort_cols:
            ordered = ordered.sort_values(sort_cols).copy()

        grouped = self._groupby_ticker(ordered)
        generated: Dict[str, pd.Series] = {}
        preserved: List[str] = []
        eps = 1e-8

        close = ordered["close"].astype(float)
        high = ordered["high"].astype(float)
        low = ordered["low"].astype(float)
        volume = ordered["volume"].astype(float)
        safe_close = close.where(close.abs() > eps, np.nan)
        prev_close = grouped["close"].shift(1)
        prev_high = grouped["high"].shift(1)
        prev_low = grouped["low"].shift(1)

        def source_or_generated(column_name: str, computed: pd.Series) -> pd.Series:
            if column_name in generated:
                return generated[column_name]
            if column_name in ordered.columns:
                preserved.append(column_name)
                return pd.to_numeric(ordered[column_name], errors="coerce")
            generated[column_name] = pd.Series(computed, index=ordered.index)
            return generated[column_name]

        rolling_highs: Dict[int, pd.Series] = {}
        rolling_lows: Dict[int, pd.Series] = {}
        for window in self.settings.windows:
            rolling_high = grouped["high"].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).max()
            )
            rolling_low = grouped["low"].transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).min()
            )
            rolling_highs[window] = rolling_high
            rolling_lows[window] = rolling_low
            source_or_generated(f"distance_to_{window}d_high", (rolling_high - close) / safe_close)
            source_or_generated(f"distance_to_{window}d_low", (close - rolling_low) / safe_close)

        for window in self.settings.breakout_windows:
            rolling_high = rolling_highs.get(window)
            rolling_low = rolling_lows.get(window)
            if rolling_high is None:
                rolling_high = grouped["high"].transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).max()
                )
            if rolling_low is None:
                rolling_low = grouped["low"].transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).min()
                )
            prior_high = rolling_high.groupby(self._group_keys(ordered), sort=False).shift(1)
            prior_low = rolling_low.groupby(self._group_keys(ordered), sort=False).shift(1)
            source_or_generated(f"breakout_{window}d", (close > prior_high).astype(np.int8))
            source_or_generated(f"breakdown_{window}d", (close < prior_low).astype(np.int8))

        higher_high = source_or_generated("higher_high", (high > prev_high).astype(np.int8))
        lower_high = source_or_generated("lower_high", (high < prev_high).astype(np.int8))
        higher_low = source_or_generated("higher_low", (low > prev_low).astype(np.int8))
        lower_low = source_or_generated("lower_low", (low < prev_low).astype(np.int8))

        count_window = self.settings.count_window
        key_for_group = self._group_keys(ordered)
        source_or_generated(
            f"higher_high_count_{count_window}",
            higher_high.groupby(key_for_group, sort=False).transform(
                lambda s, w=count_window: s.rolling(window=w, min_periods=1).sum()
            ),
        )
        source_or_generated(
            f"higher_low_count_{count_window}",
            higher_low.groupby(key_for_group, sort=False).transform(
                lambda s, w=count_window: s.rolling(window=w, min_periods=1).sum()
            ),
        )
        source_or_generated(
            f"lower_high_count_{count_window}",
            lower_high.groupby(key_for_group, sort=False).transform(
                lambda s, w=count_window: s.rolling(window=w, min_periods=1).sum()
            ),
        )
        source_or_generated(
            f"lower_low_count_{count_window}",
            lower_low.groupby(key_for_group, sort=False).transform(
                lambda s, w=count_window: s.rolling(window=w, min_periods=1).sum()
            ),
        )

        high_252 = rolling_highs[252] if 252 in rolling_highs else grouped["high"].transform(
            lambda s: s.rolling(window=252, min_periods=1).max()
        )
        low_252 = rolling_lows[252] if 252 in rolling_lows else grouped["low"].transform(
            lambda s: s.rolling(window=252, min_periods=1).min()
        )
        distance_to_52w_high = source_or_generated("distance_to_52w_high", (high_252 - close) / safe_close)
        distance_to_52w_low = source_or_generated("distance_to_52w_low", (close - low_252) / safe_close)
        threshold = self.settings.near_52w_threshold
        source_or_generated("near_52w_high", (distance_to_52w_high <= threshold).astype(np.int8))
        source_or_generated("near_52w_low", (distance_to_52w_low <= threshold).astype(np.int8))

        volume_mean = grouped["volume"].transform(
            lambda s, w=self.settings.volume_window: s.rolling(window=w, min_periods=1).mean()
        )
        breakout_any = pd.Series(0, index=ordered.index, dtype=np.int8)
        breakdown_any = pd.Series(0, index=ordered.index, dtype=np.int8)
        for window in self.settings.breakout_windows:
            breakout_any = np.maximum(
                breakout_any,
                pd.to_numeric(source_or_generated(f"breakout_{window}d", pd.Series(0, index=ordered.index)), errors="coerce").fillna(0).astype(np.int8),
            )
            breakdown_any = np.maximum(
                breakdown_any,
                pd.to_numeric(source_or_generated(f"breakdown_{window}d", pd.Series(0, index=ordered.index)), errors="coerce").fillna(0).astype(np.int8),
            )
        volume_ratio = volume / volume_mean.where(volume_mean.abs() > eps, np.nan)
        source_or_generated("breakout_volume_ratio", volume_ratio.where(breakout_any > 0, 0.0))
        source_or_generated("breakdown_volume_ratio", volume_ratio.where(breakdown_any > 0, 0.0))
        source_or_generated("volume_spike_ratio", volume_ratio)
        source_or_generated("volume_momentum", volume_ratio - 1.0)

        high_low_range = (high - low).clip(lower=0.0)
        true_range = pd.concat(
            [
                high_low_range,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series: Dict[int, pd.Series] = {}
        for window in self.settings.atr_windows:
            atr_series[window] = true_range.groupby(key_for_group, sort=False).transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )
            source_or_generated(f"atr_{window}", atr_series[window])
        atr_14 = atr_series.get(14, true_range.groupby(key_for_group, sort=False).transform(lambda s: s.rolling(window=14, min_periods=1).mean()))
        atr_20 = atr_series.get(20, true_range.groupby(key_for_group, sort=False).transform(lambda s: s.rolling(window=20, min_periods=1).mean()))
        source_or_generated("atr_ratio", atr_14 / atr_20.where(atr_20.abs() > eps, np.nan))

        returns_1d = grouped["close"].pct_change(1)
        for window in self.settings.trend_windows:
            source_or_generated(
                f"rolling_volatility_{window}",
                returns_1d.groupby(key_for_group, sort=False).transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).std(ddof=0)
                ),
            )

        rolling_high_20 = rolling_highs.get(20, grouped["high"].transform(lambda s: s.rolling(window=20, min_periods=1).max()))
        rolling_low_20 = rolling_lows.get(20, grouped["low"].transform(lambda s: s.rolling(window=20, min_periods=1).min()))
        channel_range_20 = (rolling_high_20 - rolling_low_20).where((rolling_high_20 - rolling_low_20).abs() > eps, np.nan)
        channel_position_20 = ((close - rolling_low_20) / channel_range_20).fillna(0.5)
        structure_balance_20 = (
            pd.to_numeric(source_or_generated(f"higher_high_count_{count_window}", pd.Series(0, index=ordered.index)), errors="coerce").fillna(0.0)
            + pd.to_numeric(source_or_generated(f"higher_low_count_{count_window}", pd.Series(0, index=ordered.index)), errors="coerce").fillna(0.0)
            - pd.to_numeric(source_or_generated(f"lower_high_count_{count_window}", pd.Series(0, index=ordered.index)), errors="coerce").fillna(0.0)
            - pd.to_numeric(source_or_generated(f"lower_low_count_{count_window}", pd.Series(0, index=ordered.index)), errors="coerce").fillna(0.0)
        ) / max(float(count_window), 1.0)
        trend_strength = (0.5 * structure_balance_20) + (0.5 * ((2.0 * channel_position_20) - 1.0))
        source_or_generated("trend_strength_score", trend_strength.clip(-1.0, 1.0))

        direction = np.sign(close.groupby(key_for_group, sort=False).diff()).fillna(0.0)
        for window in self.settings.trend_windows:
            source_or_generated(
                f"trend_persistence_{window}",
                direction.groupby(key_for_group, sort=False).transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
                ),
            )

        for signal_column in self.SIGNAL_COLUMNS:
            signal_series = pd.to_numeric(
                source_or_generated(signal_column, pd.Series(0, index=ordered.index)),
                errors="coerce",
            ).fillna(0.0)
            for lag in self.settings.lags:
                source_or_generated(
                    f"{signal_column}_lag_{lag}",
                    signal_series.groupby(key_for_group, sort=False).shift(lag),
                )

        generated_columns = [col for col in generated if col not in df.columns]
        generated_df = pd.DataFrame({col: generated[col] for col in generated_columns}, index=ordered.index)
        if not generated_df.empty:
            generated_df = generated_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            discrete_columns = {
                *(f"breakout_{window}d" for window in self.settings.breakout_windows),
                *(f"breakdown_{window}d" for window in self.settings.breakout_windows),
                "higher_high",
                "lower_high",
                "higher_low",
                "lower_low",
                "near_52w_high",
                "near_52w_low",
            }
            for col in generated_df.columns:
                if col in discrete_columns or any(
                    col.startswith(f"{signal}_lag_")
                    for signal in [
                        "higher_high",
                        "lower_high",
                        "higher_low",
                        "lower_low",
                        "breakout_20d",
                        "breakdown_20d",
                    ]
                ):
                    generated_df[col] = generated_df[col].astype(np.int8)
            ordered = pd.concat([ordered, generated_df], axis=1)

        if sort_cols:
            ordered = ordered.sort_index()

        metadata = {
            "enabled": True,
            "generated_features": generated_columns,
            "preserved_features": sorted(set(preserved)),
            "signal_columns": [col for col in self.SIGNAL_COLUMNS if col in ordered.columns],
            "windows": {
                "market_structure": self.settings.windows,
                "breakout": self.settings.breakout_windows,
                "trend": self.settings.trend_windows,
                "lags": self.settings.lags,
            },
        }
        return MarketStructureFeatureResult(dataframe=ordered, metadata=metadata)

    @staticmethod
    def _group_keys(df: pd.DataFrame) -> pd.Series:
        """Return the ticker grouping key or a single synthetic group."""
        if "tic" in df.columns:
            return df["tic"]
        return pd.Series("ALL", index=df.index)

    @classmethod
    def _groupby_ticker(cls, df: pd.DataFrame) -> Any:
        """Group by ticker if present, otherwise treat the frame as one series."""
        return df.groupby(cls._group_keys(df), sort=False)
