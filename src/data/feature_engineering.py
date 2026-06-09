"""
Feature engineering for Multi-Model Financial Forecasting.

This module handles:
- Technical indicator calculation (EMA, RSI, StochRSI, MACD)
- Custom candlestick feature engineering
- Time-based features (day, month)
- Target variable calculation (percent change with configurable horizon)
- External data merging (VIX, commodities, treasury yields)
- Sector-based grouping from index files
"""

import pandas as pd
import numpy as np
import talib
from stockstats import StockDataFrame as Sdf
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import json

from src.config import load_config
from src.data.feature_engineering_polars import (
    PolarsFeatureEngineeringError,
    add_fibonacci_features_polars,
    add_time_features_polars,
    merge_external_data_polars,
)
from src.utils.logger import get_logger
from src.data.financial_metrics_loader import FinancialMetricsLoader
from src.data.market_structure_features import MarketStructureFeatureBuilder


_CANDLESTICK_PATTERN_COLUMNS = [
    "doji",
    "hammer",
    "inverted_hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
    "morning_star",
    "evening_star",
]

_CANDLESTICK_ROLLING_WINDOWS = [5, 10, 20, 60]
_SEQUENCE_LAG_WINDOWS = [1, 3, 5, 10, 20]
_SEQUENCE_ROLLING_WINDOWS = [5, 10, 20, 60]


class FeatureEngineer:
    """
    Feature engineering for financial data.

        Creates features for model training including:
        - Technical indicators (EMA, RSI, StochRSI, MACD)
        - Custom candlestick features and pattern flags
        - Time features (day of month, month)
        - Target calculation (percent change)
    """

    def __init__(self, config=None, sector_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize feature engineer.

        Args:
            config: Configuration object (defaults to load_config('main') if None)
            sector_mapping: Optional dict mapping ticker -> sector/group.
                          If None, will load from index file.
        """
        if config is None:
            config = load_config('main')
        self.config = config
        self.logger = get_logger("feature_engineering", log_dir="logs")

        # Initialize financial metrics loader if enabled
        if config.data.features.FEATURE_FLAGS.get('financial_metrics', False):
            self.financial_loader = FinancialMetricsLoader(config)
        else:
            self.financial_loader = None

        # Load sector mapping from index file if not provided
        if sector_mapping is None:
            self.sector_mapping = self._load_sector_mapping_from_index()
        else:
            self.sector_mapping = sector_mapping

    def _load_sector_mapping_from_index(self) -> Dict[str, str]:
        """
        Load sector mapping from the configured index file.

        Returns:
            Dictionary mapping ticker symbol to sector name
        """
        index_path = Path(self.config.data.sources.RAW_DATA_INDEX_PATH) / self.config.data.sources.INDEX_FILE
        sector_map = {}

        try:
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index_data = json.load(f)

                if 'Components' in index_data:
                    for key in index_data['Components']:
                        component = index_data['Components'][key]
                        if 'Code' in component and 'Sector' in component:
                            sector_map[component['Code']] = component['Sector']

                self.logger.info(f"Loaded sector mapping for {len(sector_map)} stocks from {index_path.name}")
            else:
                self.logger.warning(f"Index file not found: {index_path}")
        except Exception as e:
            self.logger.warning(f"Failed to load sector mapping: {e}")

        return sector_map

    def add_group_from_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'group' column based on sector mapping from index file.

        The sector information is loaded from the index file and used to
        create a 'group' column that will be encoded as 'group_id' for
        group embeddings in the model.

        Args:
            df: DataFrame with 'tic' column

        Returns:
            DataFrame with added 'group' column
        """
        self.logger.info("Adding group (sector) information...")

        result = df.copy()

        if not self.sector_mapping:
            self.logger.warning("No sector mapping available, using 'Unknown' group")
            result['group'] = 'Unknown'
            return result

        # Map ticker to sector/group
        result['group'] = result['tic'].map(self.sector_mapping)

        # Fill missing with 'Unknown'
        result['group'] = result['group'].fillna('Unknown')

        # Log group statistics
        group_counts = result['group'].value_counts()
        self.logger.info(f"Added {len(group_counts)} unique groups:")
        for group, count in group_counts.head(10).items():
            self.logger.info(f"  {group}: {count} rows")

        unmapped = result[result['group'] == 'Unknown']['tic'].nunique()
        if unmapped > 0:
            self.logger.warning(f"  {unmapped} tickers not found in sector mapping")

        return result

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        """Compute past-only rolling z-score."""
        if min_periods is None:
            min_periods = window
        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
        safe_std = rolling_std.replace(0.0, np.nan)
        return (series - rolling_mean) / safe_std

    @staticmethod
    def _rolling_ols_beta(y: np.ndarray, x: np.ndarray) -> float:
        """Estimate hedge ratio beta from rolling OLS with intercept."""
        design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
        beta, _intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(beta)

    @staticmethod
    def _safe_adf(series: np.ndarray) -> Tuple[float, float]:
        """Run ADF safely and return (stat, pvalue)."""
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for cointegration ADF features. "
                "Install statsmodels>=0.14.0."
            ) from exc

        clean = np.asarray(series, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size < 10 or np.allclose(clean, clean[0]):
            return np.nan, np.nan

        maxlag = min(1, clean.size // 2 - 1)
        if maxlag < 0:
            return np.nan, np.nan

        try:
            stat, pvalue, *_ = adfuller(clean, maxlag=maxlag, autolag=None)
            return float(stat), float(pvalue)
        except Exception:
            return np.nan, np.nan

    @staticmethod
    def _select_pair_peer(
        target_ticker: str,
        window_returns: pd.DataFrame,
    ) -> Optional[str]:
        """Pick the same-sector peer with highest absolute return correlation."""
        if target_ticker not in window_returns.columns:
            return None

        target_returns = window_returns[target_ticker]
        correlations = {}
        for candidate in window_returns.columns:
            if candidate == target_ticker:
                continue
            candidate_returns = window_returns[candidate]
            valid_mask = target_returns.notna() & candidate_returns.notna()
            if valid_mask.sum() < 10:
                continue
            corr = target_returns[valid_mask].corr(candidate_returns[valid_mask])
            if pd.notna(corr):
                correlations[candidate] = abs(float(corr))

        if not correlations:
            return None
        return max(correlations, key=correlations.get)

    def _compute_pair_cointegration_features(
        self,
        df: pd.DataFrame,
        window: int,
        normalization_window: int,
    ) -> pd.DataFrame:
        """Create rolling pair spread features within each sector."""
        result = df.copy()
        result["pair_beta"] = np.nan
        result["spread"] = np.nan
        result["rolling_mean_spread"] = np.nan
        result["rolling_std_spread"] = np.nan
        result["spread_zscore"] = np.nan
        result["spread_norm"] = np.nan
        result["spread_adf_stat"] = np.nan
        result["spread_adf_pvalue"] = np.nan

        for group_name, group_df in result.groupby("group", sort=False):
            tickers = sorted(group_df["tic"].dropna().unique())
            if len(tickers) < 2:
                continue

            sector_df = group_df.sort_values(["date", "tic"])
            close_pivot = sector_df.pivot(index="date", columns="tic", values="close").sort_index()
            return_pivot = close_pivot.pct_change(fill_method=None)

            for ticker in tickers:
                if ticker not in close_pivot.columns:
                    continue

                pair_beta = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
                spread = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
                rolling_mean_spread = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
                rolling_std_spread = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
                spread_zscore = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
                spread_adf_stat = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
                spread_adf_pvalue = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)

                for end_idx in range(window - 1, len(close_pivot)):
                    window_slice = close_pivot.iloc[end_idx - window + 1:end_idx + 1]
                    window_returns = return_pivot.iloc[end_idx - window + 1:end_idx + 1]
                    if window_slice[ticker].isna().any():
                        continue

                    complete_window = window_slice.loc[:, window_slice.notna().all(axis=0)]
                    if ticker not in complete_window.columns or complete_window.shape[1] < 2:
                        continue

                    peer = self._select_pair_peer(ticker, window_returns[complete_window.columns])
                    if peer is None:
                        continue

                    y = complete_window[ticker].to_numpy(dtype=np.float64, copy=False)
                    x = complete_window[peer].to_numpy(dtype=np.float64, copy=False)
                    if np.nanstd(x) == 0.0:
                        continue

                    beta = self._rolling_ols_beta(y, x)
                    spread_window = y - beta * x
                    mean_spread = float(np.mean(spread_window))
                    std_spread = float(np.std(spread_window, ddof=0))
                    if std_spread == 0.0:
                        std_spread = np.nan

                    current_date = close_pivot.index[end_idx]
                    current_spread = float(spread_window[-1])

                    pair_beta.loc[current_date] = beta
                    spread.loc[current_date] = current_spread
                    rolling_mean_spread.loc[current_date] = mean_spread
                    rolling_std_spread.loc[current_date] = std_spread
                    if np.isfinite(std_spread):
                        spread_zscore.loc[current_date] = (current_spread - mean_spread) / std_spread

                    adf_stat, adf_pvalue = self._safe_adf(spread_window)
                    spread_adf_stat.loc[current_date] = adf_stat
                    spread_adf_pvalue.loc[current_date] = adf_pvalue

                spread_norm = self._rolling_zscore(spread, window=normalization_window)

                ticker_mask = (result["group"] == group_name) & (result["tic"] == ticker)
                ticker_dates = pd.to_datetime(result.loc[ticker_mask, "date"])
                result.loc[ticker_mask, "pair_beta"] = ticker_dates.map(pair_beta).to_numpy(dtype=float)
                result.loc[ticker_mask, "spread"] = ticker_dates.map(spread).to_numpy(dtype=float)
                result.loc[ticker_mask, "rolling_mean_spread"] = ticker_dates.map(rolling_mean_spread).to_numpy(dtype=float)
                result.loc[ticker_mask, "rolling_std_spread"] = ticker_dates.map(rolling_std_spread).to_numpy(dtype=float)
                result.loc[ticker_mask, "spread_zscore"] = ticker_dates.map(spread_zscore).to_numpy(dtype=float)
                result.loc[ticker_mask, "spread_norm"] = ticker_dates.map(spread_norm).to_numpy(dtype=float)
                result.loc[ticker_mask, "spread_adf_stat"] = ticker_dates.map(spread_adf_stat).to_numpy(dtype=float)
                result.loc[ticker_mask, "spread_adf_pvalue"] = ticker_dates.map(spread_adf_pvalue).to_numpy(dtype=float)
        return result

    def _compute_sector_relative_features(
        self,
        df: pd.DataFrame,
        normalization_window: int,
    ) -> pd.DataFrame:
        """Create sector-relative price and return features."""
        result = df.copy()
        result = result.sort_values(["date", "tic"]).reset_index(drop=True)

        result["stock_return_1d"] = result.groupby("tic", sort=False)["close"].pct_change(fill_method=None)

        sector_close_sum = result.groupby(["group", "date"])["close"].transform("sum")
        sector_close_count = result.groupby(["group", "date"])["close"].transform("count")
        peer_close_mean = (sector_close_sum - result["close"]) / (sector_close_count - 1).where(sector_close_count > 1, np.nan)
        result["relative_price_vs_sector"] = (result["close"] - peer_close_mean) / peer_close_mean.abs().replace(0.0, np.nan)

        sector_return_sum = result.groupby(["group", "date"])["stock_return_1d"].transform("sum")
        sector_return_count = result.groupby(["group", "date"])["stock_return_1d"].transform("count")
        peer_return_mean = (sector_return_sum - result["stock_return_1d"]) / (sector_return_count - 1).where(sector_return_count > 1, np.nan)
        result["relative_return_vs_sector"] = result["stock_return_1d"] - peer_return_mean

        result["relative_price_vs_sector_norm"] = np.nan
        for ticker, ticker_df in result.groupby("tic", sort=False):
            zscore = self._rolling_zscore(ticker_df["relative_price_vs_sector"], window=normalization_window)
            result.loc[ticker_df.index, "relative_price_vs_sector_norm"] = zscore.to_numpy(dtype=float)

        return result

    def _compute_johansen_sector_features(
        self,
        df: pd.DataFrame,
        window: int,
        normalization_window: int,
        det_order: int,
        k_ar_diff: int,
    ) -> pd.DataFrame:
        """Create sector equilibrium features with rolling Johansen vectors."""
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for Johansen cointegration features. "
                "Install statsmodels>=0.14.0."
            ) from exc

        result = df.copy()
        result["equilibrium_gap"] = np.nan
        result["equilibrium_zscore"] = np.nan
        result["equilibrium_gap_norm"] = np.nan
        result["equilibrium_adf_stat"] = np.nan
        result["equilibrium_adf_pvalue"] = np.nan

        for group_name, group_df in result.groupby("group", sort=False):
            tickers = sorted(group_df["tic"].dropna().unique())
            if len(tickers) < 2:
                continue

            sector_df = group_df.sort_values(["date", "tic"])
            close_pivot = sector_df.pivot(index="date", columns="tic", values="close").sort_index()
            log_close_pivot = np.log(close_pivot.where(close_pivot > 0))

            equilibrium_gap = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
            equilibrium_zscore = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
            equilibrium_gap_norm = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
            equilibrium_adf_stat = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)
            equilibrium_adf_pvalue = pd.Series(np.nan, index=close_pivot.index, dtype=np.float64)

            for end_idx in range(window - 1, len(log_close_pivot)):
                window_slice = log_close_pivot.iloc[end_idx - window + 1:end_idx + 1]
                complete_window = window_slice.loc[:, window_slice.notna().all(axis=0)]
                if complete_window.shape[1] < 2:
                    continue

                try:
                    johansen_result = coint_johansen(
                        complete_window.to_numpy(dtype=np.float64, copy=False),
                        det_order=det_order,
                        k_ar_diff=k_ar_diff,
                    )
                except Exception:
                    continue

                beta_vector = johansen_result.evec[:, 0]
                equilibrium_series = complete_window.to_numpy(dtype=np.float64, copy=False) @ beta_vector
                current_date = log_close_pivot.index[end_idx]
                current_gap = float(equilibrium_series[-1])
                equilibrium_gap.loc[current_date] = current_gap

                norm_window = equilibrium_series[-min(len(equilibrium_series), normalization_window):]
                mean_gap = float(np.mean(norm_window))
                std_gap = float(np.std(norm_window, ddof=0))
                if std_gap != 0.0:
                    equilibrium_zscore.loc[current_date] = (current_gap - mean_gap) / std_gap

                adf_stat, adf_pvalue = self._safe_adf(equilibrium_series)
                equilibrium_adf_stat.loc[current_date] = adf_stat
                equilibrium_adf_pvalue.loc[current_date] = adf_pvalue

            equilibrium_gap_norm = self._rolling_zscore(
                equilibrium_gap,
                window=normalization_window,
            )

            sector_mask = result["group"] == group_name
            sector_dates = pd.to_datetime(result.loc[sector_mask, "date"])
            result.loc[sector_mask, "equilibrium_gap"] = sector_dates.map(equilibrium_gap).to_numpy(dtype=float)
            result.loc[sector_mask, "equilibrium_zscore"] = sector_dates.map(equilibrium_zscore).to_numpy(dtype=float)
            result.loc[sector_mask, "equilibrium_gap_norm"] = sector_dates.map(equilibrium_gap_norm).to_numpy(dtype=float)
            result.loc[sector_mask, "equilibrium_adf_stat"] = sector_dates.map(equilibrium_adf_stat).to_numpy(dtype=float)
            result.loc[sector_mask, "equilibrium_adf_pvalue"] = sector_dates.map(equilibrium_adf_pvalue).to_numpy(dtype=float)
        return result

    def add_cointegration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pair-spread and sector-equilibrium features with rolling windows.

        All calculations use only data up to the current timestamp.
        """
        if not self.config.data.features.FEATURE_FLAGS.get("cointegration_features", False):
            self.logger.info("Skipping cointegration features (disabled in config)")
            return df

        self.logger.info("Adding rolling cointegration features...")
        result = df.copy().sort_values(["tic", "date"]).reset_index(drop=True)

        cointegration_cfg = getattr(self.config.data, "cointegration", None)
        window = int(getattr(cointegration_cfg, "ROLLING_WINDOW", 252))
        normalization_window = int(getattr(cointegration_cfg, "NORMALIZATION_WINDOW", window))
        det_order = int(getattr(cointegration_cfg, "JOHANSEN_DET_ORDER", 0))
        k_ar_diff = int(getattr(cointegration_cfg, "JOHANSEN_K_AR_DIFF", 1))

        result = self._compute_sector_relative_features(
            result,
            normalization_window=normalization_window,
        )
        result = self._compute_pair_cointegration_features(
            result,
            window=window,
            normalization_window=normalization_window,
        )
        result = self._compute_johansen_sector_features(
            result,
            window=window,
            normalization_window=normalization_window,
            det_order=det_order,
            k_ar_diff=k_ar_diff,
        )

        self.logger.info("Added cointegration features")
        return result

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.

        Args:
            df: DataFrame with OHLCV data, must have columns: open, high, low, close, volume

        Returns:
            DataFrame with added technical indicator columns
        """
        self.logger.info("Adding technical indicators...")

        result = df.copy()

        # Group by ticker and calculate indicators for each stock
        for ticker in result['tic'].unique():
            mask = result['tic'] == ticker
            stock_df = result[mask].copy()

            # Sort by date
            stock_df = stock_df.sort_values('date')

            # Calculate EMAs
            for period in self.config.data.technical_indicators.EMA_PERIODS:
                if self.config.data.features.FEATURE_FLAGS.get('ema_features', True):
                    stock_df[f'ema_{period}'] = talib.EMA(stock_df['close'], timeperiod=period)

            # Calculate RSI
            if self.config.data.features.FEATURE_FLAGS.get('rsi_features', True):
                stock_df[f'rsi_{self.config.data.technical_indicators.RSI_PERIOD}'] = talib.RSI(
                    stock_df['close'],
                    timeperiod=self.config.data.technical_indicators.RSI_PERIOD
                ) / 100.0  # Normalize to 0-1

            # Calculate Stochastic RSI
            if self.config.data.features.FEATURE_FLAGS.get('stochrsi_features', True):
                stochrsi = talib.STOCHRSI(
                    stock_df['close'],
                    timeperiod=self.config.data.technical_indicators.STOCHRSI_PERIOD,
                    fastk_period=14,
                    fastd_period=3,
                    fastd_matype=0
                )
                stock_df[f'stochrsi_{self.config.data.technical_indicators.STOCHRSI_PERIOD}'] = stochrsi[0] / 100.0

            # Calculate MACD
            if self.config.data.features.FEATURE_FLAGS.get('macd_features', True):
                macd, macdsignal, macdhist = talib.MACD(
                    stock_df['close'],
                    fastperiod=self.config.data.technical_indicators.MACD_PARAMS[0],
                    slowperiod=self.config.data.technical_indicators.MACD_PARAMS[1],
                    signalperiod=self.config.data.technical_indicators.MACD_PARAMS[2]
                )
                stock_df['macd'] = macd
                stock_df['macd_signal'] = macdsignal
                stock_df['macd_hist'] = macdhist

            # Update result
            result.loc[mask, stock_df.columns] = stock_df

        self.logger.info(f"Added technical indicators. Shape: {result.shape}")
        return result

    def add_geometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add geometric and high-level structural features.
        Inspired by QuantAgent's Trend and Indicator agents.
        
        Features:
        - ATR_14_NORM: Volatility normalized by price
        - ROC_10: Rate of change (momentum)
        - BB_WIDTH_20: Bollinger Band width (volatility squeeze)
        - SLOPE_SUP_20: Slope of support line (20-day low)
        - SLOPE_RES_20: Slope of resistance line (20-day high)
        - CHANNEL_COMPRESSION_20: Channel width normalized by price
        - CHANNEL_POSITION_20: Relative close position inside the channel
        - DIST_TO_SWING_HIGH_20: Normalized distance to rolling swing high
        - DIST_TO_SWING_LOW_20: Normalized distance to rolling swing low
        - DAYS_SINCE_SWING_HIGH_20: Bars since the most recent rolling swing high
        - DAYS_SINCE_SWING_LOW_20: Bars since the most recent rolling swing low
        - OPT_SLOPE_SUP_30: Pivot-anchored optimized support slope
        - OPT_SLOPE_RES_30: Pivot-anchored optimized resistance slope
        - OPT_CHANNEL_WIDTH_30: Width between optimized resistance/support
        """
        if not self.config.data.features.FEATURE_FLAGS.get('geometric_features', True):
            return df

        self.logger.info("Adding geometric and structural features...")
        result = df.copy()
        geometric_config = self.config.data.geometric
        channel_window = geometric_config.CHANNEL_WINDOW
        swing_window = geometric_config.SWING_WINDOW
        epsilon = 1e-8

        for ticker in result['tic'].unique():
            mask = result['tic'] == ticker
            stock_df = result[mask].copy().sort_values('date')

            if geometric_config.ENABLE_ATR_FEATURE:
                # 1. Normalized ATR (Volatility)
                atr = talib.ATR(stock_df['high'].values, stock_df['low'].values, stock_df['close'].values, timeperiod=14)
                stock_df['atr_14_norm'] = atr / stock_df['close']

            if geometric_config.ENABLE_ROC_FEATURE:
                # 2. ROC (Momentum)
                stock_df['roc_10'] = talib.ROC(stock_df['close'].values, timeperiod=10)

            if geometric_config.ENABLE_BB_WIDTH_FEATURE:
                # 3. Bollinger Band Width (Squeeze)
                upper, middle, lower = talib.BBANDS(stock_df['close'].values, timeperiod=20)
                stock_df['bb_width_20'] = (upper - lower) / middle

            needs_channel_bounds = (
                geometric_config.ENABLE_SLOPE_FEATURES
                or geometric_config.ENABLE_CHANNEL_COMPRESSION
                or geometric_config.ENABLE_CHANNEL_POSITION
            )

            if needs_channel_bounds:
                rolling_min = stock_df['low'].rolling(window=channel_window).min()
                rolling_max = stock_df['high'].rolling(window=channel_window).max()
                channel_range = rolling_max - rolling_min

            if geometric_config.ENABLE_SLOPE_FEATURES:
                # 4. Support/Resistance Slopes (Trend Geometry)
                # We use rolling min/max as proxies for support/resistance levels

                # Fill initial NaNs to allow talib to work
                stock_df['slope_sup_20'] = talib.LINEARREG_SLOPE(
                    rolling_min.bfill().values,
                    timeperiod=channel_window,
                )
                stock_df['slope_res_20'] = talib.LINEARREG_SLOPE(
                    rolling_max.bfill().values,
                    timeperiod=channel_window,
                )

            if geometric_config.ENABLE_CHANNEL_COMPRESSION:
                close_scale = stock_df['close'].abs().clip(lower=epsilon)
                stock_df['channel_compression_20'] = channel_range / close_scale

            if geometric_config.ENABLE_CHANNEL_POSITION:
                safe_range = channel_range.clip(lower=epsilon)
                stock_df['channel_position_20'] = (stock_df['close'] - rolling_min) / safe_range

            needs_swing_bounds = (
                geometric_config.ENABLE_SWING_DISTANCE
                or geometric_config.ENABLE_SWING_TIME_DISTANCE
            )

            if needs_swing_bounds:
                swing_high = stock_df['high'].rolling(window=swing_window).max()
                swing_low = stock_df['low'].rolling(window=swing_window).min()

            if geometric_config.ENABLE_SWING_DISTANCE:
                close_scale = stock_df['close'].abs().clip(lower=epsilon)
                stock_df['dist_to_swing_high_20'] = (stock_df['close'] - swing_high) / close_scale
                stock_df['dist_to_swing_low_20'] = (stock_df['close'] - swing_low) / close_scale

            if geometric_config.ENABLE_SWING_TIME_DISTANCE:
                highs = stock_df['high'].to_numpy()
                lows = stock_df['low'].to_numpy()
                days_since_swing_high = np.full(len(stock_df), np.nan, dtype=np.float64)
                days_since_swing_low = np.full(len(stock_df), np.nan, dtype=np.float64)

                for idx in range(swing_window - 1, len(stock_df)):
                    start_idx = idx - swing_window + 1
                    high_window = highs[start_idx:idx + 1]
                    low_window = lows[start_idx:idx + 1]
                    last_high_offset = np.where(high_window == np.max(high_window))[0][-1]
                    last_low_offset = np.where(low_window == np.min(low_window))[0][-1]
                    days_since_swing_high[idx] = idx - (start_idx + last_high_offset)
                    days_since_swing_low[idx] = idx - (start_idx + last_low_offset)

                stock_df['days_since_swing_high_20'] = days_since_swing_high
                stock_df['days_since_swing_low_20'] = days_since_swing_low

            if (
                geometric_config.ENABLE_OPTIMIZED_TRENDLINES
                or geometric_config.ENABLE_OPTIMIZED_CHANNEL_WIDTH
            ):
                opt_support, opt_resistance, opt_width = self._compute_optimized_trendlines(
                    low_series=stock_df['low'],
                    high_series=stock_df['high'],
                    close_series=stock_df['close'],
                    window=geometric_config.TRENDLINE_WINDOW,
                    tolerance=geometric_config.TRENDLINE_TOLERANCE,
                    max_iterations=geometric_config.TRENDLINE_MAX_ITERATIONS,
                )

                if geometric_config.ENABLE_OPTIMIZED_TRENDLINES:
                    stock_df['opt_slope_sup_30'] = opt_support
                    stock_df['opt_slope_res_30'] = opt_resistance

                if geometric_config.ENABLE_OPTIMIZED_CHANNEL_WIDTH:
                    stock_df['opt_channel_width_30'] = opt_width

            result.loc[mask, stock_df.columns] = stock_df

        return result

    @staticmethod
    def _trendline_error(
        y: np.ndarray,
        support: bool,
        pivot: int,
        slope: float,
        tolerance: float,
    ) -> float:
        """Return constrained squared error for a candidate trendline."""
        intercept = -slope * pivot + y[pivot]
        line_vals = slope * np.arange(len(y), dtype=np.float64) + intercept
        diffs = line_vals - y

        if support and diffs.max() > tolerance:
            return -1.0
        if not support and diffs.min() < -tolerance:
            return -1.0
        return float(np.square(diffs).sum())

    @classmethod
    def _optimize_trendline_slope(
        cls,
        y: np.ndarray,
        support: bool,
        pivot: int,
        initial_slope: float,
        tolerance: float,
        max_iterations: int,
    ) -> tuple[float, float]:
        """Iteratively rotate a pivot-anchored line without breaking constraints."""
        y = np.asarray(y, dtype=np.float64)
        if len(y) < 2:
            intercept = y[0] if len(y) == 1 else 0.0
            return initial_slope, intercept

        price_span = max(float(y.max() - y.min()), tolerance)
        slope_step = max(price_span / max(len(y) - 1, 1), tolerance)
        best_slope = float(initial_slope)
        best_err = cls._trendline_error(y, support, pivot, best_slope, tolerance)

        if best_err < 0:
            best_slope = 0.0
            best_err = cls._trendline_error(y, support, pivot, best_slope, tolerance)

        iterations = 0
        while slope_step > tolerance and iterations < max_iterations:
            candidates = []
            for direction in (-1.0, 1.0):
                trial_slope = best_slope + direction * slope_step
                trial_err = cls._trendline_error(y, support, pivot, trial_slope, tolerance)
                if trial_err >= 0:
                    candidates.append((trial_err, trial_slope))

            if candidates:
                trial_err, trial_slope = min(candidates, key=lambda item: item[0])
                if best_err < 0 or trial_err <= best_err:
                    best_err = trial_err
                    best_slope = trial_slope
                else:
                    slope_step *= 0.5
            else:
                slope_step *= 0.5

            iterations += 1

        intercept = -best_slope * pivot + y[pivot]
        return best_slope, intercept

    @classmethod
    def _compute_optimized_trendlines(
        cls,
        low_series: pd.Series,
        high_series: pd.Series,
        close_series: pd.Series,
        window: int,
        tolerance: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute optimized support/resistance slopes and channel width per row."""
        length = len(close_series)
        support_slopes = np.full(length, np.nan, dtype=np.float64)
        resistance_slopes = np.full(length, np.nan, dtype=np.float64)
        channel_widths = np.full(length, np.nan, dtype=np.float64)

        lows = low_series.to_numpy(dtype=np.float64)
        highs = high_series.to_numpy(dtype=np.float64)
        closes = close_series.to_numpy(dtype=np.float64)

        for idx in range(window - 1, length):
            start_idx = idx - window + 1
            low_window = lows[start_idx:idx + 1]
            high_window = highs[start_idx:idx + 1]
            close_window = closes[start_idx:idx + 1]
            x = np.arange(window, dtype=np.float64)

            seed_slope, seed_intercept = np.polyfit(x, close_window, 1)
            seed_line = seed_slope * x + seed_intercept

            support_pivot = int(np.argmin(low_window - seed_line))
            resistance_pivot = int(np.argmax(high_window - seed_line))

            support_slope, support_intercept = cls._optimize_trendline_slope(
                y=low_window,
                support=True,
                pivot=support_pivot,
                initial_slope=seed_slope,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            resistance_slope, resistance_intercept = cls._optimize_trendline_slope(
                y=high_window,
                support=False,
                pivot=resistance_pivot,
                initial_slope=seed_slope,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

            support_line_end = support_slope * (window - 1) + support_intercept
            resistance_line_end = resistance_slope * (window - 1) + resistance_intercept

            support_slopes[idx] = support_slope
            resistance_slopes[idx] = resistance_slope
            channel_widths[idx] = resistance_line_end - support_line_end

        return support_slopes, resistance_slopes, channel_widths

    def add_fibonacci_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fibonacci retracement features to DataFrame.

        Calculates swing high/low over a rolling window and computes
        Fibonacci retracement levels (38.2%, 50%, 61.8%) with normalized
        distance features for RNN compatibility.

        Args:
            df: DataFrame with OHLCV data, must have columns: high, low, close

        Returns:
            DataFrame with added Fibonacci features
        """
        if not self.config.data.features.FEATURE_FLAGS.get('fibonacci_features', False):
            self.logger.info("Skipping Fibonacci features (disabled in config)")
            return df

        self.logger.info("Adding Fibonacci retracement features...")

        result = df.copy()
        window = self.config.data.fibonacci.FIBONACCI_WINDOW
        if self.config.data.features.FEATURE_FLAGS.get('polars_fibonacci_features', False):
            self.logger.info("Using opt-in Polars implementation for Fibonacci features")
            try:
                result = add_fibonacci_features_polars(result, window=window)
                self.logger.info(f"Added Fibonacci features. Shape: {result.shape}")
                return result
            except PolarsFeatureEngineeringError as exc:
                self.logger.warning(f"{exc} Falling back to pandas Fibonacci implementation.")

        # Group by ticker and calculate Fibonacci levels for each stock
        for ticker in result['tic'].unique():
            mask = result['tic'] == ticker
            stock_df = result[mask].copy()

            # Sort by date to ensure correct rolling window
            stock_df = stock_df.sort_values('date')

            # Rolling swing high / low
            stock_df['swing_high'] = stock_df['high'].rolling(window).max()
            stock_df['swing_low'] = stock_df['low'].rolling(window).min()

            # Price range
            stock_df['fib_range'] = stock_df['swing_high'] - stock_df['swing_low']

            # Fibonacci retracement levels
            stock_df['fib_38'] = stock_df['swing_high'] - 0.382 * stock_df['fib_range']
            stock_df['fib_50'] = stock_df['swing_high'] - 0.5 * stock_df['fib_range']
            stock_df['fib_61'] = stock_df['swing_high'] - 0.618 * stock_df['fib_range']

            # Normalized distance features (RNN-friendly)
            with np.errstate(divide='ignore', invalid='ignore'):
                stock_df['dist_fib_38'] = (stock_df['close'] - stock_df['fib_38']) / stock_df['fib_range']
                stock_df['dist_fib_50'] = (stock_df['close'] - stock_df['fib_50']) / stock_df['fib_range']
                stock_df['dist_fib_61'] = (stock_df['close'] - stock_df['fib_61']) / stock_df['fib_range']

            # Break indicator (1 if close breaks below 61.8% level, 0 otherwise)
            stock_df['break_fib_61'] = (stock_df['close'] < stock_df['fib_61']).astype(int)

            # Update result
            result.loc[mask, stock_df.columns] = stock_df

        self.logger.info(f"Added Fibonacci features. Shape: {result.shape}")
        return result

    def add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add configurable market structure features.

        Existing columns are preserved. Only missing columns are generated.
        """
        if not self.config.data.features.FEATURE_FLAGS.get('geometric_features', True):
            return df

        geometric_config = self.config.data.geometric
        if not geometric_config.get("ENABLE_MARKET_STRUCTURE_FEATURES", True):
            self.logger.info("Skipping market structure features (disabled in config)")
            return df

        self.logger.info("Adding market structure features...")
        builder = MarketStructureFeatureBuilder.from_config(geometric_config)
        feature_result = builder.transform(df)
        self.logger.info(
            f"Added {len(feature_result.metadata['generated_features'])} missing market structure features"
        )
        return feature_result.dataframe

    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom candlestick, lag, return, volume, and rolling features.

        Existing columns are preserved as-is. Only missing features are added.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added candlestick feature columns and no NaN values
        """
        candlestick_cfg = self.config.data.candlestick if 'candlestick' in self.config.data else None
        use_candlestick = self.config.data.features.FEATURE_FLAGS.get('candlestick_patterns', True)
        if candlestick_cfg is not None:
            use_candlestick = use_candlestick and candlestick_cfg.get('USE_CANDLESTICK_PATTERNS', True)

        if not use_candlestick:
            self.logger.info("Skipping candlestick patterns (disabled in config)")
            return df

        self.logger.info("Adding custom candlestick and sequence features...")

        result = df.copy()
        ordered = result.sort_values(["tic", "date"]).copy()
        eps = 1e-8

        open_ = ordered["open"].astype(float)
        high = ordered["high"].astype(float)
        low = ordered["low"].astype(float)
        close = ordered["close"].astype(float)
        generated: Dict[str, pd.Series] = {}
        grouped = ordered.groupby("tic", sort=False)

        def _source_or_generated(column_name: str, computed: pd.Series) -> pd.Series:
            if column_name in ordered.columns:
                return pd.to_numeric(ordered[column_name], errors="coerce")
            generated[column_name] = pd.Series(computed, index=ordered.index)
            return computed

        body_size = _source_or_generated("body_size", (close - open_).abs())
        upper_shadow = _source_or_generated(
            "upper_shadow",
            (high - np.maximum(open_, close)).clip(lower=0.0),
        )
        lower_shadow = _source_or_generated(
            "lower_shadow",
            (np.minimum(open_, close) - low).clip(lower=0.0),
        )
        high_low_range = _source_or_generated(
            "high_low_range",
            (high - low).clip(lower=0.0),
        )
        safe_range = high_low_range.where(high_low_range > eps, np.nan)
        safe_close = close.where(close.abs() > eps, np.nan)

        candle_direction = _source_or_generated(
            "candle_direction",
            np.sign(close - open_).astype(np.int8),
        )
        body_ratio = _source_or_generated("body_ratio", body_size / safe_range)
        upper_shadow_ratio = _source_or_generated("upper_shadow_ratio", upper_shadow / safe_range)
        lower_shadow_ratio = _source_or_generated("lower_shadow_ratio", lower_shadow / safe_range)
        close_position = _source_or_generated("close_position", (close - low) / safe_range)
        open_position = _source_or_generated("open_position", (open_ - low) / safe_range)

        prev_close = grouped["close"].shift(1)
        prev_high = grouped["high"].shift(1)
        prev_low = grouped["low"].shift(1)
        prev_open = grouped["open"].shift(1)
        prev2_open = grouped["open"].shift(2)
        prev2_close = grouped["close"].shift(2)

        _source_or_generated("gap_up", (open_ > prev_high).astype(np.int8))
        _source_or_generated("gap_down", (open_ < prev_low).astype(np.int8))
        _source_or_generated("gap_size", open_ - prev_close)
        _source_or_generated(
            "overnight_return",
            (open_ - prev_close) / prev_close.where(prev_close.abs() > eps, np.nan),
        )

        grouped_body_size = body_size.groupby(ordered["tic"], sort=False)
        grouped_upper_shadow = upper_shadow.groupby(ordered["tic"], sort=False)
        grouped_lower_shadow = lower_shadow.groupby(ordered["tic"], sort=False)
        _source_or_generated("body_size_change", grouped_body_size.diff())
        _source_or_generated(
            "body_size_ema_5",
            grouped_body_size.transform(lambda s: s.ewm(span=5, adjust=False, min_periods=1).mean()),
        )
        _source_or_generated(
            "body_size_ema_20",
            grouped_body_size.transform(lambda s: s.ewm(span=20, adjust=False, min_periods=1).mean()),
        )
        _source_or_generated("upper_shadow_change", grouped_upper_shadow.diff())
        _source_or_generated("lower_shadow_change", grouped_lower_shadow.diff())

        true_range = pd.concat(
            [
                high_low_range,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = true_range.groupby(ordered["tic"], sort=False).transform(
            lambda s: s.rolling(window=14, min_periods=1).mean()
        )
        atr_20 = true_range.groupby(ordered["tic"], sort=False).transform(
            lambda s: s.rolling(window=20, min_periods=1).mean()
        )
        atr = _source_or_generated("atr", atr_14)
        _source_or_generated("atr_14", atr_14)
        _source_or_generated("atr_20", atr_20)
        _source_or_generated(
            "rolling_high_low_range_5",
            grouped["high"].transform(lambda s: s.rolling(window=5, min_periods=1).max())
            - grouped["low"].transform(lambda s: s.rolling(window=5, min_periods=1).min()),
        )
        _source_or_generated(
            "rolling_high_low_range_20",
            grouped["high"].transform(lambda s: s.rolling(window=20, min_periods=1).max())
            - grouped["low"].transform(lambda s: s.rolling(window=20, min_periods=1).min()),
        )

        _source_or_generated("body_size_pct", body_size / safe_close)
        _source_or_generated("upper_shadow_pct", upper_shadow / safe_close)
        _source_or_generated("lower_shadow_pct", lower_shadow / safe_close)
        _source_or_generated("range_pct", high_low_range / safe_close)

        returns_1d = grouped["close"].pct_change(1)
        returns_5d = grouped["close"].pct_change(5)
        returns_20d = grouped["close"].pct_change(20)
        _source_or_generated("return_1d", returns_1d)
        _source_or_generated("return_5d", returns_5d)
        _source_or_generated("return_20d", returns_20d)

        _source_or_generated(
            "rolling_volatility",
            returns_1d.groupby(ordered["tic"], sort=False).transform(
                lambda s: s.rolling(window=20, min_periods=1).std(ddof=0)
            ),
        )

        support_level = grouped["low"].transform(lambda s: s.rolling(window=20, min_periods=1).min())
        resistance_level = grouped["high"].transform(lambda s: s.rolling(window=20, min_periods=1).max())
        _source_or_generated("support_distance", (close - support_level) / safe_close)
        _source_or_generated("resistance_distance", (resistance_level - close) / safe_close)

        prev_support = support_level.groupby(ordered["tic"], sort=False).shift(1)
        prev_resistance = resistance_level.groupby(ordered["tic"], sort=False).shift(1)
        breakout_signal = np.where(close > prev_resistance, 1, np.where(close < prev_support, -1, 0))
        _source_or_generated("breakout_signal", pd.Series(breakout_signal, index=ordered.index, dtype=np.int8))

        volume = ordered["volume"].astype(float)
        volume_mean_20 = volume.groupby(ordered["tic"], sort=False).transform(
            lambda s: s.rolling(window=20, min_periods=1).mean()
        )
        volume_std_20 = volume.groupby(ordered["tic"], sort=False).transform(
            lambda s: s.rolling(window=20, min_periods=1).std(ddof=0)
        )
        _source_or_generated("volume_momentum", volume / volume_mean_20.where(volume_mean_20.abs() > eps, np.nan) - 1.0)
        volume_spike = (volume > (volume_mean_20 + 2.0 * volume_std_20.fillna(0.0))).astype(np.int8)
        _source_or_generated("volume_spike", volume_spike)

        for lag in _SEQUENCE_LAG_WINDOWS:
            _source_or_generated(f"lag_{lag}", grouped["close"].shift(lag))

        for window in _SEQUENCE_ROLLING_WINDOWS:
            _source_or_generated(
                f"rolling_mean_{window}",
                grouped["close"].transform(lambda s, w=window: s.rolling(window=w, min_periods=1).mean()),
            )
            _source_or_generated(
                f"rolling_std_{window}",
                grouped["close"].transform(lambda s, w=window: s.rolling(window=w, min_periods=1).std(ddof=0)),
            )
            _source_or_generated(
                f"rolling_min_{window}",
                grouped["close"].transform(lambda s, w=window: s.rolling(window=w, min_periods=1).min()),
            )
            _source_or_generated(
                f"rolling_max_{window}",
                grouped["close"].transform(lambda s, w=window: s.rolling(window=w, min_periods=1).max()),
            )

        for base_name, base_series in [
            ("body_size", body_size),
            ("upper_shadow", upper_shadow),
            ("lower_shadow", lower_shadow),
        ]:
            base_grouped = base_series.groupby(ordered["tic"], sort=False)
            for window in _CANDLESTICK_ROLLING_WINDOWS:
                _source_or_generated(
                    f"{base_name}_rolling_mean_{window}",
                    base_grouped.transform(lambda s, w=window: s.rolling(window=w, min_periods=1).mean()),
                )
                _source_or_generated(
                    f"{base_name}_rolling_std_{window}",
                    base_grouped.transform(lambda s, w=window: s.rolling(window=w, min_periods=1).std(ddof=0)),
                )
                _source_or_generated(
                    f"{base_name}_rolling_zscore_{window}",
                    base_grouped.transform(lambda s, w=window: self._rolling_zscore(s, window=w, min_periods=1)),
                )

        prev_body_ratio = body_ratio.groupby(ordered["tic"], sort=False).shift(1)
        prev2_body_ratio = body_ratio.groupby(ordered["tic"], sort=False).shift(2)

        small_body = body_ratio <= 0.1
        bullish = close > open_
        bearish = close < open_
        body_near_high = close_position >= 0.6
        body_near_low = close_position <= 0.4
        long_lower_shadow = lower_shadow >= (2.0 * body_size)
        long_upper_shadow = upper_shadow >= (2.0 * body_size)
        short_upper_shadow = upper_shadow <= body_size
        short_lower_shadow = lower_shadow <= body_size

        _source_or_generated("doji", small_body.astype(np.int8))
        _source_or_generated(
            "hammer",
            (long_lower_shadow & short_upper_shadow & body_near_high).astype(np.int8),
        )
        _source_or_generated(
            "inverted_hammer",
            (long_upper_shadow & short_lower_shadow & body_near_low & (candle_direction >= 0)).astype(np.int8),
        )
        _source_or_generated(
            "shooting_star",
            (long_upper_shadow & short_lower_shadow & body_near_low & (candle_direction <= 0)).astype(np.int8),
        )
        _source_or_generated(
            "bullish_engulfing",
            ((prev_close < prev_open) & bullish & (open_ <= prev_close) & (close >= prev_open)).astype(np.int8),
        )
        _source_or_generated(
            "bearish_engulfing",
            ((prev_close > prev_open) & bearish & (open_ >= prev_close) & (close <= prev_open)).astype(np.int8),
        )
        middle_of_first = (prev2_open + prev2_close) / 2.0
        _source_or_generated(
            "morning_star",
            (
                (prev2_close < prev2_open)
                & (prev2_body_ratio > 0.5)
                & (prev_body_ratio <= 0.3)
                & bullish
                & (close >= middle_of_first)
            ).astype(np.int8),
        )
        _source_or_generated(
            "evening_star",
            (
                (prev2_close > prev2_open)
                & (prev2_body_ratio > 0.5)
                & (prev_body_ratio <= 0.3)
                & bearish
                & (close <= middle_of_first)
            ).astype(np.int8),
        )

        generated_cols = [col for col in generated if col not in result.columns]
        if generated_cols:
            generated_df = pd.DataFrame({col: generated[col] for col in generated_cols}, index=ordered.index)
            generated_df = (
                generated_df
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            for col in ["candle_direction", "gap_up", "gap_down", "breakout_signal", "volume_spike", *_CANDLESTICK_PATTERN_COLUMNS]:
                if col in generated_cols:
                    generated_df[col] = generated_df[col].astype(np.int8)
            ordered = pd.concat([ordered, generated_df], axis=1)

        ordered = ordered.sort_index()
        self.logger.info(f"Added {len(generated_cols)} missing candlestick and sequence features")
        return ordered

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features (day, month).

        Args:
            df: DataFrame with date column

        Returns:
            DataFrame with added time features
        """
        if not self.config.data.features.FEATURE_FLAGS.get('time_features', True):
            self.logger.info("Skipping time features (disabled in config)")
            return df

        self.logger.info("Adding time features...")

        if self.config.data.features.FEATURE_FLAGS.get('polars_time_features', False):
            self.logger.info("Using opt-in Polars implementation for time features")
            try:
                result = add_time_features_polars(df)
                self.logger.info(f"Added time features. Shape: {result.shape}")
                return result
            except PolarsFeatureEngineeringError as exc:
                self.logger.warning(f"{exc} Falling back to pandas time feature implementation.")

        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])

        # Add day of month (1-31)
        result['day'] = result['date'].dt.day.astype(int)

        # Add month (1-12)
        result['month'] = result['date'].dt.month.astype(int)

        # Add day of week (0-6, Monday=0) - optional
        result['dayofweek'] = result['date'].dt.dayofweek.astype(int)

        self.logger.info(f"Added time features. Shape: {result.shape}")
        return result

    def calculate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate target variable (percent price change).

        target = (price[t+H] - price[t]) / price[t] * 100

        Args:
            df: DataFrame with close prices

        Returns:
            DataFrame with target column
        """
        self.logger.info(f"Calculating target (prediction horizon: {self.config.data.sequences.PREDICTION_HORIZON} days)...")

        result = df.copy()

        # Calculate target for each ticker
        for ticker in result['tic'].unique():
            mask = result['tic'] == ticker
            stock_df = result[mask].copy()

            # Sort by date
            stock_df = stock_df.sort_values('date')

            # Calculate future returns
            future_close = stock_df['close'].shift(-self.config.data.sequences.PREDICTION_HORIZON)
            target = (future_close - stock_df['close']) / stock_df['close'] * 100

            # Apply tanh normalization if enabled
            if self.config.data.sequences.NORMALIZE_TARGET:
                threshold = self.config.data.sequences.TARGET_THRESHOLD
                target = np.tanh(target / threshold)

            stock_df['target'] = target

            # Update result
            result.loc[mask, 'target'] = stock_df['target'].values

        # Drop rows where target is NaN (last H days of each stock)
        initial_rows = len(result)
        result = result.dropna(subset=['target'])
        dropped_rows = initial_rows - len(result)

        self.logger.info(f"Calculated target. Dropped {dropped_rows} rows with NaN target")

        # Log target statistics
        self.logger.info(f"Target statistics:")
        self.logger.info(f"  Mean: {result['target'].mean():.4f}")
        self.logger.info(f"  Std: {result['target'].std():.4f}")
        self.logger.info(f"  Min: {result['target'].min():.4f}")
        self.logger.info(f"  Max: {result['target'].max():.4f}")
        self.logger.info(f"  Median: {result['target'].median():.4f}")

        return result

    def merge_external_data(
        self,
        stock_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        commodities_df: Optional[pd.DataFrame] = None,
        treasury_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge external data (VIX, commodities, treasury yields) into stock DataFrame.

        Args:
            stock_df: Stock price DataFrame
            vix_df: VIX index DataFrame
            commodities_df: Commodities DataFrame
            treasury_df: Treasury yields DataFrame

        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging external data...")

        if self.config.data.features.FEATURE_FLAGS.get('polars_external_merges', False):
            self.logger.info("Using opt-in Polars implementation for external data merges")
            commodity_cols = list(self.config.data.sources.COMMODITIES.values())
            try:
                result = merge_external_data_polars(
                    stock_df=stock_df,
                    vix_df=vix_df,
                    commodities_df=commodities_df,
                    treasury_df=treasury_df,
                    include_vix=self.config.data.features.FEATURE_FLAGS.get('vix', False),
                    commodity_columns=(
                        commodity_cols if self.config.data.features.FEATURE_FLAGS.get('commodities', False) else None
                    ),
                    include_treasury=self.config.data.features.FEATURE_FLAGS.get('treasury_yields', False),
                )
                self.logger.info("Merged external data")
                return result
            except PolarsFeatureEngineeringError as exc:
                self.logger.warning(f"{exc} Falling back to pandas external-data merge implementation.")

        result = stock_df.copy()

        # Merge VIX
        if vix_df is not None and self.config.data.features.FEATURE_FLAGS.get('vix', False):
            result = pd.merge(
                result,
                vix_df[['date', 'vix']],
                on='date',
                how='left'
            )
            self.logger.info(f"Merged VIX data")

        # Merge commodities
        if commodities_df is not None and self.config.data.features.FEATURE_FLAGS.get('commodities', False):
            commodity_cols = ['date'] + list(self.config.data.sources.COMMODITIES.values())
            result = pd.merge(
                result,
                commodities_df[commodity_cols],
                on='date',
                how='left'
            )
            self.logger.info(f"Merged {len(self.config.data.sources.COMMODITIES)} commodities")

        # Merge treasury yields
        if treasury_df is not None and self.config.data.features.FEATURE_FLAGS.get('treasury_yields', False):
            result = pd.merge(
                result,
                treasury_df[['date', 'bondyield']],
                on='date',
                how='left'
            )
            self.logger.info(f"Merged treasury yields")

        return result

    def add_financial_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add financial metrics from JSON files.

        Loads financial data from raw_data/ticket_data/us/{TICKER}.json files
        and adds 8 financial metrics:
        - PE Ratio, PEG Ratio, EPS (from Highlights section)
        - ROE, ROI, Debt-to-Equity, Debt-to-Asset, Current Ratio (calculated from quarterly data)

        Args:
            df: DataFrame with 'tic' and 'date' columns

        Returns:
            DataFrame with added financial metric columns
        """
        if not self.config.data.features.FEATURE_FLAGS.get('financial_metrics', False):
            self.logger.info("Skipping financial metrics (disabled in config)")
            return df

        if self.financial_loader is None:
            self.logger.warning("Financial metrics loader not initialized")
            return df

        self.logger.info("Adding financial metrics...")

        result = df.copy()
        result['date'] = pd.to_datetime(result['date'])

        # Get unique tickers
        tickers = result['tic'].unique()

        # For each ticker, load and merge metrics
        all_metrics = []
        successful_tickers = 0

        for ticker in tickers:
            # Get daily price data for this ticker (defines date range)
            ticker_df = result[result['tic'] == ticker][['date']].copy()

            if ticker_df.empty:
                self.logger.warning(f"No price data for {ticker}")
                continue

            # Load metrics for this ticker
            metrics_df = self.financial_loader.load_metrics_for_ticker(ticker, ticker_df)

            if metrics_df is None:
                self.logger.warning(f"Failed to load metrics for {ticker}")
                continue

            all_metrics.append(metrics_df)
            successful_tickers += 1

        # Concatenate all metrics
        if all_metrics:
            metrics_combined = pd.concat(all_metrics, ignore_index=True)

            # Merge with result on [tic, date]
            result = pd.merge(
                result,
                metrics_combined,
                on=['tic', 'date'],
                how='left'
            )

            self.logger.info(f"Added financial metrics for {successful_tickers}/{len(tickers)} tickers")
            financial_metrics_columns = ['pe_ratio', 'peg_ratio', 'eps', 'dividend_flag', 'roe', 'roi',
                                         'debt_to_equity', 'debt_to_asset', 'current_ratio']
            self.logger.info(f"Added columns: {', '.join(financial_metrics_columns)}")
            self.logger.info(f"Shape: {result.shape}")
        else:
            self.logger.warning("No financial metrics added - all tickers failed to load")

        return result

    def add_all_features(
        self,
        stock_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        commodities_df: Optional[pd.DataFrame] = None,
        treasury_df: Optional[pd.DataFrame] = None,
        calculate_target: bool = True
    ) -> pd.DataFrame:
        """
        Add all features to stock DataFrame.

        This is the main method that orchestrates all feature engineering.

        Args:
            stock_df: Stock price DataFrame
            vix_df: VIX index DataFrame
            commodities_df: Commodities DataFrame
            treasury_df: Treasury yields DataFrame
            calculate_target: Whether to calculate target variable

        Returns:
            DataFrame with all features
        """
        self.logger.info("=" * 60)
        self.logger.info("FEATURE ENGINEERING PIPELINE")
        self.logger.info("=" * 60)

        result = stock_df.copy()

        # 1. Add time features
        result = self.add_time_features(result)

        # 2. Add technical indicators
        result = self.add_technical_indicators(result)

        # 2.1 Add geometric and volatility features (Inspired by QuantAgent)
        result = self.add_geometric_features(result)

        # 2.2 Add market structure features
        result = self.add_market_structure_features(result)

        # 2.5. Add Fibonacci retracement features
        result = self.add_fibonacci_features(result)

        # 3. Add candlestick patterns
        result = self.add_candlestick_patterns(result)

        # 4. Add financial metrics
        result = self.add_financial_metrics(result)

        # 5. Add group (sector) information
        result = self.add_group_from_sector(result)

        # 6. Add rolling cointegration and equilibrium features
        result = self.add_cointegration_features(result)

        # 7. Merge external data
        result = self.merge_external_data(
            result,
            vix_df=vix_df,
            commodities_df=commodities_df,
            treasury_df=treasury_df
        )

        # 8. Calculate target
        if calculate_target:
            result = self.calculate_target(result)

        self.logger.info("=" * 60)
        self.logger.info(f"Feature engineering complete. Final shape: {result.shape}")
        self.logger.info(f"Total features: {len([c for c in result.columns if c not in ['date', 'tic']])}")
        self.logger.info("=" * 60)

        return result

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding identifiers and target).

        Args:
            df: DataFrame with all features

        Returns:
            List of feature column names
        """
        exclude = {'date', 'tic', 'target'}
        features = [c for c in df.columns if c not in exclude]
        return features

    def get_feature_info(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get information about features in DataFrame.

        Args:
            df: DataFrame with all features

        Returns:
            Dictionary with feature information
        """
        feature_cols = self.get_feature_columns(df)

        # Count by type
        price_features = ['open', 'high', 'low', 'close', 'volume']
        ema_features = [c for c in feature_cols if c.startswith('ema_')]
        rsi_features = [c for c in feature_cols if 'rsi' in c.lower()]
        macd_features = [c for c in feature_cols if 'macd' in c.lower()]
        geometric_features = [
            c for c in feature_cols
            if any(
                x in c.lower()
                for x in ['atr_', 'roc_', 'slope_', 'bb_width', 'channel_compression', 'channel_position']
            )
        ]
        geometric_features.extend(
            [
                c for c in feature_cols
                if any(x in c.lower() for x in ['dist_to_swing_', 'days_since_swing_', 'opt_slope_', 'opt_channel_'])
            ]
        )
        market_structure_features = [
            c for c in feature_cols
            if any(
                x in c.lower()
                for x in [
                    'distance_to_',
                    'breakout_',
                    'breakdown_',
                    'higher_high',
                    'lower_high',
                    'higher_low',
                    'lower_low',
                    'near_52w_',
                    'trend_strength_score',
                    'trend_persistence_',
                    'rolling_volatility_',
                ]
            )
        ]
        pattern_features = [c for c in feature_cols if c in _CANDLESTICK_PATTERN_COLUMNS]
        external_features = [c for c in feature_cols if c in ['vix', 'bondyield'] or c in self.config.data.sources.COMMODITIES.values()]
        time_features = ['day', 'month', 'dayofweek']
        financial_metrics_features = ['pe_ratio', 'peg_ratio', 'eps', 'dividend_flag', 'roe', 'roi',
                                       'debt_to_equity', 'debt_to_asset', 'current_ratio']
        fibonacci_features = ['swing_high', 'swing_low', 'fib_range', 'fib_38', 'fib_50',
                               'fib_61', 'dist_fib_38', 'dist_fib_50', 'dist_fib_61', 'break_fib_61']
        regime_features = ['regime_id']
        cointegration_features = [
            'stock_return_1d',
            'relative_price_vs_sector',
            'relative_return_vs_sector',
            'relative_price_vs_sector_norm',
            'pair_beta',
            'spread',
            'rolling_mean_spread',
            'rolling_std_spread',
            'spread_zscore',
            'spread_norm',
            'spread_adf_stat',
            'spread_adf_pvalue',
            'equilibrium_gap',
            'equilibrium_zscore',
            'equilibrium_gap_norm',
            'equilibrium_adf_stat',
            'equilibrium_adf_pvalue',
        ]

        info = {
            'total_features': len(feature_cols),
            'price_features': len([f for f in price_features if f in feature_cols]),
            'ema_features': len(ema_features),
            'rsi_features': len(rsi_features),
            'macd_features': len(macd_features),
            'geometric_features': len(geometric_features),
            'market_structure_features': len(market_structure_features),
            'candlestick_patterns': len(pattern_features),
            'external_features': len(external_features),
            'time_features': len([f for f in time_features if f in feature_cols]),
            'financial_metrics': len([f for f in financial_metrics_features if f in feature_cols]),
            'fibonacci_features': len([f for f in fibonacci_features if f in feature_cols]),
            'regime_features': len([f for f in regime_features if f in feature_cols]),
            'cointegration_features': len([f for f in cointegration_features if f in feature_cols]),
            'feature_list': feature_cols
        }

        return info
