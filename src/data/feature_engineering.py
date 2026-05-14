"""
Feature engineering for Multi-Model Financial Forecasting.

This module handles:
- Technical indicator calculation (EMA, RSI, StochRSI, MACD)
- Candlestick pattern recognition using TA-Lib
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
    add_fibonacci_features_polars,
    add_time_features_polars,
    merge_external_data_polars,
)
from src.utils.logger import get_logger
from src.data.financial_metrics_loader import FinancialMetricsLoader


class FeatureEngineer:
    """
    Feature engineering for financial data.

    Creates features for model training including:
    - Technical indicators (EMA, RSI, StochRSI, MACD)
    - Candlestick patterns (~100 patterns from TA-Lib)
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
            result = add_fibonacci_features_polars(result, window=window)
            self.logger.info(f"Added Fibonacci features. Shape: {result.shape}")
            return result

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

    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all TA-Lib candlestick patterns.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added candlestick pattern columns
        """
        candlestick_cfg = self.config.data.candlestick if 'candlestick' in self.config.data else None
        use_candlestick = self.config.data.features.FEATURE_FLAGS.get('candlestick_patterns', True)
        if candlestick_cfg is not None:
            use_candlestick = use_candlestick and candlestick_cfg.get('USE_CANDLESTICK_PATTERNS', True)

        if not use_candlestick:
            self.logger.info("Skipping candlestick patterns (disabled in config)")
            return df

        self.logger.info("Adding candlestick patterns...")

        result = df.copy()
        excluded_patterns = set()
        if candlestick_cfg is not None:
            excluded_patterns = set(candlestick_cfg.get('EXCLUDE_PATTERNS', []))

        # Get all candlestick pattern functions from TA-Lib
        pattern_functions = [
            name for name in dir(talib)
            if name.startswith('CDL') and callable(getattr(talib, name))
        ]
        if excluded_patterns:
            unknown_patterns = sorted(excluded_patterns - set(pattern_functions))
            if unknown_patterns:
                self.logger.warning(
                    f"Ignoring unknown excluded candlestick patterns: {unknown_patterns}"
                )
            pattern_functions = [
                name for name in pattern_functions
                if name not in excluded_patterns
            ]

        self.logger.info(
            f"Using {len(pattern_functions)} candlestick patterns"
            + (f" after excluding {len(excluded_patterns)}" if excluded_patterns else "")
        )

        # Calculate patterns for each ticker
        for ticker in result['tic'].unique():
            mask = result['tic'] == ticker
            stock_df = result[mask].copy()

            for pattern_name in pattern_functions:
                try:
                    pattern_func = getattr(talib, pattern_name)
                    pattern_result = pattern_func(
                        stock_df['open'].values,
                        stock_df['high'].values,
                        stock_df['low'].values,
                        stock_df['close'].values
                    )

                    # Normalize to -1, 0, 1
                    # TA-Lib returns: -100 (bearish), 0 (no pattern), 100 (bullish)
                    normalized = np.where(pattern_result > 0, 1, np.where(pattern_result < 0, -1, 0))
                    stock_df[pattern_name] = normalized

                except Exception as e:
                    self.logger.warning(f"Failed to calculate {pattern_name} for {ticker}: {e}")
                    stock_df[pattern_name] = 0

            # Update result
            result.loc[mask, stock_df.columns] = stock_df

        self.logger.info(f"Added {len(pattern_functions)} candlestick patterns")
        return result

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
            result = add_time_features_polars(df)
            self.logger.info(f"Added time features. Shape: {result.shape}")
            return result

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

        # 2.5. Add Fibonacci retracement features
        result = self.add_fibonacci_features(result)

        # 3. Add candlestick patterns
        result = self.add_candlestick_patterns(result)

        # 4. Add financial metrics
        result = self.add_financial_metrics(result)

        # 5. Add group (sector) information
        result = self.add_group_from_sector(result)

        # 6. Merge external data
        result = self.merge_external_data(
            result,
            vix_df=vix_df,
            commodities_df=commodities_df,
            treasury_df=treasury_df
        )

        # 7. Calculate target
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
        pattern_features = [c for c in feature_cols if c.startswith('CDL')]
        external_features = [c for c in feature_cols if c in ['vix', 'bondyield'] or c in self.config.data.sources.COMMODITIES.values()]
        time_features = ['day', 'month', 'dayofweek']
        financial_metrics_features = ['pe_ratio', 'peg_ratio', 'eps', 'dividend_flag', 'roe', 'roi',
                                       'debt_to_equity', 'debt_to_asset', 'current_ratio']
        fibonacci_features = ['swing_high', 'swing_low', 'fib_range', 'fib_38', 'fib_50',
                               'fib_61', 'dist_fib_38', 'dist_fib_50', 'dist_fib_61', 'break_fib_61']
        regime_features = ['regime_id']

        info = {
            'total_features': len(feature_cols),
            'price_features': len([f for f in price_features if f in feature_cols]),
            'ema_features': len(ema_features),
            'rsi_features': len(rsi_features),
            'macd_features': len(macd_features),
            'geometric_features': len(geometric_features),
            'candlestick_patterns': len(pattern_features),
            'external_features': len(external_features),
            'time_features': len([f for f in time_features if f in feature_cols]),
            'financial_metrics': len([f for f in financial_metrics_features if f in feature_cols]),
            'fibonacci_features': len([f for f in fibonacci_features if f in feature_cols]),
            'regime_features': len([f for f in regime_features if f in feature_cols]),
            'feature_list': feature_cols
        }

        return info
