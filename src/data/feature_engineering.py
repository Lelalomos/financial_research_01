"""
Feature engineering for CRNN Financial Prediction Model.

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
        if not self.config.data.features.FEATURE_FLAGS.get('candlestick_patterns', True):
            self.logger.info("Skipping candlestick patterns (disabled in config)")
            return df

        self.logger.info("Adding candlestick patterns...")

        result = df.copy()

        # Get all candlestick pattern functions from TA-Lib
        pattern_functions = [
            name for name in dir(talib)
            if name.startswith('CDL') and callable(getattr(talib, name))
        ]

        self.logger.info(f"Found {len(pattern_functions)} candlestick patterns")

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
            'candlestick_patterns': len(pattern_features),
            'external_features': len(external_features),
            'time_features': len([f for f in time_features if f in feature_cols]),
            'financial_metrics': len([f for f in financial_metrics_features if f in feature_cols]),
            'fibonacci_features': len([f for f in fibonacci_features if f in feature_cols]),
            'regime_features': len([f for f in regime_features if f in feature_cols]),
            'feature_list': feature_cols
        }

        return info
