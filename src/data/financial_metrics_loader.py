"""
Financial Metrics Loader for Multi-Model Financial Forecasting.

This module loads and parses financial metrics from JSON files in raw_data/ticket_data/us/.
Each JSON file contains:
- Highlights: Single-value metrics (PE ratio, PEG ratio, EPS)
- Financials: Time-series quarterly data (Balance Sheet, Income Statement)

The loader extracts and calculates these 8 financial metrics:
1. PE Ratio - from Highlights.PERatio
2. PEG Ratio - from Highlights.PEGRatio
3. EPS - from Highlights.DilutedEpsTTM
4. ROE - calculated: netIncome / totalStockholderEquity
5. ROI - calculated: netIncome / totalAssets
6. Debt-to-Equity - calculated: totalLiab / totalStockholderEquity
7. Debt-to-Asset - calculated: totalLiab / totalAssets
8. Current Ratio - calculated: totalCurrentAssets / totalCurrentLiabilities
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

from src.config import load_config
from src.utils.logger import get_logger


class FinancialMetricsLoader:
    """
    Load and parse financial metrics from JSON files.

    Each JSON file (raw_data/ticket_data/us/{TICKER}.json) contains:
    - Highlights: Single-value metrics like PE ratio, PEG ratio, EPS
    - Financials: Quarterly time-series data for Balance Sheet and Income Statement

    This class extracts those metrics and calculates ratios from quarterly data.
    """

    def __init__(self, config):
        """
        Initialize the financial metrics loader.

        Args:
            config instance with paths and settings
        """
        self.config = config
        self.logger = get_logger("financial_metrics", log_dir="logs")
        self.raw_data_path = Path(self.config.data.financial_metrics.FINANCIAL_METRICS_SOURCE)

        if not self.raw_data_path.exists():
            self.logger.warning(f"Financial metrics source path not found: {self.raw_data_path}")

    def load_single_stock_json(self, ticker: str) -> Optional[Dict]:
        """
        Load JSON file for a single ticker.

        Args:
            ticker: Ticker symbol (e.g., "AAPL")

        Returns:
            Parsed JSON dictionary or None if file not found
        """
        json_path = self.raw_data_path / f"{ticker}.json"

        if not json_path.exists():
            self.logger.warning(f"{ticker}: JSON file not found: {json_path}")
            return None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"{ticker}: Failed to load JSON: {str(e)}")
            return None

    def extract_highlights_metrics(self, data: Dict) -> Dict[str, float]:
        """
        Extract single-value metrics from the Highlights section.

        Args:
            data: Parsed JSON data dictionary

        Returns:
            Dictionary with pe_ratio, peg_ratio, eps, dividend_flag (may contain None)

        Note:
            dividend_flag: 1 if stock pays dividend, 2 if no dividend
        """
        metrics = {
            'pe_ratio': None,
            'peg_ratio': None,
            'eps': None,
            'dividend_flag': None
        }

        if not data or 'Highlights' not in data:
            self.logger.warning("No Highlights section in JSON data")
            return metrics

        highlights = data['Highlights']

        # Extract PE Ratio
        if 'PERatio' in highlights and highlights['PERatio'] is not None:
            try:
                metrics['pe_ratio'] = float(highlights['PERatio'])
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid PE Ratio value: {highlights['PERatio']}")

        # Extract PEG Ratio
        if 'PEGRatio' in highlights and highlights['PEGRatio'] is not None:
            try:
                metrics['peg_ratio'] = float(highlights['PEGRatio'])
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid PEG Ratio value: {highlights['PEGRatio']}")

        # Extract EPS (Earnings Per Share) - use DilutedEpsTTM
        if 'DilutedEpsTTM' in highlights and highlights['DilutedEpsTTM'] is not None:
            try:
                metrics['eps'] = float(highlights['DilutedEpsTTM'])
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid EPS value: {highlights['DilutedEpsTTM']}")

        # Extract dividend flag: 1 if pays dividend, 2 if no dividend
        # Check DividendYield in Highlights
        dividend_yield = None
        if 'DividendYield' in highlights and highlights['DividendYield'] is not None:
            try:
                dividend_yield = float(highlights['DividendYield'])
            except (ValueError, TypeError):
                pass

        # Also check DividendShare as backup
        if dividend_yield is None or dividend_yield == 0:
            if 'DividendShare' in highlights and highlights['DividendShare'] is not None:
                try:
                    dividend_share = float(highlights['DividendShare'])
                    if dividend_share > 0:
                        dividend_yield = 1.0  # Has dividend
                except (ValueError, TypeError):
                    pass

        # Set flag: 1 = has dividend, 2 = no dividend
        if dividend_yield is not None and dividend_yield > 0:
            metrics['dividend_flag'] = 1
        else:
            metrics['dividend_flag'] = 2

        return metrics

    def extract_quarterly_metrics(self, data: Dict) -> pd.DataFrame:
        """
        Extract quarterly financial data and calculate financial ratios.

        Merges Balance Sheet and Income Statement quarterly data, then calculates:
        - ROE: netIncome / totalStockholderEquity
        - ROI: netIncome / totalAssets
        - Debt-to-Equity: totalLiab / totalStockholderEquity
        - Debt-to-Asset: totalLiab / totalAssets
        - Current Ratio: totalCurrentAssets / totalCurrentLiabilities

        Args:
            data: Parsed JSON data dictionary

        Returns:
            DataFrame with columns: date, roe, roi, debt_to_equity, debt_to_asset, current_ratio
        """
        # Extract Balance Sheet quarterly data
        bs_quarterly = data.get('Financials', {}).get('Balance_Sheet', {}).get('quarterly', {})

        # Extract Income Statement quarterly data
        is_quarterly = data.get('Financials', {}).get('Income_Statement', {}).get('quarterly', {})

        if not bs_quarterly or not is_quarterly:
            self.logger.warning("Missing quarterly financial data")
            return pd.DataFrame(columns=['date', 'roe', 'roi', 'debt_to_equity', 'debt_to_asset', 'current_ratio'])

        # Convert to DataFrames (dict is keyed by date string)
        bs_df = pd.DataFrame.from_dict(bs_quarterly, orient='index')
        is_df = pd.DataFrame.from_dict(is_quarterly, orient='index')

        # Ensure date column exists
        if 'date' not in bs_df.columns:
            bs_df['date'] = bs_df.index
        if 'date' not in is_df.columns:
            is_df['date'] = is_df.index

        # Convert date to datetime
        bs_df['date'] = pd.to_datetime(bs_df['date'])
        is_df['date'] = pd.to_datetime(is_df['date'])

        # Merge on date
        quarterly = pd.merge(
            bs_df,
            is_df,
            on='date',
            how='outer',
            suffixes=('_bs', '_is')
        )

        # Convert string values to float for numeric columns
        numeric_cols = [
            'totalAssets', 'totalLiab', 'totalStockholderEquity',
            'totalCurrentAssets', 'totalCurrentLiabilities',
            'netIncome'
        ]

        for col in numeric_cols:
            if col in quarterly.columns:
                quarterly[col] = pd.to_numeric(quarterly[col], errors='coerce')

        # Calculate financial ratios (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            # ROE = netIncome / totalStockholderEquity
            quarterly['roe'] = np.where(
                quarterly['totalStockholderEquity'] != 0,
                quarterly['netIncome'] / quarterly['totalStockholderEquity'],
                np.nan
            )

            # ROI = netIncome / totalAssets
            quarterly['roi'] = np.where(
                quarterly['totalAssets'] != 0,
                quarterly['netIncome'] / quarterly['totalAssets'],
                np.nan
            )

            # Debt-to-Equity = totalLiab / totalStockholderEquity
            quarterly['debt_to_equity'] = np.where(
                quarterly['totalStockholderEquity'] != 0,
                quarterly['totalLiab'] / quarterly['totalStockholderEquity'],
                np.nan
            )

            # Debt-to-Asset = totalLiab / totalAssets
            quarterly['debt_to_asset'] = np.where(
                quarterly['totalAssets'] != 0,
                quarterly['totalLiab'] / quarterly['totalAssets'],
                np.nan
            )

            # Current Ratio = totalCurrentAssets / totalCurrentLiabilities
            quarterly['current_ratio'] = np.where(
                quarterly['totalCurrentLiabilities'] != 0,
                quarterly['totalCurrentAssets'] / quarterly['totalCurrentLiabilities'],
                np.nan
            )

        # Select only the columns we need
        result = quarterly[['date', 'roe', 'roi', 'debt_to_equity', 'debt_to_asset', 'current_ratio']].copy()

        # Sort by date
        result = result.sort_values('date').reset_index(drop=True)

        return result

    def quarterly_to_daily(
        self,
        quarterly_df: pd.DataFrame,
        daily_price_df: pd.DataFrame,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Convert quarterly metrics to daily frequency using forward-fill.

        Since financial data is reported quarterly (4x/year) but price data is daily,
        we need to spread quarterly values to daily frequency. This is done by
        forward-filling: each quarter's values apply from the report date until
        the next quarter's report date.

        Args:
            quarterly_df: DataFrame with quarterly data and 'date' column
            daily_price_df: Daily price data with 'date' column (defines date range)
            fill_method: How to fill between quarters ('ffill' or 'interpolate')

        Returns:
            DataFrame with daily dates and forward-filled metrics
        """
        if quarterly_df.empty:
            return pd.DataFrame(columns=['date', 'roe', 'roi', 'debt_to_equity', 'debt_to_asset', 'current_ratio'])

        # Convert date columns to datetime
        quarterly_df['date'] = pd.to_datetime(quarterly_df['date'])
        daily_price_df['date'] = pd.to_datetime(daily_price_df['date'])

        # Get date range from daily price data
        min_date = daily_price_df['date'].min()
        max_date = daily_price_df['date'].max()

        # Create daily DataFrame with all dates in range
        daily_df = pd.DataFrame({'date': pd.date_range(start=min_date, end=max_date, freq='D')})

        # Merge with quarterly data (leaves NaN between quarters)
        merged = pd.merge(daily_df, quarterly_df, on='date', how='left')

        # Sort by date
        merged = merged.sort_values('date')

        # Forward-fill to spread quarterly data to daily
        metric_cols = [c for c in merged.columns if c != 'date']

        if fill_method == 'ffill':
            # Forward fill WITHOUT limit for financial metrics
            # Quarterly data should be carried forward until next quarter
            merged[metric_cols] = merged[metric_cols].ffill()
        elif fill_method == 'interpolate':
            # Interpolate between quarters without limit
            merged[metric_cols] = merged[metric_cols].interpolate(method='time')
        else:
            self.logger.warning(f"Unknown fill_method: {fill_method}, using 'ffill'")
            merged[metric_cols] = merged[metric_cols].ffill()

        return merged

    def load_metrics_for_ticker(
        self,
        ticker: str,
        daily_price_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Load and process all financial metrics for a single ticker.

        This is a convenience method that combines:
        1. Load JSON file
        2. Extract highlights metrics
        3. Extract and calculate quarterly metrics
        4. Convert quarterly to daily
        5. Merge highlights (broadcast to all dates)

        Args:
            ticker: Ticker symbol
            daily_price_df: Daily price data DataFrame (defines date range)

        Returns:
            DataFrame with columns: date, tic, pe_ratio, peg_ratio, eps, dividend_flag,
            roe, roi, debt_to_equity, debt_to_asset, current_ratio
            Or None if loading fails

        Note:
            dividend_flag: 1 if stock pays dividend, 2 if no dividend
        """
        # Load JSON data
        json_data = self.load_single_stock_json(ticker)
        if json_data is None:
            return None

        # Extract highlights (single values)
        highlights = self.extract_highlights_metrics(json_data)

        # Extract quarterly metrics
        quarterly_df = self.extract_quarterly_metrics(json_data)

        if quarterly_df.empty:
            self.logger.warning(f"No quarterly data for {ticker}")
            return None

        # Convert quarterly to daily
        daily_metrics = self.quarterly_to_daily(
            quarterly_df,
            daily_price_df,
            fill_method=self.config.data.financial_metrics.FINANCIAL_METRICS_FILL_METHOD
        )

        # Add highlights metrics (broadcast same value to all dates)
        daily_metrics['pe_ratio'] = highlights['pe_ratio']
        daily_metrics['peg_ratio'] = highlights['peg_ratio']
        daily_metrics['eps'] = highlights['eps']
        daily_metrics['dividend_flag'] = highlights['dividend_flag']

        # Add ticker column
        daily_metrics['tic'] = ticker

        # Reorder columns (dividend_flag goes after eps, before roe)
        cols = ['date', 'tic', 'pe_ratio', 'peg_ratio', 'eps', 'dividend_flag', 'roe', 'roi',
                'debt_to_equity', 'debt_to_asset', 'current_ratio']
        daily_metrics = daily_metrics[[c for c in cols if c in daily_metrics.columns]]

        return daily_metrics
