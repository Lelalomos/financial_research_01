"""
Data downloader for CRNN Financial Prediction Model.

This module handles downloading financial data from various sources:
- Stock data from local index files (raw_data/index/) and yfinance
- VIX index data from yfinance
- Commodity futures from yfinance
- Treasury yields from FRED (pandas_datareader)
"""

import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import requests
import json
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta
import time
from multiprocessing import Pool, cpu_count
from functools import partial

from src.config import load_config
from src.utils.logger import get_logger


def _download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    retry_attempts: int = 5,
    retry_delay: int = 5
) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
    """
    Download data for a single ticker with retry logic.

    Helper function for multiprocessing. Must be at module level for pickling.

    Args:
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        retry_attempts: Number of retry attempts (default: 5)
        retry_delay: Delay in seconds between retries (default: 5)

    Returns:
        Tuple of (ticker, DataFrame, error) - if error, ticker and error are set, DataFrame is None
    """
    last_error = None

    for attempt in range(retry_attempts):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                multi_level_index=False
            )

            if data.empty:
                last_error = "No data available"
                if attempt < retry_attempts - 1:
                    print(f"Retry {attempt + 1}/{retry_attempts} for {ticker}: No data available, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                return ticker, None, last_error

            data = data.reset_index()
            data['tic'] = ticker

            # Standardize column names
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })

            # Use adjusted close if available, otherwise use close
            if 'adj_close' in data.columns:
                data['close'] = data['adj_close']

            return ticker, data, None

        except Exception as e:
            last_error = str(e)
            # If not the last attempt, sleep and retry
            if attempt < retry_attempts - 1:
                print(f"Retry {attempt + 1}/{retry_attempts} for {ticker}: {e}, waiting {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            # Last attempt failed, return error
            return ticker, None, last_error

    # All attempts exhausted - log the final failure
    print(f"All {retry_attempts} retry attempts exhausted for {ticker}")
    return ticker, None, last_error


def _download_single_commodity(
    symbol_name: Tuple[str, str],
    start_date: str,
    end_date: str,
    retry_attempts: int = 5,
    retry_delay: int = 5
) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
    """
    Download data for a single commodity with retry logic.

    Helper function for multiprocessing. Must be at module level for pickling.

    Args:
        symbol_name: Tuple of (symbol, name)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        retry_attempts: Number of retry attempts (default: 5)
        retry_delay: Delay in seconds between retries (default: 5)

    Returns:
        Tuple of (name, DataFrame, error)
    """
    symbol, name = symbol_name
    last_error = None

    for attempt in range(retry_attempts):
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                multi_level_index=False
            )

            if data.empty:
                last_error = "No data available"
                if attempt < retry_attempts - 1:
                    print(f"Retry {attempt + 1}/{retry_attempts} for {symbol} ({name}): No data available, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                return name, None, last_error

            # Reset index and standardize column names
            data = data.reset_index()
            data.columns = [str(c).lower().replace(' ', '_') for c in data.columns]

            # Handle different possible date column names
            for col in data.columns:
                if 'date' in col.lower():
                    data['date'] = pd.to_datetime(data[col])
                    break

            # Calculate mean of OHLC
            data[name] = data[['open', 'high', 'low', 'close']].mean(axis=1)
            data = data[['date', name]]

            return name, data, None

        except Exception as e:
            last_error = str(e)
            if attempt < retry_attempts - 1:
                print(f"Retry {attempt + 1}/{retry_attempts} for {symbol} ({name}): {e}, waiting {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            return name, None, last_error

    print(f"All {retry_attempts} retry attempts exhausted for {symbol} ({name})")
    return name, None, last_error


class DataDownloader:
    """
    Download financial data from various sources.

    Supports:
    - S&P 500 constituent stocks
    - VIX index
    - Commodity futures
    - Treasury yields
    """

    def __init__(self, config=None, index_file: Optional[str] = None):
        """
        Initialize data downloader.

        Args:
            config: Configuration object (defaults to load_config('main') if None)
            index_file: Name of index file in raw_data/index/ (e.g., 'GSPC.json')
                       If None, uses config.data.sources.INDEX_FILE (default 'GSPC.json' - S&P 500)
        """
        if config is None:
            config = load_config('main')
        self.config = config
        self.logger = get_logger("downloader", log_dir="logs")
        self.session_start = time.time()

        # Set index file (default to config setting or GSPC - S&P 500)
        self.index_file = index_file or config.data.sources.INDEX_FILE
        self.index_path = Path(config.data.sources.RAW_DATA_INDEX_PATH) / self.index_file

        # Create directories
        self.raw_data_path = Path(config.data.paths.RAW_DATA_PATH)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        self.external_data_path = Path(config.data.paths.EXTERNAL_DATA_PATH)
        self.external_data_path.mkdir(parents=True, exist_ok=True)

    def _log_time(self):
        """Log elapsed time."""
        elapsed = time.time() - self.session_start
        self.logger.info(f"Elapsed time: {elapsed:.2f} seconds")

    def get_sp500_tickers(self, limit: Optional[int] = None) -> List[str]:
        """
        Get list of ticker symbols from local index file.

        Args:
            limit: Maximum number of tickers to return (None = all tickers).
                   Takes the first N tickers from the index file.

        Returns:
            List of ticker symbols

        Note:
            Reads from raw_data/index/{index_file}.json
            Default is GSPC.json (S&P 500)
        """
        self.logger.info(f"Fetching ticker list from {self.index_path}...")

        # Try to read from local index file
        try:
            if not self.index_path.exists():
                self.logger.warning(f"Index file not found: {self.index_path}")
                # List available index files
                index_dir = Path("raw_data/index")
                if index_dir.exists():
                    available = list(index_dir.glob("*.json"))
                    self.logger.info(f"Available index files: {[f.name for f in available]}")
                return self._get_fallback_tickers()

            with open(self.index_path, 'r') as f:
                index_data = json.load(f)

            # Extract tickers from Components
            tickers = []
            if 'Components' in index_data:
                for key in index_data['Components']:
                    component = index_data['Components'][key]
                    if 'Code' in component:
                        tickers.append(component['Code'])

            # Apply limit if specified
            if limit is not None and limit > 0:
                tickers = tickers[:limit]
                self.logger.info(f"Loaded {len(tickers)} tickers (limited to first {limit}) from {self.index_file}")
            else:
                self.logger.info(f"Loaded {len(tickers)} tickers from {self.index_file}")

            return tickers

        except Exception as e:
            self.logger.error(f"Failed to read index file: {e}")
            return self._get_fallback_tickers()

    def _get_fallback_tickers(self) -> List[str]:
        """Get fallback ticker list."""
        fallback_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'LLY', 'PFE', 'ABBV', 'KO',
            'PEP', 'TMO', 'MRK', 'AVGO', 'COST', 'CSCO', 'ABT', 'CRM', 'MCD', 'WMT'
        ]
        self.logger.warning(f"Using fallback ticker list with {len(fallback_tickers)} tickers")
        return fallback_tickers

    def list_available_indices(self) -> List[str]:
        """
        List all available index files in raw_data/index/.

        Returns:
            List of index filenames
        """
        index_dir = Path("raw_data/index")
        if not index_dir.exists():
            self.logger.warning(f"Index directory not found: {index_dir}")
            return []

        return [f.name for f in index_dir.glob("*.json")]

    def list_index_stocks(self, index_file: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List all stocks in an index with details.

        Args:
            index_file: Name of index file (uses default if None)

        Returns:
            List of dicts with stock info (Code, Name, Sector, Industry, Weight)
        """
        if index_file:
            path = Path("raw_data/index") / index_file
        else:
            path = self.index_path

        self.logger.info(f"Reading stocks from {path}...")

        try:
            with open(path, 'r') as f:
                index_data = json.load(f)

            stocks = []
            if 'Components' in index_data:
                for key in index_data['Components']:
                    component = index_data['Components'][key]
                    stocks.append({
                        'Code': component.get('Code', ''),
                        'Name': component.get('Name', ''),
                        'Sector': component.get('Sector', ''),
                        'Industry': component.get('Industry', ''),
                        'Weight': component.get('Weight', 0)
                    })

            self.logger.info(f"Found {len(stocks)} stocks in {path.name}")
            return stocks

        except Exception as e:
            self.logger.error(f"Failed to read index file: {e}")
            return []

    def filter_stocks_by_criteria(
        self,
        index_file: Optional[str] = None,
        sectors: Optional[List[str]] = None,
        industries: Optional[List[str]] = None,
        min_weight: Optional[float] = None,
        exclude_stocks: Optional[Set[str]] = None,
        include_stocks: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Filter stocks from an index by criteria.

        Args:
            index_file: Name of index file (uses default if None)
            sectors: List of sectors to include (None = all)
            industries: List of industries to include (None = all)
            min_weight: Minimum weight threshold (None = no threshold)
            exclude_stocks: Set of stock codes to exclude
            include_stocks: Set of stock codes to include (if set, only these are used)

        Returns:
            List of filtered ticker symbols
        """
        stocks = self.list_index_stocks(index_file)
        filtered = []

        for stock in stocks:
            code = stock['Code']

            # Explicit include list
            if include_stocks is not None:
                if code not in include_stocks:
                    continue

            # Exclude list
            if exclude_stocks and code in exclude_stocks:
                continue

            # Sector filter
            if sectors and stock['Sector'] not in sectors:
                continue

            # Industry filter
            if industries and stock['Industry'] not in industries:
                continue

            # Weight filter
            if min_weight is not None and stock['Weight'] < min_weight:
                continue

            filtered.append(code)

        self.logger.info(f"Filtered to {len(filtered)} stocks from {len(stocks)} total")
        return filtered

    def download_stock_data(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save: bool = True,
        n_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Download historical stock price data using multiprocessing.

        Args:
            tickers: List of ticker symbols (None = get S&P 500)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            save: Whether to save to disk
            n_workers: Number of parallel workers (None = use CPU count)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume, tic
        """
        if tickers is None:
            tickers = self.get_sp500_tickers()

        if start_date is None:
            start_date = self.config.data.sources.START_DATE
        if end_date is None:
            end_date = self.config.data.sources.END_DATE

        # Set number of workers
        if n_workers is None:
            n_workers = min(cpu_count(), 8)  # Limit to 8 workers by default

        self.logger.info(f"Downloading data for {len(tickers)} stocks from {start_date} to {end_date}")
        self.logger.info(f"Using {n_workers} parallel workers")

        all_data = []
        failed_tickers = []

        # Use multiprocessing pool for parallel downloads
        with Pool(processes=n_workers) as pool:
            # Use partial to pass dates and retry settings to the download function
            download_func = partial(
                _download_single_ticker,
                start_date=start_date,
                end_date=end_date,
                retry_attempts=self.config.data.download.DOWNLOAD_RETRY_ATTEMPTS,
                retry_delay=self.config.data.download.DOWNLOAD_RETRY_DELAY
            )

            # Download all tickers in parallel
            results = pool.map(download_func, tickers)

        # Process results
        for ticker, data, error in results:
            if data is not None:
                all_data.append(data)
            else:
                self.logger.warning(f"Failed to download {ticker}: {error}")
                failed_tickers.append((ticker, error))

        if not all_data:
            raise ValueError("No data downloaded successfully")

        result = pd.concat(all_data, ignore_index=True)
        result['date'] = pd.to_datetime(result['date'])

        self.logger.info(f"Downloaded {len(result)} rows for {len(all_data)} stocks")
        self.logger.info(f"Date range: {result['date'].min()} to {result['date'].max()}")

        if failed_tickers:
            self.logger.warning(f"Failed to download {len(failed_tickers)} tickers: {[t[0] for t in failed_tickers[:10]]}...")

        if save:
            save_path = self.raw_data_path / "stock_data.parquet"
            result.to_parquet(save_path, index=False)
            self.logger.info(f"Saved to {save_path}")

        self._log_time()
        return result

    def download_vix(self, save: bool = True) -> pd.DataFrame:
        """
        Download VIX index data.

        Args:
            save: Whether to save to disk

        Returns:
            DataFrame with VIX data
        """
        self.logger.info(f"Downloading VIX data ({self.config.data.sources.VIX_SYMBOL})")

        try:
            data = yf.download(
                self.config.data.sources.VIX_SYMBOL,
                start=self.config.data.sources.START_DATE,
                end=self.config.data.sources.END_DATE,
                progress=False,
                multi_level_index=False
            )

            # Reset index and standardize column names
            data = data.reset_index()
            data.columns = [str(c).lower().replace(' ', '_') for c in data.columns]

            # Handle different possible date column names
            for col in data.columns:
                if 'date' in col.lower():
                    data['date'] = pd.to_datetime(data[col])
                    break

            # Calculate mean of OHLC
            data['vix'] = data[['open', 'high', 'low', 'close']].mean(axis=1)
            data = data[['date', 'vix']]

            self.logger.info(f"Downloaded {len(data)} rows of VIX data")

            if save:
                save_path = self.external_data_path / "vix.parquet"
                data.to_parquet(save_path, index=False)
                self.logger.info(f"Saved to {save_path}")

            return data

        except Exception as e:
            self.logger.error(f"Failed to download VIX: {e}")
            raise

    def download_commodities(self, save: bool = True, n_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Download commodity futures data using multiprocessing.

        Args:
            save: Whether to save to disk
            n_workers: Number of parallel workers (None = use CPU count)

        Returns:
            DataFrame with commodity data
        """
        self.logger.info(f"Downloading commodity data for {len(self.config.data.sources.COMMODITIES)} commodities")

        # Set number of workers
        if n_workers is None:
            n_workers = min(cpu_count(), len(self.config.data.sources.COMMODITIES))

        self.logger.info(f"Using {n_workers} parallel workers")

        all_data = []

        # Convert to list of tuples for multiprocessing
        commodity_list = list(self.config.data.sources.COMMODITIES.items())

        # Use multiprocessing pool for parallel downloads
        with Pool(processes=n_workers) as pool:
            download_func = partial(_download_single_commodity,
                                    start_date=self.config.data.sources.START_DATE,
                                    end_date=self.config.data.sources.END_DATE,
                                    retry_attempts=self.config.data.download.DOWNLOAD_RETRY_ATTEMPTS,
                                    retry_delay=self.config.data.download.DOWNLOAD_RETRY_DELAY)
            results = pool.map(download_func, commodity_list)

        # Process results
        for name, data, error in results:
            if data is not None:
                all_data.append(data)
            else:
                self.logger.warning(f"Failed to download {name}: {error}")

        if not all_data:
            raise ValueError("No commodity data downloaded")

        # Merge all commodities
        result = all_data[0]
        for df in all_data[1:]:
            result = pd.merge(result, df, on='date', how='outer')

        self.logger.info(f"Downloaded {len(result)} rows of commodity data")

        if save:
            save_path = self.external_data_path / "commodities.parquet"
            result.to_parquet(save_path, index=False)
            self.logger.info(f"Saved to {save_path}")

        return result

    def download_treasury_yields(self, save: bool = True) -> pd.DataFrame:
        """
        Download treasury yield data from FRED.

        Args:
            save: Whether to save to disk

        Returns:
            DataFrame with treasury yield data
        """
        self.logger.info(f"Downloading treasury yields: {self.config.data.sources.TREASURY_YIELDS}")

        try:
            start = datetime.strptime(self.config.data.sources.START_DATE, "%Y-%m-%d")
            end = datetime.strptime(self.config.data.sources.END_DATE, "%Y-%m-%d") + timedelta(days=1)

            data = web.DataReader(
                list(self.config.data.sources.TREASURY_YIELDS),
                'fred',
                start=start,
                end=end
            )

            data = data.reset_index()
            data = data.rename(columns={'DATE': 'date'})
            data['date'] = pd.to_datetime(data['date'])

            # Calculate mean of all yields
            yield_cols = [col for col in data.columns if col != 'date']
            data['bondyield'] = data[yield_cols].mean(axis=1, skipna=True)
            data = data[['date', 'bondyield']]

            self.logger.info(f"Downloaded {len(data)} rows of treasury yield data")

            if save:
                save_path = self.external_data_path / "treasury_yields.parquet"
                data.to_parquet(save_path, index=False)
                self.logger.info(f"Saved to {save_path}")

            return data

        except Exception as e:
            self.logger.error(f"Failed to download treasury yields: {e}")
            raise

    def download_all(
        self,
        tickers: Optional[List[str]] = None,
        stock_limit: Optional[int] = None,
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all data sources.

        Args:
            tickers: List of stock tickers (None = use index file)
            stock_limit: Maximum number of stocks to download from index (None = all)
            save: Whether to save to disk

        Returns:
            Dictionary with keys: 'stocks', 'vix', 'commodities', 'treasury_yields'
        """
        self.logger.info("=" * 60)
        self.logger.info("DOWNLOADING ALL DATA SOURCES")
        self.logger.info("=" * 60)

        result = {}

        # Get tickers from index if not provided
        if tickers is None:
            tickers = self.get_sp500_tickers(limit=stock_limit)

        # Download stock data
        result['stocks'] = self.download_stock_data(tickers=tickers, save=save)

        # Download VIX
        if self.config.data.features.FEATURE_FLAGS.get('vix', True):
            result['vix'] = self.download_vix(save=save)

        # Download commodities
        if self.config.data.features.FEATURE_FLAGS.get('commodities', True):
            result['commodities'] = self.download_commodities(save=save)

        # Download treasury yields
        if self.config.data.features.FEATURE_FLAGS.get('treasury_yields', True):
            result['treasury_yields'] = self.download_treasury_yields(save=save)

        self.logger.info("=" * 60)
        self.logger.info("DATA DOWNLOAD COMPLETE")
        self.logger.info("=" * 60)
        self._log_time()

        return result

    def load_saved_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load previously downloaded data from disk.

        Returns:
            Dictionary with loaded DataFrames
        """
        self.logger.info("Loading saved data from disk...")

        result = {}

        # Load stock data
        stock_path = self.raw_data_path / "stock_data.parquet"
        if stock_path.exists():
            result['stocks'] = pd.read_parquet(stock_path)
            self.logger.info(f"Loaded stock data: {len(result['stocks'])} rows")
        else:
            self.logger.warning(f"No saved stock data found at {stock_path}")

        # Load external data
        for name, filename in [
            ('vix', 'vix.parquet'),
            ('commodities', 'commodities.parquet'),
            ('treasury_yields', 'treasury_yields.parquet')
        ]:
            path = self.external_data_path / filename
            if path.exists():
                result[name] = pd.read_parquet(path)
                self.logger.info(f"Loaded {name}: {len(result[name])} rows")

        return result
