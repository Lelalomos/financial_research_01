"""
Failed Tickers Manager for Multi-Model Financial Forecasting.

This module tracks stocks that failed to download or process, allowing them
to be skipped in future runs to avoid repeated failures.

Failed tickers are stored in a JSON file with metadata about why they failed.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

from src.config import load_config
from src.utils.logger import get_logger


class FailedTickersManager:
    """
    Manage failed tickers list - track and skip problematic stocks.

    Failed tickers are stored in a JSON file with structure:
    {
        "AAPL": {
            "reason": "No data available",
            "failed_at": "2025-01-02T10:30:00",
            "attempt_count": 1
        },
        ...
    }
    """

    def __init__(self, config):
        """
        Initialize the failed tickers manager.

        Args:
            config instance
        """
        self.config = config
        self.logger = get_logger("failed_tickers", log_dir="logs")
        self.failed_tickers_file = Path(config.FAILED_TICKERS_FILE)
        self.failed_tickers: Dict[str, Dict] = {}

        # Load existing failed tickers if file exists
        self._load()

    def _load(self):
        """Load failed tickers from JSON file."""
        if self.failed_tickers_file.exists():
            try:
                with open(self.failed_tickers_file, 'r') as f:
                    self.failed_tickers = json.load(f)
                self.logger.info(f"Loaded {len(self.failed_tickers)} failed tickers from {self.failed_tickers_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load failed tickers file: {e}")
                self.failed_tickers = {}
        else:
            self.logger.info("No existing failed tickers file found")
            self.failed_tickers = {}

    def _save(self):
        """Save failed tickers to JSON file."""
        try:
            # Ensure parent directory exists
            self.failed_tickers_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.failed_tickers_file, 'w') as f:
                json.dump(self.failed_tickers, f, indent=2)

            self.logger.debug(f"Saved {len(self.failed_tickers)} failed tickers to {self.failed_tickers_file}")
        except Exception as e:
            self.logger.error(f"Failed to save failed tickers file: {e}")

    def add_failed_ticker(self, ticker: str, reason: str):
        """
        Add a ticker to the failed list.

        Args:
            ticker: Ticker symbol that failed
            reason: Description of why it failed
        """
        if ticker in self.failed_tickers:
            # Increment attempt count
            self.failed_tickers[ticker]['attempt_count'] += 1
            self.failed_tickers[ticker]['last_failed_at'] = datetime.now().isoformat()
            self.failed_tickers[ticker]['last_reason'] = reason
            self.logger.info(f"Updated failed ticker {ticker} (attempt {self.failed_tickers[ticker]['attempt_count']})")
        else:
            # New failed ticker
            self.failed_tickers[ticker] = {
                'reason': reason,
                'last_reason': reason,
                'failed_at': datetime.now().isoformat(),
                'last_failed_at': datetime.now().isoformat(),
                'attempt_count': 1
            }
            self.logger.info(f"Added failed ticker {ticker}: {reason}")

        self._save()

    def add_failed_tickers(self, failed_list: List[str], reason: str = "Download failed"):
        """
        Add multiple tickers to the failed list.

        Args:
            failed_list: List of ticker symbols that failed
            reason: Description of why they failed
        """
        for ticker in failed_list:
            self.add_failed_ticker(ticker, reason)

        self.logger.info(f"Added {len(failed_list)} failed tickers")

    def remove_failed_ticker(self, ticker: str):
        """
        Remove a ticker from the failed list.

        Use this if a previously failed ticker is now working.

        Args:
            ticker: Ticker symbol to remove
        """
        if ticker in self.failed_tickers:
            del self.failed_tickers[ticker]
            self._save()
            self.logger.info(f"Removed {ticker} from failed list")

    def is_failed(self, ticker: str) -> bool:
        """
        Check if a ticker is in the failed list.

        Args:
            ticker: Ticker symbol to check

        Returns:
            True if ticker is in failed list, False otherwise
        """
        return ticker in self.failed_tickers

    def get_failed_tickers(self) -> Set[str]:
        """
        Get set of all failed tickers.

        Returns:
            Set of ticker symbols
        """
        return set(self.failed_tickers.keys())

    def filter_tickers(self, tickers: List[str]) -> List[str]:
        """
        Filter out failed tickers from a list.

        Args:
            tickers: List of ticker symbols

        Returns:
            List with failed tickers removed (if SKIP_FAILED_TICKERS is True)
        """
        if not self.config.SKIP_FAILED_TICKERS:
            return tickers

        failed = self.get_failed_tickers()
        filtered = [t for t in tickers if t not in failed]

        if len(filtered) < len(tickers):
            skipped = len(tickers) - len(filtered)
            self.logger.info(f"Filtered out {skipped} failed tickers from {len(tickers)} total")

        return filtered

    def get_failed_info(self, ticker: str) -> Optional[Dict]:
        """
        Get detailed information about a failed ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with failure information or None if not failed
        """
        return self.failed_tickers.get(ticker)

    def clear_all(self):
        """Clear all failed tickers from the list."""
        count = len(self.failed_tickers)
        self.failed_tickers = {}
        self._save()
        self.logger.info(f"Cleared {count} failed tickers")

    def get_summary(self) -> Dict:
        """
        Get summary of failed tickers.

        Returns:
            Dictionary with summary statistics
        """
        if not self.failed_tickers:
            return {
                'total_failed': 0,
                'reasons': {},
                'oldest_failure': None,
                'newest_failure': None
            }

        # Count by reason
        reasons = {}
        for info in self.failed_tickers.values():
            reason = info.get('reason', 'Unknown')
            reasons[reason] = reasons.get(reason, 0) + 1

        # Find oldest and newest failures
        dates = [info.get('failed_at', '') for info in self.failed_tickers.values()]
        dates = [d for d in dates if d]

        return {
            'total_failed': len(self.failed_tickers),
            'reasons': reasons,
            'oldest_failure': min(dates) if dates else None,
            'newest_failure': max(dates) if dates else None
        }
