"""
Data download integration test.

This test actually downloads real data to verify the download pipeline works.
It's marked as slow because it requires network access and takes time.
"""

import sys
import pytest
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloader import DataDownloader
from src.config import load_config


@pytest.mark.slow
@pytest.mark.integration
class TestDataDownload:
    """Test actual data download from external sources."""

    def test_download_treasury_yields_with_none_end_date(self):
        """Test treasury yields download with END_DATE=None (use today)."""
        config = load_config('main')
        config.data.sources.END_DATE = None  # Explicitly set to None

        # Create downloader with temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override paths to use temp directory
            config.data.paths.EXTERNAL_DATA_PATH = temp_dir

            downloader = DataDownloader(config)

            # This should not raise an error
            data = downloader.download_treasury_yields(save=False)

            assert data is not None
            assert len(data) > 0
            assert 'date' in data.columns
            assert 'bondyield' in data.columns
            print(f"Downloaded {len(data)} rows of treasury yield data")

    def test_download_treasury_yields_with_explicit_end_date(self):
        """Test treasury yields download with explicit END_DATE."""
        config = load_config('main')
        config.data.sources.START_DATE = "2023-01-01"
        config.data.sources.END_DATE = "2023-12-31"

        with tempfile.TemporaryDirectory() as temp_dir:
            config.data.paths.EXTERNAL_DATA_PATH = temp_dir
            downloader = DataDownloader(config)

            data = downloader.download_treasury_yields(save=False)

            assert data is not None
            assert len(data) > 0
            # FRED returns all available data, just verify 2023 data is included
            assert data[data['date'].dt.year == 2023].shape[0] > 0
            print(f"Downloaded {len(data)} rows (includes 2023 data)")

    def test_download_vix(self):
        """Test VIX download."""
        config = load_config('main')
        config.data.sources.START_DATE = "2023-01-01"
        config.data.sources.END_DATE = "2023-12-31"

        with tempfile.TemporaryDirectory() as temp_dir:
            config.data.paths.EXTERNAL_DATA_PATH = temp_dir
            downloader = DataDownloader(config)

            data = downloader.download_vix(save=False)

            assert data is not None
            assert len(data) > 0
            assert 'date' in data.columns
            assert 'vix' in data.columns
            print(f"Downloaded {len(data)} rows of VIX data")

    def test_download_commodities(self):
        """Test commodities download."""
        config = load_config('main')
        config.data.sources.START_DATE = "2023-01-01"
        config.data.sources.END_DATE = "2023-12-31"

        # Limit to just 2 commodities for faster test
        config.data.sources.COMMODITIES._data = {
            'GC=F': 'Gold',
            'SI=F': 'Silver'
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config.data.paths.EXTERNAL_DATA_PATH = temp_dir
            downloader = DataDownloader(config)

            data = downloader.download_commodities(save=False)

            assert data is not None
            assert len(data) > 0
            assert 'date' in data.columns
            assert 'Gold' in data.columns
            assert 'Silver' in data.columns
            print(f"Downloaded {len(data)} rows of commodities data")

    def test_download_all_external_data(self):
        """Test downloading all external data sources."""
        config = load_config('main')
        config.data.sources.START_DATE = "2023-01-01"
        config.data.sources.END_DATE = None  # Use today

        # Limit commodities for speed
        config.data.sources.COMMODITIES._data = {
            'GC=F': 'Gold'
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config.data.paths.EXTERNAL_DATA_PATH = temp_dir
            config.data.paths.RAW_DATA_PATH = temp_dir

            downloader = DataDownloader(config)

            # Download VIX
            vix_data = downloader.download_vix(save=False)
            assert vix_data is not None
            assert len(vix_data) > 0

            # Download commodities
            comm_data = downloader.download_commodities(save=False)
            assert comm_data is not None
            assert len(comm_data) > 0

            # Download treasury yields with None END_DATE
            treasury_data = downloader.download_treasury_yields(save=False)
            assert treasury_data is not None
            assert len(treasury_data) > 0

            print("All external data sources downloaded successfully")
