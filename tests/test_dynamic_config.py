"""
Unit tests for dynamic configuration lists.

Tests that COMMODITIES, TREASURY_YIELDS, and EMA_PERIODS
can be accessed and modified from the main config.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config


class TestDynamicConfigLists:
    """Test dynamic configuration lists in main.json."""

    @pytest.fixture
    def main_config(self):
        """Load main configuration."""
        return load_config('main')

    def test_commodities_access(self, main_config):
        """Test that COMMODITIES can be accessed from config."""
        commodities = main_config.data.sources.COMMODITIES
        # COMMODITIES is a Config object wrapping a dict
        assert hasattr(commodities, '_data')
        assert 'GC=F' in commodities  # Gold
        assert 'SI=F' in commodities  # Silver

    def test_treasury_yields_access(self, main_config):
        """Test that TREASURY_YIELDS can be accessed from config."""
        yields = main_config.data.sources.TREASURY_YIELDS
        assert isinstance(yields, list)
        # Check default yields exist
        assert 'DGS10' in yields  # 10-year Treasury
        assert 'DGS30' in yields  # 30-year Treasury

    def test_ema_periods_access(self, main_config):
        """Test that EMA_PERIODS can be accessed from config."""
        ema_periods = main_config.data.technical_indicators.EMA_PERIODS
        assert isinstance(ema_periods, list)
        # Check default EMA periods exist
        assert 50 in ema_periods
        assert 100 in ema_periods
        assert 200 in ema_periods

    def test_commodities_modification(self, main_config):
        """Test that COMMODITIES can be modified."""
        # Get original commodities
        original_commodities = dict(main_config.data.sources.COMMODITIES._data)
        original_count = len(original_commodities)

        # Add new commodity by modifying underlying dict
        main_config.data.sources.COMMODITIES._data['ZW=F'] = 'Wheat'

        # Verify it was added
        assert 'ZW=F' in main_config.data.sources.COMMODITIES
        assert main_config.data.sources.COMMODITIES['ZW=F'] == 'Wheat'
        assert len(main_config.data.sources.COMMODITIES) == original_count + 1

        # Clean up - remove the added commodity
        del main_config.data.sources.COMMODITIES._data['ZW=F']
        assert len(main_config.data.sources.COMMODITIES) == original_count

    def test_treasury_yields_modification(self, main_config):
        """Test that TREASURY_YIELDS can be modified."""
        # Get original yields
        original_yields = list(main_config.data.sources.TREASURY_YIELDS)
        original_count = len(original_yields)

        # Add new treasury yield
        main_config.data.sources.TREASURY_YIELDS.append('DGS5')  # 5-year Treasury

        # Verify it was added
        assert 'DGS5' in main_config.data.sources.TREASURY_YIELDS
        assert len(main_config.data.sources.TREASURY_YIELDS) == original_count + 1

        # Clean up - remove the added yield
        main_config.data.sources.TREASURY_YIELDS.remove('DGS5')
        assert len(main_config.data.sources.TREASURY_YIELDS) == original_count

    def test_ema_periods_modification(self, main_config):
        """Test that EMA_PERIODS can be modified."""
        # Get original periods
        original_periods = list(main_config.data.technical_indicators.EMA_PERIODS)
        original_count = len(original_periods)

        # Add new EMA period
        main_config.data.technical_indicators.EMA_PERIODS.append(20)

        # Verify it was added
        assert 20 in main_config.data.technical_indicators.EMA_PERIODS
        assert len(main_config.data.technical_indicators.EMA_PERIODS) == original_count + 1

        # Clean up - remove the added period
        main_config.data.technical_indicators.EMA_PERIODS.remove(20)
        assert len(main_config.data.technical_indicators.EMA_PERIODS) == original_count

    def test_commodities_replacement(self, main_config):
        """Test that COMMODITIES can be entirely replaced."""
        # Store original
        original_commodities = main_config.data.sources.COMMODITIES._data.copy()

        # Replace with new commodities
        new_commodities = {
            'GC=F': 'Gold',
            'SI=F': 'Silver',
            'PL=F': 'Platinum'
        }
        main_config.data.sources.COMMODITIES._data.update(new_commodities)

        # Verify replacement
        assert 'PL=F' in main_config.data.sources.COMMODITIES
        assert main_config.data.sources.COMMODITIES['PL=F'] == 'Platinum'

        # Restore original
        main_config.data.sources.COMMODITIES._data = original_commodities

    def test_treasury_yields_replacement(self, main_config):
        """Test that TREASURY_YIELDS can be entirely replaced."""
        # Store original
        original_yields = main_config.data.sources.TREASURY_YIELDS.copy()

        # Replace with new yields
        new_yields = ['DGS2', 'DGS5', 'DGS10']
        main_config.data.sources.TREASURY_YIELDS[:] = new_yields

        # Verify replacement
        assert main_config.data.sources.TREASURY_YIELDS == new_yields
        assert len(main_config.data.sources.TREASURY_YIELDS) == 3

        # Restore original
        main_config.data.sources.TREASURY_YIELDS[:] = original_yields

    def test_ema_periods_replacement(self, main_config):
        """Test that EMA_PERIODS can be entirely replaced."""
        # Store original
        original_periods = main_config.data.technical_indicators.EMA_PERIODS.copy()

        # Replace with new periods
        new_periods = [12, 26, 50]
        main_config.data.technical_indicators.EMA_PERIODS[:] = new_periods

        # Verify replacement
        assert main_config.data.technical_indicators.EMA_PERIODS == new_periods
        assert len(main_config.data.technical_indicators.EMA_PERIODS) == 3

        # Restore original
        main_config.data.technical_indicators.EMA_PERIODS[:] = original_periods


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
