"""
Unit tests for stock sampling utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sampling import sample_stocks_by_group, get_sampling_stats


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)

    n_stocks = 30
    n_groups = 5
    n_days = 100

    data = []
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    for group_id in range(n_groups):
        stocks_per_group = n_stocks // n_groups
        for stock_idx in range(stocks_per_group):
            tic = f"STOCK_{group_id}_{stock_idx}"

            for date in dates:
                data.append({
                    'tic': tic,
                    'date': date,
                    'group_id': group_id,
                    'open': np.random.randn() + 100,
                    'close': np.random.randn() + 100,
                })

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def imbalanced_data():
    """Create data with imbalanced groups."""
    np.random.seed(42)

    data = []
    dates = pd.date_range('2020-01-01', periods=50, freq='D')

    # Group 0: 10 stocks
    for i in range(10):
        for date in dates:
            data.append({'tic': f'STOCK_0_{i}', 'date': date, 'group_id': 0})

    # Group 1: 5 stocks
    for i in range(5):
        for date in dates:
            data.append({'tic': f'STOCK_1_{i}', 'date': date, 'group_id': 1})

    # Group 2: 3 stocks
    for i in range(3):
        for date in dates:
            data.append({'tic': f'STOCK_2_{i}', 'date': date, 'group_id': 2})

    return pd.DataFrame(data)


@pytest.fixture
def market_cap_metadata_dir(tmp_path):
    """Create synthetic per-ticker metadata files with market caps."""
    caps = {
        "STOCK_0_0": 1000,
        "STOCK_0_1": 900,
        "STOCK_0_2": 800,
        "STOCK_0_3": 700,
        "STOCK_0_4": 600,
        "STOCK_0_5": 500,
        "STOCK_1_0": 2000,
        "STOCK_1_1": 1900,
        "STOCK_1_2": 1800,
        "STOCK_1_3": 1700,
        "STOCK_1_4": 1600,
        "STOCK_1_5": 1500,
        "STOCK_2_0": 3000,
        "STOCK_2_1": 2900,
        "STOCK_2_2": 2800,
        "STOCK_2_3": 2700,
        "STOCK_2_4": 2600,
        "STOCK_2_5": 2500,
        "STOCK_3_0": 4000,
        "STOCK_3_1": 3900,
        "STOCK_3_2": 3800,
        "STOCK_3_3": 3700,
        "STOCK_3_4": 3600,
        "STOCK_3_5": 3500,
        "STOCK_4_0": 5000,
        "STOCK_4_1": 4900,
        "STOCK_4_2": 4800,
        "STOCK_4_3": 4700,
        "STOCK_4_4": 4600,
        "STOCK_4_5": 4500,
    }

    for ticker, market_cap in caps.items():
        payload = {"Highlights": {"MarketCapitalization": market_cap}}
        (tmp_path / f"{ticker}.json").write_text(json.dumps(payload), encoding="utf-8")

    return tmp_path


class TestSampleStocksByGroup:
    """Test sample_stocks_by_group function."""

    def test_all_groups_represented(self, sample_data):
        """Test that all group_ids are represented when sampling."""
        n_groups = sample_data['group_id'].nunique()

        # Sample more stocks than groups
        selected = sample_stocks_by_group(sample_data, n_stocks=15, seed=42)

        # Get groups of selected stocks
        selected_groups = sample_data[sample_data['tic'].isin(selected)]['group_id'].unique()

        # All groups should be represented
        assert len(selected_groups) == n_groups
        assert set(selected_groups) == set(sample_data['group_id'].unique())

    def test_exact_count(self, sample_data):
        """Test that exact number of stocks is returned."""
        for n_stocks in [5, 10, 15, 20]:
            selected = sample_stocks_by_group(sample_data, n_stocks=n_stocks, seed=42)
            assert len(selected) == n_stocks

    def test_balanced_distribution(self, sample_data):
        """Test that groups are roughly balanced."""
        n_groups = sample_data['group_id'].nunique()
        n_stocks = 20

        selected = sample_stocks_by_group(sample_data, n_stocks=n_stocks, seed=42)

        # Count stocks per group
        groups = []
        for tic in selected:
            group = sample_data[sample_data['tic'] == tic]['group_id'].iloc[0]
            groups.append(group)

        from collections import Counter
        counts = Counter(groups)

        # Each group should have roughly equal stocks
        # Expected: 20 / 5 = 4 stocks per group
        expected_per_group = n_stocks // n_groups

        for count in counts.values():
            # Allow some variance due to remainder handling
            assert expected_per_group <= count <= expected_per_group + 1

    def test_with_remainder(self, sample_data):
        """Test handling when n_stocks % n_groups != 0."""
        n_groups = sample_data['group_id'].nunique()
        n_stocks = 17  # Not evenly divisible by 5

        selected = sample_stocks_by_group(sample_data, n_stocks=n_stocks, seed=42)

        assert len(selected) == n_stocks

        # Count stocks per group
        groups = []
        for tic in selected:
            group = sample_data[sample_data['tic'] == tic]['group_id'].iloc[0]
            groups.append(group)

        from collections import Counter
        counts = Counter(groups)

        # Base allocation: 17 // 5 = 3
        # Remainder: 17 % 5 = 2
        # So 2 groups should have 4 stocks, 3 groups should have 3 stocks
        base = n_stocks // n_groups
        remainder = n_stocks % n_groups

        n_with_extra = sum(1 for count in counts.values() if count == base + 1)
        n_with_base = sum(1 for count in counts.values() if count == base)

        assert n_with_extra == remainder
        assert n_with_base == n_groups - remainder

    def test_single_group(self):
        """Test edge case with only 1 group."""
        # Create data with 1 group
        data = []
        for i in range(10):
            data.append({'tic': f'STOCK_{i}', 'group_id': 0})
        df = pd.DataFrame(data)

        selected = sample_stocks_by_group(df, n_stocks=5, seed=42)

        assert len(selected) == 5
        assert all(df[df['tic'].isin(selected)]['group_id'] == 0)

    def test_n_stocks_less_than_groups(self, sample_data):
        """Test edge case: n_stocks < n_groups."""
        n_groups = sample_data['group_id'].nunique()
        n_stocks = 3  # Less than 5 groups

        selected = sample_stocks_by_group(sample_data, n_stocks=n_stocks, seed=42)

        # Should get exactly 3 stocks from 3 different groups
        assert len(selected) == n_stocks

        # Get groups of selected stocks
        selected_groups = []
        for tic in selected:
            group = sample_data[sample_data['tic'] == tic]['group_id'].iloc[0]
            selected_groups.append(group)

        # All selected stocks should be from different groups
        assert len(set(selected_groups)) == n_stocks

    def test_reproducibility(self, sample_data):
        """Test that same seed produces same results."""
        selected1 = sample_stocks_by_group(sample_data, n_stocks=10, seed=42)
        selected2 = sample_stocks_by_group(sample_data, n_stocks=10, seed=42)

        assert set(selected1) == set(selected2)

    def test_different_seeds(self, sample_data):
        """Test that different seeds produce different results."""
        selected1 = sample_stocks_by_group(sample_data, n_stocks=10, seed=42)
        selected2 = sample_stocks_by_group(sample_data, n_stocks=10, seed=123)

        # Results should be different (very unlikely to be same)
        assert set(selected1) != set(selected2)

    def test_sorted_mode_prefers_largest_market_caps(self, sample_data, market_cap_metadata_dir):
        """Test deterministic top-market-cap selection inside each group."""
        selected = sample_stocks_by_group(
            sample_data,
            n_stocks=10,
            seed=42,
            selection_mode='sorted',
            market_cap_metadata_dir=str(market_cap_metadata_dir),
        )

        expected = {
            "STOCK_0_0", "STOCK_0_1",
            "STOCK_1_0", "STOCK_1_1",
            "STOCK_2_0", "STOCK_2_1",
            "STOCK_3_0", "STOCK_3_1",
            "STOCK_4_0", "STOCK_4_1",
        }
        assert set(selected) == expected

    def test_sorted_mode_with_remainder_uses_highest_remaining_market_caps(self, sample_data, market_cap_metadata_dir):
        """Test sorted remainder handling stays market-cap ordered."""
        selected = sample_stocks_by_group(
            sample_data,
            n_stocks=12,
            seed=42,
            selection_mode='sorted',
            market_cap_metadata_dir=str(market_cap_metadata_dir),
        )

        expected = {
            "STOCK_0_0", "STOCK_0_1",
            "STOCK_1_0", "STOCK_1_1",
            "STOCK_2_0", "STOCK_2_1",
            "STOCK_3_0", "STOCK_3_1",
            "STOCK_4_0", "STOCK_4_1",
            "STOCK_4_2", "STOCK_4_3",
        }
        assert set(selected) == expected

    def test_sorted_mode_handles_missing_market_cap_with_ticker_fallback(self, sample_data, tmp_path):
        """Test missing market-cap metadata falls back deterministically."""
        payload = {"Highlights": {"MarketCapitalization": 1000}}
        for ticker in ["STOCK_0_1", "STOCK_0_2", "STOCK_0_3", "STOCK_0_4", "STOCK_0_5"]:
            (tmp_path / f"{ticker}.json").write_text(json.dumps(payload), encoding="utf-8")

        selected = sample_stocks_by_group(
            sample_data[sample_data["group_id"] == 0],
            n_stocks=2,
            seed=42,
            selection_mode='sorted',
            market_cap_metadata_dir=str(tmp_path),
        )

        assert selected == ["STOCK_0_1", "STOCK_0_2"]

    def test_imbalanced_groups(self, imbalanced_data):
        """Test sampling from imbalanced groups."""
        # Group 0: 10 stocks, Group 1: 5 stocks, Group 2: 3 stocks
        # Request 6 stocks total
        selected = sample_stocks_by_group(imbalanced_data, n_stocks=6, seed=42)

        assert len(selected) == 6

        # All 3 groups should still be represented
        selected_groups = imbalanced_data[imbalanced_data['tic'].isin(selected)]['group_id'].unique()
        assert len(selected_groups) == 3
        assert set(selected_groups) == {0, 1, 2}

    def test_custom_columns(self):
        """Test with custom column names."""
        data = []
        for i in range(10):
            data.append({'ticker': f'STOCK_{i}', 'sector': i % 3})
        df = pd.DataFrame(data)

        selected = sample_stocks_by_group(
            df,
            n_stocks=6,
            seed=42,
            group_col='sector',
            tic_col='ticker'
        )

        assert len(selected) == 6
        assert all(t in df['ticker'].values for t in selected)

    def test_empty_dataframe(self):
        """Test error handling with empty DataFrame."""
        df = pd.DataFrame({'tic': [], 'group_id': []})

        with pytest.raises(ValueError, match="DataFrame is empty"):
            sample_stocks_by_group(df, n_stocks=5, seed=42)

    def test_missing_group_column(self):
        """Test error handling with missing group column."""
        df = pd.DataFrame({'tic': ['A', 'B', 'C']})

        with pytest.raises(ValueError, match="Column 'group_id' not found"):
            sample_stocks_by_group(df, n_stocks=2, seed=42)

    def test_missing_tic_column(self):
        """Test error handling with missing tic column."""
        df = pd.DataFrame({'group_id': [0, 0, 1]})

        with pytest.raises(ValueError, match="Column 'tic' not found"):
            sample_stocks_by_group(df, n_stocks=2, seed=42)

    def test_invalid_n_stocks(self):
        """Test error handling with invalid n_stocks."""
        df = pd.DataFrame({'tic': ['A', 'B'], 'group_id': [0, 1]})

        with pytest.raises(ValueError, match="n_stocks must be positive"):
            sample_stocks_by_group(df, n_stocks=0, seed=42)

        with pytest.raises(ValueError, match="n_stocks must be positive"):
            sample_stocks_by_group(df, n_stocks=-1, seed=42)


class TestGetSamplingStats:
    """Test get_sampling_stats function."""

    def test_stats_structure(self, sample_data):
        """Test that stats dictionary has correct structure."""
        selected = sample_stocks_by_group(sample_data, n_stocks=15, seed=42)
        stats = get_sampling_stats(sample_data, selected)

        assert 'total_selected' in stats
        assert 'total_groups' in stats
        assert 'groups' in stats
        assert 'stocks_per_group' in stats

    def test_stats_values(self, sample_data):
        """Test that stats values are correct."""
        selected = sample_stocks_by_group(sample_data, n_stocks=15, seed=42)
        stats = get_sampling_stats(sample_data, selected)

        assert stats['total_selected'] == 15
        assert stats['total_groups'] == sample_data['group_id'].nunique()
        assert len(stats['stocks_per_group']) == stats['total_groups']

    def test_stocks_per_group_count(self, sample_data):
        """Test stocks_per_group counts are correct."""
        selected = sample_stocks_by_group(sample_data, n_stocks=15, seed=42)
        stats = get_sampling_stats(sample_data, selected)

        # Manually verify counts
        for group_id, count in stats['stocks_per_group'].items():
            actual_count = len([
                tic for tic in selected
                if sample_data[sample_data['tic'] == tic]['group_id'].iloc[0] == group_id
            ])
            assert count == actual_count

    def test_custom_columns(self):
        """Test with custom column names."""
        data = []
        for i in range(10):
            data.append({'ticker': f'STOCK_{i}', 'sector': i % 3})
        df = pd.DataFrame(data)

        selected = sample_stocks_by_group(
            df,
            n_stocks=6,
            seed=42,
            group_col='sector',
            tic_col='ticker'
        )

        stats = get_sampling_stats(df, selected, group_col='sector', tic_col='ticker')

        assert stats['total_selected'] == 6
        assert 'stocks_per_group' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
