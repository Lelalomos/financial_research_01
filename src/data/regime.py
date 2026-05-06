"""
Market regime detection utilities.

The first implementation is dependency-light and deterministic. It fits
quantile thresholds on training data only, then applies those thresholds to all
splits so validation/test rows cannot influence regime boundaries.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class MarketRegimeDetector:
    """Quantile-based market regime detector."""

    def __init__(
        self,
        proxy_column: str = "vix",
        n_regimes: int = 3,
        low_quantile: float = 0.33,
        high_quantile: float = 0.66,
        default_regime: int = 1,
    ):
        if n_regimes not in (2, 3):
            raise ValueError("n_regimes must be 2 or 3")
        if not 0 <= default_regime < n_regimes:
            raise ValueError("default_regime must be within the regime range")
        if n_regimes == 3 and low_quantile >= high_quantile:
            raise ValueError("low_quantile must be less than high_quantile")

        self.proxy_column = proxy_column
        self.n_regimes = n_regimes
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.default_regime = default_regime
        self.thresholds: Optional[List[float]] = None

    @classmethod
    def from_config(cls, config) -> "MarketRegimeDetector":
        """Create a detector from config.data.regime."""
        regime_config = config.data.regime
        return cls(
            proxy_column=regime_config.PROXY_COLUMN,
            n_regimes=regime_config.N_REGIMES,
            low_quantile=regime_config.LOW_QUANTILE,
            high_quantile=regime_config.HIGH_QUANTILE,
            default_regime=regime_config.DEFAULT_REGIME,
        )

    def fit(self, train_df: pd.DataFrame) -> "MarketRegimeDetector":
        """
        Fit regime thresholds from training data only.

        Args:
            train_df: Training split containing the proxy column.
        """
        if self.proxy_column not in train_df.columns:
            raise ValueError(f"Missing regime proxy column: {self.proxy_column}")

        proxy = pd.to_numeric(train_df[self.proxy_column], errors="coerce").dropna()
        if proxy.empty:
            raise ValueError(f"No valid values available in regime proxy column: {self.proxy_column}")

        quantiles = [0.5] if self.n_regimes == 2 else [self.low_quantile, self.high_quantile]
        self.thresholds = [float(value) for value in np.quantile(proxy.to_numpy(), quantiles)]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime_id using already-fitted thresholds.

        Lower proxy values map to lower regime IDs. Missing proxy values use the
        configured default regime.
        """
        if self.thresholds is None:
            raise ValueError("MarketRegimeDetector must be fit before transform")
        if self.proxy_column not in df.columns:
            raise ValueError(f"Missing regime proxy column: {self.proxy_column}")

        result = df.copy()
        proxy = pd.to_numeric(result[self.proxy_column], errors="coerce")
        regime = np.searchsorted(np.asarray(self.thresholds), proxy.to_numpy(), side="right")
        regime = regime.astype(np.int32)
        regime[proxy.isna().to_numpy()] = self.default_regime
        result["regime_id"] = regime
        return result

    def fit_transform_splits(
        self,
        splits: Dict[str, pd.DataFrame],
        train_split: str = "train",
    ) -> Dict[str, pd.DataFrame]:
        """Fit on the train split and transform every split."""
        if train_split not in splits or splits[train_split].empty:
            raise ValueError("A non-empty train split is required to fit market regimes")

        self.fit(splits[train_split])
        return {
            split_name: self.transform(split_df) if not split_df.empty else split_df.copy()
            for split_name, split_df in splits.items()
        }

    def to_metadata(self) -> Dict[str, object]:
        """Return weights-only-safe metadata for checkpoints or preprocessing info."""
        if self.thresholds is None:
            raise ValueError("MarketRegimeDetector must be fit before exporting metadata")
        return {
            "method": "quantile",
            "proxy_column": self.proxy_column,
            "n_regimes": self.n_regimes,
            "thresholds": list(self.thresholds),
            "default_regime": self.default_regime,
        }

    @classmethod
    def from_metadata(cls, metadata: Dict[str, object]) -> "MarketRegimeDetector":
        """Restore a detector from exported metadata."""
        detector = cls(
            proxy_column=str(metadata["proxy_column"]),
            n_regimes=int(metadata["n_regimes"]),
            default_regime=int(metadata.get("default_regime", 1)),
        )
        detector.thresholds = [float(value) for value in metadata["thresholds"]]
        return detector
