"""
Data configuration for CRNN Financial Prediction Model.

This module defines all data-related configuration including:
- Data sources (stocks, VIX, commodities, treasury yields)
- Feature engineering parameters
- Data split ratios
- Normalization settings
- Feature flags for ablation studies
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class DataConfig:
    """
    Configuration for data pipeline.

    Attributes:
        # Data sources
        START_DATE: Start date for historical data download
        END_DATE: End date for historical data (None = current date)
        SP500_TICKER_SOURCE: Source for S&P 500 ticker list ('yfinance', 'wikipedia', or file path)
        USE_YFINANCE_LIVE: Fetch live data from yfinance if True

        # Data splits (time-based)
        TRAIN_RATIO: Ratio of training data (default 0.70)
        TEST_RATIO: Ratio of test data (default 0.20)
        VAL_RATIO: Ratio of validation data (default 0.10)

        # Sequence parameters
        SEQUENCE_LENGTH: Lookback window size in days (default 30)
        PREDICTION_HORIZON: Days ahead to predict (default 5, configurable)
        TARGET_THRESHOLD: Threshold for tanh normalization (default 10.0 = +/-10%)

        # Technical indicators
        EMA_PERIODS: EMA periods to calculate (default: 50, 100, 200)
        RSI_PERIOD: RSI calculation period (default 14)
        STOCHRSI_PERIOD: Stochastic RSI period (default 14)

        # Candlestick patterns
        USE_CANDLESTICK_PATTERNS: Whether to include TA-Lib candlestick patterns

        # External data sources
        VIX_SYMBOL: VIX index symbol (default ^VIX)
        COMMODITIES: Dictionary of commodity symbols and names
        TREASURY_YIELDS: List of FRED treasury yield symbols

        # Sector/group classification
        SECTOR_MAPPING_SOURCE: Path to sector mapping JSON file or 'default'
        USE_SECTOR_EMBEDDING: Whether to use sector/group embeddings

        # Feature flags for ablation studies
        FEATURE_FLAGS: Dictionary to enable/disable feature groups

        # Normalization
        NORMALIZATION_METHOD: Method for normalization ('log_transform', 'standard', 'minmax')
        LOG_TRANSFORM_OFFSET: Offset for log transform to handle negative/zeros

        # Data filtering
        MIN_TRADING_DAYS: Minimum trading days required for a stock (default 252 = 1 year)
        MAX_MISSING_RATIO: Maximum ratio of missing values allowed (default 0.1)

        # External data handling
        EXTERNAL_DATA_FILL_METHOD: How to handle missing external data ('ffill', 'interpolate', 'drop')
        MAX_EXTERNAL_FILL_DAYS: Maximum days to forward fill external data (default 5)

        # Paths
        RAW_DATA_PATH: Path to store raw downloaded data
        PROCESSED_DATA_PATH: Path to store processed data
        SPLITS_PATH: Path to store train/val/test splits
        EXTERNAL_DATA_PATH: Path to store external data (VIX, commodities, etc.)
    """

    # Data sources
    START_DATE: str = "2000-01-01"
    END_DATE: Optional[str] = None
    SP500_TICKER_SOURCE: str = "wikipedia"  # 'wikipedia', 'yfinance', or file path
    USE_YFINANCE_LIVE: bool = True
    INDEX_FILE: str = "GSPC.json"  # Index file in raw_data/index/ (default: S&P 500)
    RAW_DATA_INDEX_PATH: str = "raw_data/index"  # Path to index files

    # Data splits (time-based)
    TRAIN_RATIO: float = 0.70
    TEST_RATIO: float = 0.20
    VAL_RATIO: float = 0.10

    # Sequence parameters
    SEQUENCE_LENGTH: int = 30  # Lookback window in trading days
    PREDICTION_HORIZON: int = 5  # Days ahead to predict (configurable!)
    TARGET_THRESHOLD: float = 10.0  # For tanh normalization: +/-10%

    # Technical indicators
    EMA_PERIODS: tuple = (50, 100, 200)
    RSI_PERIOD: int = 14
    STOCHRSI_PERIOD: int = 14
    MACD_PARAMS: tuple = (12, 26, 9)  # fast, slow, signal

    # Candlestick patterns
    USE_CANDLESTICK_PATTERNS: bool = True

    # External data sources
    VIX_SYMBOL: str = "^VIX"
    COMMODITIES: Dict[str, str] = field(default_factory=lambda: {
        'GC=F': 'Gold',
        'HG=F': 'Copper',
        'ZC=F': 'Corn',
        'ZS=F': 'Soybeans',
        'CC=F': 'Cocoa',
        'SI=F': 'Silver'
    })
    TREASURY_YIELDS: tuple = ('DGS10', 'DGS30', 'DGS2')

    # Sector/group classification
    SECTOR_MAPPING_SOURCE: str = "default"
    USE_SECTOR_EMBEDDING: bool = True

    # Default sector mapping (can be overridden by file)
    DEFAULT_SECTOR_MAPPING: Dict[str, List[str]] = field(default_factory=dict)

    # Feature flags for ablation studies
    FEATURE_FLAGS: Dict[str, bool] = field(default_factory=lambda: {
        'price_features': True,      # OHLCV
        'ema_features': True,
        'rsi_features': True,
        'stochrsi_features': True,
        'macd_features': True,
        'candlestick_patterns': True,
        'vix': True,
        'commodities': True,
        'treasury_yields': True,
        'time_features': True,       # day, month
        'financial_metrics': True,   # EPS, ROE, ROI, debt ratios, PE, PEG, current ratio
    })

    # Normalization
    NORMALIZATION_METHOD: str = "log_transform"  # 'log_transform', 'standard', 'minmax', 'robust'
    LOG_TRANSFORM_OFFSET: float = 1.0  # Add this before log to handle negatives

    # Data filtering
    MIN_TRADING_DAYS: int = 252  # Minimum 1 year of data
    MAX_MISSING_RATIO: float = 0.1  # Maximum 10% missing values

    # External data handling
    EXTERNAL_DATA_FILL_METHOD: str = "ffill"  # 'ffill', 'interpolate', 'drop'
    MAX_EXTERNAL_FILL_DAYS: int = 5

    # Financial metrics
    USE_FINANCIAL_METRICS: bool = True
    FINANCIAL_METRICS_SOURCE: str = "raw_data/ticket_data/us"
    FINANCIAL_METRICS_FILL_METHOD: str = "ffill"

    # Failed stocks tracking
    FAILED_TICKERS_FILE: str = "data/failed_tickers.json"
    SKIP_FAILED_TICKERS: bool = True

    # Paths
    RAW_DATA_PATH: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"
    SPLITS_PATH: str = "data/splits"
    EXTERNAL_DATA_PATH: str = "data/external"

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Validate split ratios
        if not abs((self.TRAIN_RATIO + self.TEST_RATIO + self.VAL_RATIO) - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {self.TRAIN_RATIO + self.TEST_RATIO + self.VAL_RATIO}")

        # Set end date to current if not specified
        if self.END_DATE is None:
            self.END_DATE = datetime.now().strftime("%Y-%m-%d")

        # Validate prediction horizon
        if self.PREDICTION_HORIZON < 1:
            raise ValueError(f"PREDICTION_HORIZON must be >= 1, got {self.PREDICTION_HORIZON}")

        # Validate sequence length
        if self.SEQUENCE_LENGTH < 1:
            raise ValueError(f"SEQUENCE_LENGTH must be >= 1, got {self.SEQUENCE_LENGTH}")

        # Validate normalization method
        valid_methods = ['log_transform', 'standard', 'minmax', 'robust']
        if self.NORMALIZATION_METHOD not in valid_methods:
            raise ValueError(f"NORMALIZATION_METHOD must be one of {valid_methods}, got {self.NORMALIZATION_METHOD}")

    @property
    def ema_columns(self) -> List[str]:
        """Return EMA column names."""
        return [f'ema_{period}' for period in self.EMA_PERIODS]

    @property
    def commodity_columns(self) -> List[str]:
        """Return commodity column names."""
        return list(self.COMMODITIES.values())

    @property
    def treasury_columns(self) -> List[str]:
        """Return treasury yield column names."""
        return ['bondyield']  # Averaged value

    @property
    def financial_metrics_columns(self) -> List[str]:
        """Return financial metric column names."""
        return ['pe_ratio', 'peg_ratio', 'eps', 'roe', 'roi',
                'debt_to_equity', 'debt_to_asset', 'current_ratio']

    @property
    def enabled_feature_groups(self) -> List[str]:
        """Return list of enabled feature groups."""
        return [k for k, v in self.FEATURE_FLAGS.items() if v]

    def get_enabled_features(self) -> List[str]:
        """
        Get list of all enabled feature column names.

        Returns:
            List of feature names that are enabled via FEATURE_FLAGS
        """
        features = []

        if self.FEATURE_FLAGS.get('price_features', False):
            features.extend(['open', 'high', 'low', 'close', 'volume'])

        if self.FEATURE_FLAGS.get('ema_features', False):
            features.extend(self.ema_columns)

        if self.FEATURE_FLAGS.get('rsi_features', False):
            features.append(f'rsi_{self.RSI_PERIOD}')

        if self.FEATURE_FLAGS.get('stochrsi_features', False):
            features.append(f'stochrsi_{self.STOCHRSI_PERIOD}')

        if self.FEATURE_FLAGS.get('macd_features', False):
            features.extend(['macd', 'macd_signal', 'macd_hist'])

        if self.FEATURE_FLAGS.get('candlestick_patterns', False):
            import talib
            # Get all candlestick pattern functions
            patterns = [name for name in dir(talib) if name.startswith('CDL')]
            features.extend(patterns)

        if self.FEATURE_FLAGS.get('vix', False):
            features.append('vix')

        if self.FEATURE_FLAGS.get('commodities', False):
            features.extend(self.commodity_columns)

        if self.FEATURE_FLAGS.get('treasury_yields', False):
            features.extend(self.treasury_columns)

        if self.FEATURE_FLAGS.get('time_features', False):
            features.extend(['day', 'month'])

        # Financial metrics
        if self.FEATURE_FLAGS.get('financial_metrics', False):
            features.extend(self.financial_metrics_columns)

        return features


# Default configuration instance
default_data_config = DataConfig()
