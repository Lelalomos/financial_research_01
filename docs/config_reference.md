# Config Reference

## Context

This document explains the JSON config files under `config/` as they exist in
the current repository. It is based on:

- `config/*.json`
- `docs/configuration.md`
- `docs/run_guide.md`
- `docs/model_architecture.md`
- `docs/data_pipeline.md`
- `docs/lightning_backend.md`
- `src/config/schemas.py`
- runtime usage in `src/` and `scripts/`

Use this file as the field-by-field reference. For operational steps, see
`docs/run_guide.md`.

## Important Notes

- `config/main.json` and `config/model.json` are the main runtime configs.
- `config/hyperparameter.json` is used by Optuna tuning workflows.
- `config/test.json`, `config/deploy.json`, and `config/validate.json` exist,
  but current runtime usage appears limited compared with `main.json` and
  `model.json`.
- The schema in `src/config/schemas.py` is the source of truth for validated
  fields in `main.json` and `model.json`.
- Live config can differ from older documentation. Example:
  `model.loss.DIRECTIONAL_ALPHA` is `1.0` in the current `config/model.json`,
  while older docs mention `0.1`.

## How To Read This Reference

Each field includes:

- `Meaning`: what the field controls
- `How to set`: valid or practical values
- `When to change`: when you should tune it

## `config/main.json`

Controls data sources, feature engineering, preprocessing, and dataset layout.

### `data.sources`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `START_DATE` | First date to download/use for market data. | ISO date string like `2000-01-01`. | Change to limit history or focus on a recent market regime. |
| `END_DATE` | Last date to include. `null` means use latest available data. | ISO date string or `null`. | Set for reproducible experiments on a fixed cutoff date. |
| `SP500_TICKER_SOURCE` | Source used to build the stock universe. | Current value is `wikipedia`; keep to supported sources only. | Change only if you implement another ticker source. |
| `USE_YFINANCE_LIVE` | Whether to fetch live market/external data from yfinance. | `true` or `false`. | Disable for offline or cached-only workflows. |
| `INDEX_FILE` | File name for index metadata. | String file name. | Change if you maintain multiple index metadata snapshots. |
| `RAW_DATA_INDEX_PATH` | Directory containing saved index metadata. | Relative path string. | Change if you reorganize raw index storage. |
| `VIX_SYMBOL` | Symbol used for volatility proxy download. | Usually `^VIX`. | Change if you want another volatility proxy. |
| `COMMODITIES` | Map of market symbol to output feature name. Keys are provider tickers; values become readable column names. | JSON object like `{ "GC=F": "Gold" }`. | Change when adding/removing external commodity features. |
| `TREASURY_YIELDS` | FRED series IDs for treasury yield features. | List of strings like `["DGS10"]`. | Change when you want different interest-rate tenors. |

### `data.splits`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `TRAIN_RATIO` | Fraction of dates assigned to train. | Float between `0` and `1`; all split ratios must sum to `1.0`. | Increase when you need more training data. |
| `TEST_RATIO` | Fraction of dates assigned to test. | Float between `0` and `1`. | Increase when you want a more conservative final holdout. |
| `VAL_RATIO` | Fraction of dates assigned to validation. | Float between `0` and `1`. | Increase when model selection stability matters more than train size. |

### `data.sequences`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `SEQUENCE_LENGTH` | Number of past bars per sample. | Positive integer. Common values: `20`, `30`, `60`, `90`. | Increase for longer temporal context; decrease for faster training. |
| `PREDICTION_HORIZON` | Future offset used to compute the target return. | Positive integer measured in bars/days. | Change to switch between short-horizon and swing-horizon prediction. |
| `TARGET_THRESHOLD` | Target scaling or threshold parameter used in preprocessing logic. | Positive float. | Change only if your target engineering or labeling strategy depends on it. |
| `NORMALIZE_TARGET` | Whether target values are normalized for training/inference compatibility. | `true` or `false`. | Keep consistent across training and prediction; change only with care. |
| `STRIDE` | Step size for sliding-window sequence generation. | Positive integer; `1` means every possible window. | Increase to reduce sample count and speed up preprocessing. |

### `data.dataset`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `MODE` | Controls how sequence windows are prepared for training. | `precomputed_sequences` or `on_the_fly_sequences`. | Use `precomputed_sequences` to create and save sequence arrays during preprocessing; use `on_the_fly_sequences` for lazy/streaming window generation during training when RAM is limited. |

### Wrapper routing note

Shell wrappers under `scripts/` now resolve the fallback model type from
`config/model.json -> model.selection.DEFAULT_MODEL_TYPE` when `--model-type`
is omitted. They then map special model families to these processed data
directories unless `--data-dir` overrides them:

- `chronos2` -> `data/processed_chronos2`
- `chronos_rich` -> `data/processed_chronos_rich`
- `kronos_rich` -> `data/processed_kronos_rich`
- all other model types -> `data/processed`

### `data.technical_indicators`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `EMA_PERIODS` | EMA windows to generate. | List of positive integers. | Change to test faster or slower trend-following features. |
| `RSI_PERIOD` | RSI lookback period. | Positive integer, commonly `14`. | Change for more reactive or smoother momentum. |
| `STOCHRSI_PERIOD` | StochRSI lookback period. | Positive integer. | Change only when tuning oscillator sensitivity. |
| `MACD_PARAMS` | MACD configuration `[fast, slow, signal]`. | List of three positive integers, usually `[12, 26, 9]`. | Change when experimenting with alternative momentum settings. |

### `data.fibonacci`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `FIBONACCI_WINDOW` | Lookback window used to find swing high/low for Fibonacci features. | Positive integer. | Increase for broader structure; decrease for more local retracement behavior. |

### `data.geometric`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `CHANNEL_WINDOW` | Window for rolling channel min/max and slope features. | Integer `>= 2`. | Change when tuning short- vs medium-term structure. |
| `SWING_WINDOW` | Lookback used for swing high/low distance features. | Integer `>= 2`. | Change when tuning structural swing sensitivity. |
| `TRENDLINE_WINDOW` | Window used by optimized trendline solver. | Integer `>= 2`. | Increase for smoother structural lines. |
| `TRENDLINE_TOLERANCE` | Numeric tolerance for optimized trendline fitting. | Positive float. | Change only if trendline solver is unstable or too strict. |
| `TRENDLINE_MAX_ITERATIONS` | Maximum iterations for optimized trendline search. | Integer `>= 1`. | Increase if solver needs more convergence room. |
| `ENABLE_ATR_FEATURE` | Generates normalized ATR feature. | `true` or `false`. | Disable if you want a smaller feature set. |
| `ENABLE_ROC_FEATURE` | Generates rate-of-change feature. | `true` or `false`. | Disable if ROC is noisy in your experiments. |
| `ENABLE_BB_WIDTH_FEATURE` | Generates Bollinger Band width feature. | `true` or `false`. | Disable if volatility-width features do not help. |
| `ENABLE_SLOPE_FEATURES` | Generates support/resistance slope features. | `true` or `false`. | Disable for simpler geometric feature sets. |
| `ENABLE_CHANNEL_COMPRESSION` | Generates normalized channel-width feature. | `true` or `false`. | Enable when you want squeeze/compression signals. |
| `ENABLE_CHANNEL_POSITION` | Generates relative close position inside channel. | `true` or `false`. | Enable when you want mean-reversion or breakout context. |
| `ENABLE_SWING_DISTANCE` | Generates distance-to-swing-high/low features. | `true` or `false`. | Enable when structural proximity matters. |
| `ENABLE_SWING_TIME_DISTANCE` | Generates bars-since-swing-high/low features. | `true` or `false`. | Enable when recency of structure matters. |
| `ENABLE_OPTIMIZED_TRENDLINES` | Generates optimized support/resistance slopes. | `true` or `false`. | Enable only if you want more advanced structural features. |
| `ENABLE_OPTIMIZED_CHANNEL_WIDTH` | Generates width between optimized trendlines. | `true` or `false`. | Enable with optimized trendlines to capture structural spread. |

### `data.candlestick`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `USE_CANDLESTICK_PATTERNS` | Master switch for TA-Lib candlestick features. | `true` or `false`. | Disable to reduce feature sparsity and noise. |
| `EXCLUDE_PATTERNS` | Pattern names to skip even when candlesticks are enabled. | List of `CDL*` pattern names. | Add sparse or unhelpful patterns here instead of disabling all candlestick features. |

### `data.sector`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `SECTOR_MAPPING_SOURCE` | Where sector/group mapping comes from. | String; current value `default`. | Change only if you add another grouping source. |
| `USE_SECTOR_EMBEDDING` | Whether sector/group IDs are used as categorical inputs. | `true` or `false`. | Disable to test models without sector context. |
| `DEFAULT_SECTOR_MAPPING` | Fallback explicit mapping dictionary. | JSON object. | Fill this if automatic grouping is incomplete or wrong. |

### `data.features.FEATURE_FLAGS`

These are top-level feature switches. They decide whether feature families are
included at all.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `price_features` | Include raw OHLCV-style price features. | `true` or `false`. | Keep `true` for almost all runs. |
| `ema_features` | Include EMA-derived features. | `true` or `false`. | Disable only for ablation tests. |
| `rsi_features` | Include RSI feature(s). | `true` or `false`. | Disable for indicator ablation. |
| `stochrsi_features` | Include StochRSI feature(s). | `true` or `false`. | Disable if oscillators add noise. |
| `macd_features` | Include MACD-derived features. | `true` or `false`. | Disable for simpler momentum experiments. |
| `geometric_features` | Include structural/geometric features. | `true` or `false`. | Disable to compare against pure technical baselines. |
| `fibonacci_features` | Include Fibonacci retracement features. | `true` or `false`. | Enable only when explicitly testing Fibonacci structure. |
| `candlestick_patterns` | Include candlestick pattern features. | `true` or `false`. | Disable if sparse discrete patterns hurt generalization. |
| `vix` | Include volatility index feature. | `true` or `false`. | Disable for price-only experiments. |
| `commodities` | Include commodity external features. | `true` or `false`. | Disable when external data quality is poor. |
| `treasury_yields` | Include treasury yield features. | `true` or `false`. | Disable for technical-only experiments. |
| `time_features` | Include day/month/day-of-week features. | `true` or `false`. | Disable if calendar seasonality is not desired. |
| `financial_metrics` | Include company fundamental metrics. | `true` or `false`. | Disable when raw fundamentals are unavailable or unreliable. |
| `market_regime` | Include `regime_id` feature. | `true` or `false`. | Enable together with `data.regime.ENABLED`. |
| `cointegration_features` | Include rolling pair-spread, sector-relative, and Johansen equilibrium features. | `true` or `false`. | Enable for richer continuous state features for attention models and attribution analysis. |
| `polars_fibonacci_features` | Use Polars path for Fibonacci feature engineering where supported. | `true` or `false`. | Enable only when validating Polars migration/performance. |
| `polars_time_features` | Use Polars path for time feature generation where supported. | `true` or `false`. | Enable when profiling time-feature generation. |
| `polars_external_merges` | Use Polars path for external-data merges where supported. | `true` or `false`. | Enable only if parity has been validated. |

### `data.cointegration`

These fields control the rolling past-only cointegration feature family. The
pipeline uses these settings only when
`data.features.FEATURE_FLAGS.cointegration_features = true`.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `ROLLING_WINDOW` | Historical lookback used for rolling pair OLS spread features and rolling Johansen sector equilibrium features. | Integer such as `252` or `504`. | Increase for slower, more stable equilibrium estimates; decrease for faster reaction. |
| `NORMALIZATION_WINDOW` | Historical lookback used for rolling normalization of `spread`, `equilibrium_gap`, and `relative_price_vs_sector`. | Integer such as `252` or `504`. | Keep aligned with `ROLLING_WINDOW` unless you explicitly want a different smoothing horizon. |
| `JOHANSEN_DET_ORDER` | Deterministic term setting passed to `statsmodels.tsa.vector_ar.vecm.coint_johansen`. | Integer `-1`, `0`, or `1`. | Change only if you intentionally want a different deterministic assumption in the sector equilibrium model. |
| `JOHANSEN_K_AR_DIFF` | Lag-difference order passed to `coint_johansen`. | Positive integer. | Increase only when you intentionally want a slower VECM lag structure. |

### `data.regime`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `ENABLED` | Whether regime detection logic runs. | `true` or `false`. | Enable only if you want `regime_id` as a feature. |
| `METHOD` | Regime detection method. | Currently only `quantile` is supported by schema. | Change only if new methods are implemented. |
| `PROXY_COLUMN` | Column used to derive regime thresholds. | Existing feature column, typically `vix`. | Change to another market stress proxy if justified. |
| `N_REGIMES` | Number of regimes. | Integer `2` or `3`; current schema allows up to `3`. | Usually keep `3` for low/mid/high regimes. |
| `LOW_QUANTILE` | Lower threshold quantile for 3-regime mode. | Float between `0` and `1`. | Change if you want more or less aggressive low-vol regime bounds. |
| `HIGH_QUANTILE` | Upper threshold quantile for 3-regime mode. | Float between `0` and `1`; must be greater than `LOW_QUANTILE`. | Change when rebalancing mid/high regime boundaries. |
| `DEFAULT_REGIME` | Fallback regime ID if data is missing or unavailable. | Integer less than `N_REGIMES`. | Change if your neutral regime should not be the middle bucket. |

### `data.normalization`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `NORMALIZATION_METHOD` | Feature normalization strategy. | One of `log_transform`, `standard`, `minmax`, `robust`. | Use `robust` for outlier-heavy financial data; use `standard` for conventional scaling tests. |
| `LOG_TRANSFORM_OFFSET` | Offset used by log-based normalization to avoid invalid values. | Positive float. | Increase only if log-transform input can approach invalid ranges. |

### `data.filtering`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `MIN_TRADING_DAYS` | Minimum history required for a stock to be kept. | Positive integer. | Increase for stricter data quality; decrease to keep more symbols. |
| `MAX_MISSING_RATIO` | Maximum tolerated missing-data fraction before filtering. | Float between `0` and `1`. | Lower for stricter data cleanliness. |

### `data.sampling`

These fields control balanced stock selection when preprocessing runs with
`--stocks N`.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `STOCK_SELECTION_MODE` | How stocks are chosen inside each group. | `random` or `sorted`. | Use `random` to keep current behavior; use `sorted` to select the largest-market-cap stocks first inside each group. |
| `MARKET_CAP_METADATA_DIR` | Directory of per-ticker JSON metadata files used to read market cap in `sorted` mode. | Relative path string like `raw_data/ticket_data/us`. | Change if ticker metadata is stored elsewhere. |

### `data.external_data`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `EXTERNAL_DATA_FILL_METHOD` | Fill strategy for merged external series. | Common values: `ffill`, `bfill`, or custom supported logic. | Use `ffill` for slowly changing macro data. |
| `MAX_EXTERNAL_FILL_DAYS` | Maximum forward-fill gap accepted for external features. | Non-negative integer. | Lower to avoid stale macro data leakage. |

### `data.download`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `DOWNLOAD_RETRY_ATTEMPTS` | Number of retry attempts for downloads. | Integer `>= 1`. | Increase for flaky data sources. |
| `DOWNLOAD_RETRY_DELAY` | Delay between retries in seconds. | Integer `>= 0`. | Increase to be gentler on remote APIs. |

### `data.financial_metrics`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `USE_FINANCIAL_METRICS` | Whether to merge fundamental/company metrics. | `true` or `false`. | Disable if your fundamental dataset is incomplete. |
| `FINANCIAL_METRICS_SOURCE` | Path to stored company metrics. | Relative path string. | Change when data storage layout changes. |
| `FINANCIAL_METRICS_FILL_METHOD` | Fill method for fundamental metrics. | Commonly `ffill`. | Change if you need stricter handling of stale fundamentals. |

### `data.validation`

These fields describe expected columns and load-time validation behavior.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `REQUIRED_COLUMNS.identifiers` | Columns that must exist for identity. | List of strings. | Change only if upstream schema changes. |
| `REQUIRED_COLUMNS.price_data` | Core market columns required by the pipeline. | List of strings. | Change only if the feature pipeline changes. |
| `REQUIRED_COLUMNS.target` | Required target columns. | List of strings. | Change only if target naming changes. |
| `OPTIONAL_COLUMNS.financial_metrics` | Optional fundamental columns that may appear. | List of strings. | Extend if you add more fundamentals. |
| `OPTIONAL_COLUMNS.geometric` | Optional structural feature columns. | List of strings. | Update when geometric feature outputs change. |
| `OPTIONAL_COLUMNS.fibonacci` | Optional Fibonacci feature columns. | List of strings. | Update if Fibonacci outputs change. |
| `OPTIONAL_COLUMNS.time_features` | Optional calendar feature columns. | List of strings. | Extend if you add more calendar features. |
| `OPTIONAL_COLUMNS.external` | Optional merged macro/external columns. | List of strings. | Extend if you add more external series. |
| `OPTIONAL_COLUMNS.market_regime` | Optional regime columns. | List of strings. | Keep aligned with regime feature names. |
| `OPTIONAL_COLUMNS.cointegration` | Optional rolling spread and equilibrium columns. | List of strings. | Keep aligned with the generated cointegration feature names so validation and attribution stay correct. |
| `OPTIONAL_COLUMNS.grouping` | Optional grouping columns like sector/group. | List of strings. | Extend if grouping schema changes. |
| `OPTIONAL_COLUMNS.categorical_encoded` | Optional encoded categorical ID columns. | List of strings. | Keep aligned with encoder outputs. |
| `VALIDATE_ON_LOAD` | Whether validation should run when loading prepared data. | `true` or `false`. | Disable only for debugging broken intermediate data. |
| `WARN_ON_MISSING_OPTIONAL` | Whether to warn when optional columns are absent. | `true` or `false`. | Keep `true` unless warnings are too noisy. |

### `data.paths`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `RAW_DATA_PATH` | Base directory for raw downloaded market data. | Relative path string. | Change if data storage moves. |
| `PROCESSED_DATA_PATH` | Base directory for processed sequence datasets. | Relative path string. | Change if you version processed outputs separately. |
| `SPLITS_PATH` | Directory for saved split metadata or artifacts. | Relative path string. | Change if split artifacts move. |
| `EXTERNAL_DATA_PATH` | Base directory for external series data. | Relative path string. | Change if external data storage moves. |

### `data.chronos2_preparation`

These fields configure the separate Chronos2-style data preparation script that
builds future target paths for quantile training experiments without replacing
the repo's normal scalar-target dataset.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `OUTPUT_DIR` | Output directory for Chronos2-prepared sequence arrays and metadata. | Relative path string like `data/processed_chronos2`. | Change when you want Chronos2 data stored separately from the standard processed dataset. |
| `TARGET_COLUMN` | Column used to build the future target path for each sample. | Existing normalized split column name, typically `close`. | Change only if you intentionally want Chronos2 quantiles over another series. |
| `INCLUDE_SCALAR_TARGET` | Whether to also save the repo's scalar horizon return target alongside the future path arrays. | `true` or `false`. | Keep `true` if you want backward-compatible diagnostics or hybrid training experiments. |
| `TARGET_MODE` | How `future_target` is generated. Current supported mode is `trend_extension`. | String. | Keep `trend_extension` when you want synthetic future paths continued from recent trend instead of real observed future values. |
| `TREND_LOOKBACK` | Number of recent values from `TARGET_COLUMN` used to estimate the continuation trend. | Integer `>= 2`; common choice `7`. | Increase for smoother trend estimates; decrease for more reactive continuation. |
| `TREND_METHOD` | Method used to turn recent values into one continuation gap. Current supported method is `mean_gap`. | String. | Keep `mean_gap` unless another continuation rule is implemented. |

### `data.chronos_rich_preparation`

These fields configure the separate Chronos-rich data preparation script that
builds the same richer future targets used by `kronos_rich`, but writes them to
a Chronos-specific processed dataset for the `chronos_rich` model family.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `OUTPUT_DIR` | Output directory for Chronos-rich prepared arrays and metadata. | Relative path string like `data/processed_chronos_rich`. | Change when you want Chronos-rich data stored separately from other prepared datasets. |
| `OHLCV_COLUMNS` | Future market columns saved into `future_ohlcv`. | List like `["open", "high", "low", "close", "volume"]`. | Keep aligned with columns present in the normalized split cache. |
| `INCLUDE_SCALAR_TARGET` | Whether to also save the repo's scalar horizon return target. | `true` or `false`. | Keep `true` for compatibility with existing diagnostics or hybrid experiments. |
| `INCLUDE_RETURN_PATH` | Whether to save `future_return_path`. | `true` or `false`. | Disable if you only want OHLCV rows and regime labels. |
| `INCLUDE_REGIME_LABEL` | Whether to save `future_regime`. | `true` or `false`. | Disable for pure sequence-only experiments. |
| `REGIME_SOURCE` | How future regime labels are chosen. Current supported value is `column_or_realized_volatility`. | String. | Keep default unless another regime labeling mode is implemented. |
| `VOLATILITY_LOW_QUANTILE` | Low quantile cut for fallback realized-volatility regime labels. | Float in `(0, 1)`; common value `0.33`. | Change when you want a different low-volatility bucket boundary. |
| `VOLATILITY_HIGH_QUANTILE` | High quantile cut for fallback realized-volatility regime labels. | Float in `(0, 1)`; common value `0.66`. | Change when you want a different high-volatility bucket boundary. |

### `data.kronos_rich_preparation`

These fields configure the separate Kronos-rich data preparation script that
builds future OHLCV rows, future paths, and future regime/volatility labels
without replacing the repo's normal scalar-target dataset.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `OUTPUT_DIR` | Output directory for Kronos-rich prepared arrays and metadata. | Relative path string like `data/processed_kronos_rich`. | Change when you want Kronos-rich data stored separately from other prepared datasets. |
| `OHLCV_COLUMNS` | Future market columns saved into `future_ohlcv`. | List like `["open", "high", "low", "close", "volume"]`. | Keep aligned with columns present in the normalized split cache. |
| `INCLUDE_SCALAR_TARGET` | Whether to also save the repo's scalar horizon return target. | `true` or `false`. | Keep `true` for compatibility with existing diagnostics or hybrid experiments. |
| `INCLUDE_RETURN_PATH` | Whether to save `future_return_path`. | `true` or `false`. | Disable if you only want OHLCV rows and regime labels. |
| `INCLUDE_REGIME_LABEL` | Whether to save `future_regime`. | `true` or `false`. | Disable for pure sequence-only experiments. |
| `REGIME_SOURCE` | How future regime labels are chosen. Current supported value is `column_or_realized_volatility`. | String. | Keep default unless another regime labeling mode is implemented. |
| `VOLATILITY_LOW_QUANTILE` | Low quantile cut for fallback realized-volatility regime labels. | Float in `(0, 1)`; common value `0.33`. | Change when you want a different low-volatility bucket boundary. |
| `VOLATILITY_HIGH_QUANTILE` | High quantile cut for fallback realized-volatility regime labels. | Float in `(0, 1)`; common value `0.66`. | Change when you want a different high-volatility bucket boundary. |

## `config/model.json`

Controls model architecture, training behavior, device settings, and
experiment tracking.

### `model.embeddings`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `EMBEDDING_DIM_STOCK` | Embedding size for stock ID categorical input. | Positive integer. | Increase for large stock universes; decrease to reduce parameters. |
| `EMBEDDING_DIM_GROUP` | Embedding size for sector/group ID input. | Positive integer. | Increase only if group structure is informative. |
| `EMBEDDING_DIM_DAY` | Embedding size for day-of-month input. | Positive integer. | Keep small; increase only if calendar effects seem useful. |
| `EMBEDDING_DIM_MONTH` | Embedding size for month input. | Positive integer. | Keep small. |
| `EMBEDDING_DIM_DIVIDEND_FLAG` | Embedding size for dividend flag categorical input. | Positive integer. | Keep small unless dividend-state information is important. |
| `DROPOUT_EMBEDDING` | Dropout applied to embedding block. | Float in `[0, 1)`. | Increase if embeddings overfit. |

### `model.training`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LEARNING_RATE` | Base optimizer learning rate. | Positive float. Common range: `1e-5` to `1e-3`. | Lower if training is unstable; raise carefully if under-training. |
| `WEIGHT_DECAY` | L2-style regularization in optimizer. | Non-negative float. | Increase to reduce overfitting. |
| `BATCH_SIZE` | Training batch size. | Positive integer. | Increase for throughput if memory allows; reduce if OOM occurs. |
| `NUM_EPOCHS` | Maximum training epochs. | Positive integer. | Increase if learning is still improving. |
| `EARLY_STOPPING_PATIENCE` | Epochs without improvement allowed before stopping. | Integer `>= 0`. | Increase for noisy validation curves. |
| `GRADIENT_CLIP_VALUE` | Global gradient clip value. | Non-negative float. | Lower if exploding gradients appear. |
| `ACCUMULATION_STEPS` | Gradient accumulation steps before optimizer update. | Positive integer. | Increase to simulate larger effective batch size. |
| `OPTIMIZER` | Optimizer choice. | One of `adam`, `adamw`, `sgd`, `rmsprop`. | `adam`/`adamw` are typical defaults. |
| `SCHEDULER` | LR scheduler type. | `reduce_on_plateau`, `cosine`, `step`, or `null`. | Use `reduce_on_plateau` for validation-driven decay. |
| `SCHEDULER_PARAMS.reduce_on_plateau.mode` | Target direction for plateau detection. | Usually `min`. | Change only if monitoring a metric where higher is better. |
| `SCHEDULER_PARAMS.reduce_on_plateau.factor` | LR decay factor on plateau. | Float in `(0, 1)`. | Lower for more aggressive LR reduction. |
| `SCHEDULER_PARAMS.reduce_on_plateau.patience` | Plateau epochs before LR reduction. | Positive integer. | Increase if validation is noisy. |
| `SCHEDULER_PARAMS.cosine.T_max` | Cosine schedule cycle length. | Positive integer. | Align with expected epoch count. |
| `SCHEDULER_PARAMS.cosine.eta_min` | Minimum LR under cosine annealing. | Non-negative float. | Raise if LR decays too low. |
| `SCHEDULER_PARAMS.step.step_size` | Epoch interval between step decays. | Positive integer. | Change for manual LR schedule experiments. |
| `SCHEDULER_PARAMS.step.gamma` | Multiplicative LR decay factor for step schedule. | Float in `(0, 1)` typically. | Lower for stronger LR drops. |
| `USE_MIXED_PRECISION` | Whether to use mixed precision training where supported. | `true` or `false`. | Enable for GPU speed/memory gains after stability checks. |

### `model.training_backend`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `DEFAULT` | Default training backend. | `lightning` or `custom`; current default is `lightning`. | Change only if you want custom trainer as the default path. |
| `FALLBACK` | Backup backend when default is unavailable. | Currently fixed to `custom`. | Leave as-is unless fallback behavior is redesigned. |
| `ALLOW_CUSTOM_FALLBACK` | Whether automatic fallback to custom backend is allowed. | `true` or `false`. | Disable if you want hard failure when Lightning is unavailable. |

### `model.selection`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `DEFAULT_MODEL_TYPE` | Default model family when CLI does not specify one. This fallback is used by training, test, validation, and backtest scripts. | Must match a key under `model.models`. | Set to your most trusted baseline. In the current repo, the default is `chronos_rich`; switch it if you want another model family to drive the default runtime path. |

### `model.loss`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LOSS_TYPE` | Training loss function. | One of `mse`, `mae`, `smooth_l1`, `huber`, `directional`, `sharpe`, `directional_mse`, `directional_huber`, `quantile_loss`, `pinball_loss`, `multi_part_rich_loss`. | Use `directional_mse` or `directional_huber` when sign accuracy matters, `quantile_loss` or `pinball_loss` for asymmetric quantile regression, and `multi_part_rich_loss` for `chronos_rich`. |
| `HUBER_DELTA` | Delta threshold for Huber loss. | Positive float. | Relevant when `LOSS_TYPE = "huber"` or `LOSS_TYPE = "directional_huber"`. |
| `DIRECTIONAL_ALPHA` | Weight of wrong-direction penalty inside directional hybrid losses. `directional_mse` uses `MSE + alpha * mean(relu(-pred * sign(target)))`; `directional_huber` uses `Huber + alpha * mean(relu(-pred * sign(target)))`. | Non-negative float. `0.0` behaves like plain base regression loss; `0.1` to `0.5` is mild; `1.0` is strong. | Increase when direction matters more than exact return magnitude. |
| `SHARPE_EPSILON` | Stability epsilon in Sharpe-style loss denominator. | Positive float. | Change only if numerical stability requires it. |
| `QUANTILE` | Quantile used by `quantile_loss` and `pinball_loss`. | Float in `(0, 1)`, such as `0.1`, `0.5`, or `0.9`. | Change when you want the model to penalize under-prediction or over-prediction asymmetrically. |

Loss notes:

- `quantile_loss` and `pinball_loss` use the same formula in this repo
- `multi_part_rich_loss` is a structured rich-output loss, not a plain scalar loss
- `multi_part_rich_loss` should be used with `chronos_rich`

### `model.device`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `DEVICE` | Preferred runtime device. | `cuda` or `cpu`. | Set `cpu` for CPU-only environments. |
| `NUM_WORKERS` | DataLoader worker count. | Integer `>= 0`. | Reduce if container/host multiprocessing is unstable. |
| `PIN_MEMORY` | Whether DataLoader pins host memory for faster GPU transfer. | `true` or `false`. | Usually keep `true` for CUDA. |
| `PREFETCH_FACTOR` | Prefetch factor for worker-based DataLoader loading. | Integer `>= 1`. | Lower if memory pressure is high. |

### `model.checkpointing`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `CHECKPOINT_DIR` | Directory where model checkpoints are written. | Relative path string. | Change when organizing experiments or storage locations. |
| `SAVE_BEST_ONLY` | Whether only best-improving checkpoints are kept as stable paths. | `true` or `false`. | Set `false` if you also want periodic overwrite checkpoints. |
| `SAVE_LAST_N` | Legacy retention count from older checkpoint workflows. | Integer `>= 1`. | Mostly backward-compatibility; change only if legacy tooling relies on it. |
| `CHECKPOINT_FREQUENCY` | Periodic save interval in epochs when periodic checkpointing is active. | Integer `>= 1`. | Increase to reduce I/O. |

### `model.ensemble`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `ENABLED` | Whether ensemble prediction mode is enabled. | `true` or `false`. | Enable only when you actually serve multiple checkpoints together. |
| `CHECKPOINT_PATHS` | List of checkpoints to ensemble. | At least two paths when enabled. | Populate when building model ensembles. |
| `WEIGHTS` | Optional ensemble weights. | `null` or list matching `CHECKPOINT_PATHS`. | Set when you want weighted averaging instead of equal weighting. |
| `REQUIRE_MATCHING_FEATURES` | Guard that ensemble members use compatible feature sets. | `true` or `false`. | Keep `true` unless you explicitly handle feature mismatch. |
| `REQUIRE_MATCHING_TARGET_NORMALIZATION` | Guard that ensemble members use compatible target normalization. | `true` or `false`. | Keep `true` to avoid inconsistent prediction scales. |

### `model.logging`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LOG_FREQUENCY` | Batch/step logging frequency during training. | Positive integer. | Increase to reduce log noise. |
| `TENSORBOARD_DIR` | Output directory for TensorBoard logs. | Path string or `null`. | Change when segregating experiments. |
| `WANDB_PROJECT` | Optional Weights & Biases project name. | String or `null`. | Set only if W&B integration is intentionally used. |

### `model.experiment_tracking`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `ENABLED` | Whether local experiment tracking is active. | `true` or `false`. | Disable if you want a lighter local workflow. |
| `BACKEND` | Tracking backend. | Current supported value is `mlflow`. | Change only if new backend support is implemented. |
| `MLFLOW_TRACKING_URI` | Tracking storage URI. Schema requires a local path or `file:` URI. | Example: `file:./mlruns`. | Change to another local storage location. |
| `EXPERIMENT_NAME` | MLflow experiment name. | Non-empty string. | Change to separate projects or phases. |
| `LOG_PARAMS` | Whether to log config/hyperparameters. | `true` or `false`. | Keep `true` for reproducibility. |
| `LOG_METRICS` | Whether to log metrics. | `true` or `false`. | Keep `true` unless you want minimal logging. |
| `LOG_ARTIFACTS` | Whether to log artifacts such as outputs/checkpoints. | `true` or `false`. | Disable if disk usage is a concern. |

### `model.validation`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `VAL_FREQUENCY` | How often validation runs during training, in epochs. | Positive integer. | Increase to speed training when validation is expensive. |
| `VAL_BATCH_SIZE` | Evaluation batch size override. `null` falls back to `training.BATCH_SIZE`. | Positive integer or `null`. | Increase for faster eval if memory allows. |

### `model.nan_handling`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `CHECK_INPUTS` | Validate model inputs for NaN/Inf. | `true` or `false`. | Keep `true` when stabilizing the pipeline. |
| `SANITIZE_INPUTS` | Replace problematic numeric inputs instead of only failing. | `true` or `false`. | Disable if you prefer hard failure on bad data. |
| `CHECK_GRADIENTS` | Inspect gradients for NaN/Inf. | `true` or `false`. | Keep `true` when debugging unstable training. |
| `STOP_ON_NAN` | Abort on NaN conditions. | `true` or `false`. | Keep `true` for safety. |
| `LOG_NAN_DETAILS` | Emit detailed NaN diagnostics. | `true` or `false`. | Keep `true` unless logs are too verbose. |
| `MAX_GRAD_VALUE` | Threshold for unusually large gradients. | Positive float. | Lower if you want stricter instability detection. |
| `REPLACE_VALUE` | Numeric replacement used during sanitization. | Float. | Usually keep `0.0`; change only with a clear imputation reason. |

### `model.reproducibility`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `RANDOM_SEED` | Seed for reproducible training components. | Integer. | Change to test robustness across seeds. |
| `DETERMINISTIC` | Whether to prefer deterministic execution where supported. | `true` or `false`. | Enable for reproducibility; disable for maximum performance. |

### `model.models.crnn`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `CNN_CHANNELS` | Output channel sizes for stacked Conv1D layers. | List of positive integers. | Increase for more feature extraction capacity. |
| `CNN_KERNEL_SIZE` | Convolution kernel width. | Positive integer, typically `3`. | Increase for wider local context. |
| `CNN_POOL_SIZE` | Pooling size after CNN stack. | Positive integer. | Change only if sequence downsampling should differ. |
| `CNN_USE_BATCH_NORM` | Whether CNN layers use batch normalization. | `true` or `false`. | Enable when training deeper conv blocks. |
| `RNN_HIDDEN_SIZE` | Hidden size of recurrent block after CNN. | Positive integer. | Increase for more temporal capacity. |
| `RNN_NUM_LAYERS` | Number of recurrent layers. | Positive integer. | Increase for deeper temporal modeling. |
| `RNN_DROPOUT` | Dropout in recurrent stack. | Float in `[0, 1)`. | Increase if overfitting. |
| `USE_LAYER_NORM` | Whether to apply layer normalization around recurrent outputs. | `true` or `false`. | Enable if recurrent activations are unstable. |
| `FC_HIDDEN_SIZES` | Hidden sizes of dense prediction head. | List of positive integers. | Adjust if head is too weak or too large. |
| `FC_DROPOUT` | Dropout in dense head. | Float in `[0, 1)`. | Increase for regularization. |
| `FC_USE_BATCH_NORM` | Whether dense head uses batch normalization. | `true` or `false`. | Enable only if it empirically helps. |

### `model.models.rnn`

Same semantics as the `crnn` recurrent/head fields, but without CNN settings.

### `model.models.rnn_attention`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `RNN_HIDDEN_SIZE` | Recurrent hidden size before attention. | Positive integer. | Increase for more sequence capacity. |
| `RNN_NUM_LAYERS` | Number of recurrent layers. | Positive integer. | Increase for deeper sequence modeling. |
| `RNN_DROPOUT` | Recurrent dropout. | Float in `[0, 1)`. | Increase if overfitting. |
| `USE_LAYER_NORM` | Apply layer norm around recurrent block. | `true` or `false`. | Enable when recurrent outputs are unstable. |
| `USE_ATTENTION` | Whether attention block is active. | `true` or `false`. | Usually keep `true` for this model family. |
| `ATTENTION_HEADS` | Number of attention heads. | Positive integer compatible with attention dimension. | Increase only if model dimension supports it cleanly. |
| `ATTENTION_DROPOUT` | Dropout inside attention. | Float in `[0, 1)`. | Increase if attention overfits. |
| `FC_HIDDEN_SIZES` | Dense head sizes. | List of positive integers. | Tune head capacity. |
| `FC_DROPOUT` | Dense head dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `FC_USE_BATCH_NORM` | Batch norm in dense head. | `true` or `false`. | Enable only if empirically useful. |

### `model.models.crnn_attention`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `CNN_CHANNELS` | Conv1D channel sizes. | List of positive integers. | Increase for stronger local feature extraction. |
| `CNN_KERNEL_SIZE` | Convolution width. | Positive integer. | Change for wider/narrower local patterns. |
| `CNN_POOL_SIZE` | Pooling size after CNN. | Positive integer. | Change only if temporal downsampling should differ. |
| `CNN_USE_BATCH_NORM` | Batch norm in CNN stack. | `true` or `false`. | Enable for deeper conv stabilization. |
| `LSTM4_HIDDEN_SIZES` | Hidden sizes across the 4-layer BiLSTM stack. | List of four positive integers. | Tune model depth/capacity. |
| `LSTM4_DROPOUT` | Dropout in BiLSTM stack. | Float in `[0, 1)`. | Increase if overfitting. |
| `LSTM4_ATTENTION_HEADS` | Multi-head attention head count after BiLSTM. | Positive integer. | Change only if dimensions remain compatible. |
| `LSTM4_ATTENTION_DROPOUT` | Attention dropout. | Float in `[0, 1)`. | Increase for stronger regularization. |

### `model.models.transformer`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LSTM4_HIDDEN_SIZES` | Hidden sizes for pre-transformer BiLSTM stack. | List of four positive integers. | Tune recurrent front-end capacity. |
| `LSTM4_DROPOUT` | Dropout in BiLSTM front-end. | Float in `[0, 1)`. | Adjust regularization. |
| `TRANSFORMER_NUM_LAYERS` | Number of transformer encoder layers. | Positive integer. | Increase for deeper attention stack. |
| `TRANSFORMER_NUM_HEADS` | Number of transformer attention heads. | Positive integer compatible with `D_MODEL`. | Tune only with dimension compatibility in mind. |
| `TRANSFORMER_D_MODEL` | Transformer model dimension. | Positive integer. | Increase for capacity; raises compute cost. |
| `TRANSFORMER_DIM_FEEDFORWARD` | Feed-forward hidden dimension inside transformer blocks. | Positive integer. | Increase for more transformer capacity. |
| `TRANSFORMER_DROPOUT` | Transformer dropout. | Float in `[0, 1)`. | Increase if transformer overfits. |

### `model.models.lstm3`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LSTM3_HIDDEN_SIZE` | Hidden size of the 3-layer LSTM. | Positive integer. | Increase for more temporal capacity. |
| `LSTM3_NUM_LAYERS` | Number of LSTM layers. | Positive integer; this family is intended for `3`. | Change only if the implementation supports it cleanly. |
| `LSTM3_DROPOUT` | LSTM dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `LSTM3_USE_LAYER_NORM` | Apply layer norm around recurrent outputs. | `true` or `false`. | Enable for training stability. |
| `FC_HIDDEN_SIZES` | Dense head sizes. | List of positive integers. | Tune head complexity. |
| `FC_DROPOUT` | Dense head dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `FC_USE_BATCH_NORM` | Batch norm in dense head. | `true` or `false`. | Enable only if it helps. |

### `model.models.lstm3_attention`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LSTM3_HIDDEN_SIZE` | Hidden size of recurrent stack. | Positive integer. | Increase for more sequence capacity. |
| `LSTM3_NUM_LAYERS` | Number of LSTM layers. | Positive integer, typically `3`. | Change only if implementation assumptions allow it. |
| `LSTM3_DROPOUT` | Recurrent dropout. | Float in `[0, 1)`. | Increase if overfitting. |
| `LSTM3_USE_LAYER_NORM` | Layer normalization toggle. | `true` or `false`. | Enable for stability. |
| `LSTM3_ATTENTION_HIDDEN_SIZE` | Optional explicit attention hidden size. `null` means infer/default behavior. | Positive integer or `null`. | Set only when custom attention dimension is needed. |
| `LSTM3_ATTENTION_HEADS` | Attention head count. | Positive integer. | Change only with dimension compatibility in mind. |
| `LSTM3_ATTENTION_DROPOUT` | Attention dropout. | Float in `[0, 1)`. | Increase if attention overfits. |
| `FC_HIDDEN_SIZES` | Dense head sizes. | List of positive integers. | Tune head capacity. |
| `FC_DROPOUT` | Dense head dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `FC_USE_BATCH_NORM` | Batch norm in dense head. | `true` or `false`. | Enable only if beneficial. |

### `model.models.bilstm4_attention`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LSTM4_HIDDEN_SIZES` | Hidden sizes of 4-layer BiLSTM stack. | List of four positive integers. | Tune depth/capacity. |
| `LSTM4_DROPOUT` | BiLSTM dropout. | Float in `[0, 1)`. | Increase if overfitting. |
| `LSTM4_ATTENTION_HEADS` | Attention head count. | Positive integer. | Tune carefully with dimension compatibility. |
| `LSTM4_ATTENTION_DROPOUT` | Attention dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `FC_HIDDEN_SIZES` | Hidden sizes of the timestep-wise MLP applied after attention and before pooling. | List of positive integers. | Tune post-attention feature capacity. |
| `FC_DROPOUT` | Dropout used inside the timestep-wise MLP head. | Float in `[0, 1)`. | Increase for regularization. |
| `FC_USE_BATCH_NORM` | Whether the timestep-wise MLP uses batch norm on flattened `(batch * seq)` activations. | `true` or `false`. | Enable only if it helps. |

### `model.models.multi_branch_bilstm`

This experimental model separates features into technical, geometric, and
macro/financial branches before fusion.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `USE_EMBEDDINGS` | Whether categorical embeddings are included in the model. | `true` or `false`. | Disable only for ablation. |
| `INCLUDE_UNASSIGNED_IN_TECHNICAL` | Whether features not matched to a branch are placed in the technical branch. | `true` or `false`. | Keep `true` unless strict branch assignment is required. |
| `BRANCH_POOLING` | Pooling method used per branch. | Current value `mean`; keep to implemented options only. | Change only if another pooling mode exists in code. |
| `TECHNICAL_HIDDEN_SIZE` | Hidden size of technical branch BiLSTM. | Positive integer. | Increase if technical branch is underpowered. |
| `TECHNICAL_NUM_LAYERS` | Number of layers in technical branch. | Positive integer. | Increase for deeper technical modeling. |
| `TECHNICAL_DROPOUT` | Technical branch dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `TECHNICAL_USE_LAYER_NORM` | Layer norm in technical branch. | `true` or `false`. | Enable for branch stability. |
| `GEOMETRIC_HIDDEN_SIZE` | Hidden size of geometric branch. | Positive integer. | Increase if structural features are important. |
| `GEOMETRIC_NUM_LAYERS` | Layer count of geometric branch. | Positive integer. | Increase for deeper geometric modeling. |
| `GEOMETRIC_DROPOUT` | Geometric branch dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `GEOMETRIC_USE_LAYER_NORM` | Layer norm in geometric branch. | `true` or `false`. | Enable for stability. |
| `MACRO_HIDDEN_SIZE` | Hidden size of macro/fundamental branch. | Positive integer. | Increase if macro/fundamental information is central. |
| `MACRO_NUM_LAYERS` | Layer count of macro/fundamental branch. | Positive integer. | Increase for deeper macro modeling. |
| `MACRO_DROPOUT` | Macro branch dropout. | Float in `[0, 1)`. | Adjust regularization. |
| `MACRO_USE_LAYER_NORM` | Layer norm in macro branch. | `true` or `false`. | Enable for stability. |
| `FUSION_HIDDEN_SIZES` | Dense layer sizes after branch fusion. | List of positive integers. | Tune final predictor capacity. |
| `FUSION_DROPOUT` | Dropout after branch fusion. | Float in `[0, 1)`. | Increase if fusion head overfits. |
| `FUSION_USE_BATCH_NORM` | Batch norm in fusion head. | `true` or `false`. | Enable only if it improves results. |
| `TECHNICAL_EXACT_FEATURES` | Exact feature names assigned to the technical branch. | List of strings. | Update when feature naming changes. |
| `TECHNICAL_PREFIX_FEATURES` | Prefix rules assigning features to the technical branch. | List of string prefixes. | Extend when new technical indicators are added. |
| `GEOMETRIC_EXACT_FEATURES` | Exact feature names assigned to geometric branch. | List of strings. | Update if branch taxonomy changes. |
| `GEOMETRIC_PREFIX_FEATURES` | Prefix rules assigning geometric/structural features. | List of prefixes. | Keep aligned with generated geometric feature names. |
| `MACRO_FINANCIAL_EXACT_FEATURES` | Exact macro/fundamental feature names assigned to macro branch. | List of strings. | Update when external/fundamental columns change. |
| `MACRO_FINANCIAL_PREFIX_FEATURES` | Prefix rules for macro/fundamental branch. | List of prefixes. | Use if new macro feature families share naming prefixes. |

### `model.models.chronos2`

This block configures the repo-integrated Chronos2-style forecasting adapter.
It uses patchified `close` history, a transformer encoder, a quantile forecast
head, and a final scalar return head.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `D_MODEL` | Transformer hidden size for patch tokens. | Positive integer. | Increase for more model capacity. |
| `D_KV` | Reserved compatibility field from the Chronos2-style design. | Positive integer. | Keep aligned with your experiment setup; current adapter does not use it directly. |
| `D_FF` | Feed-forward width in the transformer and forecast head. | Positive integer. | Increase if the encoder is underpowered. |
| `NUM_LAYERS` | Number of transformer encoder layers. | Positive integer. | Increase for deeper sequence modeling. |
| `NUM_HEADS` | Number of attention heads. | Positive integer compatible with `D_MODEL`. | Tune attention capacity carefully. |
| `DROPOUT_RATE` | Dropout used in the patch encoder and forecast head. | Float in `[0, 1)`. | Increase if Chronos2 overfits. |
| `ATTN_IMPLEMENTATION` | Reserved compatibility field for attention backend selection. | String. | Keep for future adapter changes; current adapter uses PyTorch transformer layers. |
| `INPUT_PATCH_SIZE` | Number of timesteps per patch. | Positive integer. | Increase to give each token more local context. |
| `INPUT_PATCH_STRIDE` | Step between consecutive patches. | Positive integer. | Reduce for more overlap; increase for fewer tokens. |
| `USE_REG_TOKEN` | Reserved compatibility flag for a regression token. | `true` or `false`. | Kept for config compatibility; current adapter does not use it directly. |
| `USE_ARCSINH` | Reserved compatibility flag for alternate value scaling. | `true` or `false`. | Keep `false` unless adapter logic is extended. |
| `USE_STOCK_EMBEDDING` | Whether Chronos2 includes `stock_id` embeddings. | `true` or `false`. | Disable for ablation or if stock identity hurts generalization. |
| `USE_GROUP_EMBEDDING` | Whether Chronos2 includes `group_id` embeddings. | `true` or `false`. | Disable for ablation or if sector identity is noisy. |
| `STOCK_EMB_DIM` | Chronos2-specific stock embedding size. | Positive integer. | Increase for large ticker universes; decrease to save parameters. |
| `GROUP_EMB_DIM` | Chronos2-specific group embedding size. | Positive integer. | Increase only if sector information is important. |
| `DAY_EMB_DIM` | Chronos2-specific day-of-month embedding size. | Positive integer. | Keep small; increase only if calendar effects matter. |
| `MONTH_EMB_DIM` | Chronos2-specific month embedding size. | Positive integer. | Keep small. |
| `DIVIDEND_FLAG_EMB_DIM` | Chronos2-specific dividend flag embedding size. | Positive integer. | Keep small unless dividend state matters strongly. |
| `DROPOUT_EMBEDDING` | Dropout applied to the Chronos2 categorical embedding block. | Float in `[0, 1)`. | Increase if Chronos2 embeddings overfit. |
| `QUANTILES` | Quantiles predicted across the horizon before scalar summarization. | List of floats in `(0, 1)`. | Add/remove quantiles for richer or cheaper forecast summaries. |
| `HEAD_HIDDEN_SIZES` | Hidden sizes of the final scalar prediction head. | List of positive integers. | Tune final prediction capacity. |
| `HEAD_DROPOUT` | Dropout used in the final scalar prediction head. | Float in `[0, 1)`. | Increase for stronger regularization. |

### `model.models.chronos_rich`

This block configures the repo-integrated Chronos-rich adapter. It keeps the
Chronos2-style patch pipeline, adds explicit time attention plus group
attention across same-`group_id` series in the batch, and predicts richer
future outputs:

- `future_ohlcv`
- `future_return_path`
- `future_regime`

It also exposes the final return-path value as the scalar `prediction` so the
existing direct-model evaluation and backtest flow can still operate. The repo
embedding block (`stock_id`, `group_id`, `day`, `month`, `dividend_flag`) is
kept as additional metadata context on top of the attention backbone.

The core architecture fields have the same meaning as `model.models.chronos2`:

- `D_MODEL`
- `D_KV`
- `D_FF`
- `NUM_LAYERS`
- `NUM_HEADS`
- `DROPOUT_RATE`
- `ATTN_IMPLEMENTATION`
- `INPUT_PATCH_SIZE`
- `INPUT_PATCH_STRIDE`
- `USE_REG_TOKEN`
- `USE_ARCSINH`
- `USE_STOCK_EMBEDDING`
- `USE_GROUP_EMBEDDING`
- `STOCK_EMB_DIM`
- `GROUP_EMB_DIM`
- `DAY_EMB_DIM`
- `MONTH_EMB_DIM`
- `DIVIDEND_FLAG_EMB_DIM`
- `DROPOUT_EMBEDDING`
- `ACTIVATION`
- `NORM_TYPE`
- `USE_BIAS`
- `QUANTILES`
- `HEAD_HIDDEN_SIZES`
- `HEAD_DROPOUT`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `ACTIVATION` | Activation function used in the Chronos-rich feed-forward blocks and MLP heads. `geglu` and `swiglu` use gated feed-forward behavior instead of a plain pointwise nonlinearity. | One of `relu`, `gelu`, `silu`, `leaky_relu`, `geglu`, `swiglu`. | Change when you want a different nonlinearity for training behavior or forecast smoothness. |
| `NORM_TYPE` | Normalization layer used in the Chronos-rich attention blocks, feed-forward blocks, and final encoder norm. | One of `layernorm`, `rmsnorm`. | Change when you want to compare standard LayerNorm against RMSNorm-style scaling. |
| `USE_BIAS` | Whether Chronos-rich linear layers and attention projections use bias terms. | `true` or `false`. | Disable when you want a bias-free variant for ablations or to match another architecture more closely. |

Additional Chronos-rich loss-balance fields:

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `SCALAR_LOSS_TYPE` | Loss used for the scalar `prediction` output. | Regression loss name such as `directional_huber`, `directional_mse`, `mse`, `mae`, `huber`, `quantile_loss`, or `pinball_loss`. | Change when you want the final scalar target to optimize a different objective from other models. |
| `SCALAR_HUBER_DELTA` | Huber delta for `SCALAR_LOSS_TYPE` when it uses Huber. | Positive float. | Change only when `SCALAR_LOSS_TYPE` is `huber` or `directional_huber`. |
| `SCALAR_DIRECTIONAL_ALPHA` | Direction penalty weight for scalar directional losses. | Non-negative float. | Change when scalar sign accuracy matters more or less. |
| `SCALAR_QUANTILE` | Quantile used by scalar quantile or pinball loss. | Float in `(0, 1)`. | Change when you want asymmetric scalar regression. |
| `SCALAR_LOSS_WEIGHT` | Weight on the scalar horizon-return loss. | Non-negative float. | Increase when the final return matters more than the richer side targets. |
| `OHLCV_LOSS_TYPE` | Loss used for `future_ohlcv`. | Regression loss name such as `mse`, `mae`, `smooth_l1`, `huber`, `quantile_loss`, or `pinball_loss`. | Change when full path reconstruction should use a different regression objective. |
| `OHLCV_HUBER_DELTA` | Huber delta for `OHLCV_LOSS_TYPE` when it uses Huber. | Positive float. | Change only when `OHLCV_LOSS_TYPE = "huber"`. |
| `OHLCV_DIRECTIONAL_ALPHA` | Direction penalty weight for OHLCV directional losses if you intentionally use them. | Non-negative float. | Usually leave at default unless you deliberately want a directional hybrid loss on OHLCV. |
| `OHLCV_QUANTILE` | Quantile used by OHLCV quantile or pinball loss. | Float in `(0, 1)`. | Change when you want asymmetric OHLCV regression. |
| `OHLCV_LOSS_WEIGHT` | Weight on the `future_ohlcv` regression loss. | Non-negative float. | Increase when full path reconstruction matters more. |
| `RETURN_PATH_LOSS_TYPE` | Loss used for `future_return_path`. | Regression loss name such as `mse`, `mae`, `directional_mse`, `directional_huber`, `huber`, `quantile_loss`, or `pinball_loss`. | Change when the return path should optimize a different shape or directional objective. |
| `RETURN_PATH_HUBER_DELTA` | Huber delta for `RETURN_PATH_LOSS_TYPE` when it uses Huber. | Positive float. | Change only when `RETURN_PATH_LOSS_TYPE` is `huber` or `directional_huber`. |
| `RETURN_PATH_DIRECTIONAL_ALPHA` | Direction penalty weight for directional return-path losses. | Non-negative float. | Increase when return-path sign accuracy matters more. |
| `RETURN_PATH_QUANTILE` | Quantile used by return-path quantile or pinball loss. | Float in `(0, 1)`. | Change when you want asymmetric return-path regression. |
| `RETURN_PATH_LOSS_WEIGHT` | Weight on the `future_return_path` regression loss. | Non-negative float. | Increase when the shape of the return path matters more. |
| `REGIME_LOSS_TYPE` | Loss used for `future_regime`. | Currently `cross_entropy`. | Keep as `cross_entropy` unless regime-head training is redesigned. |
| `REGIME_LABEL_SMOOTHING` | Label smoothing used by the regime classifier loss. | Float in `[0, 1)`. | Increase slightly if regime classification is overconfident. |
| `REGIME_LOSS_WEIGHT` | Weight on the `future_regime` classification loss. | Non-negative float. | Increase when regime prediction matters more. |

Chronos-rich training now builds its component losses from the `chronos_rich`
config block itself. The total loss is the weighted sum of:

- scalar target loss from `SCALAR_LOSS_TYPE`
- `future_ohlcv` loss from `OHLCV_LOSS_TYPE`
- `future_return_path` loss from `RETURN_PATH_LOSS_TYPE`
- `future_regime` loss from `REGIME_LOSS_TYPE`

### `model.models.kronos`

This block configures the Kronos tokenizer, Kronos generator network, and
Kronos predictor wrapper. In the current codebase these settings are used by:

- `src.models.create_kronos_tokenizer()`
- `src.models.create_kronos_model()`
- `src.models.create_kronos_predictor()`
- the Kronos branch inside `scripts/train.py`
- the Kronos branches inside `scripts/test.py`
- the Kronos branches inside `scripts/validate.py`
- the Kronos branches inside `scripts/backtest.py`

They are still not used by the generic `src.models.create_model()` registry.
Kronos uses dedicated runtime branches because it trains and evaluates as a
token-generation model instead of a direct scalar regressor.

Reference / credit:

- Original project: `Kronos`
- Original repository: `https://github.com/shiyu-coder/Kronos`
- Original paper: `Kronos: A Foundation Model for the Language of Financial Markets`
- Authors listed in the upstream citation:
  - Yu Shi
  - Zongliang Fu
  - Shuo Chen
  - Bohan Zhao
  - Wei Xu
  - Changshui Zhang
  - Jian Li
- Paper link: `https://arxiv.org/abs/2508.02739`
- Upstream license: `MIT`

### `model.models.kronos.tokenizer`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `D_IN` | Number of continuous input features passed into the tokenizer. The current default assumes `open, high, low, close, volume, amount`. | Positive integer. Keep aligned with the real predictor input width. | Change when Kronos should tokenize a different feature set. |
| `D_MODEL` | Internal Transformer width used inside the tokenizer. | Positive integer. | Increase for more tokenizer capacity; decrease to reduce memory and parameters. |
| `N_HEADS` | Number of attention heads in tokenizer Transformer blocks. | Positive integer that divides `D_MODEL`. | Change only with dimension compatibility in mind. |
| `FF_DIM` | Feed-forward hidden size in tokenizer Transformer blocks. | Positive integer, often `2x` to `4x` `D_MODEL`. | Increase for more nonlinear capacity. |
| `N_ENC_LAYERS` | Number of tokenizer encoder layers. | Positive integer. | Increase if the tokenizer needs a deeper encoder. |
| `N_DEC_LAYERS` | Number of tokenizer decoder layers. | Positive integer. | Increase if token-to-feature reconstruction is too weak. |
| `FFN_DROPOUT_P` | Dropout in tokenizer feed-forward blocks. | Float in `[0, 1)`. | Increase if tokenizer overfits. |
| `ATTN_DROPOUT_P` | Dropout inside tokenizer attention blocks. | Float in `[0, 1)`. | Increase for stronger regularization. |
| `RESID_DROPOUT_P` | Dropout on tokenizer residual outputs. | Float in `[0, 1)`. | Increase when deeper tokenizer stacks overfit. |
| `ACTIVATION` | Activation function used inside tokenizer feed-forward blocks. `geglu` and `swiglu` use gated feed-forward behavior. | One of `relu`, `gelu`, `silu`, `leaky_relu`, `geglu`, `swiglu`. | Change when you want a different tokenizer nonlinearity. |
| `NORM_TYPE` | Normalization layer used inside tokenizer transformer blocks. | One of `layernorm`, `rmsnorm`. | Change when you want to compare tokenizer normalization behavior. |
| `USE_BIAS` | Whether tokenizer linear layers and attention projections use bias terms. | `true` or `false`. | Disable when you want a bias-free tokenizer variant. |
| `S1_BITS` | Number of bits used for the coarse token. This sets coarse token vocabulary size to `2 ** S1_BITS`. | Positive integer. | Increase if the coarse code is too small to represent the data well. |
| `S2_BITS` | Number of bits used for the fine token. This sets fine token vocabulary size to `2 ** S2_BITS`. | Positive integer. | Increase if the fine code needs more detail. |
| `BETA` | Commitment-loss weight in the binary spherical quantizer. | Non-negative float. | Increase when token assignments drift too much from pre-quantized activations. |
| `GAMMA0` | Weight on per-sample entropy term in the quantizer. | Non-negative float. | Change only when tuning code usage behavior. |
| `GAMMA` | Weight on codebook entropy term in the quantizer. | Non-negative float. | Change when tuning how evenly the codebook is used. |
| `ZETA` | Overall weight multiplier for the entropy penalty term. | Non-negative float. | Increase if codebook regularization is too weak. |
| `GROUP_SIZE` | Sub-code group size used by the quantizer entropy approximation. It must divide `S1_BITS + S2_BITS`. | Positive integer divisor of total bits. | Change only if you understand the quantizer grouping tradeoff. |

### `model.models.kronos.network`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `S1_BITS` | Coarse token bit width expected by the Kronos generator. Must match tokenizer `S1_BITS`. | Positive integer. | Keep equal to tokenizer value unless you intentionally redesign the token interface. |
| `S2_BITS` | Fine token bit width expected by the Kronos generator. Must match tokenizer `S2_BITS`. | Positive integer. | Keep equal to tokenizer value unless you intentionally redesign the token interface. |
| `N_LAYERS` | Number of causal Transformer blocks in the Kronos generator. | Positive integer. | Increase for deeper sequence modeling; decrease to reduce compute. |
| `D_MODEL` | Token embedding and hidden size in the Kronos generator. | Positive integer. | Increase for model capacity; decrease for speed and memory savings. |
| `N_HEADS` | Number of attention heads in the Kronos generator. | Positive integer that divides `D_MODEL`. | Change only if hidden dimension stays compatible. |
| `FF_DIM` | Feed-forward hidden size inside each Kronos Transformer block. | Positive integer. | Increase when the generator head is underpowered. |
| `FFN_DROPOUT_P` | Feed-forward dropout in the generator blocks. | Float in `[0, 1)`. | Increase if the generator overfits. |
| `ATTN_DROPOUT_P` | Attention dropout in the generator blocks. | Float in `[0, 1)`. | Increase for stronger regularization. |
| `RESID_DROPOUT_P` | Residual dropout in the generator blocks. | Float in `[0, 1)`. | Adjust if deep-token modeling overfits. |
| `ACTIVATION` | Activation function used inside generator feed-forward blocks. `geglu` and `swiglu` use gated feed-forward behavior. | One of `relu`, `gelu`, `silu`, `leaky_relu`, `geglu`, `swiglu`. | Change when you want a different generator nonlinearity. |
| `NORM_TYPE` | Normalization layer used inside generator transformer blocks, dependency layer, and final generator norm. | One of `layernorm`, `rmsnorm`. | Change when you want to compare generator normalization behavior. |
| `USE_BIAS` | Whether generator linear layers and attention projections use bias terms. | `true` or `false`. | Disable when you want a bias-free generator variant. |
| `TOKEN_DROPOUT_P` | Dropout applied after token and time embeddings are added. | Float in `[0, 1)`. | Increase to regularize token embeddings. |
| `LEARN_TE` | Whether time embeddings are learned instead of fixed sinusoidal-style embeddings. | `true` or `false`. | Set `true` to let the model learn calendar embeddings; set `false` for fixed embeddings. |
| `USE_STOCK_EMBEDDING` | Whether Kronos adds `stock_id` embeddings from prepared data. | `true` or `false`. | Enable when stock identity is useful to the model. |
| `USE_GROUP_EMBEDDING` | Whether Kronos adds `group_id` embeddings from prepared data. | `true` or `false`. | Enable when sector/group context is useful. |
| `STOCK_EMB_DIM` | Embedding width for `stock_id`. | Positive integer. | Increase for large stock universes; decrease to save parameters. |
| `GROUP_EMB_DIM` | Embedding width for `group_id`. | Positive integer. | Keep smaller than stock embedding unless group structure is very important. |

### `model.models.kronos_rich`

This block is a separate Kronos-family config path reserved for the
`kronos_rich` branch. It stays close to upstream Kronos generator behavior
while leaving the original local `kronos` path unchanged.

In the current codebase it uses the same field structure as `model.models.kronos`:

- `tokenizer`
- `network`
- `predictor`

The meaning of each field is the same as the `kronos` tables above with one
important runtime rule:

- `kronos` keeps the repo's optional `stock_id` / `group_id` generator context embeddings
- `kronos_rich` uses the upstream-style generator path and does not apply those extra repo embeddings even if the config fields are present

Use this block when running `--model-type kronos_rich`.

Additional Kronos-rich training-loss fields:

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `RECON_LOSS_TYPE` | Loss used for tokenizer reconstruction output `z_full -> features`. | Regression loss name such as `mse`, `mae`, `smooth_l1`, `huber`, `quantile_loss`, or `pinball_loss`. | Change when full reconstruction should use a different regression objective. |
| `RECON_HUBER_DELTA` | Huber delta for `RECON_LOSS_TYPE` when it uses Huber. | Positive float. | Change only when `RECON_LOSS_TYPE = "huber"`. |
| `RECON_QUANTILE` | Quantile used by reconstruction quantile or pinball loss. | Float in `(0, 1)`. | Change when you want asymmetric reconstruction regression. |
| `RECON_LOSS_WEIGHT` | Weight on reconstruction loss. | Non-negative float. | Increase when exact reconstruction matters more. |
| `PRE_LOSS_TYPE` | Loss used for the pre-quantized reconstruction output `z_pre -> features`. | Regression loss name such as `mse`, `mae`, `smooth_l1`, `huber`, `quantile_loss`, or `pinball_loss`. | Change when the pre-quantized path needs a different objective. |
| `PRE_HUBER_DELTA` | Huber delta for `PRE_LOSS_TYPE` when it uses Huber. | Positive float. | Change only when `PRE_LOSS_TYPE = "huber"`. |
| `PRE_QUANTILE` | Quantile used by pre-reconstruction quantile or pinball loss. | Float in `(0, 1)`. | Change when you want asymmetric pre-reconstruction regression. |
| `PRE_LOSS_WEIGHT` | Weight on pre-reconstruction loss. | Non-negative float. | Increase when the coarse reconstruction path matters more. |
| `TOKEN_LOSS_TYPE` | Loss used for the token prediction head. | Currently `cross_entropy`. | Keep as `cross_entropy` unless token-head training is redesigned. |
| `TOKEN_LABEL_SMOOTHING` | Label smoothing used in token cross-entropy. | Float in `[0, 1)`. | Increase slightly if token logits are too overconfident. |
| `TOKEN_LOSS_WEIGHT` | Weight on token prediction loss. | Non-negative float. | Increase when token prediction matters more than reconstruction. |
| `BSQ_LOSS_WEIGHT` | Weight on the quantizer BSQ loss returned by the tokenizer. | Non-negative float. | Increase when codebook commitment and quantizer regularization need stronger pressure. |

Kronos-rich does not share the same outputs as `chronos_rich`. Its train loss is
the weighted sum of:

- reconstruction loss from `RECON_LOSS_TYPE`
- pre-reconstruction loss from `PRE_LOSS_TYPE`
- quantizer `bsq_loss`
- token loss from `TOKEN_LOSS_TYPE`

Kronos derives stock/group vocabulary sizes from dataset metadata at runtime.
`NUM_STOCKS` and `NUM_GROUPS` are intentionally not stored in
`config/model.json`.

### `model.models.kronos.predictor`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `MAX_CONTEXT` | Maximum token history window used during autoregressive generation and final decode. | Positive integer. | Increase for longer context if memory allows. |
| `CLIP` | Absolute clipping value used on normalized input features before tokenization and generation. | Positive float. | Lower if outliers destabilize tokenization; raise if clipping removes too much signal. |

### Runtime notes for Kronos

- Kronos can now be selected from the normal shell flows with:
  - `--model-type kronos`
  - or `model.selection.DEFAULT_MODEL_TYPE = "kronos"`
- `scripts/train.py` supports extra quick-smoke arguments for Kronos:
  - `--max-train-batches`
  - `--max-val-batches`
- `scripts/test.py`, `scripts/validate.py`, and `scripts/backtest.py` support:
  - `--max-samples`
- In evaluation and backtest, Kronos does not produce the saved scalar target directly.
  The runtime generates future rows and converts the generated future `close`
  path back into the same horizon return target used by the rest of the repo.

## `config/hyperparameter.json`

Used by `scripts/optuna_tune.py` and `src/hyperparameter/optimizer.py`.

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `N_TRIALS` | Number of Optuna trials. | Positive integer. | Increase for better search quality. |
| `TIMEOUT` | Global Optuna timeout in seconds. | Integer or `null`. | Set when you want time-bounded tuning. |
| `N_JOBS` | Number of parallel Optuna jobs. | Positive integer. | Increase only if your environment supports safe parallel runs. |
| `MODEL_TYPE` | Model family to tune. | Valid model type string. | Set to the architecture you want to optimize. |
| `HPARAM_STOCKS` | Number of stocks used in the small tuning dataset. | Positive integer. | Lower for speed; raise for more representative tuning data. |
| `HPARAM_YEARS` | Number of years of data for tuning subset. | Integer or `null`. | Set to focus tuning on a shorter period. |
| `HPARAM_ALL_YEARS` | Whether to use all years instead of a limited subset. | `true` or `false`. | Keep aligned with `HPARAM_YEARS`. |
| `HPARAM_START_DATE` | Optional fixed start date for tuning dataset creation. | ISO date string or `null`. | Use for reproducible subset generation. |
| `HPARAM_MAX_EPOCHS` | Max epochs per Optuna trial. | Positive integer. | Lower for faster coarse search. |
| `HPARAM_ES_PATIENCE` | Early stopping patience inside each trial. | Non-negative integer. | Lower to shorten weak trials. |
| `LEARNING_RATE_RANGE` | Search range for learning rate. | Two-element float list `[low, high]`. | Adjust if best values hit range edges. |
| `LSTM_HIDDEN_SIZE_RANGE` | Search range for hidden size. | Two-element integer list. | Increase for larger-capacity search. |
| `LSTM_NUM_LAYERS_RANGE` | Search range for recurrent layer count. | Two-element integer list. | Narrow if deeper models are too expensive. |
| `DROPOUT_RANGE` | Search range for dropout. | Two-element float list. | Adjust if tuned dropout clusters at boundaries. |
| `WEIGHT_DECAY_RANGE` | Search range for weight decay. | Two-element float list. | Adjust when regularization search is too narrow. |
| `SEQUENCE_LENGTH_CHOICES` | Candidate sequence lengths. | List of positive integers. | Extend if you want longer/shorter temporal context search. |
| `BATCH_SIZE_CHOICES` | Candidate batch sizes. | List of positive integers. | Remove sizes that do not fit in memory. |
| `BEST_PARAMS_PATH` | Output file path for best tuning result JSON. | Path string. | Change when organizing tuning results by experiment. |

## `config/test.json`

This file exists and is loadable through `load_config('test')`, but current
main test/evaluation scripts appear to prefer explicit CLI arguments and
`config/model.json` over these values. Treat this file as a convenience config
or legacy helper unless you verify a specific workflow depends on it.

### `test.data`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `START_DATE` | Suggested start date for test dataset generation. | ISO date string. | Change when generating a different sample test period. |
| `END_DATE` | Suggested end date for test dataset generation. | ISO date string or `null`. | Set for reproducible bounded test windows. |
| `N_STOCKS` | Suggested number of stocks for test workflows. | Positive integer. | Lower for faster smoke-style runs. |
| `N_DAYS` | Suggested number of days in synthetic or sampled test data. | Positive integer. | Lower for quick checks; raise for broader samples. |
| `SEQUENCE_LENGTH` | Suggested sequence length for test runs. | Positive integer. | Keep aligned with the model you want to test. |
| `BATCH_SIZE` | Suggested test batch size. | Positive integer. | Lower if memory is limited. |

### `test.model`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `MODEL_TYPE` | Suggested model type for test runs. | Valid model type string. | Change to test another architecture quickly. |
| `NUM_EPOCHS` | Suggested training epochs in test mode. | Positive integer. | Keep very small for smoke tests. |
| `EARLY_STOPPING_PATIENCE` | Suggested early stopping patience in test mode. | Non-negative integer. | Lower for fast failure. |
| `BATCH_SIZE` | Suggested model-side batch size in test mode. | Positive integer. | Keep aligned with data batch size if used. |

### `test.paths`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `TEST_DATA_DIR` | Suggested directory for test data artifacts. | Path string. | Change when isolating test outputs. |
| `TEST_CHECKPOINT_DIR` | Suggested directory for test checkpoints. | Path string. | Change when isolating smoke-test checkpoints. |

## `config/deploy.json`

This file is also loadable through `load_config('deploy')`, but current deploy,
test, and backtest scripts mostly accept explicit CLI arguments or reuse
`config/model.json`. Treat this as lightweight deployment metadata unless you
wire it into a production entrypoint.

### `deploy.model`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `MODEL_PATH` | Default checkpoint path intended for deployment. | Path string. | Change to the model you actually want to serve. |
| `DEVICE` | Preferred deployment device. | `cuda` or `cpu`. | Set `cpu` in CPU-only environments. |
| `BATCH_SIZE` | Inference batch size for deployment workflows. | Positive integer. | Tune for latency vs throughput. |
| `MODEL_TYPE` | Model family expected by the deployment checkpoint. | Valid model type string. | Keep aligned with the checkpoint metadata. |

### `deploy.paths`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LOG_DIR` | Intended log directory for deployment workflows. | Path string. | Change when segregating deployment logs. |
| `CHECKPOINT_DIR` | Intended directory for deployment checkpoint assets. | Path string. | Change when deployment artifact storage moves. |

## `config/validate.json`

This file defines validation-oriented preferences, but current validation logic
appears to compute metrics directly and use `config/model.json` for most loader
and checkpoint settings. Treat it as workflow policy/config rather than a fully
enforced schema.

### `validate.data`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `VAL_BATCH_SIZE` | Intended validation batch size. | Positive integer. | Increase for faster evaluation if memory allows. |
| `STRIDE` | Intended validation sequence stride. | Positive integer. | Increase to thin validation samples. |

### `validate.metrics`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `metrics` | List of metrics you intend to monitor/report. | List of strings such as `loss`, `rmse`, `directional_accuracy`. | Extend when your reporting standard changes. |

### `validate.thresholds`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `MIN_DIRECTIONAL_ACCURACY` | Intended minimum acceptable directional hit rate. | Float between `0` and `1`. | Raise for stricter directional acceptance criteria. |
| `MAX_RMSE` | Intended maximum acceptable RMSE. | Positive float. | Lower for stricter regression quality gates. |

## Practical Recommendations

### If you only change a few things for a normal training run

Focus on:

- `data.sources.START_DATE`
- `data.sequences.SEQUENCE_LENGTH`
- `data.sequences.PREDICTION_HORIZON`
- `data.features.FEATURE_FLAGS.*`
- `model.selection.DEFAULT_MODEL_TYPE`
- `model.training.LEARNING_RATE`
- `model.training.BATCH_SIZE`
- `model.training.NUM_EPOCHS`
- `model.loss.LOSS_TYPE`
- `model.loss.DIRECTIONAL_ALPHA`

### If you want a safer baseline

- Keep `LOSS_TYPE = "directional_huber"` for the current default baseline
- Start `DIRECTIONAL_ALPHA` around `0.1` to `0.25`
- Keep `NORMALIZE_TARGET = true`
- Keep `DEFAULT = "lightning"`
- Keep `ALLOW_CUSTOM_FALLBACK = true`
- Keep `CHECK_INPUTS`, `CHECK_GRADIENTS`, and `STOP_ON_NAN` enabled

### If you want a smaller, faster experiment

- Reduce `START_DATE` history span
- Disable some external and structural features
- Lower `SEQUENCE_LENGTH`
- Lower `BATCH_SIZE` only if memory requires it; otherwise keep it moderate
- Use `rnn` or `lstm3` instead of the heavier models

## Known Mismatches And Cautions

- Existing older docs mention `DIRECTIONAL_ALPHA = 0.1`, but live
  `config/model.json` currently uses `1.0`.
- `config/test.json`, `config/deploy.json`, and `config/validate.json` are
  present, but current code paths appear to rely more heavily on CLI arguments
  and `main.json` / `model.json`. Verify workflow usage before assuming these
  files are authoritative.
- `model.experiment_tracking.ENABLED = true` and `LOG_ARTIFACTS = true` can
  create substantial local artifact storage.
