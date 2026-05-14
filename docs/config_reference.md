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
| `polars_fibonacci_features` | Use Polars path for Fibonacci feature engineering where supported. | `true` or `false`. | Enable only when validating Polars migration/performance. |
| `polars_time_features` | Use Polars path for time feature generation where supported. | `true` or `false`. | Enable when profiling time-feature generation. |
| `polars_external_merges` | Use Polars path for external-data merges where supported. | `true` or `false`. | Enable only if parity has been validated. |

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
| `DEFAULT_MODEL_TYPE` | Default model family when CLI does not specify one. | Must match a key under `model.models`. | Set to your most trusted baseline. |

### `model.loss`

| Field | Meaning | How to set | When to change |
|---|---|---|---|
| `LOSS_TYPE` | Training loss function. | One of `mse`, `mae`, `smooth_l1`, `huber`, `directional`, `sharpe`, `directional_mse`. | Use `directional_mse` when sign accuracy matters. |
| `HUBER_DELTA` | Delta threshold for Huber loss. | Positive float. | Relevant only when `LOSS_TYPE = "huber"`. |
| `DIRECTIONAL_ALPHA` | Weight of wrong-direction penalty inside `directional_mse`. Actual formula is `MSE + alpha * mean(relu(-pred * sign(target)))`. | Non-negative float. `0.0` behaves like plain MSE; `0.1` to `0.5` is mild; `1.0` is strong. | Increase when direction matters more than exact return magnitude. |
| `SHARPE_EPSILON` | Stability epsilon in Sharpe-style loss denominator. | Positive float. | Change only if numerical stability requires it. |

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
| `FC_HIDDEN_SIZES` | Dense head sizes. | List of positive integers. | Tune prediction-head capacity. |
| `FC_DROPOUT` | Dense head dropout. | Float in `[0, 1)`. | Increase for regularization. |
| `FC_USE_BATCH_NORM` | Dense head batch norm toggle. | `true` or `false`. | Enable only if it helps. |

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

- Keep `LOSS_TYPE = "directional_mse"`
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
