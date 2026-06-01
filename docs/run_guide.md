# Project Run Guide

## Overview

This guide explains how to run the project end to end:

1. Start the Docker environment
2. Configure data and model settings
3. Preprocess real market data
4. Train a model
5. Run prediction
6. Run backtesting

This guide is operational. It is meant to be used together with:

- `docs/configuration.md`
- `docs/data_pipeline.md`
- `docs/model_architecture.md`
- `docs/lightning_backend.md`

The main CLI entrypoints are:

- `scripts/preprocess_data.py`
- `scripts/prepare_chronos2_data.py`
- `scripts/prepare_kronos_rich_data.py`
- `scripts/train.py`
- `scripts/test.py`
- `scripts/validate.py`
- `scripts/predict.py`
- `scripts/backtest.py`


## Environment Setup

The project is designed to run inside Docker.

Start the container:

```bash
docker compose up -d
```

Open a shell inside the main container:

```bash
docker exec -it crnn_predictor bash
```

You can also run commands directly from the host without entering the shell:

```bash
docker exec crnn_predictor python scripts/train.py --model-type crnn_attention
```

The main runtime container name is `crnn_predictor`.


## Important Config Files

Two config files control almost everything:

### `config/main.json`

This file controls:

- data sources
- download date range
- feature flags
- train/val/test split ratios
- sequence length
- prediction horizon
- target normalization
- optional regime features

Important defaults:

- `data.sources.START_DATE = "2000-01-01"`
- `data.splits.TRAIN_RATIO = 0.7`
- `data.splits.VAL_RATIO = 0.1`
- `data.splits.TEST_RATIO = 0.2`
- `data.sequences.SEQUENCE_LENGTH = 30`
- `data.sequences.PREDICTION_HORIZON = 5`
- `data.sequences.NORMALIZE_TARGET = true`

### `config/model.json`

This file controls:

- training hyperparameters
- optimizer and scheduler
- training backend
- checkpoint behavior
- experiment tracking
- model-specific architecture parameters

Important defaults:

- `model.training.LEARNING_RATE = 0.0001`
- `model.training.BATCH_SIZE = 128`
- `model.training.NUM_EPOCHS = 30`
- `model.training.OPTIMIZER = "adam"`
- `model.training.SCHEDULER = "reduce_on_plateau"`
- `model.loss.LOSS_TYPE = "directional_mse"`
- `model.training_backend.DEFAULT = "lightning"`


## Available Model Types

The current model registry supports these model types:

- `crnn`
- `rnn`
- `rnn_attention`
- `crnn_attention`
- `transformer`
- `lstm3`
- `lstm3_attention`
- `bilstm4_attention`
- `multi_branch_bilstm`
- `kronos`
- `kronos_rich`

Practical guidance:

- `rnn`: smallest baseline
- `crnn_attention`: strong default starting point
- `lstm3_attention`: deeper recurrent model
- `bilstm4_attention`: larger recurrent-attention model
- `transformer`: heavier model, usually worth testing after a baseline
- `multi_branch_bilstm`: experimental branch-based model that separates
  technical, geometric, and macro/financial feature streams
- `kronos`: experimental tokenized generative model; slower but useful when you
  want autoregressive multi-feature forecasting
- `chronos_rich`: Chronos-family multi-target model that predicts future OHLCV,
  future return path, and future regime while still exposing a scalar return
  prediction for the normal direct-model evaluation flow. It uses a
  Chronos-2-style patch backbone with time attention and group attention across
  same-`group_id` series in the batch, while still keeping the repo's metadata
  embedding path.
- `kronos_rich`: separate Kronos-family branch reserved for richer
  market-behavior experiments while keeping the generator path closer to
  upstream Kronos than the repo-adapted `kronos` branch

Preprocessing note:

- `--model-type chronos2` uses `scripts/prepare_chronos2_data.py`
- `--model-type chronos_rich` uses `scripts/prepare_chronos_rich_data.py`
- `--model-type kronos_rich` uses `scripts/prepare_kronos_rich_data.py`
- other model types use `scripts/preprocess_data.py`
- shell wrappers resolve the default model type from `config/model.json` when
  `--model-type` is omitted
- shell wrappers auto-select matching processed directories for `chronos2`,
  `chronos_rich`, and `kronos_rich` unless you override `--data-dir`
- full-pipeline wrappers like `run_all.sh` and `run_all_in_container.sh` now
  forward the chosen model type into preprocess, train, validate, test, and
  backtest so all steps stay on the same dataset path

If you want one sensible first model, use `crnn_attention` or `lstm3_attention`.

Kronos credit:

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


## How Model Selection Works

There are two separate concepts:

1. Model family selection at runtime
2. Model architecture settings in `config/model.json`

### Runtime selection

You choose the model family with:

```bash
--model-type crnn_attention
```

If you do not pass `--model-type`, the scripts fall back to:

```json
model.selection.DEFAULT_MODEL_TYPE
```

### Architecture selection

Once the model type is chosen, the script reads that model's parameter block
from `config/model.json`.

Examples:

- `model.models.crnn_attention`
- `model.models.lstm3_attention`
- `model.models.bilstm4_attention`
- `model.models.kronos`

This means:

- `--model-type crnn_attention` activates the CRNN attention model
- its hidden sizes, dropout, and attention settings come from the matching JSON block
- `--model-type kronos` activates the Kronos tokenizer + generator branch
- `--model-type chronos_rich` activates the Chronos-family rich direct model
- it predicts `future_ohlcv`, `future_return_path`, and `future_regime`
- it also exposes the final predicted return-path value as the scalar
  backtest/test prediction
- `--model-type kronos_rich` activates the separate Kronos-rich tokenizer +
  generator branch
- unlike `kronos`, `kronos_rich` uses the upstream-style generator path without
  the repo's extra `stock_id` / `group_id` generator embeddings
- if `DEFAULT_MODEL_TYPE` is set to `kronos`, the same train, test, validate,
  and backtest scripts use Kronos without any extra wrapper


## Recommended First Configuration

For a first real-data run, keep the setup conservative.

Recommended data settings in `config/main.json`:

- keep `USE_YFINANCE_LIVE = true`
- keep external features enabled only if your downloads are stable
- keep `market_regime = false` at first
- keep Polars flags disabled at first
- keep Fibonacci disabled at first unless you are explicitly testing it

Recommended training settings in `config/model.json`:

- `NUM_EPOCHS = 30`
- `BATCH_SIZE = 64` or `128`
- `LEARNING_RATE = 0.0001`
- `LOSS_TYPE = "directional_mse"`
- keep backend as `lightning`


## Real Data Preprocessing

Preprocessing does all of the following:

1. Downloads or loads market data
2. Engineers features
3. Splits data by time
4. Fits normalization on the training split only
5. Stops after normalized splits, leaving training to decide how sequence windows are built
6. Saves processed metadata and whichever artifacts that mode requires

### Basic real-data preprocessing

```bash
docker exec crnn_predictor python scripts/preprocess_data.py --start-date 2015-01-01
```

### Use a limited number of stocks

```bash
docker exec crnn_predictor python scripts/preprocess_data.py --start-date 2015-01-01 --stock-limit 100
```

### Use balanced sampling across groups

```bash
docker exec crnn_predictor python scripts/preprocess_data.py --start-date 2015-01-01 --stocks 150
```

### Use specific tickers only

```bash
docker exec crnn_predictor python scripts/preprocess_data.py --start-date 2015-01-01 --tickers AAPL MSFT GOOGL
```

### Reuse existing downloaded data

```bash
docker exec crnn_predictor python scripts/preprocess_data.py --skip-download
```

### Skip precomputed sequence arrays

```bash
docker exec crnn_predictor python scripts/preprocess_data.py --skip-download --skip-sequences
```

This skips writing any saved sequence arrays. Training can still run because both dataset modes consume the normalized split cache, but they differ in how sequences are built at training time.

### Output generated by preprocessing

The default output directory is `data/processed`.

Important files created:

- `data/processed/info.json`
- `data/processed/feature_columns.txt`
- `data/processed/.cache/normalized_splits/train.parquet`
- `data/processed/.cache/normalized_splits/val.parquet`
- `data/processed/.cache/normalized_splits/test.parquet`
- `data/pre_normalized.parquet`
- `data/normalized_data.parquet`

Saved `.npy` sequence files are no longer required for the documented training modes. The normalized split parquet cache and metadata are the important artifacts used by training.

`info.json` is especially important because training uses it to recover:

- number of features
- feature column names
- preprocessing metadata
- optional regime metadata


## Train a Model

Once preprocessing is complete, train from the processed directory.

### Basic training

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type crnn_attention \
  --data-dir data/processed \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.0001
```

Training data mode is controlled by `config/main.json`:

- `data.dataset.MODE = "precomputed_sequences"`: preprocessing creates and saves sequence arrays, and training loads those saved arrays first
- `data.dataset.MODE = "on_the_fly_sequences"`: training loads normalized split parquet caches and streams window slices lazily during training

For `precomputed_sequences`, training still has a fallback path that rebuilds
sequences from normalized split caches if saved arrays are missing.

### Force Lightning backend explicitly

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type crnn_attention \
  --backend lightning
```

### Use the custom trainer instead

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type crnn_attention \
  --backend custom
```

Kronos uses a custom training branch inside the same `scripts/train.py` file.
If you request Lightning with Kronos, the script falls back to the custom path.

### Force CPU

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type crnn_attention \
  --force-cpu
```

### Small Kronos smoke training

These extra flags are useful when testing Kronos quickly:

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type kronos \
  --backend custom \
  --device cpu \
  --epochs 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

These flags are available in `scripts/train.py` and
`scripts/train_in_container.sh`:

- `--max-train-batches`
- `--max-val-batches`

### Resume or fine-tune

Resume from checkpoint:

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type crnn_attention \
  --resume models/checkpoints/your_checkpoint.pth
```

Fine-tune from checkpoint:

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type crnn_attention \
  --fine-tune models/checkpoints/your_checkpoint.pth
```

Fine-tune only on selected stocks:

```bash
docker exec crnn_predictor python scripts/train.py \
  --model-type crnn_attention \
  --fine-tune models/checkpoints/your_checkpoint.pth \
  --stocks AAPL MSFT \
  --freeze-embeddings
```

### Where checkpoints go

By default:

```text
models/checkpoints/
```

Lightning remains the default training backend. It writes custom-compatible
`.pth` checkpoints so prediction and backtesting still use the existing
checkpoint contract.

Checkpoint cadence is controlled by `config/model.json`:
- `SAVE_BEST_ONLY = true`: overwrite the stable best checkpoint only
- `SAVE_BEST_ONLY = false`: stable best checkpoint plus a stable periodic
  checkpoint overwritten every `N` epochs
- `CHECKPOINT_FREQUENCY = N`: periodic overwrite cadence in epochs
- `SAVE_LAST_N`: legacy retention setting for older timestamped workflows

### Training monitoring

You can monitor training with TensorBoard and optional local MLflow.

Host wrapper:

```bash
./scripts/train.sh --model-type crnn_attention --monitor
./scripts/train.sh --model-type crnn_attention --mlflow
./scripts/train.sh --model-type crnn_attention --monitor-all
```

Useful host-wrapper options:

- `--monitor`: start TensorBoard on the host
- `--mlflow`: start MLflow UI on the host
- `--monitor-all`: start both
- `--tensorboard-port N`: override the default TensorBoard port `6006`
- `--mlflow-port N`: override the default MLflow port `5000`

In-container wrapper:

```bash
./scripts/train_in_container.sh --model-type crnn_attention
```

The same wrapper also works for Kronos:

```bash
./scripts/train_in_container.sh --model-type kronos --backend custom
```

Relevant paths:

- TensorBoard logs: `logs/tensorboard`
- MLflow runs: `mlruns/`

For full local MLflow setup and UI usage, see
`docs/experiment_tracking.md`.

### What to watch during training

Standard metrics:

- `train/loss`
- `val/loss`
- `val/mse`
- `val/mae`
- `val/rmse`
- `val/directional_accuracy`

Validation prediction-health metrics are especially important for this project:

- `val/pred_positive_rate`
- `val/pred_negative_rate`
- `val/pred_std`
- `val/pred_mean`
- `val/pred_target_corr`
- `val/collapse_penalty`
- `val/is_collapsed`

These help detect the known failure mode where the model predicts positive
returns for nearly all samples. Healthy validation output should not show:

- `pred_positive_rate` near `1.0`
- `pred_negative_rate` near `1.0`
- `pred_std` near zero
- `is_collapsed = 1`

Lightning best-checkpoint selection now uses these diagnostics during
selection, not raw `val/loss` alone. See `docs/lightning_backend.md` for the
selection details.


## Prediction

Use prediction mainly for inference or model inspection.

### Show model info

```bash
docker exec crnn_predictor python scripts/predict.py info \
  --model models/checkpoints/your_checkpoint.pth
```

### Predict one row

```bash
docker exec crnn_predictor python scripts/predict.py single \
  --model models/checkpoints/your_checkpoint.pth \
  --ticker AAPL \
  --date 2024-01-15 \
  --input "open=150,high=152,low=149,close=151.5,volume=50000000"
```

### Batch prediction from file

```bash
docker exec crnn_predictor python scripts/predict.py batch \
  --model models/checkpoints/your_checkpoint.pth \
  --input data/prediction_input.csv \
  --output outputs/predictions.csv \
  --format csv
```


## Testing

Use testing to evaluate a trained checkpoint on `train`, `val`, or `test`.

### Basic test

```bash
docker exec crnn_predictor python scripts/test.py \
  --model best \
  --model-type crnn_attention \
  --data-dir data/processed \
  --split test
```

### In-container wrapper

```bash
./scripts/test_in_container.sh --model best --model-type crnn_attention
```

### Quick smoke test

```bash
./scripts/test_in_container.sh \
  --model best \
  --model-type kronos \
  --force-cpu \
  --max-samples 8
```

Useful wrapper option:

- `--max-samples N`: limit the number of evaluated samples for a quick smoke run

For direct models, test compares scalar predictions to the saved target.
For Kronos, test generates future rows, converts generated close prices into the
same horizon return target, and then reports the usual metrics.


## Validation

Validation uses the same idea as testing, but defaults to the validation split.

### Basic validation

```bash
docker exec crnn_predictor python scripts/validate.py \
  --model best \
  --model-type crnn_attention \
  --data-dir data/processed \
  --split val
```

### In-container wrapper

```bash
./scripts/validate_in_container.sh --model best --model-type crnn_attention
```

### Quick Kronos validation smoke run

```bash
./scripts/validate_in_container.sh \
  --model best \
  --model-type kronos \
  --force-cpu \
  --max-samples 8
```


## Backtesting

Backtesting in this project runs the trained model over one processed split and
simulates a simple threshold-based trading strategy.

It reports:

- total return
- final capital
- Sharpe ratio
- Sortino ratio
- max drawdown
- risk-adjusted return
- win rate
- profit factor
- average turnover
- total turnover
- transaction cost
- direction accuracy by sector

### Basic backtest on test split

```bash
docker exec crnn_predictor python scripts/backtest.py \
  --model best \
  --model-type crnn_attention \
  --data-dir data/processed \
  --split test \
  --output outputs/backtest_report.xlsx
```

### Backtest with a prediction threshold

```bash
docker exec crnn_predictor python scripts/backtest.py \
  --model best \
  --model-type crnn_attention \
  --data-dir data/processed \
  --split test \
  --threshold 0.5 \
  --initial-capital 100000 \
  --output outputs/backtest_report.xlsx
```

### Output formats

Supported output formats:

- `excel`
- `csv`
- `json`

Example:

```bash
docker exec crnn_predictor python scripts/backtest.py \
  --model best \
  --model-type crnn_attention \
  --data-dir data/processed \
  --split test \
  --output outputs/backtest_report.json \
  --output-format json
```

### In-container wrapper

```bash
./scripts/backtest_in_container.sh --model best --model-type crnn_attention
```

### Quick Kronos backtest smoke run

```bash
./scripts/backtest_in_container.sh \
  --model final \
  --model-type kronos \
  --force-cpu \
  --max-samples 8 \
  --output outputs/kronos_backtest.json \
  --output-format json
```

Useful wrapper options:

- `--max-samples N`: limit evaluated samples
- `--output-format excel|csv|json`: choose report format

For direct models, backtest uses the predicted scalar return directly.
For Kronos, backtest derives the return signal from generated future close
prices over the configured prediction horizon.


## Recommended End-to-End Workflow

If you want one clean baseline workflow, use this:

```bash
docker compose up -d
docker exec crnn_predictor python scripts/preprocess_data.py --start-date 2015-01-01 --stocks 150
docker exec crnn_predictor python scripts/train.py --model-type crnn_attention --backend lightning --epochs 30 --batch-size 128 --lr 0.0001
docker exec crnn_predictor python scripts/test.py --model best --model-type crnn_attention --data-dir data/processed --split test
docker exec crnn_predictor python scripts/validate.py --model best --model-type crnn_attention --data-dir data/processed --split val
docker exec crnn_predictor python scripts/backtest.py --model best --model-type crnn_attention --data-dir data/processed --split test --threshold 0.5 --output outputs/backtest_report.xlsx
```


## Common Problems

### No training data found

Cause:

- preprocessing was not run
- wrong `--data-dir`

Fix:

- run `scripts/preprocess_data.py` first
- confirm `data/processed/info.json` exists
- confirm `data/processed/.cache/normalized_splits/` contains split parquet files

### Permission denied in processed directories

Cause:

- previous Docker run created root-owned files

Fix:

- remove or repair ownership of the affected output directory
- or use a different output directory

### Download failure

Cause:

- live source unavailable
- network issue

Fix:

- retry preprocessing
- use `--skip-download` if cached data already exists

### Kronos evaluation is slow

Cause:

- Kronos generates future steps autoregressively
- CPU runs can be much slower than direct scalar models

Fix:

- use `--max-samples` for smoke runs
- use GPU for larger evaluation runs
- keep full-split backtests for final checks only


## Notes on Interpretation

This project predicts future percentage change over the configured prediction
horizon. The backtest uses a simplified strategy based on the sign and
magnitude of model predictions. Treat it as research evaluation, not as a
production trading system.
