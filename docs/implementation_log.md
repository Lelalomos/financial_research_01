# Implementation Log

This log records roadmap implementation work so progress is recoverable if a
session is interrupted.

## Phase 1: Foundation and Safety

Status: complete

Approved scope:
- Add typed Pydantic validation while preserving JSON config compatibility.
- Add custom financial loss functions.
- Add walk-forward and purged time-series split utilities.
- Add unit tests for each new behavior.
- Run unit tests, small data test, full-flow test, and full pytest after the phase.

### Tasks

- [x] Pydantic config validation
- [x] Custom financial losses
- [x] Walk-forward and purged split utilities
- [x] Unit tests for Phase 1
- [x] Validation test runs

### Notes

- Existing JSON files remain the source of configuration.
- New config validation should fail early for invalid types/ranges without
  breaking existing `Config` attribute access.
- New loss functions must be optional and selected through `config/model.json`.

### Implemented Files

- `src/config/schemas.py`: Pydantic schemas for `main.json` and `model.json`.
- `src/config/config_loader.py`: validates known configs during `load_config()`.
- `src/training/losses.py`: directional, directional-MSE, and Sharpe-style losses.
- `src/training/trainer.py`: supports `LOSS_TYPE` values `directional`, `directional_mse`, and `sharpe`.
- `src/data/time_series_split.py`: walk-forward and purged time-series split utilities.
- `tests/test_config_schemas.py`: config validation tests.
- `tests/test_financial_losses.py`: loss function and trainer integration tests.
- `tests/test_time_series_split.py`: walk-forward and purged split tests.

### Validation Progress

- Syntax compile: passed.
- Phase 1 tests: `12 passed`.
- Installed `pydantic>=2.0.0` in the active container for validation; dependency
  is also recorded in `requirements.txt` and `setup.py`.
- Broader unit suite: `149 passed`.
- Full small data test: `12 passed`.
- Full-flow integration: `1 passed`.
- Full pytest: `207 passed`, `23 warnings`.

## Phase 2: Evaluation and Modeling Extensions

Status: complete

Approved scope:
- Add model ensembling as an optional prediction mode selected through
  `config/model.json`.
- Preserve existing single-model prediction behavior.
- Add compatibility checks for feature columns, feature counts, and target
  normalization metadata before averaging predictions.
- Add unit tests and run validation in the Docker container.
- Add dependency-light market regime detection as an optional preprocessing
  feature selected through `config/main.json`.
- Fit regime thresholds on training dates only, then apply them to all splits.
- Keep regime features disabled by default.

### Tasks

- [x] Task 2.1: Model ensembling
- [x] Task 2.2: Market regime detection utilities
- [x] Task 2.3: Meta-labeling utilities

### Task 2.1 Notes

- Ensemble mode is configured by `model.ensemble` in `config/model.json`.
- Ensemble mode is disabled by default.
- When enabled, `scripts/predict.py` builds an ensemble predictor from config
  checkpoint paths instead of requiring a single `--model` path.
- Single-model prediction remains unchanged when ensemble mode is disabled.
- Ensemble weights are optional. If omitted, predictions are averaged equally.
- Incompatible ensemble members fail early with clear errors.

### Implemented Files

- `config/model.json`: added disabled-by-default `model.ensemble` settings.
- `src/config/schemas.py`: validates ensemble checkpoint paths, weights, and
  enabled-state requirements.
- `src/prediction/ensemble.py`: adds `EnsemblePredictor` and config-based
  factory functions.
- `src/prediction/predictor.py`: exposes checkpoint metadata needed for
  ensemble compatibility checks.
- `src/prediction/__init__.py`: exports ensemble prediction utilities.
- `scripts/predict.py`: uses ensemble config when `model.ensemble.ENABLED` is
  true.
- `tests/test_ensemble.py`: ensemble averaging and incompatibility tests.
- `tests/test_config_schemas.py`: ensemble config validation tests.

### Validation Progress

- Targeted container tests:
  `pytest tests/test_ensemble.py tests/test_config_schemas.py tests/test_prediction.py -q`
  passed with `25 passed`, `4 warnings`.
- Full-flow container test:
  `pytest tests/test_full_flow.py -q` passed with `1 passed`, `6 warnings`.
- Full container pytest:
  `pytest -q` passed with `215 passed`, `23 warnings`.

### Task 2.2 Notes

- Market regime detection is configured by `data.regime` and
  `data.features.FEATURE_FLAGS.market_regime` in `config/main.json`.
- Regime mode is disabled by default.
- The current method is quantile-based and dependency-light.
- `DataPreprocessor.preprocess_pipeline()` fits thresholds on the train split
  only, then adds `regime_id` to train, validation, and test splits.
- `regime_id` is included as a normal model feature when enabled and is not
  normalized.
- Regime parameters are included in preprocessing info so inference/checkpoints
  can reuse the same thresholds.
- Prediction preparation can apply loaded regime parameters when a checkpoint
  expects `regime_id`.

### Task 2.2 Implemented Files

- `config/main.json`: added disabled-by-default `data.regime` settings and
  `market_regime` feature flag.
- `src/config/schemas.py`: validates regime config values and quantiles.
- `src/data/regime.py`: quantile-based train-only regime detector.
- `src/data/preprocessing.py`: fits train-only regime thresholds and appends
  `regime_id` before normalization/sequence creation.
- `src/data/prediction_prep.py`: applies loaded regime parameters for inference
  when `regime_id` is part of checkpoint feature columns.
- `src/prediction/predictor.py`: loads `regime_params` into the prediction
  preparator when present.
- `src/training/trainer.py`: supports additional checkpoint metadata.
- `scripts/preprocess_data.py`: writes `regime_params` into `info.json`.
- `scripts/train.py`: carries preprocessing metadata into checkpoints.
- `tests/test_regime.py`: detector and enabled-preprocessing tests.
- `tests/test_config_schemas.py`: regime config validation tests.

### Task 2.2 Validation Progress

- Targeted container tests:
  `pytest tests/test_regime.py tests/test_config_schemas.py tests/test_data_pipeline.py -q`
  passed with `22 passed`, `4 warnings`.
- Prediction/training container tests:
  `pytest tests/test_prediction.py tests/test_training.py tests/test_training_integration.py -q`
  passed with `24 passed`, `7 warnings`.
- Final metadata-path targeted container tests:
  `pytest tests/test_prediction.py tests/test_training.py tests/test_training_integration.py tests/test_regime.py -q`
  passed with `28 passed`, `7 warnings`.
- Full-flow container test:
  `pytest tests/test_full_flow.py -q` passed with `1 passed`, `6 warnings`.
- Full container pytest:
  `pytest -q` passed with `221 passed`, `23 warnings`.

### Task 2.3 Notes

- Meta-labeling is implemented as an offline analysis utility.
- `create_meta_labels()` labels `1` when predicted direction matches realized
  target direction, else `0`.
- Zero predictions/targets use direction `0`, so zero only matches zero.
- Prediction and target dates/tickers are validated row-by-row when supplied.
- `prediction_source` records whether labels came from `out_of_sample`,
  `out_of_fold`, `walk_forward`, `purged_cv`, or `in_sample` predictions.
- In-sample predictions emit a warning and can be rejected with
  `require_out_of_sample=True`.
- Confidence filtering uses `abs(prediction)` and keeps rows above the chosen
  threshold.
- No second-stage production model is trained in this task.

### Task 2.3 Implemented Files

- `src/evaluation/meta_labeling.py`: meta-label creation, alignment checks,
  confidence filtering, and prediction-source safeguards.
- `src/evaluation/__init__.py`: exports meta-labeling utilities.
- `tests/test_meta_labeling.py`: direction labeling, alignment rejection,
  confidence filtering, in-sample warnings, and feature merge tests.
- `docs/meta_labeling.md`: usage and leakage guidance.

### Task 2.3 Validation Progress

- Targeted container tests:
  `pytest tests/test_meta_labeling.py tests/test_time_series_split.py tests/test_config_schemas.py -q`
  passed with `22 passed`, `4 warnings`.
- Full-flow container test:
  `pytest tests/test_full_flow.py -q` passed with `1 passed`, `6 warnings`.
- Full container pytest:
  `pytest -q` passed with `230 passed`, `23 warnings`.

## Phase 3: Interpretability and Feature Pruning

Status: complete

Approved scope:
- Add attention weight extraction while preserving existing `forward()`
  behavior.
- Add reporting utilities to aggregate attention by sequence position.
- Document that attention is diagnostic and not causal proof.
- Add unit tests and run validation in the Docker container.

### Tasks

- [x] Task 3.1: Attention weight extraction
- [x] Task 3.2: Captum or SHAP feature attribution
- [x] Task 3.3: Feature pruning report

### Task 3.1 Notes

- Attention models now expose `forward_with_attention()` separately from
  `forward()`.
- Existing training and prediction paths continue to call `forward()` unchanged.
- Attention weights are requested with `average_attn_weights=False`, producing
  per-head tensors.
- Interpretability utilities aggregate attention received by sequence position.
- Attention summaries are diagnostic only and should not be interpreted as
  causal explanations.

### Task 3.1 Implemented Files

- `src/models/rnn_attention.py`: added `forward_with_attention()`.
- `src/models/lstm3_attn_model.py`: added `forward_with_attention()`.
- `src/models/crnn_attention.py`: added `forward_with_attention()`.
- `src/models/bilstm4_attn_model.py`: added `forward_with_attention()`.
- `src/evaluation/interpretability.py`: attention conversion, aggregation,
  summary, and report helpers.
- `src/evaluation/__init__.py`: exports interpretability utilities.
- `tests/test_interpretability.py`: attention shape and report tests.
- `docs/interpretability.md`: usage and caution notes.

### Task 3.1 Validation Progress

- Targeted container tests:
  `pytest tests/test_interpretability.py tests/test_models.py tests/test_lstm3_models.py -q`
  passed with `73 passed`, `4 warnings`.
- Full-flow container test:
  `pytest tests/test_full_flow.py -q` passed with `1 passed`, `6 warnings`.
- Full container pytest:
  `pytest -q` passed with `240 passed`, `23 warnings`.

### Task 3.2 Notes

- Feature attribution is implemented as an optional Captum-based utility.
- Captum is imported only when attribution is requested; missing Captum raises
  `AttributionDependencyError` with install guidance.
- Supported methods are `integrated_gradients` and `feature_ablation`.
- Attribution targets the numeric `features` tensor while categorical model
  inputs are fixed from the batch.
- Reports include raw sequence-level attributions, feature-level mean absolute
  importance, and metadata for method, baseline, sample count, sequence length,
  and interpretation guidance.
- Attribution runs on sampled data by default through `max_batches` and
  `max_samples`.
- No feature removal or config mutation is performed in this task.

### Task 3.2 Implemented Files

- `src/evaluation/feature_attribution.py`: optional Captum attribution helpers,
  feature aggregation, sampled report generation, and missing dependency error.
- `src/evaluation/__init__.py`: exports feature attribution utilities.
- `tests/test_feature_attribution.py`: dependency handling, report schema,
  sorting, and sampling tests without requiring Captum.
- `docs/feature_attribution.md`: usage and interpretation guidance.

### Task 3.2 Validation Progress

- Targeted container tests:
  `pytest tests/test_feature_attribution.py tests/test_interpretability.py tests/test_models.py -q`
  passed with `67 passed`, `4 warnings`.
- Full-flow container test:
  `pytest tests/test_full_flow.py -q` passed with `1 passed`, `6 warnings`.
- Full container pytest:
  `pytest -q` passed with `246 passed`, `23 warnings`.

### Task 3.3 Notes

- Feature pruning is implemented as a report-only utility.
- `create_feature_pruning_report()` consumes a feature importance DataFrame and
  marks bottom-percentile and optional low-threshold features as pruning
  candidates.
- Reports include ranked features, exact pruning candidates, retained features,
  and a manual config patch suggestion.
- No config files, feature lists, checkpoints, or preprocessing artifacts are
  mutated by the pruning utility.
- Reports reject `test`, `holdout`, and `production` split names by default to
  reduce evaluation leakage risk.
- Test/holdout reports require `allow_test_data=True` and include warnings that
  they must not be used for model selection.
- Low attribution importance is documented as diagnostic rather than causal
  proof that a feature is useless.

### Task 3.3 Implemented Files

- `src/evaluation/feature_pruning.py`: report-only feature pruning
  recommendations, split provenance guardrails, and manual patch suggestions.
- `src/evaluation/__init__.py`: exports feature pruning utility.
- `tests/test_feature_pruning.py`: ranking, threshold selection, split leakage
  guard, schema validation, and no-mutation tests.
- `docs/feature_pruning.md`: usage, leakage controls, and interpretation
  guidance.
- `docs/feature_attribution.md`: links attribution output to pruning reports.

### Task 3.3 Validation Progress

- Targeted container tests:
  `pytest tests/test_feature_pruning.py tests/test_feature_attribution.py -q`
  passed with `12 passed`, `4 warnings`.
- Phase 3 evaluation container tests:
  `pytest tests/test_feature_pruning.py tests/test_feature_attribution.py tests/test_interpretability.py -q`
  passed with `22 passed`, `4 warnings`.

### Phase 3 Final Validation

- Full small data container test:
  `pytest tests/test_small_dataset.py -q` passed with `12 passed`,
  `2552 warnings`.
- Full-flow container test:
  `pytest tests/test_full_flow.py -q` passed with `1 passed`,
  `747 warnings`.
- Full container pytest:
  `pytest -q` passed with `252 passed`, `3647 warnings`.

## Phase 4: MLOps and Performance

Status: in progress

Approved scope:
- Implement Task 4.1 as local MLflow experiment tracking only.
- Keep experiment tracking disabled by default.
- Preserve existing TensorBoard logging and training behavior.
- Use no API keys, no cloud account, and no remote MLflow tracking URI.
- Keep MLflow optional so disabled training does not require the package.
- Add docs explaining how to enable local MLflow, run training, start the local
  UI, and inspect runs.

### Tasks

- [x] Task 4.1: Local MLflow experiment tracking
- [x] Task 4.2: Polars profiling and migration

### Task 4.1 Notes

- `model.experiment_tracking` config controls optional tracking.
- `ENABLED=false` is the default.
- Local MLflow uses `MLFLOW_TRACKING_URI=file:./mlruns`.
- Remote MLflow URIs are rejected by config validation.
- No secrets or API keys are stored in config.
- Artifact logging is disabled by default to avoid unexpectedly storing large
  files.
- Tests use no-op or injected mock trackers and do not call external services.

### Task 4.1 Implemented Files

- `config/model.json`: adds disabled-by-default local MLflow tracking config.
- `src/config/schemas.py`: validates local-only MLflow tracking settings and
  rejects remote tracking URIs.
- `src/training/experiment_tracking.py`: no-op tracker, optional local MLflow
  tracker, config factory, and parameter/metric helpers.
- `src/training/trainer.py`: logs parameters and epoch metrics through the
  optional tracker while preserving TensorBoard behavior.
- `src/training/__init__.py`: exports tracking helpers.
- `tests/test_experiment_tracking.py`: no-op tracker, local config validation,
  remote URI rejection, and injected tracker logging tests.
- `docs/experiment_tracking.md`: local MLflow enablement, training command, UI
  usage, Docker port mapping note, and safety guidance.
- `docs/configuration.md`: documents `model.experiment_tracking`.

### Task 4.1 Validation Progress

- Targeted container tests:
  `pytest tests/test_experiment_tracking.py tests/test_config_schemas.py tests/test_training.py -q`
  passed with `20 passed`, `4 warnings`.
- Training integration container tests:
  `pytest tests/test_training_integration.py tests/test_full_flow.py -q`
  passed with `8 passed`, `943 warnings`.
- Combined tracking/training regression container tests:
  `pytest tests/test_experiment_tracking.py tests/test_config_schemas.py tests/test_training.py tests/test_training_integration.py tests/test_full_flow.py -q`
  passed with `28 passed`, `943 warnings`.

### Task 4.2 Notes

- Polars migration is incremental and opt-in.
- Polars implementations were added where parity is straightforward:
  time features, Fibonacci rolling features, and external data joins.
- The public pipeline still returns pandas DataFrames to preserve downstream
  preprocessing, prediction, evaluation, and reporting contracts.
- TA-Lib indicators, stockstats integration, financial metrics loading, sklearn
  preprocessing, prediction/evaluation APIs, and Excel reports remain pandas.
- `polars_fibonacci_features`, `polars_time_features`, and
  `polars_external_merges` are disabled by default.
- Fibonacci profiling helper compares pandas and Polars timings on the same
  input.
- Polars was installed in the active container for validation.

### Task 4.2 Implemented Files

- `config/main.json`: adds disabled-by-default `polars_*` feature flags.
- `requirements.txt` and `setup.py`: add Polars dependency.
- `src/data/feature_engineering_polars.py`: optional Polars time feature,
  Fibonacci, external merge, and profiling helpers.
- `src/data/feature_engineering.py`: routes opt-in Polars feature engineering
  paths while preserving pandas defaults.
- `tests/test_feature_engineering_polars.py`: pandas/Polars parity,
  order-preservation, profiling, and missing-dependency tests.
- `docs/polars_migration.md`: enablement, profiling, validation, and retained
  pandas scope.
- `docs/data_pipeline.md` and `docs/configuration.md`: document opt-in flags.

### Task 4.2 Validation Progress

- Initial targeted container tests caught a parity difference where Polars kept
  nulls for `break_fib_61` while pandas treated `close < NaN` as false.
- Fixed Polars Fibonacci break indicator to match pandas behavior.
- Targeted container tests:
  `pytest tests/test_feature_engineering_polars.py tests/test_data_pipeline.py tests/test_config_schemas.py -q`
  passed with `26 passed`, `4 warnings`.
- Full-flow regression container test:
  `pytest tests/test_feature_engineering_polars.py tests/test_data_pipeline.py tests/test_config_schemas.py tests/test_full_flow.py -q`
  passed with `27 passed`, `747 warnings`.

## Phase 5: PyTorch Lightning Migration

Status: in progress

Approved scope:
- Make Lightning the default training backend.
- Keep the existing custom `Trainer` as the backup path.
- Add `--backend lightning|custom` to select the backend explicitly.
- Reuse existing model, loss, optimizer, scheduler, and batch contracts.
- Save custom-compatible `.pth` checkpoints from Lightning so prediction
  compatibility is preserved.
- Add tests wherever Lightning behavior is introduced.

### Tasks

- [x] Task 5.1: Lightning default path with custom trainer backup
- [ ] Task 5.2: Lightning parity and migration decision

### Task 5.1 Notes

- `model.training_backend.DEFAULT` is `lightning`.
- `model.training_backend.FALLBACK` is `custom`.
- `ALLOW_CUSTOM_FALLBACK=true` permits explicit fallback when Lightning is
  unavailable.
- `scripts/train.py --backend custom` forces the backup trainer.
- Lightning writes custom-compatible `.pth` checkpoints with
  `metadata.training_backend = "lightning"`.
- Lightning `--resume` and `--fine-tune` load model weights from custom
  checkpoints; full optimizer resume remains a Task 5.2 parity topic.

### Task 5.1 Implemented Files

- `config/model.json`: adds `model.training_backend` with Lightning default.
- `requirements.txt` and `setup.py`: add Lightning dependency.
- `src/training/common.py`: shared loss factory for both backends.
- `src/training/lightning_module.py`: LightningModule wrapper, Lightning
  trainer factory, and custom-compatible checkpoint callback.
- `src/training/trainer.py`: reuses shared loss factory.
- `src/training/__init__.py`: exports Lightning backend helpers.
- `scripts/train.py`: backend selection, Lightning default, custom fallback,
  and model-weight loading for Lightning fine-tune/resume.
- `tests/test_lightning_backend.py`: config validation, dependency behavior,
  training/validation step, optimizer, smoke fit, and custom checkpoint tests.
- `docs/lightning_backend.md`: backend usage, fallback, dependency, and
  checkpoint compatibility.
- `docs/configuration.md` and `docs/roadmap_execution_plan.md`: document the
  changed default backend strategy.

### Task 5.1 Validation Progress

- Installed `lightning>=2.2.0` in the active container for validation.
- Targeted container tests:
  `pytest tests/test_lightning_backend.py tests/test_config_schemas.py tests/test_training.py -q`
  passed with `19 passed`, `5 warnings`.
- Lightning/custom training regression container tests:
  `pytest tests/test_lightning_backend.py tests/test_config_schemas.py tests/test_training.py tests/test_training_integration.py -q`
  passed with `27 passed`, `204 warnings`.
- Full-flow container test:
  `pytest tests/test_full_flow.py -q` passed with `1 passed`, `747 warnings`.
- Full container pytest:
  `docker exec d633c5977c4f pytest -q` passed with `270 passed`,
  `3679 warnings`.
