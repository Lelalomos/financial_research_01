# Lightning Training Backend

Phase 5 makes PyTorch Lightning the default training backend while keeping the
existing custom `Trainer` as the backup path.

## Backend Selection

The default backend is configured in `config/model.json`:

```json
{
  "model": {
    "training_backend": {
      "DEFAULT": "lightning",
      "FALLBACK": "custom",
      "ALLOW_CUSTOM_FALLBACK": true
    }
  }
}
```

Run training with the default backend:

```bash
docker exec crnn_predictor python scripts/train.py
```

Force the custom trainer backup:

```bash
docker exec crnn_predictor python scripts/train.py --backend custom
```

Force Lightning explicitly:

```bash
docker exec crnn_predictor python scripts/train.py --backend lightning
```

## Dependency

Lightning is a project dependency. For an existing container:

```bash
docker exec -it crnn_predictor pip install "lightning>=2.2.0"
```

If Lightning is unavailable and `ALLOW_CUSTOM_FALLBACK=true`, `scripts/train.py`
logs the issue and falls back to the custom `Trainer`. The fallback is explicit
and not silent.

## Compatibility

The Lightning backend wraps the existing PyTorch model and reuses:

- existing batch format
- configured loss function
- optimizer settings
- scheduler settings
- gradient clipping settings

Lightning checkpoints are not used as the production prediction artifact.
Instead, the Lightning path writes a custom-compatible `.pth` checkpoint with:

- `model_state_dict`
- optimizer state when available
- model type
- feature metadata
- target normalization metadata
- `metadata.training_backend = "lightning"`
- validation metrics for the selected checkpoint

This keeps the current prediction/checkpoint loading contract intact.

## Validation Health and Best-Checkpoint Selection

Lightning now tracks validation prediction-health diagnostics at the end of
each validation epoch:

- `val/pred_positive_rate`
- `val/pred_negative_rate`
- `val/pred_std`
- `val/pred_mean`
- `val/target_positive_rate`
- `val/pred_target_corr`
- `val/collapse_penalty`
- `val/is_collapsed`

Best-checkpoint selection is no longer based on raw `val/loss` alone. The
saved `*_best_lightning.pth` path uses:

```text
selection_score = val_loss + collapse_penalty
```

This is intended to avoid promoting obviously collapsed checkpoints, such as
near-constant or one-sided positive-only validation predictions, as the
production `best_lightning` artifact.

The custom-compatible checkpoint now includes:

- `selection_score`
- `val_metrics`
- `metadata.is_collapsed`

## Parity Decision

Task 5.2 keeps Lightning as the default backend. The parity scope verified:

- the same configured loss factory is used by both backends
- optimizer and scheduler settings are created from the same config values
- Lightning now passes `GRADIENT_CLIP_VALUE` into its trainer
- a same-seed single-batch training comparison produces matching model weights
- Lightning custom-format checkpoints load through the existing `Predictor`

## Backup Trainer

The existing custom `Trainer` remains available through `--backend custom`.
Use it when debugging low-level training behavior, if Lightning dependency
issues occur, or when a workflow needs full optimizer-state resume from an
existing custom checkpoint.

## Current Limitations

- Full optimizer resume for Lightning from existing custom checkpoints is not
  implemented; Lightning loads model weights for `--resume` and `--fine-tune`.
- The custom `Trainer` should not be deleted because it remains the fallback
  and the explicit path for optimizer-resume workflows.
- Collapse-aware selection improves checkpoint choice, but it does not solve a
  weak underlying dataset or feature set by itself. Retraining is still
  required for existing bad checkpoints.
