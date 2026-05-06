# Lightning Training Backend

Phase 5 Task 5.1 makes PyTorch Lightning the default training backend while
keeping the existing custom `Trainer` as the backup path.

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

Lightning checkpoints are not used as the production prediction artifact.
Instead, the Lightning path writes a custom-compatible `.pth` checkpoint with:

- `model_state_dict`
- optimizer state when available
- model type
- feature metadata
- target normalization metadata
- `metadata.training_backend = "lightning"`

This keeps the current prediction/checkpoint loading contract intact.

## Backup Trainer

The existing custom `Trainer` remains available through `--backend custom`.
Use it when debugging low-level training behavior or if Lightning dependency
issues occur.

## Current Limitations

- Full optimizer resume for Lightning from existing custom checkpoints is not
  implemented in Task 5.1; Lightning loads model weights for `--resume` and
  `--fine-tune`.
- Lightning remains subject to Task 5.2 parity analysis before removing the
  custom trainer backup.
- The custom `Trainer` should not be deleted until parity and checkpoint
  behavior are proven on realistic runs.
