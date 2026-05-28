from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_container_wrappers_default_to_chronos2():
    files = [
        "scripts/train_in_container.sh",
        "scripts/test_in_container.sh",
        "scripts/backtest_in_container.sh",
    ]
    for path in files:
        assert 'MODEL_TYPE="chronos2"' in _read(path)


def test_batch_runner_and_tuner_defaults_use_chronos2():
    files = [
        "scripts/run_all.sh",
        "scripts/run_all_in_container.sh",
        "scripts/optuna_tune.sh",
        "scripts/optuna_tune_in_container.sh",
    ]
    for path in files:
        assert 'MODEL_TYPE="chronos2"' in _read(path)
