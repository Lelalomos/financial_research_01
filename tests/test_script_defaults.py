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
        contents = _read(path)
        assert 'source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"' in contents
        assert 'MODEL_TYPE=""' in contents
        assert 'resolve_model_type "$MODEL_TYPE"' in contents


def test_train_in_container_disables_monitors_by_default():
    contents = _read("scripts/train_in_container.sh")
    assert "START_TENSORBOARD=0" in contents
    assert "START_MLFLOW=0" in contents


def test_batch_runner_and_tuner_defaults_use_chronos2():
    files = [
        "scripts/run_all.sh",
        "scripts/run_all_in_container.sh",
        "scripts/optuna_tune.sh",
        "scripts/optuna_tune_in_container.sh",
    ]
    for path in files:
        contents = _read(path)
        assert 'source "$(dirname "${BASH_SOURCE[0]}")/common_model_routing.sh"' in contents
        assert 'MODEL_TYPE=""' in contents
        assert 'resolve_model_type "$MODEL_TYPE"' in contents


def test_run_all_scripts_forward_model_type_to_every_pipeline_step():
    for path in ["scripts/run_all.sh", "scripts/run_all_in_container.sh"]:
        contents = _read(path)
        assert 'preprocess' in contents
        assert '--model-type "$MODEL_TYPE"' in contents
