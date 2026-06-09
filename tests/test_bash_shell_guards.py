import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent

BASH_ONLY_SCRIPTS = [
    "scripts/analyze_correlation_in_container.sh",
    "scripts/backtest.sh",
    "scripts/backtest_in_container.sh",
    "scripts/common_model_routing.sh",
    "scripts/optuna_tune.sh",
    "scripts/optuna_tune_in_container.sh",
    "scripts/plot_latest_output_in_container.sh",
    "scripts/preprocess.sh",
    "scripts/preprocess_in_container.sh",
    "scripts/run_all.sh",
    "scripts/run_all_in_container.sh",
    "scripts/test.sh",
    "scripts/test_all.sh",
    "scripts/test_in_container.sh",
    "scripts/test_nan_fix.sh",
    "scripts/test_small_dataset.sh",
    "scripts/test_small_dataset_in_container.sh",
    "scripts/test_unit.sh",
    "scripts/test_unit_in_container.sh",
    "scripts/train.sh",
    "scripts/train_in_container.sh",
    "scripts/validate.sh",
    "scripts/validate_in_container.sh",
]


@pytest.mark.parametrize("script_path", BASH_ONLY_SCRIPTS)
def test_bash_only_scripts_fail_clearly_when_run_with_sh(script_path):
    result = subprocess.run(
        ["sh", script_path, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "This script must be run with bash:" in result.stderr
    assert f"bash {script_path}" in result.stderr
    assert "Syntax error" not in result.stderr
