import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_in_container.sh"


def test_train_in_container_installs_missing_einops_before_training(tmp_path):
    log_path = tmp_path / "python_calls.log"
    state_path = tmp_path / "einops_check_state"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> \"{log_path}\"\n"
        "if [ \"$1\" = \"-\" ]; then\n"
        f"  if [ ! -f \"{state_path}\" ]; then\n"
        f"    touch \"{state_path}\"\n"
        "    exit 1\n"
        "  fi\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ] && [ \"$3\" = \"install\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "--model-type",
            "rnn",
            "--data-dir",
            "data/processed",
            "--epochs",
            "1",
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
            "--force-cpu",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert any(call == "- einops" for call in calls)
    assert any(call == "-m pip install --user einops>=0.8.1" for call in calls)
    assert any(call.startswith("scripts/train.py ") for call in calls)


def test_train_in_container_allows_mlflow_file_store_by_default():
    contents = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"' in contents
