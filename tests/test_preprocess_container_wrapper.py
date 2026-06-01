import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "preprocess_in_container.sh"


def _default_model_type() -> str:
    import json
    with (REPO_ROOT / "config" / "model.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)["model"]["selection"]["DEFAULT_MODEL_TYPE"]


def _make_fake_python(tmp_path: Path) -> tuple[Path, Path]:
    log_path = tmp_path / "python_args.txt"
    python_path = tmp_path / "python"
    python_path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-\" ]; then\n"
        "  if [ \"$2\" = \"statsmodels\" ]; then\n"
        "    exit 0\n"
        "  fi\n"
        f"  exec {sys.executable} \"$@\"\n"
        "fi\n"
        f"printf '%s\\n' \"$@\" > \"{log_path}\"\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    return python_path, log_path


def _run_wrapper(tmp_path: Path, extra_args: list[str]) -> list[str]:
    _fake_python, log_path = _make_fake_python(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    subprocess.run(
        ["bash", str(SCRIPT_PATH), *extra_args],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return log_path.read_text(encoding="utf-8").splitlines()


def test_preprocess_wrapper_routes_chronos2_to_prepare_script(tmp_path):
    args = _run_wrapper(tmp_path, ["--model-type", "chronos2", "--skip-download"])

    assert args[0] == "scripts/prepare_chronos2_data.py"
    assert "--skip-download" in args


def test_preprocess_wrapper_routes_kronos_rich_to_prepare_script(tmp_path):
    args = _run_wrapper(tmp_path, ["--model-type", "kronos_rich", "--skip-download"])

    assert args[0] == "scripts/prepare_kronos_rich_data.py"
    assert "--skip-download" in args


def test_preprocess_wrapper_routes_chronos_rich_to_prepare_script(tmp_path):
    args = _run_wrapper(tmp_path, ["--model-type", "chronos_rich", "--skip-download"])

    assert args[0] == "scripts/prepare_chronos_rich_data.py"
    assert "--skip-download" in args


def test_preprocess_wrapper_uses_default_model_type_from_config(tmp_path):
    args = _run_wrapper(tmp_path, ["--skip-download"])

    expected = {
        "chronos2": "scripts/prepare_chronos2_data.py",
        "chronos_rich": "scripts/prepare_chronos_rich_data.py",
        "kronos_rich": "scripts/prepare_kronos_rich_data.py",
    }.get(_default_model_type(), "scripts/preprocess_data.py")
    assert args[0] == expected
    assert "--skip-download" in args


def test_preprocess_wrapper_does_not_force_default_stocks_when_tickers_are_passed(tmp_path):
    args = _run_wrapper(tmp_path, ["--skip-download", "--tickers", "AAPL", "MSFT"])

    expected = {
        "chronos2": "scripts/prepare_chronos2_data.py",
        "chronos_rich": "scripts/prepare_chronos_rich_data.py",
        "kronos_rich": "scripts/prepare_kronos_rich_data.py",
    }.get(_default_model_type(), "scripts/preprocess_data.py")
    assert args[0] == expected
    assert "--tickers" in args
    assert "AAPL" in args
    assert "MSFT" in args
    assert "--stocks" not in args


def test_preprocess_wrapper_routes_other_models_to_standard_preprocess(tmp_path):
    args = _run_wrapper(tmp_path, ["--model-type", "rnn", "--skip-download"])

    assert args[0] == "scripts/preprocess_data.py"
    assert "--skip-download" in args


def test_preprocess_wrapper_installs_missing_statsmodels_before_running(tmp_path):
    log_path = tmp_path / "python_calls.log"
    state_path = tmp_path / "statsmodels_check_state"
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
        ["bash", str(SCRIPT_PATH), "--model-type", "rnn", "--skip-download"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    calls = log_path.read_text(encoding="utf-8").splitlines()
    assert any(call == "- statsmodels" for call in calls)
    assert any(call == "-m pip install --user statsmodels>=0.14.0" for call in calls)
    assert any(call.startswith("scripts/preprocess_data.py ") for call in calls)
