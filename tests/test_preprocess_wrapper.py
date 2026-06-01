import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "preprocess.sh"


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


def test_preprocess_wrapper_routes_kronos_rich_to_prepare_script(tmp_path):
    args = _run_wrapper(tmp_path, ["--model-type", "kronos_rich"])
    assert args[0] == "scripts/prepare_kronos_rich_data.py"


def test_preprocess_wrapper_routes_chronos_rich_to_prepare_script(tmp_path):
    args = _run_wrapper(tmp_path, ["--model-type", "chronos_rich"])
    assert args[0] == "scripts/prepare_chronos_rich_data.py"


def test_preprocess_wrapper_uses_default_model_type_from_config(tmp_path):
    args = _run_wrapper(tmp_path, [])
    expected = {
        "chronos2": "scripts/prepare_chronos2_data.py",
        "chronos_rich": "scripts/prepare_chronos_rich_data.py",
        "kronos_rich": "scripts/prepare_kronos_rich_data.py",
    }.get(_default_model_type(), "scripts/preprocess_data.py")
    assert args[0] == expected


def test_preprocess_wrapper_routes_default_model_to_standard_preprocess(tmp_path):
    args = _run_wrapper(tmp_path, ["--model-type", "rnn"])
    assert args[0] == "scripts/preprocess_data.py"
