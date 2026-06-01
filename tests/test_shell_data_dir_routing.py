import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _default_model_type() -> str:
    import json
    with (REPO_ROOT / "config" / "model.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)["model"]["selection"]["DEFAULT_MODEL_TYPE"]


def _expected_data_dir_for_default_model() -> str:
    return {
        "chronos2": "data/processed_chronos2",
        "chronos_rich": "data/processed_chronos_rich",
        "kronos_rich": "data/processed_kronos_rich",
    }.get(_default_model_type(), "data/processed")


def _make_fake_binary(tmp_path: Path, name: str) -> Path:
    log_path = tmp_path / f"{name}_args.txt"
    binary_path = tmp_path / name
    if name == "python":
        binary_path.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = \"-\" ]; then\n"
            f"  exec {sys.executable} \"$@\"\n"
            "fi\n"
            f"printf '%s\\n' \"$@\" > \"{log_path}\"\n",
            encoding="utf-8",
        )
    else:
        binary_path.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > \"{log_path}\"\n",
            encoding="utf-8",
        )
    binary_path.chmod(0o755)
    return log_path


def _run_with_fake_python(script_name: str, tmp_path: Path, extra_args: list[str]) -> list[str]:
    log_path = _make_fake_binary(tmp_path, "python")
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / script_name), *extra_args],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return log_path.read_text(encoding="utf-8").splitlines()


def _run_with_fake_docker(script_name: str, tmp_path: Path, extra_args: list[str]) -> list[str]:
    log_path = _make_fake_binary(tmp_path, "docker")
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / script_name), *extra_args],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return log_path.read_text(encoding="utf-8").splitlines()


def test_test_sh_routes_kronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_python("test.sh", tmp_path, ["--model-type", "kronos_rich", "--no-cleanup"])
    assert "--data-dir" in args
    assert "data/processed_kronos_rich" in args


def test_test_sh_routes_chronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_python("test.sh", tmp_path, ["--model-type", "chronos_rich", "--no-cleanup"])
    assert "--data-dir" in args
    assert "data/processed_chronos_rich" in args


def test_validate_sh_routes_kronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_python("validate.sh", tmp_path, ["--model-type", "kronos_rich"])
    assert "--data-dir" in args
    assert "data/processed_kronos_rich" in args


def test_validate_sh_routes_chronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_python("validate.sh", tmp_path, ["--model-type", "chronos_rich"])
    assert "--data-dir" in args
    assert "data/processed_chronos_rich" in args


def test_backtest_sh_routes_kronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_python("backtest.sh", tmp_path, ["--model-type", "kronos_rich"])
    assert "--data-dir" in args
    assert "data/processed_kronos_rich" in args


def test_backtest_sh_routes_chronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_python("backtest.sh", tmp_path, ["--model-type", "chronos_rich"])
    assert "--data-dir" in args
    assert "data/processed_chronos_rich" in args


def test_train_sh_routes_kronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_docker("train.sh", tmp_path, ["--model-type", "kronos_rich"])
    assert "python" in args
    assert "--data-dir" in args
    assert "data/processed_kronos_rich" in args


def test_train_sh_routes_chronos_rich_to_special_data_dir(tmp_path):
    args = _run_with_fake_docker("train.sh", tmp_path, ["--model-type", "chronos_rich"])
    assert "python" in args
    assert "--data-dir" in args
    assert "data/processed_chronos_rich" in args


def test_train_sh_uses_default_model_type_from_config_for_data_dir(tmp_path):
    args = _run_with_fake_docker("train.sh", tmp_path, [])
    assert "python" in args
    assert "--data-dir" in args
    assert _expected_data_dir_for_default_model() in args


def test_test_sh_uses_default_model_type_from_config_for_data_dir(tmp_path):
    args = _run_with_fake_python("test.sh", tmp_path, ["--no-cleanup"])
    assert "--data-dir" in args
    assert _expected_data_dir_for_default_model() in args
