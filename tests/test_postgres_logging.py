import importlib
import json
from argparse import Namespace

import pytest

from src.utils.postgres_logging import PostgresRunLogger, _serialize_payload


class FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeAppLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


class RecordingRunLogger:
    def __init__(self):
        self.calls = []

    def log_run(self, table_name, payload):
        self.calls.append((table_name, payload))
        return True


def test_serialize_payload_converts_json_columns_to_json_strings():
    payload = {
        "run_uuid": "run-1",
        "status": "completed",
        "main_config_json": {"a": 1, "path": "x"},
        "feature_cols_json": ["close", "volume"],
        "num_features": 2,
    }

    serialized = _serialize_payload(payload)

    assert serialized["run_uuid"] == "run-1"
    assert json.loads(serialized["main_config_json"]) == {"a": 1, "path": "x"}
    assert json.loads(serialized["feature_cols_json"]) == ["close", "volume"]
    assert serialized["num_features"] == 2


def test_postgres_run_logger_inserts_jsonb_payload(monkeypatch):
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    logger = PostgresRunLogger(logger=FakeAppLogger(), env={"POSTGRES_PASSWORD": "secret"})
    monkeypatch.setattr(logger, "_connect", lambda: connection)

    success = logger.log_run(
        "test_runs",
        {
            "run_uuid": "run-1",
            "status": "completed",
            "main_config_json": {"model": "rnn"},
            "feature_cols_json": ["close"],
            "num_features": 1,
        },
    )

    assert success is True
    assert len(cursor.calls) == 2
    insert_sql, params = cursor.calls[1]
    assert "INSERT INTO test_runs" in insert_sql
    assert "::jsonb" in insert_sql
    assert json.loads(params[2]) == {"model": "rnn"}
    assert json.loads(params[3]) == ["close"]
    assert connection.committed is True


def test_postgres_run_logger_is_best_effort_on_failure(monkeypatch):
    app_logger = FakeAppLogger()
    logger = PostgresRunLogger(logger=app_logger, env={"POSTGRES_PASSWORD": "secret"})

    def _boom():
        raise RuntimeError("db offline")

    monkeypatch.setattr(logger, "_connect", _boom)

    success = logger.log_run("validation_runs", {"run_uuid": "run-1", "status": "failed"})

    assert success is False
    assert any("db offline" in message for message in app_logger.messages)


def test_postgres_run_logger_stays_disabled_when_config_flag_is_off():
    logger = PostgresRunLogger(
        logger=FakeAppLogger(),
        env={"POSTGRES_PASSWORD": "secret"},
        enabled=False,
    )

    assert logger.enabled is False
    assert logger.log_run("test_runs", {"run_uuid": "run-1", "status": "completed"}) is False


@pytest.mark.parametrize(
    ("module_name", "args", "table_name"),
    [
        (
            "scripts.test",
            Namespace(
                model="best",
                model_type=None,
                data_dir="data/processed",
                raw_data_dir=None,
                split="test",
                device=None,
                force_cpu=True,
                output=None,
                excel_report=None,
                max_samples=None,
            ),
            "test_runs",
        ),
        (
            "scripts.validate",
            Namespace(
                model="best",
                model_type=None,
                data_dir="data/processed",
                split="val",
                device=None,
                force_cpu=True,
                output=None,
                excel_report=None,
                max_samples=None,
            ),
            "validation_runs",
        ),
        (
            "scripts.backtest",
            Namespace(
                model="best",
                model_type=None,
                data_dir="data/processed",
                raw_data_dir=None,
                split="test",
                device=None,
                force_cpu=True,
                output="outputs/backtest_report.xlsx",
                output_format="excel",
                threshold=0.0,
                initial_capital=100000.0,
                max_samples=None,
            ),
            "backtest_runs",
        ),
    ],
)
def test_eval_scripts_log_failed_row_when_split_data_is_missing(monkeypatch, module_name, args, table_name):
    module = importlib.import_module(module_name)
    recording_logger = RecordingRunLogger()

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "make_run_logger", lambda logger, enabled=True: recording_logger)
    monkeypatch.setattr(module, "make_run_uuid", lambda: "run-1")
    monkeypatch.setattr(module, "resolve_device", lambda **kwargs: "cpu")
    monkeypatch.setattr(module, "get_device_info", lambda verbose=False: {"cuda_available": False, "cuda_working": False})
    monkeypatch.setattr(module, "load_sequences", lambda *a, **k: None)

    status = module.main()

    assert status == 1
    assert len(recording_logger.calls) == 1
    logged_table, payload = recording_logger.calls[0]
    assert logged_table == table_name
    assert payload["status"] == "failed"
    assert "No" in payload["error_message"]


def test_train_script_logs_failed_row_when_training_data_is_missing(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.train")
    recording_logger = RecordingRunLogger()
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    args = Namespace(
        model_type="rnn",
        data_dir=str(data_dir),
        config=None,
        epochs=1,
        batch_size=8,
        lr=0.001,
        device=None,
        force_cpu=True,
        resume=None,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        stocks=None,
        fine_tune=None,
        freeze_embeddings=False,
        backend="custom",
        max_train_batches=None,
        max_val_batches=None,
    )

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "make_run_logger", lambda logger, enabled=True: recording_logger)
    monkeypatch.setattr(module, "make_run_uuid", lambda: "run-1")
    monkeypatch.setattr(module, "get_device", lambda force_cpu=False, verbose=True: "cpu")
    monkeypatch.setattr(module, "get_device_info", lambda verbose=False: {"cuda_available": False, "cuda_working": False})
    monkeypatch.setattr(module, "load_sequences", lambda *a, **k: None)

    status = module.main()

    assert status == 1
    assert len(recording_logger.calls) == 1
    logged_table, payload = recording_logger.calls[0]
    assert logged_table == "training_runs"
    assert payload["status"] == "failed"
    assert "Missing feature_cols" in payload["error_message"]
