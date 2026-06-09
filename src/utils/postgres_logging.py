"""
Best-effort PostgreSQL logging for runtime summaries.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


_JSON_COLUMNS = {
    "stocks_filter",
    "main_config_json",
    "model_config_json",
    "cli_args_json",
    "dataset_info_json",
    "feature_cols_json",
    "regime_params_json",
    "sector_stats_json",
}


def _read_dotenv(dotenv_path: Optional[Path] = None) -> Dict[str, str]:
    path = dotenv_path or (Path(__file__).resolve().parents[2] / ".env")
    if not path.exists():
        return {}

    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _default_host() -> str:
    return "postgres" if Path("/.dockerenv").exists() else "localhost"


def _resolve_db_config(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env_map = dict(_read_dotenv())
    env_map.update(env or {})
    merged = {**os.environ, **env_map}

    password = (
        merged.get("POSTGRES_PASSWORD")
        or merged.get("PGPASSWORD")
        or merged.get("DB_PASSWORD")
    )
    if not password:
        return {}

    return {
        "host": merged.get("POSTGRES_HOST") or merged.get("PGHOST") or merged.get("DB_HOST") or _default_host(),
        "port": int(merged.get("POSTGRES_PORT") or merged.get("PGPORT") or merged.get("DB_PORT") or 5432),
        "dbname": merged.get("POSTGRES_DB") or merged.get("PGDATABASE") or merged.get("DB_NAME") or "stockdb",
        "user": merged.get("POSTGRES_USER") or merged.get("PGUSER") or merged.get("DB_USER") or "admin",
        "password": password,
    }


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _serialize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in _JSON_COLUMNS:
            serialized[key] = None if value is None else json.dumps(_safe_json(value))
        else:
            serialized[key] = _safe_scalar(value)
    return serialized


class PostgresRunLogger:
    """Insert run summaries into PostgreSQL without breaking the main flow."""

    def __init__(self, logger=None, env: Optional[Dict[str, str]] = None, enabled: bool = True):
        self.logger = logger
        self._enabled = enabled
        self.db_config = _resolve_db_config(env=env)

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self.db_config)

    def log_run(self, table_name: str, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        try:
            self._insert_row(table_name, payload)
            return True
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(f"PostgreSQL logging skipped for {table_name}: {exc}")
            return False

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is required for PostgreSQL runtime logging."
            ) from exc
        return psycopg.connect(**self.db_config)

    def _insert_row(self, table_name: str, payload: Dict[str, Any]) -> None:
        serialized = _serialize_payload(payload)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_table_ddl(table_name))

                columns = []
                placeholders = []
                values = []
                for key, value in serialized.items():
                    columns.append(key)
                    if key in _JSON_COLUMNS:
                        placeholders.append("%s::jsonb")
                    else:
                        placeholders.append("%s")
                    values.append(value)

                query = (
                    f"INSERT INTO {table_name} "
                    f"({', '.join(columns)}) "
                    f"VALUES ({', '.join(placeholders)})"
                )
                cur.execute(query, values)
            conn.commit()


def _table_ddl(table_name: str) -> str:
    if table_name == "training_runs":
        return """
        CREATE TABLE IF NOT EXISTS training_runs (
          id BIGSERIAL PRIMARY KEY,
          run_uuid TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          status TEXT NOT NULL,
          model_type TEXT,
          training_backend TEXT,
          device TEXT,
          data_dir TEXT,
          checkpoint_dir TEXT,
          best_checkpoint_path TEXT,
          final_checkpoint_path TEXT,
          resume_checkpoint_path TEXT,
          fine_tune_checkpoint_path TEXT,
          stocks_filter JSONB,
          max_train_batches INTEGER,
          max_val_batches INTEGER,
          num_epochs_requested INTEGER,
          batch_size_requested INTEGER,
          learning_rate_requested DOUBLE PRECISION,
          main_config_json JSONB,
          model_config_json JSONB,
          cli_args_json JSONB,
          dataset_info_json JSONB,
          feature_cols_json JSONB,
          num_features INTEGER,
          num_stocks INTEGER,
          num_groups INTEGER,
          sequence_length INTEGER,
          prediction_horizon INTEGER,
          normalize_target BOOLEAN,
          target_threshold DOUBLE PRECISION,
          regime_params_json JSONB,
          train_samples INTEGER,
          val_samples INTEGER,
          best_val_loss DOUBLE PRECISION,
          final_train_loss DOUBLE PRECISION,
          final_val_loss DOUBLE PRECISION,
          notes TEXT,
          error_message TEXT
        )
        """

    if table_name == "test_runs":
        return """
        CREATE TABLE IF NOT EXISTS test_runs (
          id BIGSERIAL PRIMARY KEY,
          run_uuid TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          status TEXT NOT NULL,
          model_type TEXT,
          device TEXT,
          data_dir TEXT,
          split TEXT,
          checkpoint_path TEXT,
          checkpoint_epoch INTEGER,
          raw_data_dir TEXT,
          output_json_path TEXT,
          excel_report_path TEXT,
          max_samples INTEGER,
          main_config_json JSONB,
          model_config_json JSONB,
          cli_args_json JSONB,
          dataset_info_json JSONB,
          feature_cols_json JSONB,
          num_features INTEGER,
          num_stocks INTEGER,
          num_groups INTEGER,
          sequence_length INTEGER,
          prediction_horizon INTEGER,
          normalize_target BOOLEAN,
          target_threshold DOUBLE PRECISION,
          sample_count INTEGER,
          mse DOUBLE PRECISION,
          rmse DOUBLE PRECISION,
          mae DOUBLE PRECISION,
          r2 DOUBLE PRECISION,
          mape DOUBLE PRECISION,
          directional_accuracy DOUBLE PRECISION,
          hit_rate DOUBLE PRECISION,
          sharpe_ratio DOUBLE PRECISION,
          max_drawdown DOUBLE PRECISION,
          sortino_ratio DOUBLE PRECISION,
          total_return DOUBLE PRECISION,
          error_message TEXT
        )
        """

    if table_name == "validation_runs":
        return """
        CREATE TABLE IF NOT EXISTS validation_runs (
          id BIGSERIAL PRIMARY KEY,
          run_uuid TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          status TEXT NOT NULL,
          model_type TEXT,
          device TEXT,
          data_dir TEXT,
          split TEXT,
          checkpoint_path TEXT,
          checkpoint_epoch INTEGER,
          output_json_path TEXT,
          excel_report_path TEXT,
          max_samples INTEGER,
          main_config_json JSONB,
          model_config_json JSONB,
          cli_args_json JSONB,
          dataset_info_json JSONB,
          feature_cols_json JSONB,
          num_features INTEGER,
          num_stocks INTEGER,
          num_groups INTEGER,
          sequence_length INTEGER,
          prediction_horizon INTEGER,
          normalize_target BOOLEAN,
          target_threshold DOUBLE PRECISION,
          sample_count INTEGER,
          mse DOUBLE PRECISION,
          rmse DOUBLE PRECISION,
          mae DOUBLE PRECISION,
          r2 DOUBLE PRECISION,
          mape DOUBLE PRECISION,
          directional_accuracy DOUBLE PRECISION,
          hit_rate DOUBLE PRECISION,
          sharpe_ratio DOUBLE PRECISION,
          max_drawdown DOUBLE PRECISION,
          sortino_ratio DOUBLE PRECISION,
          total_return DOUBLE PRECISION,
          error_message TEXT
        )
        """

    if table_name == "backtest_runs":
        return """
        CREATE TABLE IF NOT EXISTS backtest_runs (
          id BIGSERIAL PRIMARY KEY,
          run_uuid TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          status TEXT NOT NULL,
          model_type TEXT,
          device TEXT,
          data_dir TEXT,
          split TEXT,
          checkpoint_path TEXT,
          checkpoint_epoch INTEGER,
          raw_data_dir TEXT,
          output_path TEXT,
          output_format TEXT,
          prediction_threshold DOUBLE PRECISION,
          initial_capital DOUBLE PRECISION,
          max_samples INTEGER,
          main_config_json JSONB,
          model_config_json JSONB,
          cli_args_json JSONB,
          dataset_info_json JSONB,
          feature_cols_json JSONB,
          num_features INTEGER,
          num_stocks INTEGER,
          num_groups INTEGER,
          sequence_length INTEGER,
          prediction_horizon INTEGER,
          normalize_target BOOLEAN,
          target_threshold DOUBLE PRECISION,
          sample_count INTEGER,
          initial_capital_result DOUBLE PRECISION,
          final_capital DOUBLE PRECISION,
          total_return_pct DOUBLE PRECISION,
          total_return_value DOUBLE PRECISION,
          sharpe_ratio DOUBLE PRECISION,
          sortino_ratio DOUBLE PRECISION,
          max_drawdown_pct DOUBLE PRECISION,
          risk_adjusted_return DOUBLE PRECISION,
          num_trades INTEGER,
          num_position_changes INTEGER,
          win_rate_pct DOUBLE PRECISION,
          avg_win_pct DOUBLE PRECISION,
          avg_loss_pct DOUBLE PRECISION,
          profit_factor DOUBLE PRECISION,
          average_turnover DOUBLE PRECISION,
          total_turnover DOUBLE PRECISION,
          commission_rate DOUBLE PRECISION,
          total_transaction_cost_pct DOUBLE PRECISION,
          total_transaction_cost_value DOUBLE PRECISION,
          sector_stats_json JSONB,
          error_message TEXT
        )
        """

    raise ValueError(f"Unsupported table name: {table_name}")


def make_run_uuid() -> str:
    return str(uuid.uuid4())


def make_run_logger(
    logger=None,
    env: Optional[Dict[str, str]] = None,
    enabled: bool = True,
) -> PostgresRunLogger:
    return PostgresRunLogger(logger=logger, env=env, enabled=enabled)
