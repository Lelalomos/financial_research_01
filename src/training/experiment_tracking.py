"""
Optional experiment tracking utilities.

The default tracker is a no-op. Local MLflow tracking is only initialized when
explicitly enabled in config and never requires an API key.
"""

from typing import Any, Dict, Optional


class ExperimentTrackingError(RuntimeError):
    """Raised when an enabled experiment tracker cannot be initialized."""


class NoOpTracker:
    """Experiment tracker that intentionally does nothing."""

    enabled = False

    def start_run(self, run_name: Optional[str] = None) -> None:
        return None

    def log_params(self, params: Dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        return None

    def end_run(self, status: str = "FINISHED") -> None:
        return None


class LocalMLflowTracker:
    """Local file-backed MLflow tracker."""

    enabled = True

    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        experiment_name: str = "multi-model-financial-forecasting",
        log_params_enabled: bool = True,
        log_metrics_enabled: bool = True,
        log_artifacts_enabled: bool = False,
    ):
        if not _is_local_tracking_uri(tracking_uri):
            raise ExperimentTrackingError(
                "Only local MLflow tracking URIs are allowed. Use a file URI "
                "such as `file:./mlruns` or a local path."
            )

        try:
            import mlflow
        except ImportError as exc:
            raise ExperimentTrackingError(
                "MLflow is required when experiment tracking is enabled. "
                "Install it with `pip install mlflow`, or keep "
                "model.experiment_tracking.ENABLED=false."
            ) from exc

        self.mlflow = mlflow
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.log_params_enabled = log_params_enabled
        self.log_metrics_enabled = log_metrics_enabled
        self.log_artifacts_enabled = log_artifacts_enabled
        self._active_run = None

        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> None:
        if self._active_run is None:
            self._active_run = self.mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.log_params_enabled:
            return
        safe_params = _stringify_params(params)
        if safe_params:
            self.mlflow.log_params(safe_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.log_metrics_enabled:
            return
        safe_metrics = _numeric_metrics(metrics)
        if safe_metrics:
            self.mlflow.log_metrics(safe_metrics, step=step)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run is not None:
            self.mlflow.end_run(status=status)
            self._active_run = None


def create_experiment_tracker(config) -> NoOpTracker:
    """
    Build an experiment tracker from model config.

    Tracking is disabled unless `model.experiment_tracking.ENABLED` is true.
    """
    tracking_config = getattr(config.model, "experiment_tracking", None)
    if tracking_config is None or not tracking_config.ENABLED:
        return NoOpTracker()

    backend = tracking_config.BACKEND
    if backend != "mlflow":
        raise ExperimentTrackingError(f"Unsupported experiment tracking backend: {backend}")

    return LocalMLflowTracker(
        tracking_uri=tracking_config.MLFLOW_TRACKING_URI,
        experiment_name=tracking_config.EXPERIMENT_NAME,
        log_params_enabled=tracking_config.LOG_PARAMS,
        log_metrics_enabled=tracking_config.LOG_METRICS,
        log_artifacts_enabled=tracking_config.LOG_ARTIFACTS,
    )


def training_params(config, model_type: str) -> Dict[str, Any]:
    """Extract stable scalar params for experiment tracking."""
    params = {"model_type": model_type}

    for section_name in ("training", "loss"):
        section = getattr(config.model, section_name)
        section_data = section.to_dict() if hasattr(section, "to_dict") else section._data
        for key, value in section_data.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                params[f"{section_name}.{key}"] = value

    return params


def _is_local_tracking_uri(uri: str) -> bool:
    if not uri:
        return False
    lowered = uri.lower()
    if lowered.startswith(("http://", "https://", "databricks://")):
        return False
    return lowered.startswith("file:") or "://" not in lowered


def _stringify_params(params: Dict[str, Any]) -> Dict[str, str]:
    return {
        str(key): str(value)[:250]
        for key, value in params.items()
        if value is not None
    }


def _numeric_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    safe_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            safe_metrics[str(key)] = float(value)
    return safe_metrics
