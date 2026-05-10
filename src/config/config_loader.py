"""
JSON configuration loader.

Replaces Python dataclass configs with JSON-based configuration.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, ItemsView, KeysView, ValuesView

from .schemas import validate_config_data


class Config:
    """Base config class that provides attribute access to dict."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Config):
            value = value.to_dict()
        self._data[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            # Always wrap dicts as Config objects for nested access
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def keys(self) -> KeysView:
        """Return dictionary keys."""
        return self._data.keys()

    def values(self) -> ValuesView:
        """Return dictionary values."""
        return self._data.values()

    def items(self) -> ItemsView:
        """Return dictionary items."""
        return self._data.items()

    def __iter__(self):
        """Make Config iterable over its keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of keys."""
        return len(self._data)

    # Model config helper methods for backward compatibility
    def get_scheduler_params(self) -> Dict[str, Any]:
        """
        Get scheduler parameters for the configured scheduler type.

        This is a helper method for backward compatibility with the old ModelConfig.
        Assumes the config has model.training.SCHEDULER and model.training.SCHEDULER_PARAMS.
        """
        try:
            scheduler = self.model.training.SCHEDULER
            scheduler_params = self.model.training.SCHEDULER_PARAMS
            if scheduler_params and scheduler in scheduler_params._data:
                return scheduler_params._data[scheduler]
            return {}
        except AttributeError:
            return {}


_config_cache: Dict[str, Config] = {}


def load_config(
    config_name: str,
    config_dir: Optional[str] = None,
    validate: bool = True
) -> Config:
    """
    Load configuration from JSON file.

    Args:
        config_name: Name of config ('model', 'main', 'test', 'deploy', 'validate')
        config_dir: Directory containing config files (default: 'config/')
        validate: Whether to validate known config files with Pydantic schemas

    Returns:
        Config object with attribute access

    Example:
        config = load_config('model')
        print(config.model.architecture.D_MODEL)
    """
    if config_name in _config_cache:
        return _config_cache[config_name]

    if config_dir is None:
        # Default to project root/config
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config"
    else:
        config_dir = Path(config_dir)

    config_file = config_dir / f"{config_name}.json"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        data = json.load(f)

    if validate:
        validate_config_data(config_name, data)

    config = Config(data)
    _config_cache[config_name] = config
    return config


def reload_config(config_name: str, config_dir: Optional[str] = None) -> Config:
    """Reload configuration from file (bypass cache)."""
    if config_name in _config_cache:
        del _config_cache[config_name]
    return load_config(config_name, config_dir)


# Convenience functions for backward compatibility during transition
def get_model_config() -> Config:
    """Get model configuration."""
    return load_config('model')


def get_main_config() -> Config:
    """Get main pipeline configuration."""
    return load_config('main')


def get_test_config() -> Config:
    """Get test configuration."""
    return load_config('test')


def get_deploy_config() -> Config:
    """Get deployment configuration."""
    return load_config('deploy')


def get_validate_config() -> Config:
    """Get validation configuration."""
    return load_config('validate')


def get_hyperparameter_config() -> Config:
    """Get hyperparameter search configuration."""
    return load_config('hyperparameter')
