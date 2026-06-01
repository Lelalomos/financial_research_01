from types import SimpleNamespace
from pathlib import Path

from src.config import load_config
from scripts.prepare_chronos_rich_data import resolve_chronos_rich_prep_settings


def test_resolve_chronos_rich_prep_settings_reads_config_defaults():
    config = load_config("main")
    args = SimpleNamespace(
        processed_dir=None,
        output_dir=None,
        skip_scalar_target=False,
        skip_return_path=False,
        skip_regime_label=False,
    )

    settings = resolve_chronos_rich_prep_settings(config, args)

    assert settings["processed_dir"] == Path("data/processed")
    assert settings["output_dir"] == Path("data/processed_chronos_rich")
    assert settings["ohlcv_columns"] == ["open", "high", "low", "close", "volume"]
    assert settings["include_return_path"] is True
    assert settings["include_regime_label"] is True
