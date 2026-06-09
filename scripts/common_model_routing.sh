#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    echo "This script must be run with bash:" >&2
    echo "bash $0 $*" >&2
    exit 2
fi


get_default_model_type() {
    local script_dir repo_root
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"

    python - "$repo_root" <<'PY'
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
config_path = repo_root / "config" / "model.json"
with config_path.open("r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data["model"]["selection"]["DEFAULT_MODEL_TYPE"])
PY
}

resolve_model_type() {
    local explicit_model_type="$1"
    if [ -n "$explicit_model_type" ]; then
        printf '%s\n' "$explicit_model_type"
    else
        get_default_model_type
    fi
}

resolve_data_dir_for_model_type() {
    case "$1" in
        chronos2)
            printf '%s\n' "data/processed_chronos2"
            ;;
        chronos_rich)
            printf '%s\n' "data/processed_chronos_rich"
            ;;
        kronos_rich)
            printf '%s\n' "data/processed_kronos_rich"
            ;;
        *)
            printf '%s\n' "data/processed"
            ;;
    esac
}
