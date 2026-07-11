from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    if base_path is None:
        base_path = Path(config_path).parent.parent / "base.yaml"

    base = load_yaml(base_path) if Path(base_path).exists() else {}
    experiment = load_yaml(config_path)

    return deep_merge(base, experiment)
