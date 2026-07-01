"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML configuration and attach useful runtime paths."""
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config["project_root"] = str(PROJECT_ROOT)
    config["config_path"] = str(path)
    return config
