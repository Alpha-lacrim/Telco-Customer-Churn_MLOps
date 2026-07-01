"""Shared utility functions."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_path

LOGGER = logging.getLogger(__name__)


def ensure_project_directories(config: dict[str, Any]) -> None:
    """Create all directories used by the pipeline."""
    paths = [
        config["data"]["raw_dir"],
        "data/v1",
        "data/v2",
        "data/v3",
        "models",
        config["artifacts"]["dir"],
        config["artifacts"]["confusion_matrix_dir"],
        config["artifacts"]["feature_importance_dir"],
        "notebooks",
        "mlruns",
    ]
    for path in paths:
        resolve_path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a dataframe to CSV and return the resolved output path."""
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    LOGGER.info("Saved dataframe to %s with shape %s.", output_path, df.shape)
    return output_path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataframe from a project-relative path."""
    input_path = resolve_path(path)
    LOGGER.info("Loading dataframe from %s.", input_path)
    return pd.read_csv(input_path)


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON with stable indentation."""
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    LOGGER.info("Saved JSON artifact to %s.", output_path)
    return output_path


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact."""
    input_path = resolve_path(path)
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_name(name: str) -> str:
    """Create a filesystem-safe artifact name."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip().lower()).strip("_")


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary for experiment tracking parameters."""
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened
