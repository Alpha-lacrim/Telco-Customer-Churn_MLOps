"""Raw data loading and dataset version 1 creation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_path
from src.utils import save_dataframe, write_json

LOGGER = logging.getLogger(__name__)


def _find_source_excel(config: dict[str, Any]) -> Path:
    """Find the IBM Telco Excel file in the configured raw folder or project root."""
    raw_filename = config["data"]["raw_filename"]
    candidates = [
        resolve_path(config["data"]["raw_dir"]) / raw_filename,
        resolve_path(raw_filename),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Raw dataset was not found. Checked: {candidate_text}")


def load_and_version_raw_data(config: dict[str, Any]) -> Path:
    """Load the Excel dataset and persist version 1 without semantic changes."""
    LOGGER.info("Loading raw dataset.")
    source_path = _find_source_excel(config)
    raw_dir = resolve_path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_copy_path = raw_dir / config["data"]["raw_filename"]

    if source_path.resolve() != raw_copy_path.resolve():
        shutil.copy2(source_path, raw_copy_path)
        LOGGER.info("Copied raw Excel file to %s.", raw_copy_path)

    dataframe = pd.read_excel(raw_copy_path)
    v1_config = config["data"]["versions"]["v1"]
    output_path = save_dataframe(dataframe, v1_config["path"])

    metadata = {
        "dataset_version": "v1",
        "source_file": str(raw_copy_path),
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": list(dataframe.columns),
    }
    write_json(v1_config["metadata_path"], metadata)
    LOGGER.info("Created dataset version v1.")
    return output_path
