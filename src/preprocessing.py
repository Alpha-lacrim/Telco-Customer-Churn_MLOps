"""Data cleaning, missing value handling, and categorical encoding."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils import load_dataframe, save_dataframe, write_json

LOGGER = logging.getLogger(__name__)


def _coerce_configured_numeric_columns(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Convert configured numeric columns, including Excel object columns."""
    df = dataframe.copy()
    for column in config["schema"]["numeric_columns"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def clean_raw_dataframe(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    fit_metadata: bool = True,
    include_target: bool = True,
    log_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw records and return cleaned data plus preprocessing metadata."""
    target_column = config["schema"]["target_column"]
    drop_columns = set(config["schema"]["drop_columns"])

    df = dataframe.copy()
    df = _coerce_configured_numeric_columns(df, config)

    columns_to_drop = [column for column in drop_columns if column in df.columns]
    df = df.drop(columns=columns_to_drop)

    if not include_target and target_column in df.columns:
        df = df.drop(columns=[target_column])

    if include_target:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing from the dataset.")
        df[target_column] = pd.to_numeric(df[target_column], errors="raise").astype(int)

    if fit_metadata:
        feature_columns = [column for column in df.columns if column != target_column]
        numeric_columns = [
            column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])
        ]
        categorical_columns = [column for column in feature_columns if column not in numeric_columns]

        numeric_impute_values = {
            column: float(df[column].median()) if not np.isnan(df[column].median()) else 0.0
            for column in numeric_columns
        }
        categorical_impute_values = {}
        for column in categorical_columns:
            mode = df[column].dropna().mode()
            categorical_impute_values[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"

        metadata = {
            "target_column": target_column,
            "dropped_columns": sorted(columns_to_drop),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "numeric_impute_values": numeric_impute_values,
            "categorical_impute_values": categorical_impute_values,
        }
    elif metadata is None:
        raise ValueError("Preprocessing metadata is required when fit_metadata=False.")

    numeric_values = metadata["numeric_impute_values"]
    categorical_values = metadata["categorical_impute_values"]

    for column, value in numeric_values.items():
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(float(value))

    for column, value in categorical_values.items():
        if column not in df.columns:
            df[column] = np.nan
        df[column] = (
            df[column]
            .replace(r"^\s*$", np.nan, regex=True)
            .fillna(value)
            .astype(str)
            .str.strip()
        )

    ordered_columns = list(numeric_values) + list(categorical_values)
    if include_target:
        ordered_columns.append(target_column)
    df = df[ordered_columns]
    if log_progress:
        LOGGER.info("Cleaned dataframe with shape %s.", df.shape)
    return df, metadata


def encode_categorical_features(
    dataframe: pd.DataFrame,
    target_column: str | None,
) -> pd.DataFrame:
    """One-hot encode categorical features while preserving the target column."""
    df = dataframe.copy()
    target = None
    if target_column and target_column in df.columns:
        target = df[target_column].astype(int)
        df = df.drop(columns=[target_column])

    categorical_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_categorical_dtype(df[column])
    ]
    encoded = pd.get_dummies(df, columns=categorical_columns, dtype=int)

    for column in encoded.select_dtypes(include=["bool"]).columns:
        encoded[column] = encoded[column].astype(int)

    if target is not None:
        encoded[target_column] = target
    LOGGER.info("Encoded categorical columns. Output shape: %s.", encoded.shape)
    return encoded


def align_to_feature_columns(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Align encoded inference records to the training feature schema."""
    aligned = dataframe.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    aligned = aligned[feature_columns]
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0)


def preprocess_dataset(config: dict[str, Any], input_path: str) -> pd.DataFrame:
    """Create dataset version 2 with cleaned and encoded features."""
    LOGGER.info("Creating dataset version v2.")
    raw_dataframe = load_dataframe(input_path)
    cleaned_dataframe, metadata = clean_raw_dataframe(
        raw_dataframe,
        config=config,
        fit_metadata=True,
        include_target=True,
    )
    encoded_dataframe = encode_categorical_features(
        cleaned_dataframe,
        target_column=config["schema"]["target_column"],
    )

    v2_config = config["data"]["versions"]["v2"]
    save_dataframe(encoded_dataframe, v2_config["path"])
    metadata["dataset_version"] = "v2"
    metadata["feature_columns"] = [
        column
        for column in encoded_dataframe.columns
        if column != config["schema"]["target_column"]
    ]
    write_json(v2_config["metadata_path"], metadata)
    LOGGER.info("Created dataset version v2.")
    return encoded_dataframe
