"""Feature engineering for dataset version 3 and inference transformations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data_registry import write_data_version_manifest
from src.preprocessing import (
    align_to_feature_columns,
    clean_raw_dataframe,
    encode_categorical_features,
)
from src.utils import load_dataframe, save_dataframe, write_json

LOGGER = logging.getLogger(__name__)


def _numeric_series(dataframe: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series for a column or a default-valued fallback."""
    if column not in dataframe.columns:
        return pd.Series(default, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(default)


def _yes_indicator(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Convert a Yes/No style service column into a binary indicator."""
    if column not in dataframe.columns:
        return pd.Series(0, index=dataframe.index, dtype="int64")
    return dataframe[column].astype(str).str.strip().str.lower().eq("yes").astype(int)


def add_domain_features(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    log_progress: bool = True,
) -> pd.DataFrame:
    """Add realistic churn-oriented domain features."""
    df = dataframe.copy()
    target_column = config["schema"]["target_column"]
    target = df[target_column] if target_column in df.columns else None
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    tenure = _numeric_series(df, "Tenure Months")
    monthly_charges = _numeric_series(df, "Monthly Charges")
    total_charges = _numeric_series(df, "Total Charges")

    service_columns = config["feature_engineering"]["service_columns"]
    protection_columns = config["feature_engineering"]["protection_columns"]
    streaming_columns = config["feature_engineering"]["streaming_columns"]

    df["Average Monthly Spend"] = np.where(
        tenure.gt(0),
        total_charges / tenure.replace(0, np.nan),
        monthly_charges,
    )
    df["Average Monthly Spend"] = pd.Series(df["Average Monthly Spend"]).fillna(monthly_charges)

    df["Tenure Group"] = pd.cut(
        tenure,
        bins=[-0.01, 12, 24, 48, 60, np.inf],
        labels=["0-12", "13-24", "25-48", "49-60", "61+"],
    ).astype("object")

    service_indicators = [_yes_indicator(df, column) for column in service_columns]
    df["Service Count"] = sum(service_indicators) if service_indicators else 0

    protection_indicators = [_yes_indicator(df, column) for column in protection_columns]
    df["Protection Score"] = sum(protection_indicators) if protection_indicators else 0

    streaming_indicators = [_yes_indicator(df, column) for column in streaming_columns]
    df["Streaming Score"] = sum(streaming_indicators) if streaming_indicators else 0

    internet_service = df.get("Internet Service", pd.Series("No", index=df.index)).astype(str)
    contract = df.get("Contract", pd.Series("Unknown", index=df.index)).astype(str)
    paperless = df.get("Paperless Billing", pd.Series("No", index=df.index)).astype(str)
    payment_method = df.get("Payment Method", pd.Series("Unknown", index=df.index)).astype(str)

    df["Has Internet Service"] = internet_service.str.lower().ne("no").astype(int)
    df["Has Fiber Optic"] = internet_service.str.lower().eq("fiber optic").astype(int)
    df["Is Month To Month"] = contract.str.lower().eq("month-to-month").astype(int)
    df["Has Long Term Contract"] = contract.str.lower().isin(["one year", "two year"]).astype(int)
    df["Contract Length Months"] = contract.map(
        config["feature_engineering"]["contract_length_months"]
    ).fillna(0)
    df["Paperless Electronic Payment"] = (
        paperless.str.lower().eq("yes")
        & payment_method.str.lower().eq("electronic check")
    ).astype(int)
    df["Charges Per Service"] = monthly_charges / df["Service Count"].clip(lower=1)

    if target is not None:
        df[target_column] = target.astype(int)

    if log_progress:
        LOGGER.info("Added domain features. Output shape: %s.", df.shape)
    return df


def build_feature_dataset(config: dict[str, Any], input_path: str) -> pd.DataFrame:
    """Create dataset version 3 with engineered, encoded model features."""
    LOGGER.info("Creating dataset version v3.")
    raw_dataframe = load_dataframe(input_path)
    cleaned_dataframe, metadata = clean_raw_dataframe(
        raw_dataframe,
        config=config,
        fit_metadata=True,
        include_target=True,
    )
    featured_dataframe = add_domain_features(cleaned_dataframe, config)
    encoded_dataframe = encode_categorical_features(
        featured_dataframe,
        target_column=config["schema"]["target_column"],
    )

    target_column = config["schema"]["target_column"]
    feature_columns = [column for column in encoded_dataframe.columns if column != target_column]

    v3_config = config["data"]["versions"]["v3"]
    save_dataframe(encoded_dataframe, v3_config["path"])

    metadata.update(
        {
            "dataset_version": "v3",
            "created_features": [
                "Average Monthly Spend",
                "Tenure Group",
                "Service Count",
                "Protection Score",
                "Streaming Score",
                "Has Internet Service",
                "Has Fiber Optic",
                "Is Month To Month",
                "Has Long Term Contract",
                "Contract Length Months",
                "Paperless Electronic Payment",
                "Charges Per Service",
            ],
            "feature_columns": feature_columns,
        }
    )

    write_json(v3_config["metadata_path"], metadata)
    write_json(config["artifacts"]["preprocessing_metadata"], metadata)
    write_json(config["artifacts"]["feature_columns"], feature_columns)
    write_data_version_manifest(
        config=config,
        dataset_version="v3",
        dataset_path=v3_config["path"],
        source_paths=[input_path],
        metadata_path=v3_config["metadata_path"],
        stage="feature_engineering",
        rows=int(encoded_dataframe.shape[0]),
        columns=int(encoded_dataframe.shape[1]),
        extra={"parent_version": "v1"},
    )
    LOGGER.info("Created dataset version v3.")
    return encoded_dataframe


def transform_raw_records_for_inference(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    metadata: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Apply the training feature transformation to raw prediction records."""
    raw_dataframe = pd.DataFrame.from_records(records)
    cleaned_dataframe, _ = clean_raw_dataframe(
        raw_dataframe,
        config=config,
        metadata=metadata,
        fit_metadata=False,
        include_target=False,
    )
    featured_dataframe = add_domain_features(cleaned_dataframe, config)
    encoded_dataframe = encode_categorical_features(featured_dataframe, target_column=None)
    return align_to_feature_columns(encoded_dataframe, feature_columns)
