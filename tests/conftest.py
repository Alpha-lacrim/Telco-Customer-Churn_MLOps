from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest


@pytest.fixture()
def base_config() -> dict[str, Any]:
    """Minimal config for unit tests."""
    return {
        "schema": {
            "target_column": "Churn Value",
            "numeric_columns": [
                "Latitude",
                "Longitude",
                "Tenure Months",
                "Monthly Charges",
                "Total Charges",
            ],
            "drop_columns": [
                "CustomerID",
                "Count",
                "Country",
                "State",
                "City",
                "Zip Code",
                "Lat Long",
                "Churn Label",
                "Churn Score",
                "CLTV",
                "Churn Reason",
            ],
        },
        "feature_engineering": {
            "service_columns": [
                "Phone Service",
                "Multiple Lines",
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies",
            ],
            "protection_columns": [
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
            ],
            "streaming_columns": ["Streaming TV", "Streaming Movies"],
            "contract_length_months": {
                "Month-to-month": 1,
                "One year": 12,
                "Two year": 24,
            },
        },
    }


@pytest.fixture()
def raw_customer_frame() -> pd.DataFrame:
    """Small raw customer dataset with the inference columns."""
    return pd.DataFrame(
        [
            {
                "CustomerID": "A",
                "Gender": "Female",
                "Senior Citizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure Months": 12,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "Yes",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "Yes",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 89.1,
                "Total Charges": 1069.2,
                "Latitude": 34.05,
                "Longitude": -118.24,
                "Churn Value": 1,
            },
            {
                "CustomerID": "B",
                "Gender": "Male",
                "Senior Citizen": "No",
                "Partner": "No",
                "Dependents": "Yes",
                "Tenure Months": 24,
                "Phone Service": "Yes",
                "Multiple Lines": "Yes",
                "Internet Service": "DSL",
                "Online Security": "Yes",
                "Online Backup": "No",
                "Device Protection": "Yes",
                "Tech Support": "Yes",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "One year",
                "Paperless Billing": "No",
                "Payment Method": "Mailed check",
                "Monthly Charges": 55.0,
                "Total Charges": 1320.0,
                "Latitude": 35.0,
                "Longitude": -119.0,
                "Churn Value": 0,
            },
        ]
    )


def with_artifact_paths(config: dict[str, Any], tmp_path: Any) -> dict[str, Any]:
    """Attach temporary paths for functions that write artifacts."""
    copied = deepcopy(config)
    copied["data"] = {
        "raw_filename": "Telco_customer_churn.xlsx",
        "raw_dir": str(tmp_path / "raw"),
        "versions": {
            "v1": {
                "path": str(tmp_path / "v1" / "telco_churn_v1.csv"),
                "metadata_path": str(tmp_path / "v1" / "metadata.json"),
            },
            "v2": {
                "path": str(tmp_path / "v2" / "telco_churn_v2_clean_encoded.csv"),
                "metadata_path": str(tmp_path / "v2" / "preprocessing_metadata.json"),
            },
            "v3": {
                "path": str(tmp_path / "v3" / "telco_churn_v3_features.csv"),
                "metadata_path": str(tmp_path / "v3" / "feature_metadata.json"),
            },
        },
    }
    copied["data_versioning"] = {
        "registry_dir": str(tmp_path / "registry"),
        "checksum_algorithm": "sha256",
        "remote_uri": "",
    }
    copied["artifacts"] = {
        "dir": str(tmp_path / "artifacts"),
        "comparison_table": str(tmp_path / "artifacts" / "model_comparison.csv"),
        "best_model_summary": str(tmp_path / "artifacts" / "best_model_summary.json"),
        "preprocessing_metadata": str(tmp_path / "artifacts" / "preprocessing_metadata.json"),
        "feature_columns": str(tmp_path / "artifacts" / "feature_columns.json"),
        "confusion_matrix_dir": str(tmp_path / "artifacts" / "confusion_matrices"),
        "feature_importance_dir": str(tmp_path / "artifacts" / "feature_importance"),
    }
    return copied
