from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


SOURCE_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]


def _synthetic_telco_frame(rows: int = 40) -> pd.DataFrame:
    records = []
    for index in range(rows):
        churned = int(index % 2 == 0)
        tenure = 2 + index
        monthly_charges = 45.0 + (index % 10) * 4.0
        latitude = 34.0 + index / 1000.0
        longitude = -118.0 - index / 1000.0
        records.append(
            {
                "CustomerID": f"CUST-{index:04d}",
                "Count": 1,
                "Country": "United States",
                "State": "California",
                "City": "Los Angeles",
                "Zip Code": 90000 + index,
                "Lat Long": f"{latitude}, {longitude}",
                "Latitude": latitude,
                "Longitude": longitude,
                "Gender": "Female" if index % 2 == 0 else "Male",
                "Senior Citizen": "Yes" if index % 5 == 0 else "No",
                "Partner": "Yes" if index % 3 == 0 else "No",
                "Dependents": "No" if churned else "Yes",
                "Tenure Months": tenure,
                "Phone Service": "Yes",
                "Multiple Lines": "Yes" if index % 3 == 0 else "No",
                "Internet Service": "Fiber optic" if churned else "DSL",
                "Online Security": "No" if churned else "Yes",
                "Online Backup": "Yes" if index % 4 == 0 else "No",
                "Device Protection": "No" if churned else "Yes",
                "Tech Support": "No" if churned else "Yes",
                "Streaming TV": "Yes" if index % 2 == 0 else "No",
                "Streaming Movies": "Yes" if index % 4 == 0 else "No",
                "Contract": "Month-to-month" if churned else "One year",
                "Paperless Billing": "Yes" if churned else "No",
                "Payment Method": "Electronic check" if churned else "Mailed check",
                "Monthly Charges": monthly_charges,
                "Total Charges": monthly_charges * tenure,
                "Churn Label": "Yes" if churned else "No",
                "Churn Value": churned,
                "Churn Score": 80 if churned else 20,
                "CLTV": 3000 + index * 10,
                "Churn Reason": "Competitor made better offer" if churned else "",
            }
        )
    return pd.DataFrame(records, columns=SOURCE_COLUMNS)


def _write_smoke_config(tmp_path: Path) -> Path:
    with Path("config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["data"]["raw_dir"] = str(tmp_path / "raw")
    config["data"]["raw_filename"] = "Telco_customer_churn.xlsx"
    config["data"]["versions"]["v1"]["path"] = str(tmp_path / "data" / "v1.csv")
    config["data"]["versions"]["v1"]["metadata_path"] = str(tmp_path / "data" / "v1.json")
    config["data"]["versions"]["v2"]["path"] = str(tmp_path / "data" / "v2.csv")
    config["data"]["versions"]["v2"]["metadata_path"] = str(tmp_path / "data" / "v2.json")
    config["data"]["versions"]["v3"]["path"] = str(tmp_path / "data" / "v3.csv")
    config["data"]["versions"]["v3"]["metadata_path"] = str(tmp_path / "data" / "v3.json")
    config["data_versioning"]["registry_dir"] = str(tmp_path / "registry")

    config["artifacts"]["dir"] = str(tmp_path / "artifacts")
    config["artifacts"]["comparison_table"] = str(tmp_path / "artifacts" / "comparison.csv")
    config["artifacts"]["best_model_summary"] = str(tmp_path / "artifacts" / "summary.json")
    config["artifacts"]["preprocessing_metadata"] = str(tmp_path / "artifacts" / "preprocessing.json")
    config["artifacts"]["feature_columns"] = str(tmp_path / "artifacts" / "features.json")
    config["artifacts"]["confusion_matrix_dir"] = str(tmp_path / "artifacts" / "confusion")
    config["artifacts"]["feature_importance_dir"] = str(tmp_path / "artifacts" / "importance")
    config["deployment"]["model_uri"] = str(tmp_path / "models" / "best_model")
    config["mlflow"]["tracking_uri"] = str(tmp_path / "mlruns")

    config["training"]["test_size"] = 0.2
    config["training"]["validation_size"] = 0.2
    config["training"]["cv_splits"] = 2
    config["training"]["n_jobs"] = 1
    config["training"]["decision_threshold_search"] = {"min": 0.3, "max": 0.7, "step": 0.1}
    config["training"]["scoring"] = {
        "roc_auc": "roc_auc",
        "overall_score": "overall_score",
    }
    config["training"]["hyperparameter_refit_metric"] = "overall_score"

    config["models"]["logistic_regression"]["enabled"] = True
    config["models"]["logistic_regression"]["param_grid"] = {
        "classifier__C": [1.0],
        "classifier__solver": ["liblinear"],
        "classifier__penalty": ["l2"],
        "classifier__class_weight": [None],
    }
    for model_name in ["random_forest", "xgboost", "catboost"]:
        config["models"][model_name]["enabled"] = False

    config_path = tmp_path / "smoke_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return config_path


def test_run_pipeline_skip_mlflow_smoke(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    _synthetic_telco_frame().to_excel(raw_dir / "Telco_customer_churn.xlsx", index=False)
    config_path = _write_smoke_config(tmp_path)

    completed = subprocess.run(
        [sys.executable, "run_pipeline.py", "--config", str(config_path), "--skip-mlflow"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout

    summary = json.loads((tmp_path / "artifacts" / "summary.json").read_text(encoding="utf-8"))
    registry = json.loads((tmp_path / "registry" / "manifest.json").read_text(encoding="utf-8"))

    assert summary["selection_metric"] == "validation_overall_score"
    assert summary["final_model_fit_scope"] == "train_plus_validation_after_selection"
    assert {"v1", "v2", "v3"}.issubset(registry["versions"])
