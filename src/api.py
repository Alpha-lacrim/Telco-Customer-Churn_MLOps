"""FastAPI prediction service for the selected churn model."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import load_config, resolve_path
from src.logger import configure_logging
from src.utils import read_json

configure_logging()
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Telco Customer Churn API", version="1.0.0")


YesNo = Literal["No", "Yes"]


class TelcoCustomerRecord(BaseModel):
    """Validated inference payload using IBM Telco column aliases."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    gender: Literal["Female", "Male"] = Field(..., alias="Gender")
    senior_citizen: YesNo = Field(..., alias="Senior Citizen")
    partner: YesNo = Field(..., alias="Partner")
    dependents: YesNo = Field(..., alias="Dependents")
    tenure_months: int = Field(..., alias="Tenure Months", ge=0, le=120)
    phone_service: YesNo = Field(..., alias="Phone Service")
    multiple_lines: Literal["No", "No phone service", "Yes"] = Field(
        ...,
        alias="Multiple Lines",
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        alias="Internet Service",
    )
    online_security: Literal["No", "No internet service", "Yes"] = Field(
        ...,
        alias="Online Security",
    )
    online_backup: Literal["No", "No internet service", "Yes"] = Field(
        ...,
        alias="Online Backup",
    )
    device_protection: Literal["No", "No internet service", "Yes"] = Field(
        ...,
        alias="Device Protection",
    )
    tech_support: Literal["No", "No internet service", "Yes"] = Field(
        ...,
        alias="Tech Support",
    )
    streaming_tv: Literal["No", "No internet service", "Yes"] = Field(
        ...,
        alias="Streaming TV",
    )
    streaming_movies: Literal["No", "No internet service", "Yes"] = Field(
        ...,
        alias="Streaming Movies",
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        alias="Contract",
    )
    paperless_billing: YesNo = Field(..., alias="Paperless Billing")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., alias="Payment Method")
    monthly_charges: float = Field(
        ...,
        alias="Monthly Charges",
        ge=0.0,
        le=1000.0,
        allow_inf_nan=False,
    )
    total_charges: float = Field(
        ...,
        alias="Total Charges",
        ge=0.0,
        le=100000.0,
        allow_inf_nan=False,
    )
    latitude: float = Field(
        ...,
        alias="Latitude",
        ge=-90.0,
        le=90.0,
        allow_inf_nan=False,
    )
    longitude: float = Field(
        ...,
        alias="Longitude",
        ge=-180.0,
        le=180.0,
        allow_inf_nan=False,
    )


class PredictionRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    records: list[TelcoCustomerRecord] = Field(..., min_length=1, max_length=1000)


class PredictionItem(BaseModel):
    """Single prediction response."""

    predicted_class: Literal[0, 1]
    decision_threshold: float = Field(..., ge=0.0, le=1.0)
    probability_stayed: float = Field(..., ge=0.0, le=1.0)
    probability_churned: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok"]
    model_uri: str


def _decision_threshold(config: dict[str, Any]) -> float:
    """Load threshold from env first, then optional training summary."""
    threshold_override = os.getenv("DECISION_THRESHOLD")
    if threshold_override is not None:
        return float(threshold_override)

    summary_path = resolve_path(config["artifacts"]["best_model_summary"])
    if summary_path.exists():
        return float(read_json(summary_path).get("decision_threshold", 0.5))
    return 0.5


def _resolved_model_uri(model_uri: str) -> str:
    """Resolve local model paths while preserving MLflow registry/run URIs."""
    if "://" in model_uri or model_uri.startswith(("runs:", "models:")):
        return model_uri

    resolved_path = resolve_path(model_uri)
    if not Path(resolved_path).exists():
        raise RuntimeError(
            "MLflow model was not found at "
            f"{resolved_path}. Run the training pipeline and mount/export the model, "
            "or set MODEL_URI to an MLflow model registry/run URI."
        )
    return str(resolved_path)


@lru_cache(maxsize=1)
def _runtime_resources() -> dict[str, Any]:
    """Load config, threshold metadata, and MLflow model once per process."""
    import mlflow.sklearn

    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    model_uri = os.getenv("MODEL_URI", config["deployment"]["model_uri"])
    model_uri = _resolved_model_uri(model_uri)
    decision_threshold = _decision_threshold(config)

    model = mlflow.sklearn.load_model(model_uri)
    LOGGER.info("Loaded MLflow model from %s.", model_uri)
    return {
        "config": config,
        "model": model,
        "model_uri": model_uri,
        "decision_threshold": decision_threshold,
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API health and model source."""
    resources = _runtime_resources()
    return HealthResponse(status="ok", model_uri=resources["model_uri"])


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict churn probabilities and classes for one or more customers."""
    resources = _runtime_resources()
    features = pd.DataFrame.from_records(
        [record.model_dump(by_alias=True) for record in request.records]
    )
    model = resources["model"]

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model does not expose predict_proba.")

    probabilities = np.asarray(model.predict_proba(features))
    decision_threshold = resources["decision_threshold"]
    predicted_classes = (probabilities[:, 1] >= decision_threshold).astype(int)

    predictions = []
    for index, predicted_class in enumerate(predicted_classes):
        predictions.append(
            PredictionItem(
                predicted_class=int(predicted_class),
                decision_threshold=float(decision_threshold),
                probability_stayed=float(probabilities[index, 0]),
                probability_churned=float(probabilities[index, 1]),
            )
        )
    return PredictionResponse(predictions=predictions)
