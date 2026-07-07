"""FastAPI prediction service for the selected churn model."""

from __future__ import annotations

import logging
import math
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
MODEL_METADATA_FILENAME = "model_metadata.json"


class PredictionRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(..., min_length=1)


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

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_uri: str


def _decision_threshold(model_metadata: dict[str, Any]) -> float:
    """Load threshold from env first, then bundled model metadata."""
    threshold_override = os.getenv("DECISION_THRESHOLD")
    if threshold_override is not None:
        return float(threshold_override)

    threshold = model_metadata.get("decision_threshold")
    if threshold is not None:
        return float(threshold)

    LOGGER.warning("Model metadata has no decision_threshold; falling back to 0.5.")
    return 0.5


def _input_validation_schema(
    config: dict[str, Any],
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Prefer model-bundled schema, falling back to config."""
    schema = model_metadata.get("api_input_schema") or config.get("api", {}).get("input_schema")
    if not schema or "fields" not in schema:
        raise RuntimeError("API input schema is missing from model metadata and config.")

    input_columns = model_metadata.get("input_columns")
    if not input_columns:
        return schema

    fields_by_name = {field["name"]: field for field in schema["fields"]}
    ordered_fields = []
    missing_fields = []
    for column in input_columns:
        field = fields_by_name.get(column)
        if field is None:
            missing_fields.append(column)
        else:
            ordered_fields.append(field)
    if missing_fields:
        raise RuntimeError(
            "Model input columns are missing from API input schema: "
            + ", ".join(missing_fields)
        )
    return {"fields": ordered_fields}


def _max_batch_size(config: dict[str, Any], model_metadata: dict[str, Any]) -> int:
    """Load max request batch size from model metadata or config."""
    return int(
        model_metadata.get(
            "api_max_batch_size",
            config.get("api", {}).get("max_batch_size", 1000),
        )
    )


def _validation_error(index: int, field_name: str, message: str) -> dict[str, Any]:
    """Format a FastAPI-style validation error."""
    return {
        "loc": ["body", "records", index, field_name],
        "msg": message,
        "type": "value_error",
    }


def _validate_numeric_value(
    value: Any,
    field: dict[str, Any],
    index: int,
) -> tuple[float | int | None, dict[str, Any] | None]:
    """Validate and coerce numeric API input."""
    field_name = field["name"]
    field_type = field["type"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, _validation_error(index, field_name, "Input should be numeric.")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None, _validation_error(index, field_name, "Input should be finite.")
    if "min" in field and numeric_value < float(field["min"]):
        return None, _validation_error(index, field_name, f"Input should be >= {field['min']}.")
    if "max" in field and numeric_value > float(field["max"]):
        return None, _validation_error(index, field_name, f"Input should be <= {field['max']}.")
    if field_type == "integer":
        if not numeric_value.is_integer():
            return None, _validation_error(index, field_name, "Input should be an integer.")
        return int(numeric_value), None
    return numeric_value, None


def _validate_prediction_records(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    max_batch_size: int,
) -> list[dict[str, Any]]:
    """Validate records against the configured or model-bundled API schema."""
    if len(records) > max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "records"],
                    "msg": f"Batch size must be <= {max_batch_size}.",
                    "type": "value_error",
                }
            ],
        )

    fields = schema["fields"]
    field_names = [field["name"] for field in fields]
    allowed_field_names = set(field_names)
    errors = []
    validated_records: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        extra_fields = sorted(set(record) - allowed_field_names)
        for field_name in extra_fields:
            errors.append(_validation_error(index, field_name, "Extra field is not permitted."))

        validated_record: dict[str, Any] = {}
        for field in fields:
            field_name = field["name"]
            if field.get("required", True) and field_name not in record:
                errors.append(_validation_error(index, field_name, "Field required."))
                continue
            if field_name not in record:
                continue

            value = record[field_name]
            field_type = field["type"]
            if field_type == "categorical":
                if not isinstance(value, str):
                    errors.append(_validation_error(index, field_name, "Input should be a string."))
                    continue
                normalized_value = value.strip()
                allowed_values = {str(value) for value in field.get("allowed_values", [])}
                if normalized_value not in allowed_values:
                    errors.append(
                        _validation_error(
                            index,
                            field_name,
                            f"Input should be one of: {sorted(allowed_values)}.",
                        )
                    )
                    continue
                validated_record[field_name] = normalized_value
            elif field_type in {"integer", "number"}:
                numeric_value, error = _validate_numeric_value(value, field, index)
                if error is not None:
                    errors.append(error)
                    continue
                validated_record[field_name] = numeric_value
            else:
                errors.append(_validation_error(index, field_name, f"Unsupported field type: {field_type}."))

        validated_records.append(validated_record)

    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return validated_records


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


def _download_model_artifact(model_uri: str) -> Path | None:
    """Return a local model artifact path for metadata inspection."""
    if "://" not in model_uri and not model_uri.startswith(("runs:", "models:")):
        return Path(model_uri)

    try:
        import mlflow.artifacts

        return Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
    except Exception:
        LOGGER.warning("Could not download MLflow model metadata artifact.", exc_info=True)
        return None


def _load_model_metadata(model_uri: str) -> dict[str, Any]:
    """Load metadata packaged with an MLflow model artifact."""
    local_model_path = _download_model_artifact(model_uri)
    if local_model_path is None:
        return {}

    metadata: dict[str, Any] = {}
    try:
        from mlflow.models import Model

        mlflow_model = Model.load(str(local_model_path))
        metadata.update(dict(mlflow_model.metadata or {}))
    except Exception:
        LOGGER.warning("Could not read MLflow model metadata from MLmodel.", exc_info=True)

    sidecar_path = local_model_path / MODEL_METADATA_FILENAME
    if sidecar_path.exists():
        try:
            metadata.update(read_json(sidecar_path))
        except Exception:
            LOGGER.warning("Could not read bundled model metadata sidecar.", exc_info=True)
    return metadata


@lru_cache(maxsize=1)
def _runtime_resources() -> dict[str, Any]:
    """Load config, model metadata, and MLflow model once per process."""
    import mlflow.sklearn

    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    model_uri = os.getenv("MODEL_URI", config["deployment"]["model_uri"])
    model_uri = _resolved_model_uri(model_uri)
    model_metadata = _load_model_metadata(model_uri)
    input_schema = _input_validation_schema(config, model_metadata)
    max_batch_size = _max_batch_size(config, model_metadata)
    decision_threshold = _decision_threshold(model_metadata)

    model = mlflow.sklearn.load_model(model_uri)
    LOGGER.info("Loaded MLflow model from %s.", model_uri)
    return {
        "config": config,
        "model": model,
        "model_uri": model_uri,
        "model_metadata": model_metadata,
        "input_schema": input_schema,
        "max_batch_size": max_batch_size,
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
    records = _validate_prediction_records(
        request.records,
        resources["input_schema"],
        resources["max_batch_size"],
    )
    features = pd.DataFrame.from_records(records)
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
