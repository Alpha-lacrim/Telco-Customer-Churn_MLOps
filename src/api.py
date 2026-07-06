"""FastAPI prediction service for the selected churn model."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from src.config import load_config, resolve_path
from src.logger import configure_logging
from src.utils import read_json

configure_logging()
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Telco Customer Churn API", version="1.0.0")


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


def _parse_records(payload: Any) -> list[dict[str, Any]]:
    """Accept either a single JSON object, a list, or an object with records."""
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise HTTPException(status_code=422, detail="Request body must be a JSON object or list.")

    if not records or not all(isinstance(record, dict) for record in records):
        raise HTTPException(status_code=422, detail="Each prediction record must be a JSON object.")
    return records


@app.get("/health")
def health() -> dict[str, str]:
    """Report API health and model source."""
    resources = _runtime_resources()
    return {"status": "ok", "model_uri": resources["model_uri"]}


@app.post("/predict")
def predict(payload: Any = Body(...)) -> dict[str, Any]:
    """Predict churn probabilities and classes for one or more customers."""
    resources = _runtime_resources()
    records = _parse_records(payload)
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
            {
                "predicted_class": int(predicted_class),
                "decision_threshold": float(decision_threshold),
                "probability_stayed": float(probabilities[index, 0]),
                "probability_churned": float(probabilities[index, 1]),
            }
        )
    return {"predictions": predictions}
