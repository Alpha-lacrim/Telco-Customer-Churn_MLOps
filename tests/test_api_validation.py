from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api import PredictionRequest, _validate_prediction_records
from src.config import load_config


def valid_payload() -> dict:
    return {
        "records": [
            {
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
            }
        ]
    }


def test_prediction_request_accepts_valid_records():
    request = PredictionRequest.model_validate(valid_payload())
    schema = load_config()["api"]["input_schema"]
    records = _validate_prediction_records(request.records, schema, max_batch_size=1000)

    assert records[0]["Contract"] == "Month-to-month"


def test_prediction_request_rejects_missing_required_fields():
    payload = valid_payload()
    del payload["records"][0]["Total Charges"]
    schema = load_config()["api"]["input_schema"]
    request = PredictionRequest.model_validate(payload)

    with pytest.raises(HTTPException):
        _validate_prediction_records(request.records, schema, max_batch_size=1000)


def test_prediction_request_rejects_extra_or_leaky_fields():
    payload = valid_payload()
    payload["records"][0]["Churn Value"] = 1
    schema = load_config()["api"]["input_schema"]
    request = PredictionRequest.model_validate(payload)

    with pytest.raises(HTTPException):
        _validate_prediction_records(request.records, schema, max_batch_size=1000)


def test_prediction_request_rejects_invalid_categories():
    payload = valid_payload()
    payload["records"][0]["Contract"] = "Weekly"
    schema = load_config()["api"]["input_schema"]
    request = PredictionRequest.model_validate(payload)

    with pytest.raises(HTTPException):
        _validate_prediction_records(request.records, schema, max_batch_size=1000)


def test_prediction_request_still_rejects_invalid_top_level_shape():
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({"records": []})
