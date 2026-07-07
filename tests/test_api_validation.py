from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api import PredictionRequest


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

    assert request.records[0].contract == "Month-to-month"


def test_prediction_request_rejects_missing_required_fields():
    payload = valid_payload()
    del payload["records"][0]["Total Charges"]

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_prediction_request_rejects_extra_or_leaky_fields():
    payload = valid_payload()
    payload["records"][0]["Churn Value"] = 1

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_prediction_request_rejects_invalid_categories():
    payload = valid_payload()
    payload["records"][0]["Contract"] = "Weekly"

    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)
