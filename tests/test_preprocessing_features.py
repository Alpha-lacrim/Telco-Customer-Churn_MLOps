from __future__ import annotations

import pandas as pd

from src.features import add_domain_features
from src.preprocessing import clean_raw_dataframe


def test_clean_raw_dataframe_fits_and_reuses_imputation_metadata(
    base_config,
    raw_customer_frame,
):
    cleaned, metadata = clean_raw_dataframe(raw_customer_frame, base_config)

    assert "CustomerID" not in cleaned.columns
    assert metadata["numeric_impute_values"]["Monthly Charges"] == 72.05

    inference_frame = raw_customer_frame.drop(columns=["Monthly Charges", "Churn Value"])
    transformed, _ = clean_raw_dataframe(
        inference_frame,
        base_config,
        metadata=metadata,
        fit_metadata=False,
        include_target=False,
    )

    assert transformed["Monthly Charges"].tolist() == [72.05, 72.05]
    assert "Churn Value" not in transformed.columns


def test_add_domain_features_creates_expected_churn_features(base_config):
    cleaned = pd.DataFrame(
        [
            {
                "Tenure Months": 10,
                "Monthly Charges": 80.0,
                "Total Charges": 800.0,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Online Security": "Yes",
                "Online Backup": "Yes",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "Yes",
                "Streaming Movies": "No",
                "Internet Service": "Fiber optic",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Churn Value": 1,
            }
        ]
    )

    featured = add_domain_features(cleaned, base_config, log_progress=False)

    assert featured.loc[0, "Average Monthly Spend"] == 80.0
    assert featured.loc[0, "Service Count"] == 4
    assert featured.loc[0, "Protection Score"] == 2
    assert featured.loc[0, "Streaming Score"] == 1
    assert featured.loc[0, "Has Fiber Optic"] == 1
    assert featured.loc[0, "Is Month To Month"] == 1
    assert featured.loc[0, "Contract Length Months"] == 1
    assert featured.loc[0, "Paperless Electronic Payment"] == 1
    assert featured.loc[0, "Charges Per Service"] == 20.0
