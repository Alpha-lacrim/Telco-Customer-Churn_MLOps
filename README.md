# Telco Customer Churn MLOps Pipeline

Production-style MLOps project for customer churn prediction using the IBM Telco Customer Churn dataset. The project versions data, cleans and engineers features, trains multiple models, logs experiments with MLflow, registers the best model, and serves predictions through a Dockerized FastAPI service.

## Dataset

The source file is `Telco_customer_churn.xlsx`, copied into `data/raw/` by the pipeline. The target is `Churn Value`, where `1` means churned and `0` means stayed.

The pipeline creates three dataset versions:

- `data/v1/telco_churn_v1.csv`: raw Excel data loaded as-is.
- `data/v2/telco_churn_v2_clean_encoded.csv`: cleaned, imputed, and one-hot encoded data.
- `data/v3/telco_churn_v3_features.csv`: realistic domain features plus encoded modeling columns.

Scaling is fitted inside sklearn pipelines during cross-validation and training to avoid data leakage.

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |-- v1/
|   |-- v2/
|   `-- v3/
|-- src/
|   |-- api.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- features.py
|   |-- logger.py
|   |-- mlflow_utils.py
|   |-- preprocessing.py
|   |-- train.py
|   `-- utils.py
|-- models/
|-- artifacts/
|-- notebooks/
|-- mlruns/
|-- config.yaml
|-- Dockerfile
|-- requirements.txt
|-- run_pipeline.py
`-- README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run MLflow

In a separate terminal:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000`.

## Run the Full Pipeline

```bash
python run_pipeline.py --config config.yaml
```

This single command:

- creates data versions v1, v2, and v3
- performs cleaning and feature engineering
- creates stratified train, validation, and test splits
- trains Logistic Regression, Random Forest, XGBoost, and CatBoost
- runs stratified K-fold `GridSearchCV`
- evaluates accuracy, precision, recall, F1, ROC AUC, and confusion matrices
- tunes the classification decision threshold on the validation set for the configured objective
- logs runs, metrics, parameters, artifacts, and models to MLflow
- registers the best model by the configured validation selection metric
- saves a deployable MLflow model under `models/best_model`

Useful outputs:

- `artifacts/model_comparison.csv`
- `artifacts/best_model_summary.json`
- `artifacts/confusion_matrices/`
- `artifacts/feature_importance/`

## Docker Deployment

Run the pipeline first so `models/best_model` exists, then build the API image:

```bash
docker build -t telco-churn-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 telco-churn-api
```

The prediction service is available at `http://127.0.0.1:8000`.

## Prediction API

Endpoint:

```text
POST /predict
```

Example request:

```json
{
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
      "Longitude": -118.24
    }
  ]
}
```

Example response:

```json
{
  "predictions": [
    {
      "predicted_class": 1,
      "decision_threshold": 0.55,
      "probability_stayed": 0.23,
      "probability_churned": 0.77
    }
  ]
}
```

## Configuration

All project constants live in `config.yaml`, including paths, random seed, split sizes, MLflow settings, model hyperparameter grids, and feature engineering rules.

The pipeline can select the best model with `training.model_selection_metric`. The current configuration uses `overall_score`, a weighted validation metric that combines ROC AUC, balanced accuracy, F1, accuracy, and recall. Each model's probability threshold is tuned on the validation split using `training.decision_threshold_metric`, so API class predictions use the same threshold chosen during training without using the test set for model selection.

The best model is selected by validation ROC AUC; test metrics are reported after selection.
