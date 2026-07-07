# Telco Customer Churn MLOps Pipeline

Production-style MLOps project for customer churn prediction using the IBM Telco Customer Churn dataset. The project versions data, cleans and engineers features, trains multiple models, logs experiments with MLflow, registers the best model, and serves predictions through a Dockerized FastAPI service.

## Dataset

The source file is `Telco_customer_churn.xlsx`, copied into `data/raw/` by the pipeline. The target is `Churn Value`, where `1` means churned and `0` means stayed.

The pipeline creates three dataset versions:

- `data/v1/telco_churn_v1.csv`: raw Excel data loaded as-is.
- `data/v2/telco_churn_v2_clean_encoded.csv`: cleaned, imputed, and one-hot encoded data.
- `data/v3/telco_churn_v3_features.csv`: realistic domain features plus encoded modeling columns.

The saved v2 and v3 files are assignment artifacts. Model training loads v1, performs the train/validation/test split first, then fits imputation, feature engineering, one-hot encoding, and scaling inside sklearn pipelines during cross-validation, model selection, and the final train+validation refit.

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |-- registry/
|   |-- v1/
|   |-- v2/
|   `-- v3/
|-- src/
|   |-- api.py
|   |-- config.py
|   |-- data_loader.py
|   |-- data_registry.py
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
|-- tests/
|-- .github/workflows/ci.yml
|-- config.yaml
|-- Dockerfile
|-- requirements.lock
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

`requirements.txt` installs the exact package versions pinned in
`requirements.lock`, which is also used by Docker and CI.

Use a clean virtual environment for this project. Mixing conda base packages with user-site packages can produce binary incompatibilities in NumPy, SciPy, or scikit-learn.

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
- writes SHA-256 data registry manifests for each version
- performs cleaning and feature engineering
- creates stratified train, validation, and test splits
- fits model preprocessing inside sklearn pipelines after splitting to avoid leakage
- trains Logistic Regression, Random Forest, XGBoost, and CatBoost
- runs stratified K-fold multi-metric `GridSearchCV`
- evaluates accuracy, precision, recall, F1, ROC AUC, and confusion matrices
- tunes the classification decision threshold on the validation set for the configured objective
- refits the selected pipeline on train+validation before final test reporting and deployment
- logs runs, metrics, parameters, artifacts, model signatures, input examples, and model metadata to MLflow
- registers the best model by the configured validation selection metric
- saves a deployable MLflow model under `models/best_model`

Useful outputs:

- `artifacts/model_comparison.csv`
- `artifacts/best_model_summary.json`
- `models/best_model/model_metadata.json`
- `data/registry/manifest.json`
- `artifacts/confusion_matrices/`
- `artifacts/feature_importance/`

## Data Versioning

Generated data files stay out of Git, but every pipeline run writes
Git-trackable manifests under `data/registry/`. Each manifest records SHA-256
checksums, file sizes, source/parent artifacts, row and column counts, and
optional remote object storage URIs.

To map manifests to remote storage, set `DATA_REMOTE_URI` before running the
pipeline, or set `data_versioning.remote_uri` in `config.yaml`:

```bash
export DATA_REMOTE_URI=s3://your-bucket/telco-customer-churn
python run_pipeline.py --config config.yaml
```

Upload the ignored data files to the same relative paths under that remote URI.
The checksums in `data/registry/manifest.json` are the reproducibility contract
for restoring or auditing a data version.

## Tests and CI

Run the test suite locally:

```bash
python -m pytest -q
```

The suite covers preprocessing metadata reuse, feature engineering,
threshold optimization, strict API request validation, and a smoke run of
`run_pipeline.py --skip-mlflow` with a synthetic dataset. GitHub Actions runs
the same checks on pushes and pull requests to `main`.

## Docker Deployment

The Docker image is reproducible from a fresh GitHub clone. It copies only source
code and configuration; generated training outputs such as `models/`, `mlruns/`,
and `artifacts/` are intentionally excluded from the image context.

```bash
docker build -t telco-churn-api .
```

To serve a locally trained model, run the pipeline first, then mount the exported
MLflow model into the container:

```bash
python run_pipeline.py
docker run --rm -p 8000:8000 \
  -v "${PWD}/models/best_model:/app/models/best_model:ro" \
  telco-churn-api
```

On Windows PowerShell, use:

```powershell
python run_pipeline.py
docker run --rm -p 8000:8000 `
  -v "${PWD}\models\best_model:/app/models/best_model:ro" `
  telco-churn-api
```

To load from an MLflow registry or run URI instead of a mounted local model:

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_URI=models:/TelcoCustomerChurnBestModel@champion \
  telco-churn-api
```

The prediction service is available at `http://127.0.0.1:8000`. The API reads
the decision threshold from metadata packaged inside the MLflow model artifact.
Set `DECISION_THRESHOLD` only when you intentionally need an operational
override; otherwise the threshold, input schema, preprocessing metadata, and
feature schema travel with the model.

## Prediction API

Endpoint:

```text
POST /predict
```

Requests must use a JSON object with a non-empty `records` list. Each record
must include the required IBM Telco inference columns shown below. Unknown
fields, missing fields, invalid categories, and out-of-range numeric values
return FastAPI `422` validation errors.

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

The pipeline can select the best model with `training.model_selection_metric`. The current configuration uses `overall_score`, a weighted validation metric that combines ROC AUC, balanced accuracy, F1, accuracy, and recall. Hyperparameter search is also aligned to this objective through multi-metric `training.scoring` and `training.hyperparameter_refit_metric`. Validation ROC AUC is still logged and reported as a component metric, but it is not the configured selection or refit objective. Each model's probability threshold is tuned on the validation split using `training.decision_threshold_metric`, so API class predictions use the same threshold chosen during training without using the test set for model selection.

After validation-based model and threshold selection, the winning sklearn pipeline is cloned and refit on train+validation. Test metrics are reported from that final refit while the test split remains held out from preprocessing, hyperparameter search, threshold tuning, and model selection.
