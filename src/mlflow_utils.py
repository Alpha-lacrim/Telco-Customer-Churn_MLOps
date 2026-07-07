"""MLflow experiment tracking and model registry integration."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from src.config import resolve_path
from src.train import ModelResult
from src.utils import flatten_dict, read_json, write_json

LOGGER = logging.getLogger(__name__)
MODEL_METADATA_FILENAME = "model_metadata.json"


def _resolve_tracking_uri(config: dict[str, Any]) -> str:
    """Resolve local tracking paths into MLflow-compatible file URIs."""
    configured_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    if "://" in configured_uri or configured_uri.startswith("databricks"):
        if configured_uri.startswith("file://"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        return configured_uri
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return resolve_path(configured_uri).as_uri()


def _log_params_safely(mlflow_module: Any, params: dict[str, Any]) -> None:
    """Log MLflow params after converting values to bounded strings."""
    for key, value in params.items():
        mlflow_module.log_param(key, str(value)[:500])


def _log_metrics(mlflow_module: Any, prefix: str, metrics: dict[str, Any]) -> None:
    """Log scalar metrics under a prefix."""
    for name, value in metrics.items():
        if name == "confusion_matrix":
            continue
        mlflow_module.log_metric(f"{prefix}_{name}", float(value))


def _read_json_if_exists(path: str | Path) -> Any:
    """Read a JSON file when present."""
    resolved_path = resolve_path(path)
    if not resolved_path.exists():
        return None
    return read_json(resolved_path)


def _input_schema_from_example(result: ModelResult) -> list[dict[str, str]]:
    """Describe the raw inference columns used by the model package."""
    return [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in result.input_example.dtypes.items()
    ]


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep scalar metrics suitable for model metadata."""
    return {
        name: float(value)
        for name, value in metrics.items()
        if name != "confusion_matrix"
    }


def _model_metadata(config: dict[str, Any], result: ModelResult) -> dict[str, Any]:
    """Create serving metadata that travels with the MLflow model artifact."""
    return {
        "model_name": result.model_name,
        "training_timestamp_utc": result.training_timestamp,
        "dataset_version": config["training"]["dataset_version"],
        "decision_threshold": float(result.decision_threshold),
        "threshold_metric": result.threshold_metric,
        "threshold_metric_score": float(result.threshold_metric_score),
        "cv_refit_metric": result.cv_refit_metric,
        "cv_scores": result.cv_scores,
        "validation_metrics": _scalar_metrics(result.validation_metrics),
        "test_metrics": _scalar_metrics(result.test_metrics),
        "input_columns": list(result.input_example.columns),
        "input_schema": _input_schema_from_example(result),
        "target_column": config["schema"]["target_column"],
        "drop_columns": config["schema"]["drop_columns"],
        "preprocessing_metadata": _read_json_if_exists(
            config["artifacts"]["preprocessing_metadata"]
        ),
        "feature_columns": _read_json_if_exists(config["artifacts"]["feature_columns"]),
    }


def _infer_signature(mlflow_module: Any, result: ModelResult) -> Any:
    """Infer a probability-output signature for MLflow model serving."""
    predictions = result.estimator.predict_proba(result.input_example)
    return mlflow_module.models.infer_signature(result.input_example, predictions)


def _log_single_result(
    mlflow_module: Any,
    config: dict[str, Any],
    result: ModelResult,
) -> str:
    """Log one trained model run and return its MLflow run id."""
    with mlflow_module.start_run(run_name=result.model_name) as run:
        mlflow_module.set_tag("model_name", result.model_name)
        mlflow_module.set_tag("training_timestamp_utc", result.training_timestamp)
        mlflow_module.log_param("dataset_version", config["training"]["dataset_version"])
        mlflow_module.log_param("model_name", result.model_name)
        mlflow_module.log_param("random_seed", config["training"]["random_seed"])
        mlflow_module.log_param("cv_splits", config["training"]["cv_splits"])
        mlflow_module.log_param("cv_refit_metric", result.cv_refit_metric)
        mlflow_module.log_param("decision_threshold", result.decision_threshold)
        mlflow_module.log_param("threshold_metric", result.threshold_metric)
        mlflow_module.log_metric("cv_best_score", result.cv_best_score)
        for metric_name, metric_value in result.cv_scores.items():
            mlflow_module.log_metric(f"cv_best_{metric_name}", float(metric_value))
        mlflow_module.log_metric(
            f"threshold_{result.threshold_metric}",
            result.threshold_metric_score,
        )
        mlflow_module.log_metric("execution_time_seconds", result.execution_time_seconds)

        _log_params_safely(mlflow_module, result.best_params)
        _log_metrics(mlflow_module, "validation", result.validation_metrics)
        _log_metrics(mlflow_module, "test", result.test_metrics)
        mlflow_module.log_dict(
            {"confusion_matrix": result.test_metrics["confusion_matrix"]},
            "confusion_matrix.json",
        )

        for artifact_path in result.artifact_paths.values():
            if Path(artifact_path).exists():
                mlflow_module.log_artifact(artifact_path)

        comparison_path = resolve_path(config["artifacts"]["comparison_table"])
        if comparison_path.exists():
            mlflow_module.log_artifact(str(comparison_path))

        flattened_config = flatten_dict(config)
        _log_params_safely(
            mlflow_module,
            {
                key: value
                for key, value in flattened_config.items()
                if key.startswith("training.") or key.startswith("mlflow.")
            },
        )

        model_metadata = _model_metadata(config, result)
        signature = _infer_signature(mlflow_module, result)
        mlflow_module.sklearn.log_model(
            result.estimator,
            artifact_path="model",
            signature=signature,
            input_example=result.input_example,
            metadata=model_metadata,
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow_module.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        mlflow_module.log_dict(model_metadata, f"model/{MODEL_METADATA_FILENAME}")
        LOGGER.info("Logged %s to MLflow run %s.", result.model_name, run.info.run_id)
        return run.info.run_id


def _register_best_model(
    mlflow_module: Any,
    config: dict[str, Any],
    best_result: ModelResult,
) -> dict[str, Any]:
    """Register the best MLflow model and assign a champion alias when supported."""
    if best_result.run_id is None:
        raise ValueError("Best result has no MLflow run id.")

    registered_model_name = config["mlflow"]["registered_model_name"]
    model_uri = f"runs:/{best_result.run_id}/model"
    model_version = mlflow_module.register_model(model_uri, registered_model_name)

    try:
        client = mlflow_module.tracking.MlflowClient()
        client.set_registered_model_alias(
            registered_model_name,
            "champion",
            model_version.version,
        )
    except Exception:
        LOGGER.warning("Could not set MLflow champion alias.", exc_info=True)

    return {
        "registered_model_name": registered_model_name,
        "registered_model_version": model_version.version,
        "model_uri": model_uri,
    }


def _save_local_mlflow_model(
    mlflow_module: Any,
    config: dict[str, Any],
    best_result: ModelResult,
) -> Path:
    """Save the selected model as a local MLflow model for Docker deployment."""
    model_path = resolve_path(config["deployment"]["model_uri"])
    if model_path.exists():
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_metadata = _model_metadata(config, best_result)
    signature = _infer_signature(mlflow_module, best_result)
    mlflow_module.sklearn.save_model(
        best_result.estimator,
        path=str(model_path),
        signature=signature,
        input_example=best_result.input_example,
        metadata=model_metadata,
        pyfunc_predict_fn="predict_proba",
        serialization_format=mlflow_module.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    write_json(model_path / MODEL_METADATA_FILENAME, model_metadata)
    LOGGER.info("Saved best model for deployment to %s.", model_path)
    return model_path


def log_results_to_mlflow(
    config: dict[str, Any],
    results: list[ModelResult],
    best_result: ModelResult,
) -> None:
    """Log all model runs, register the winner, and save deployment artifacts."""
    import mlflow
    import mlflow.sklearn

    tracking_uri = _resolve_tracking_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    LOGGER.info("MLflow tracking URI: %s.", tracking_uri)

    for result in results:
        result.run_id = _log_single_result(mlflow, config, result)

    registry_metadata = _register_best_model(mlflow, config, best_result)
    local_model_path = _save_local_mlflow_model(mlflow, config, best_result)

    summary_path = config["artifacts"]["best_model_summary"]
    summary = read_json(summary_path)
    summary.update(
        {
            "mlflow": registry_metadata,
            "deployment_model_path": str(local_model_path),
        }
    )
    write_json(summary_path, summary)
    LOGGER.info("Registered best model with MLflow.")
