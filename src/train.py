"""Model training, validation, and best-model selection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import resolve_path
from src.evaluate import (
    compute_classification_metrics,
    optimize_decision_threshold,
    predict_positive_class_scores,
    save_confusion_matrix_artifact,
    save_feature_importance_artifact,
)
from src.utils import load_dataframe, save_dataframe, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Training specification for one model family."""

    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


@dataclass
class ModelResult:
    """Collected outputs for a trained model."""

    model_name: str
    estimator: Pipeline
    best_params: dict[str, Any]
    cv_best_score: float
    decision_threshold: float
    threshold_metric: str
    threshold_metric_score: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    execution_time_seconds: float
    training_timestamp: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-safe preprocessing step for already encoded features."""
    binary_columns = [
        column for column in features.columns if features[column].dropna().nunique() <= 2
    ]
    continuous_columns = [column for column in features.columns if column not in binary_columns]

    transformers: list[tuple[str, Any, list[str]]] = []
    if continuous_columns:
        transformers.append(("scaler", StandardScaler(), continuous_columns))
    if binary_columns:
        transformers.append(("binary", "passthrough", binary_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_pipeline(estimator: BaseEstimator, features: pd.DataFrame) -> Pipeline:
    """Combine preprocessing and estimator into a sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(features)),
            ("classifier", estimator),
        ]
    )


def _positive_class_weight(target: pd.Series) -> float:
    """Calculate the XGBoost scale_pos_weight value from training labels."""
    counts = target.value_counts()
    positives = float(counts.get(1, 1.0))
    negatives = float(counts.get(0, 1.0))
    return negatives / max(positives, 1.0)


def _resolve_param_grid(
    param_grid: dict[str, list[Any]],
    positive_class_weight: float,
) -> dict[str, list[Any]]:
    """Resolve config shortcuts in model parameter grids."""
    resolved: dict[str, list[Any]] = {}
    for key, values in param_grid.items():
        resolved_values: list[Any] = []
        for value in values:
            if key.endswith("scale_pos_weight") and value == "auto":
                resolved_values.append(positive_class_weight)
            else:
                resolved_values.append(value)
        resolved[key] = resolved_values
    return resolved


def _make_model_specs(
    config: dict[str, Any],
    training_target: pd.Series,
) -> list[ModelSpec]:
    """Instantiate configured model families."""
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    random_seed = int(config["training"]["random_seed"])
    n_jobs = int(config["training"]["n_jobs"])
    model_config = config["models"]
    positive_class_weight = _positive_class_weight(training_target)

    specs: list[ModelSpec] = []

    if model_config["logistic_regression"]["enabled"]:
        specs.append(
            ModelSpec(
                name="Logistic Regression",
                estimator=LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_seed,
                ),
                param_grid=_resolve_param_grid(
                    model_config["logistic_regression"]["param_grid"],
                    positive_class_weight,
                ),
            )
        )

    if model_config["random_forest"]["enabled"]:
        specs.append(
            ModelSpec(
                name="Random Forest",
                estimator=RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=n_jobs,
                    random_state=random_seed,
                ),
                param_grid=_resolve_param_grid(
                    model_config["random_forest"]["param_grid"],
                    positive_class_weight,
                ),
            )
        )

    if model_config["xgboost"]["enabled"]:
        scale_pos_weight = (
            positive_class_weight
            if model_config["xgboost"].get("use_scale_pos_weight", False)
            else 1.0
        )
        specs.append(
            ModelSpec(
                name="XGBoost",
                estimator=XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    n_jobs=n_jobs,
                    random_state=random_seed,
                    scale_pos_weight=scale_pos_weight,
                    tree_method="hist",
                ),
                param_grid=_resolve_param_grid(
                    model_config["xgboost"]["param_grid"],
                    positive_class_weight,
                ),
            )
        )

    if model_config["catboost"]["enabled"]:
        specs.append(
            ModelSpec(
                name="CatBoost",
                estimator=CatBoostClassifier(
                    allow_writing_files=False,
                    auto_class_weights="Balanced",
                    eval_metric="AUC",
                    loss_function="Logloss",
                    random_seed=random_seed,
                    verbose=False,
                ),
                param_grid=_resolve_param_grid(
                    model_config["catboost"]["param_grid"],
                    positive_class_weight,
                ),
            )
        )

    return specs


def _split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train, validation, and test splits."""
    random_seed = int(config["training"]["random_seed"])
    test_size = float(config["training"]["test_size"])
    validation_size = float(config["training"]["validation_size"])

    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_seed,
        stratify=target,
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=relative_validation_size,
        random_state=random_seed,
        stratify=y_train_validation,
    )

    LOGGER.info(
        "Created stratified splits: train=%s, validation=%s, test=%s.",
        x_train.shape,
        x_validation.shape,
        x_test.shape,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def _train_single_model(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
) -> ModelResult:
    """Run CV hyperparameter search and evaluate one model."""
    LOGGER.info("Training %s.", spec.name)
    start_time = time.perf_counter()
    random_seed = int(config["training"]["random_seed"])
    cv = StratifiedKFold(
        n_splits=int(config["training"]["cv_splits"]),
        shuffle=True,
        random_state=random_seed,
    )
    pipeline = _make_pipeline(spec.estimator, x_train)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid,
        scoring=config["training"]["scoring"],
        cv=cv,
        n_jobs=int(config["training"]["n_jobs"]),
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)

    best_estimator = search.best_estimator_
    threshold_config = config["training"]["decision_threshold_search"]
    threshold_metric = config["training"]["decision_threshold_metric"]
    overall_score_weights = config["training"].get("overall_score_weights")
    validation_scores = predict_positive_class_scores(best_estimator, x_validation)
    decision_threshold, threshold_metric_score = optimize_decision_threshold(
        positive_scores=validation_scores,
        target=y_validation,
        metric=threshold_metric,
        minimum=float(threshold_config["min"]),
        maximum=float(threshold_config["max"]),
        step=float(threshold_config["step"]),
        overall_score_weights=overall_score_weights,
    )
    validation_metrics = compute_classification_metrics(
        best_estimator,
        x_validation,
        y_validation,
        decision_threshold=decision_threshold,
        overall_score_weights=overall_score_weights,
    )
    test_metrics = compute_classification_metrics(
        best_estimator,
        x_test,
        y_test,
        decision_threshold=decision_threshold,
        overall_score_weights=overall_score_weights,
    )

    execution_time = time.perf_counter() - start_time
    timestamp = datetime.now(timezone.utc).isoformat()

    confusion_matrix_path = save_confusion_matrix_artifact(
        spec.name,
        test_metrics["confusion_matrix"],
        config["artifacts"]["confusion_matrix_dir"],
    )
    feature_importance_path = save_feature_importance_artifact(
        spec.name,
        best_estimator,
        list(x_train.columns),
        config["artifacts"]["feature_importance_dir"],
    )

    artifact_paths = {"confusion_matrix": str(confusion_matrix_path)}
    if feature_importance_path is not None:
        artifact_paths["feature_importance"] = str(feature_importance_path)

    LOGGER.info(
        "Finished %s. validation_roc_auc=%.4f test_roc_auc=%.4f",
        spec.name,
        validation_metrics["roc_auc"],
        test_metrics["roc_auc"],
    )

    return ModelResult(
        model_name=spec.name,
        estimator=best_estimator,
        best_params=dict(search.best_params_),
        cv_best_score=float(search.best_score_),
        decision_threshold=decision_threshold,
        threshold_metric=threshold_metric,
        threshold_metric_score=threshold_metric_score,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        execution_time_seconds=float(execution_time),
        training_timestamp=timestamp,
        artifact_paths=artifact_paths,
    )


def _comparison_table(results: list[ModelResult], config: dict[str, Any]) -> pd.DataFrame:
    """Build a concise model comparison table."""
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "model_name": result.model_name,
            "cv_best_roc_auc": result.cv_best_score,
            "decision_threshold": result.decision_threshold,
            "threshold_metric": result.threshold_metric,
            "threshold_metric_score": result.threshold_metric_score,
            "execution_time_seconds": result.execution_time_seconds,
        }
        for metric_name, metric_value in result.validation_metrics.items():
            if metric_name != "confusion_matrix":
                row[f"validation_{metric_name}"] = metric_value
        for metric_name, metric_value in result.test_metrics.items():
            if metric_name != "confusion_matrix":
                row[f"test_{metric_name}"] = metric_value
        rows.append(row)

    comparison = pd.DataFrame(rows)
    selection_metric = config["training"].get("model_selection_metric", "roc_auc")
    sort_column = f"validation_{selection_metric}"
    if sort_column not in comparison.columns:
        sort_column = "validation_roc_auc"
    return comparison.sort_values(sort_column, ascending=False)


def _select_best_model(results: list[ModelResult], config: dict[str, Any]) -> ModelResult:
    """Select the best model using the configured validation metric."""
    if not results:
        raise ValueError("No model results were produced.")
    selection_metric = config["training"].get("model_selection_metric", "roc_auc")
    return max(
        results,
        key=lambda result: (
            result.validation_metrics[selection_metric],
            result.validation_metrics["roc_auc"],
            result.validation_metrics["f1"],
        ),
    )


def train_models(config: dict[str, Any]) -> tuple[list[ModelResult], ModelResult]:
    """Train all configured models and persist evaluation artifacts."""
    LOGGER.info("Training configured model families.")
    dataset_version = config["training"]["dataset_version"]
    dataset_path = config["data"]["versions"][dataset_version]["path"]
    dataset = load_dataframe(dataset_path)

    target_column = config["schema"]["target_column"]
    target = dataset[target_column].astype(int)
    features = dataset.drop(columns=[target_column])

    x_train, x_validation, x_test, y_train, y_validation, y_test = _split_dataset(
        features,
        target,
        config,
    )

    specs = _make_model_specs(config, y_train)
    results = [
        _train_single_model(
            spec=spec,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            x_test=x_test,
            y_test=y_test,
            config=config,
        )
        for spec in specs
    ]

    comparison = _comparison_table(results, config)
    save_dataframe(comparison, config["artifacts"]["comparison_table"])

    best_result = _select_best_model(results, config)
    selection_metric = config["training"].get("model_selection_metric", "roc_auc")
    summary = {
        "best_model": best_result.model_name,
        "best_params": best_result.best_params,
        "selection_metric": f"validation_{selection_metric}",
        "selection_score": best_result.validation_metrics[selection_metric],
        "decision_threshold": best_result.decision_threshold,
        "threshold_metric": best_result.threshold_metric,
        "threshold_metric_score": best_result.threshold_metric_score,
        "overall_score_weights": config["training"].get("overall_score_weights"),
        "validation_roc_auc": best_result.validation_metrics["roc_auc"],
        "validation_overall_score": best_result.validation_metrics["overall_score"],
        "validation_balanced_accuracy": best_result.validation_metrics["balanced_accuracy"],
        "validation_accuracy": best_result.validation_metrics["accuracy"],
        "validation_precision": best_result.validation_metrics["precision"],
        "validation_recall": best_result.validation_metrics["recall"],
        "validation_f1": best_result.validation_metrics["f1"],
        "test_roc_auc": best_result.test_metrics["roc_auc"],
        "test_overall_score": best_result.test_metrics["overall_score"],
        "test_balanced_accuracy": best_result.test_metrics["balanced_accuracy"],
        "test_accuracy": best_result.test_metrics["accuracy"],
        "test_precision": best_result.test_metrics["precision"],
        "test_recall": best_result.test_metrics["recall"],
        "test_f1": best_result.test_metrics["f1"],
        "dataset_version": dataset_version,
        "comparison_table": str(resolve_path(config["artifacts"]["comparison_table"])),
    }
    write_json(config["artifacts"]["best_model_summary"], summary)
    LOGGER.info("Selected best model: %s.", best_result.model_name)
    return results, best_result
