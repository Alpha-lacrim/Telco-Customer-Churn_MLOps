"""Model evaluation and artifact creation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import resolve_path
from src.utils import safe_name, save_dataframe

LOGGER = logging.getLogger(__name__)


DEFAULT_OVERALL_SCORE_WEIGHTS = {
    "roc_auc": 0.30,
    "balanced_accuracy": 0.25,
    "f1": 0.25,
    "accuracy": 0.10,
    "recall": 0.10,
}


def predict_positive_class_scores(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores for ROC AUC."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        return probabilities[:, 1]
    if hasattr(estimator, "decision_function"):
        decision_scores = estimator.decision_function(features)
        return 1.0 / (1.0 + np.exp(-decision_scores))
    return estimator.predict(features)


def predict_classes_from_scores(
    positive_scores: np.ndarray,
    decision_threshold: float,
) -> np.ndarray:
    """Convert positive-class scores into class predictions."""
    return (positive_scores >= decision_threshold).astype(int)


def compute_overall_score(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Combine normalized classification metrics into one balanced objective."""
    metric_weights = weights or DEFAULT_OVERALL_SCORE_WEIGHTS
    weight_sum = sum(float(weight) for weight in metric_weights.values())
    if weight_sum <= 0:
        raise ValueError("Overall score weights must sum to a positive value.")

    score = 0.0
    for metric_name, weight in metric_weights.items():
        if metric_name not in metrics:
            raise ValueError(f"Metric '{metric_name}' is not available for overall scoring.")
        score += float(weight) * float(metrics[metric_name])
    return score / weight_sum


def _metrics_from_predictions(
    target: pd.Series,
    predictions: np.ndarray,
    positive_scores: np.ndarray,
    overall_score_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute scalar metrics from predictions and positive-class scores."""
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, positive_scores)),
    }
    metrics["overall_score"] = compute_overall_score(metrics, overall_score_weights)
    return metrics


def optimize_decision_threshold(
    positive_scores: np.ndarray,
    target: pd.Series,
    metric: str = "accuracy",
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.005,
    overall_score_weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Choose a decision threshold on validation data without using the test set."""
    if step <= 0:
        raise ValueError("Threshold search step must be positive.")

    thresholds = np.arange(minimum, maximum + step / 2, step)
    best_threshold = 0.5
    best_score = -np.inf
    best_candidate = (-np.inf, -np.inf, -np.inf)

    for threshold in thresholds:
        predictions = predict_classes_from_scores(positive_scores, float(threshold))
        metrics = _metrics_from_predictions(
            target,
            predictions,
            positive_scores,
            overall_score_weights,
        )
        if metric not in metrics:
            raise ValueError(f"Unsupported threshold optimization metric: {metric}")

        score = metrics[metric]
        tie_breaker = metrics["f1"] if metric != "f1" else metrics["balanced_accuracy"]
        candidate = (float(score), float(tie_breaker), -abs(float(threshold) - 0.5))
        if candidate > best_candidate:
            best_threshold = float(threshold)
            best_score = float(score)
            best_candidate = candidate

    return best_threshold, best_score


def compute_classification_metrics(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    decision_threshold: float = 0.5,
    overall_score_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics for a fitted estimator."""
    positive_scores = predict_positive_class_scores(estimator, features)
    predictions = predict_classes_from_scores(positive_scores, decision_threshold)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])

    metrics = _metrics_from_predictions(
        target,
        predictions,
        positive_scores,
        overall_score_weights,
    )
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def save_confusion_matrix_artifact(
    model_name: str,
    matrix: list[list[int]],
    output_dir: str | Path,
) -> Path:
    """Save a model confusion matrix as a CSV artifact."""
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=["actual_stayed", "actual_churned"],
        columns=["predicted_stayed", "predicted_churned"],
    )
    output_path = resolve_path(output_dir) / f"{safe_name(model_name)}_confusion_matrix.csv"
    return save_dataframe(matrix_dataframe, output_path, index=True)


def _pipeline_feature_names(estimator: Any, original_columns: list[str]) -> list[str]:
    """Extract post-preprocessing feature names from a sklearn pipeline."""
    preprocessor = estimator.named_steps.get("preprocessor") if hasattr(estimator, "named_steps") else None
    if preprocessor is None:
        return original_columns
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        LOGGER.debug("Could not read feature names from preprocessor.", exc_info=True)
        return original_columns


def extract_feature_importance(
    estimator: Any,
    original_columns: list[str],
) -> pd.DataFrame | None:
    """Extract feature importance or coefficient magnitude when available."""
    classifier = estimator.named_steps.get("classifier") if hasattr(estimator, "named_steps") else estimator
    feature_names = _pipeline_feature_names(estimator, original_columns)

    if hasattr(classifier, "feature_importances_"):
        importance_values = np.asarray(classifier.feature_importances_, dtype=float)
        value_column = "importance"
    elif hasattr(classifier, "coef_"):
        importance_values = np.abs(np.asarray(classifier.coef_).reshape(-1))
        value_column = "absolute_coefficient"
    else:
        return None

    if len(feature_names) != len(importance_values):
        feature_names = original_columns[: len(importance_values)]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            value_column: importance_values,
        }
    ).sort_values(value_column, ascending=False)
    return importance


def save_feature_importance_artifact(
    model_name: str,
    estimator: Any,
    original_columns: list[str],
    output_dir: str | Path,
) -> Path | None:
    """Save feature importance if the estimator exposes it."""
    importance = extract_feature_importance(estimator, original_columns)
    if importance is None:
        LOGGER.info("No feature importance available for %s.", model_name)
        return None
    output_path = resolve_path(output_dir) / f"{safe_name(model_name)}_feature_importance.csv"
    return save_dataframe(importance, output_path, index=False)
