"""Model training, validation, and best-model selection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import resolve_path
from src.evaluate import (
    compute_classification_metrics,
    optimize_decision_threshold,
    predict_positive_class_scores,
    save_confusion_matrix_artifact,
    save_feature_importance_artifact,
)
from src.features import add_domain_features
from src.preprocessing import clean_raw_dataframe
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
    cv_refit_metric: str
    cv_scores: dict[str, float]
    decision_threshold: float
    threshold_metric: str
    threshold_metric_score: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    execution_time_seconds: float
    training_timestamp: str
    artifact_paths: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None


class CleanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Fit cleaning metadata and domain features inside the sklearn pipeline."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> "CleanFeatureEngineer":
        """Fit split-local imputation metadata."""
        _, metadata = clean_raw_dataframe(
            pd.DataFrame(features).copy(),
            config=self.config,
            fit_metadata=True,
            include_target=False,
            log_progress=False,
        )
        self.metadata_ = metadata
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and engineer features using fitted metadata."""
        if not hasattr(self, "metadata_"):
            raise ValueError("CleanFeatureEngineer must be fitted before transform.")

        cleaned_features, _ = clean_raw_dataframe(
            pd.DataFrame(features).copy(),
            config=self.config,
            metadata=self.metadata_,
            fit_metadata=False,
            include_target=False,
            log_progress=False,
        )
        return add_domain_features(
            cleaned_features,
            self.config,
            log_progress=False,
        )


@dataclass(frozen=True)
class OverallScoreScorer:
    """Business-objective scorer for GridSearchCV."""

    weights: dict[str, float] | None = None

    def __call__(self, estimator: Any, features: pd.DataFrame, target: pd.Series) -> float:
        metrics = compute_classification_metrics(
            estimator,
            features,
            target,
            decision_threshold=0.5,
            overall_score_weights=self.weights,
        )
        return float(metrics["overall_score"])


def _to_float_array(features: Any) -> np.ndarray:
    """Convert mixed transformer output to a plain numeric array for estimators."""
    return np.asarray(features, dtype=np.float64)


def _scorer_from_name(name: str, config: dict[str, Any]) -> str | OverallScoreScorer:
    """Return a sklearn scorer by configured metric name."""
    supported_scorers = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }
    if name == "overall_score":
        return OverallScoreScorer(config["training"].get("overall_score_weights"))
    if name in supported_scorers:
        return supported_scorers[name]
    raise ValueError(f"Unsupported GridSearchCV scoring metric: {name}")


def _grid_search_scoring(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Build multi-metric scoring and the configured refit key."""
    scoring_config = config["training"].get("scoring", "roc_auc")

    if isinstance(scoring_config, str):
        metric_names = [scoring_config]
        scoring = {scoring_config: _scorer_from_name(scoring_config, config)}
    elif isinstance(scoring_config, list):
        metric_names = [str(metric_name) for metric_name in scoring_config]
        scoring = {
            metric_name: _scorer_from_name(metric_name, config)
            for metric_name in metric_names
        }
    elif isinstance(scoring_config, dict):
        metric_names = [str(metric_name) for metric_name in scoring_config]
        scoring = {
            str(metric_name): _scorer_from_name(str(scorer_name), config)
            for metric_name, scorer_name in scoring_config.items()
        }
    else:
        raise ValueError("training.scoring must be a metric name, list, or mapping.")

    refit_metric = str(
        config["training"].get(
            "hyperparameter_refit_metric",
            config["training"].get("model_selection_metric", metric_names[0]),
        )
    )
    if refit_metric not in scoring:
        scoring[refit_metric] = _scorer_from_name(refit_metric, config)

    return scoring, refit_metric


def _build_preprocessor(features: pd.DataFrame, config: dict[str, Any]) -> ColumnTransformer:
    """Build preprocessing for the engineered feature space."""
    feature_engineer = CleanFeatureEngineer(config)
    engineered_sample = feature_engineer.fit(features).transform(features)
    numeric_columns = [
        column
        for column in engineered_sample.columns
        if pd.api.types.is_numeric_dtype(engineered_sample[column])
    ]
    categorical_columns = [
        column for column in engineered_sample.columns if column not in numeric_columns
    ]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_pipeline(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    config: dict[str, Any],
) -> Pipeline:
    """Combine preprocessing and estimator into a sklearn pipeline."""
    return Pipeline(
        steps=[
            ("features", CleanFeatureEngineer(config)),
            ("preprocessor", _build_preprocessor(features, config)),
            ("to_float", FunctionTransformer(_to_float_array, validate=False)),
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


def _prepare_leakage_safe_modeling_data(
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    dict[str, Any],
    list[str],
]:
    """Split raw data first and leave learned preprocessing inside sklearn pipelines."""
    target_column = config["schema"]["target_column"]
    target = dataset[target_column].astype(int)
    raw_features = dataset.drop(columns=[target_column])

    x_train_raw, x_validation_raw, x_test_raw, y_train_raw, y_validation_raw, y_test_raw = (
        _split_dataset(raw_features, target, config)
    )

    feature_columns = list(x_train_raw.columns)
    preprocessing_metadata = {
        "fit_scope": "inside_sklearn_pipeline_after_split",
        "selection_fit_scope": "train_only_grid_search_cv",
        "final_model_fit_scope": "train_plus_validation_after_selection",
        "raw_feature_columns": feature_columns,
        "target_column": target_column,
        "dropped_columns": config["schema"]["drop_columns"],
        "training_rows": int(x_train_raw.shape[0]),
        "validation_rows": int(x_validation_raw.shape[0]),
        "final_training_rows": int(x_train_raw.shape[0] + x_validation_raw.shape[0]),
        "test_rows": int(x_test_raw.shape[0]),
    }

    write_json(config["artifacts"]["preprocessing_metadata"], preprocessing_metadata)
    write_json(config["artifacts"]["feature_columns"], feature_columns)
    LOGGER.info(
        "Prepared leakage-safe raw modeling splits: "
        "train=%s, validation=%s, test=%s.",
        x_train_raw.shape,
        x_validation_raw.shape,
        x_test_raw.shape,
    )
    return (
        x_train_raw,
        x_validation_raw,
        x_test_raw,
        y_train_raw,
        y_validation_raw,
        y_test_raw,
        preprocessing_metadata,
        feature_columns,
    )


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
    pipeline = _make_pipeline(spec.estimator, x_train, config)
    scoring, refit_metric = _grid_search_scoring(config)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=int(config["training"]["n_jobs"]),
        refit=refit_metric,
        verbose=0,
    )
    search.fit(x_train, y_train)

    best_estimator = search.best_estimator_
    cv_scores = {
        metric_name: float(search.cv_results_[f"mean_test_{metric_name}"][search.best_index_])
        for metric_name in scoring
    }
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
        cv_refit_metric=refit_metric,
        cv_scores=cv_scores,
        decision_threshold=decision_threshold,
        threshold_metric=threshold_metric,
        threshold_metric_score=threshold_metric_score,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        execution_time_seconds=float(execution_time),
        training_timestamp=timestamp,
        artifact_paths=artifact_paths,
    )


def _refit_selected_model_on_train_validation(
    best_result: ModelResult,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any],
) -> ModelResult:
    """Refit the selected pipeline on train+validation before deployment."""
    LOGGER.info(
        "Refitting selected model %s on train+validation before deployment.",
        best_result.model_name,
    )
    start_time = time.perf_counter()
    x_train_validation = pd.concat([x_train, x_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)

    final_estimator = clone(best_result.estimator)
    final_estimator.fit(x_train_validation, y_train_validation)
    best_result.estimator = final_estimator
    best_result.execution_time_seconds += float(time.perf_counter() - start_time)

    overall_score_weights = config["training"].get("overall_score_weights")
    best_result.test_metrics = compute_classification_metrics(
        final_estimator,
        x_test,
        y_test,
        decision_threshold=best_result.decision_threshold,
        overall_score_weights=overall_score_weights,
    )

    confusion_matrix_path = save_confusion_matrix_artifact(
        best_result.model_name,
        best_result.test_metrics["confusion_matrix"],
        config["artifacts"]["confusion_matrix_dir"],
    )
    feature_importance_path = save_feature_importance_artifact(
        best_result.model_name,
        final_estimator,
        list(x_train.columns),
        config["artifacts"]["feature_importance_dir"],
    )

    best_result.artifact_paths = {"confusion_matrix": str(confusion_matrix_path)}
    if feature_importance_path is not None:
        best_result.artifact_paths["feature_importance"] = str(feature_importance_path)

    LOGGER.info(
        "Refit %s on %s train+validation rows. final_test_roc_auc=%.4f",
        best_result.model_name,
        x_train_validation.shape[0],
        best_result.test_metrics["roc_auc"],
    )
    return best_result


def _comparison_table(results: list[ModelResult], config: dict[str, Any]) -> pd.DataFrame:
    """Build a concise model comparison table."""
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "model_name": result.model_name,
            "cv_refit_metric": result.cv_refit_metric,
            "cv_best_score": result.cv_best_score,
            "decision_threshold": result.decision_threshold,
            "threshold_metric": result.threshold_metric,
            "threshold_metric_score": result.threshold_metric_score,
            "execution_time_seconds": result.execution_time_seconds,
        }
        for metric_name, metric_value in result.cv_scores.items():
            row[f"cv_best_{metric_name}"] = metric_value
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

    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
        preprocessing_metadata,
        feature_columns,
    ) = _prepare_leakage_safe_modeling_data(dataset, config)

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

    best_result = _select_best_model(results, config)
    best_result = _refit_selected_model_on_train_validation(
        best_result=best_result,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        config=config,
    )

    comparison = _comparison_table(results, config)
    save_dataframe(comparison, config["artifacts"]["comparison_table"])

    selection_metric = config["training"].get("model_selection_metric", "roc_auc")
    summary = {
        "best_model": best_result.model_name,
        "best_params": best_result.best_params,
        "cv_refit_metric": best_result.cv_refit_metric,
        "cv_best_score": best_result.cv_best_score,
        "cv_scores": best_result.cv_scores,
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
        "preprocessing_fit_scope": preprocessing_metadata["fit_scope"],
        "selection_fit_scope": preprocessing_metadata["selection_fit_scope"],
        "final_model_fit_scope": preprocessing_metadata["final_model_fit_scope"],
        "final_model_training_rows": preprocessing_metadata["final_training_rows"],
        "test_metrics_scope": "final_refit_model_on_held_out_test",
        "feature_count": len(feature_columns),
        "comparison_table": str(resolve_path(config["artifacts"]["comparison_table"])),
    }
    write_json(config["artifacts"]["best_model_summary"], summary)
    LOGGER.info("Selected best model: %s.", best_result.model_name)
    return results, best_result
