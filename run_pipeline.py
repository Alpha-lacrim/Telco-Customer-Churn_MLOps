"""Run the complete Telco churn MLOps pipeline from one command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.data_loader import load_and_version_raw_data
from src.features import build_feature_dataset
from src.logger import configure_logging
from src.mlflow_utils import log_results_to_mlflow
from src.preprocessing import preprocess_dataset
from src.train import train_models
from src.utils import ensure_project_directories

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Telco churn MLOps pipeline.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Run data and training steps without logging to MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate dataset versioning, training, evaluation, and model packaging."""
    args = parse_args()
    config = load_config(Path(args.config))
    configure_logging()
    ensure_project_directories(config)

    LOGGER.info("Starting Telco customer churn pipeline.")
    v1_path = load_and_version_raw_data(config)
    preprocess_dataset(config=config, input_path=v1_path)
    build_feature_dataset(config=config, input_path=v1_path)

    results, best_result = train_models(config)

    if args.skip_mlflow:
        LOGGER.warning("Skipping MLflow logging because --skip-mlflow was provided.")
    else:
        log_results_to_mlflow(config=config, results=results, best_result=best_result)

    LOGGER.info("Pipeline completed. Best model: %s", best_result.model_name)


if __name__ == "__main__":
    main()
