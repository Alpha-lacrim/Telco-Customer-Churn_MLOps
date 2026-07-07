from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_overall_score, optimize_decision_threshold


def test_compute_overall_score_uses_configured_weights():
    metrics = {"accuracy": 0.8, "recall": 0.6}
    score = compute_overall_score(metrics, {"accuracy": 0.25, "recall": 0.75})

    assert score == pytest.approx(0.65)


def test_optimize_decision_threshold_prefers_best_validation_metric():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    target = pd.Series([0, 0, 1, 1])

    threshold, score = optimize_decision_threshold(
        positive_scores=scores,
        target=target,
        metric="accuracy",
        minimum=0.3,
        maximum=0.7,
        step=0.1,
    )

    assert threshold == pytest.approx(0.5)
    assert score == pytest.approx(1.0)
