"""
Tests for metrics functions and calibration logic.
No GPU or model download required.
"""

import numpy as np
import pytest

from deception_guardrail.probes.evaluate import compute_metrics
from deception_guardrail.probes.calibration import FPR_TARGETS, _threshold_for_fpr


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    metrics = compute_metrics(y_true, y_score)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_compute_metrics_returns_all_keys():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.4, 0.6, 0.3, 0.7])
    metrics = compute_metrics(y_true, y_score)
    required = [
        "accuracy", "auroc", "auprc", "f1", "precision", "recall",
        "confusion_matrix", "mean_score_deceptive", "mean_score_honest",
        "score_separation",
    ]
    for key in required:
        assert key in metrics, f"Missing key '{key}' in metrics"


def test_compute_metrics_random_chance():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_score = rng.uniform(0, 1, 200)
    metrics = compute_metrics(y_true, y_score)
    # AUROC should be near 0.5 for random scores
    assert 0.3 < metrics["auroc"] < 0.7


def test_score_separation_sign():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.3, 0.7, 0.8])
    metrics = compute_metrics(y_true, y_score)
    assert metrics["score_separation"] > 0, "Deceptive mean should be higher than honest mean"
    assert metrics["mean_score_deceptive"] > metrics["mean_score_honest"]


def test_threshold_for_fpr_1pct():
    # Create control scores: 99 low, 1 high => 1% FPR should be near the high one
    ctrl_scores = np.array([0.1] * 99 + [0.95])
    threshold = _threshold_for_fpr(ctrl_scores, 0.01)
    # At this threshold, approximately 1% of controls should score above it
    fpr = np.mean(ctrl_scores > threshold)
    assert fpr <= 0.02, f"FPR={fpr:.3f} too high for 1% target"


def test_threshold_for_fpr_monotone():
    rng = np.random.default_rng(42)
    ctrl_scores = rng.uniform(0, 1, 500)
    t_1 = _threshold_for_fpr(ctrl_scores, 0.01)
    t_5 = _threshold_for_fpr(ctrl_scores, 0.05)
    t_10 = _threshold_for_fpr(ctrl_scores, 0.10)
    # Higher FPR target → lower threshold
    assert t_1 >= t_5 >= t_10, (
        f"Thresholds not monotone: 1%={t_1:.3f}, 5%={t_5:.3f}, 10%={t_10:.3f}"
    )


def test_tpr_at_fixed_fpr_all_deceptive_above_threshold():
    """If deceptive scores are all very high and threshold is low, TPR should be 1."""
    ctrl_scores = np.array([0.1, 0.2, 0.15, 0.3])
    decep_scores = np.array([0.95, 0.97, 0.93])
    threshold = _threshold_for_fpr(ctrl_scores, 0.25)  # loose threshold
    tpr = float(np.mean(decep_scores > threshold))
    assert tpr == pytest.approx(1.0)


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.3, 0.7, 0.4, 0.8])
    metrics = compute_metrics(y_true, y_score)
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2


def test_metrics_balanced_dataset():
    n = 100
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))
    # Slightly better than random
    rng = np.random.default_rng(0)
    y_score = rng.beta(3, 1, n // 2).tolist() + rng.beta(1, 3, n // 2).tolist()
    y_score = 1 - np.array(y_score)  # flip so deceptive (label=1) has higher scores
    # We just check the function runs and returns valid values
    metrics = compute_metrics(y_true, y_score)
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
