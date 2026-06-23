"""
Metrics computation for layer-wise probes.

All metrics are computed from (y_true, y_score) where:
    y_score = predicted probability of the deceptive class (label=1)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Compute classification metrics from true labels and predicted scores.

    Args:
        y_true: binary labels (0=honest, 1=deceptive)
        y_score: predicted probability of the deceptive class

    Returns:
        dict with keys: accuracy, auroc, auprc, f1, precision, recall,
                        confusion_matrix, mean_score_deceptive, mean_score_honest,
                        score_separation
    """
    y_pred = (y_score >= 0.5).astype(int)

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    deceptive_mask = y_true == 1
    honest_mask = y_true == 0
    mean_score_deceptive = float(y_score[deceptive_mask].mean()) if deceptive_mask.any() else float("nan")
    mean_score_honest = float(y_score[honest_mask].mean()) if honest_mask.any() else float("nan")
    score_separation = mean_score_deceptive - mean_score_honest

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "mean_score_deceptive": mean_score_deceptive,
        "mean_score_honest": mean_score_honest,
        "score_separation": score_separation,
    }


def compute_scores(
    clf,
    scaler,
    X: np.ndarray,
) -> np.ndarray:
    """Apply scaler and return deceptive-class probability scores."""
    X_scaled = scaler.transform(X)
    return clf.predict_proba(X_scaled)[:, 1]
