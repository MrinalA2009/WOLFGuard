"""
Control-based calibration: compute FPR and TPR at fixed FPR thresholds.

For each layer, we:
1. Score benign controls with the trained probe.
2. Compute FPR at threshold 0.5 (default).
3. Find thresholds that achieve 1%, 5%, 10% FPR on controls.
4. Report TPR on held-out deceptive test samples at those thresholds.

High control FPR means the probe fires on benign prompts — it is detecting
prompt style, persuasion, roleplay, or fact-retrieval mode rather than
deception. This is the key validity check beyond raw AUROC.
"""

import numpy as np

from deception_guardrail.probes.evaluate import compute_scores
from deception_guardrail.probes.train import LayerProbeResult
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)

FPR_TARGETS = [0.01, 0.05, 0.10]


def _threshold_for_fpr(control_scores: np.ndarray, target_fpr: float) -> float:
    """Return the score threshold at which approximately `target_fpr` of controls score above it."""
    # threshold t such that P(score > t | benign) = target_fpr
    # i.e., t = (1 - target_fpr)-th percentile of control scores
    return float(np.percentile(control_scores, (1.0 - target_fpr) * 100))


def calibrate_layer(
    result: LayerProbeResult,
    control_activations: np.ndarray,
    deceptive_test_activations: np.ndarray,
    deceptive_test_labels: np.ndarray,
) -> dict:
    """
    Calibrate one layer probe against benign controls.

    Args:
        result: trained LayerProbeResult for this layer
        control_activations: [N_ctrl, D] activations for this layer
        deceptive_test_activations: [N_test, D] activations (deceptive only)
        deceptive_test_labels: [N_test] labels (should all be 1)

    Returns:
        dict with fpr metrics and tpr at fixed fpr
    """
    ctrl_scores = compute_scores(result.classifier, result.scaler, control_activations)
    decep_scores = compute_scores(result.classifier, result.scaler, deceptive_test_activations)

    fpr_at_0_5 = float(np.mean(ctrl_scores > 0.5))

    rows = {
        "layer_index": result.layer_index,
        "fpr_at_threshold_0_5": fpr_at_0_5,
        "n_controls": len(ctrl_scores),
        "n_deceptive_test": len(decep_scores),
        "mean_control_score": float(ctrl_scores.mean()),
        "mean_deceptive_score": float(decep_scores.mean()),
    }

    for target_fpr in FPR_TARGETS:
        threshold = _threshold_for_fpr(ctrl_scores, target_fpr)
        tpr = float(np.mean(decep_scores > threshold))
        pct = int(target_fpr * 100)
        rows[f"threshold_at_{pct}pct_fpr"] = float(threshold)
        rows[f"tpr_at_{pct}pct_fpr"] = tpr

    return rows


def calibrate_all_layers(
    results: list[LayerProbeResult],
    control_artifact: dict,
    test_artifact: dict,
) -> list[dict]:
    """Run calibration for all layers and return a list of row dicts."""
    ctrl_acts = control_artifact["activations"].numpy()  # [N_ctrl, L, D]
    test_acts = test_artifact["activations"].numpy()
    test_labels = np.array(test_artifact["labels"])

    deceptive_mask = test_labels == 1
    test_acts_deceptive = test_acts[deceptive_mask]
    test_labels_deceptive = test_labels[deceptive_mask]

    rows = []
    for result in results:
        ti = result.tensor_index
        ctrl_layer = ctrl_acts[:, ti, :]
        test_layer = test_acts_deceptive[:, ti, :]

        row = calibrate_layer(
            result,
            ctrl_layer,
            test_layer,
            test_labels_deceptive,
        )
        rows.append(row)
        logger.info(
            f"Layer {result.layer_index:02d} | "
            f"ctrl_fpr@0.5={row['fpr_at_threshold_0_5']:.4f} | "
            f"tpr@1%fpr={row['tpr_at_1pct_fpr']:.4f}"
        )

    return rows
