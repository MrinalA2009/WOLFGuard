"""
Publication-quality figures for the layer-wise probe experiment.
One plot per file. No seaborn. Clean matplotlib only.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deception_guardrail.probes.train import LayerProbeResult
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)

DPI = 150
FIGSIZE = (8, 5)


def _layer_numbers(results: list[LayerProbeResult]) -> list[int]:
    return [r.layer_index for r in results]


def plot_layer_vs_auroc(
    results: list[LayerProbeResult],
    split: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = _layer_numbers(results)
    aurocs = [r.test_metrics["auroc"] if split == "test" else r.val_metrics["auroc"] for r in results]
    best_idx = int(np.argmax(aurocs))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(layers, aurocs, marker="o", linewidth=1.5, markersize=4, color="steelblue")
    ax.axvline(layers[best_idx], color="tomato", linestyle="--", linewidth=1.2,
               label=f"Best layer {layers[best_idx]} (AUROC={aurocs[best_idx]:.4f})")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Layer-wise AUROC ({split} set)")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_layer_vs_auprc(
    results: list[LayerProbeResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = _layer_numbers(results)
    auprcs = [r.test_metrics["auprc"] for r in results]
    best_idx = int(np.argmax(auprcs))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(layers, auprcs, marker="s", linewidth=1.5, markersize=4, color="darkorange")
    ax.axvline(layers[best_idx], color="tomato", linestyle="--", linewidth=1.2,
               label=f"Best layer {layers[best_idx]} (AUPRC={auprcs[best_idx]:.4f})")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("AUPRC")
    ax.set_title("Layer-wise AUPRC (test set)")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_layer_vs_control_fpr(
    calibration_rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [row["layer_index"] for row in calibration_rows]
    fprs = [row["fpr_at_threshold_0_5"] for row in calibration_rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(layers, fprs, color="mediumorchid", alpha=0.8, width=0.8)
    ax.axhline(0.05, color="tomato", linestyle="--", linewidth=1.2,
               label="5% FPR reference")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("False Positive Rate on Benign Controls")
    ax.set_title("Control FPR at threshold=0.5 by Layer")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_tpr_at_fixed_fpr(
    calibration_rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = [row["layer_index"] for row in calibration_rows]
    tpr_1 = [row["tpr_at_1pct_fpr"] for row in calibration_rows]
    tpr_5 = [row["tpr_at_5pct_fpr"] for row in calibration_rows]
    tpr_10 = [row["tpr_at_10pct_fpr"] for row in calibration_rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(layers, tpr_1, marker="o", linewidth=1.5, label="TPR @ 1% FPR", color="steelblue")
    ax.plot(layers, tpr_5, marker="s", linewidth=1.5, label="TPR @ 5% FPR", color="darkorange")
    ax.plot(layers, tpr_10, marker="^", linewidth=1.5, label="TPR @ 10% FPR", color="forestgreen")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("True Positive Rate (deceptive samples)")
    ax.set_title("TPR at Fixed FPR by Layer (calibrated on benign controls)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_score_distributions(
    best_result: LayerProbeResult,
    test_artifact: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from deception_guardrail.probes.evaluate import compute_scores

    acts = test_artifact["activations"].numpy()  # [N, L, D]
    labels = np.array(test_artifact["labels"])
    ti = best_result.tensor_index

    X = acts[:, ti, :]
    scores = compute_scores(best_result.classifier, best_result.scaler, X)

    honest_scores = scores[labels == 0]
    deceptive_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, 40)
    ax.hist(honest_scores, bins=bins, alpha=0.6, label="Honest (label=0)", color="steelblue", density=True)
    ax.hist(deceptive_scores, bins=bins, alpha=0.6, label="Deceptive (label=1)", color="tomato", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold=0.5")
    ax.set_xlabel("Probe Score (deceptive probability)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Score Distributions at Best Layer {best_result.layer_index} "
        f"(test AUROC={best_result.test_metrics['auroc']:.4f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_auroc_tpr_panel(
    results: list[LayerProbeResult],
    calibration_rows: list[dict],
    output_path: Path,
) -> None:
    """
    Two-panel figure:
      top   : val AUROC (dashed) and test AUROC (solid) by layer
      bottom : TPR@1%FPR by layer
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layers = _layer_numbers(results)
    val_aurocs = [r.val_metrics["auroc"] for r in results]
    test_aurocs = [r.test_metrics["auroc"] for r in results]
    best_idx = int(np.argmax(test_aurocs))

    cal_by_layer = {row["layer_index"]: row for row in calibration_rows}
    tpr_1 = [cal_by_layer.get(l, {}).get("tpr_at_1pct_fpr", 0.0) for l in layers]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_top.plot(layers, val_aurocs, marker="o", linewidth=1.5, markersize=4,
                color="steelblue", linestyle="--", label="Val AUROC")
    ax_top.plot(layers, test_aurocs, marker="o", linewidth=1.5, markersize=4,
                color="steelblue", label="Test AUROC")
    ax_top.axvline(layers[best_idx], color="tomato", linestyle="--", linewidth=1.2,
                   label=f"Best layer {layers[best_idx]}")
    ax_top.set_ylabel("AUROC")
    ax_top.set_title("Layer-wise AUROC and TPR@1%FPR")
    ax_top.set_ylim(0.4, 1.05)
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(layers, tpr_1, marker="^", linewidth=1.5, markersize=4,
                color="forestgreen", label="TPR @ 1% FPR")
    ax_bot.axvline(layers[best_idx], color="tomato", linestyle="--", linewidth=1.2)
    ax_bot.set_xlabel("Transformer Layer")
    ax_bot.set_ylabel("TPR @ 1% FPR")
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.legend()
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def make_all_plots(
    results: list[LayerProbeResult],
    calibration_rows: list[dict],
    best_result: LayerProbeResult,
    test_artifact: dict,
    plot_paths: dict,
) -> None:
    plot_layer_vs_auroc(
        results, "test", Path(plot_paths["layer_vs_test_auroc"])
    )
    plot_layer_vs_auroc(
        results, "validation", Path(plot_paths["layer_vs_validation_auroc"])
    )
    plot_layer_vs_auprc(results, Path(plot_paths["layer_vs_test_auprc"]))
    plot_layer_vs_control_fpr(calibration_rows, Path(plot_paths["layer_vs_control_fpr"]))
    plot_tpr_at_fixed_fpr(calibration_rows, Path(plot_paths["tpr_at_fixed_fpr"]))
    plot_score_distributions(best_result, test_artifact, Path(plot_paths["score_distributions"]))
    plot_auroc_tpr_panel(results, calibration_rows, Path(plot_paths["auroc_tpr_panel"]))
