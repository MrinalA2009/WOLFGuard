"""
Run summary: collects metrics into a structured JSON artifact and prints
a human-readable report.
"""

import datetime
import json
import platform
from pathlib import Path
from typing import Optional

import torch

from deception_guardrail.probes.train import LayerProbeResult
from deception_guardrail.utils.io import git_commit_hash
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


def _interpret_auroc(auroc: float) -> str:
    if auroc > 0.95:
        return "Very strong signal under controlled conditions."
    elif auroc > 0.85:
        return "Promising signal — worth investigating robustness."
    elif auroc > 0.65:
        return "Moderate signal — check probe design, token position, and data quality."
    else:
        return "Weak signal — check prompt templates, activation capture, and model loading."


def build_run_summary(
    run_id: str,
    model_name: str,
    model_short_name: str,
    exp_config_path: str,
    model_config_path: str,
    exp_config_hash: str,
    model_config_hash: str,
    best_result: LayerProbeResult,
    calibration_rows: list[dict],
    train_artifact: dict,
    val_artifact: dict,
    test_artifact: dict,
    control_artifact: dict,
) -> dict:
    best_cal = next(
        (r for r in calibration_rows if r["layer_index"] == best_result.layer_index),
        calibration_rows[0] if calibration_rows else {},
    )

    summary = {
        "run_id": run_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "model_name": model_name,
        "model_short_name": model_short_name,
        "experiment_config_path": exp_config_path,
        "model_config_path": model_config_path,
        "experiment_config_hash": exp_config_hash,
        "model_config_hash": model_config_hash,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "activation_shapes": {
            "train": list(train_artifact["activations"].shape),
            "validation": list(val_artifact["activations"].shape),
            "test": list(test_artifact["activations"].shape),
            "controls": list(control_artifact["activations"].shape),
        },
        "layer_indices": train_artifact.get("layer_indices", []),
        "token_position": train_artifact.get("token_position", "final_prompt_token"),
        "best_layer": {
            "layer_index": best_result.layer_index,
            "best_c": best_result.best_c,
            "val_auroc": best_result.val_metrics["auroc"],
            "val_auprc": best_result.val_metrics["auprc"],
            "test_auroc": best_result.test_metrics["auroc"],
            "test_auprc": best_result.test_metrics["auprc"],
            "test_accuracy": best_result.test_metrics["accuracy"],
            "test_f1": best_result.test_metrics["f1"],
            "control_fpr_at_0_5": best_cal.get("fpr_at_threshold_0_5"),
            "tpr_at_1pct_fpr": best_cal.get("tpr_at_1pct_fpr"),
            "tpr_at_5pct_fpr": best_cal.get("tpr_at_5pct_fpr"),
        },
        "interpretation": _interpret_auroc(best_result.test_metrics["auroc"]),
    }
    return summary


def save_run_summary(summary: dict, metadata_dir: Path) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    run_id = summary["run_id"]
    path = metadata_dir / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Run metadata saved → {path}")
    return path


def print_run_summary(summary: dict) -> None:
    bl = summary["best_layer"]
    lines = [
        "",
        "=" * 60,
        f"  RUN SUMMARY   {summary['run_id']}",
        "=" * 60,
        f"  Model        : {summary['model_name']}",
        f"  Timestamp    : {summary['timestamp']}",
        f"  Git commit   : {summary.get('git_commit', 'N/A')}",
        "",
        f"  Best layer   : {bl['layer_index']}",
        f"  Best C       : {bl['best_c']}",
        f"  Val AUROC    : {bl['val_auroc']:.4f}",
        f"  Test AUROC   : {bl['test_auroc']:.4f}",
        f"  Test AUPRC   : {bl['test_auprc']:.4f}",
        f"  Test Accuracy: {bl['test_accuracy']:.4f}",
        f"  Test F1      : {bl['test_f1']:.4f}",
        "",
        f"  Ctrl FPR@0.5 : {bl.get('control_fpr_at_0_5', 'N/A')}",
        f"  TPR@1%FPR    : {bl.get('tpr_at_1pct_fpr', 'N/A')}",
        f"  TPR@5%FPR    : {bl.get('tpr_at_5pct_fpr', 'N/A')}",
        "",
        f"  Interpretation: {summary['interpretation']}",
        "",
        "  IMPORTANT: This experiment does not prove that a model cannot",
        "  deceive. It tests whether direct factual deception, under controlled",
        "  prompt-pair conditions, is linearly detectable from internal activations.",
        "=" * 60,
        "",
    ]
    print("\n".join(lines))


def save_layerwise_csv(results: list[LayerProbeResult], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_index", "tensor_index", "best_c",
        "val_auroc", "val_auprc", "val_accuracy", "val_f1",
        "test_auroc", "test_auprc", "test_accuracy", "test_f1",
        "test_precision", "test_recall",
        "test_mean_score_deceptive", "test_mean_score_honest", "test_score_separation",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "layer_index": r.layer_index,
                "tensor_index": r.tensor_index,
                "best_c": r.best_c,
                "val_auroc": r.val_metrics["auroc"],
                "val_auprc": r.val_metrics["auprc"],
                "val_accuracy": r.val_metrics["accuracy"],
                "val_f1": r.val_metrics["f1"],
                "test_auroc": r.test_metrics["auroc"],
                "test_auprc": r.test_metrics["auprc"],
                "test_accuracy": r.test_metrics["accuracy"],
                "test_f1": r.test_metrics["f1"],
                "test_precision": r.test_metrics["precision"],
                "test_recall": r.test_metrics["recall"],
                "test_mean_score_deceptive": r.test_metrics["mean_score_deceptive"],
                "test_mean_score_honest": r.test_metrics["mean_score_honest"],
                "test_score_separation": r.test_metrics["score_separation"],
            })
    logger.info(f"Saved layerwise metrics CSV → {path}")


def save_best_layer_json(best_result: LayerProbeResult, best_cal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "layer_index": best_result.layer_index,
        "tensor_index": best_result.tensor_index,
        "best_c": best_result.best_c,
        "val_metrics": best_result.val_metrics,
        "test_metrics": best_result.test_metrics,
        "calibration": best_cal,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved best layer summary → {path}")


def save_calibration_csv(calibration_rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not calibration_rows:
        return
    fieldnames = list(calibration_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calibration_rows)
    logger.info(f"Saved calibration CSV → {path}")
