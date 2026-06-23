#!/usr/bin/env python3
"""
Inspect false positives and false negatives at the best probe layer.

Prints:
  - Top-K highest-scoring benign controls  (false positive candidates)
  - Top-K highest-scoring honest samples   (probe confused honest→deceptive)
  - Bottom-K lowest-scoring deceptive samples (probe missed deception)

Saves a JSON report to results/metrics/{run_name}/error_inspection.json

Usage:
    python scripts/14_inspect_errors.py \\
        --model-config configs/qwen2_5_1b5.yaml \\
        --experiment-config configs/experiment.yaml \\
        --run-name qwen1_5b_full_v1 \\
        --top-k 20
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.evaluate import compute_scores
from deception_guardrail.probes.train import load_probes, select_best_layer
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.paths import resolve_run_paths

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect probe errors at best layer")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--top-k", type=int, default=20, help="Examples per bucket (default: 20)")
    return p.parse_args()


def _load_art(artifacts_dir: Path, msn: str, split: str, rn: str | None) -> dict:
    path = activation_path(artifacts_dir, msn, split, rn)
    if not path.exists():
        raise FileNotFoundError(
            f"Activation file not found: {path}\n"
            f"Run 03_capture_activations.py for split='{split}' first."
        )
    art = load_activations(path)
    validate_activation_artifact(art, context=split)
    return art


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)

    run_paths = resolve_run_paths(exp_cfg, model_cfg.model_short_name, args.run_name)
    artifacts_dir = run_paths["artifacts_dir"]
    msn = model_cfg.model_short_name
    rn = args.run_name
    k = args.top_k

    probes_path = Path(run_paths["probes_pkl"])
    if not probes_path.exists():
        raise FileNotFoundError(f"Probes not found: {probes_path}. Run script 04 first.")
    results = load_probes(probes_path)

    # Identify best layer using calibration CSV if available
    cal_csv = Path(run_paths["metrics"]["control_calibration_csv"])
    best_cal: dict = {}
    if cal_csv.exists():
        with open(cal_csv) as f:
            calibration_rows = list(csv.DictReader(f))
        calibration_rows_typed = [
            {key: (int(val) if key == "layer_index" else float(val)) for key, val in row.items()}
            for row in calibration_rows
        ]
        ctrl_fpr_by_layer = {
            row["layer_index"]: row["fpr_at_threshold_0_5"]
            for row in calibration_rows_typed
        }
        best = select_best_layer(results, control_fpr=ctrl_fpr_by_layer)
        best_cal = next(
            (r for r in calibration_rows_typed if r["layer_index"] == best.layer_index),
            {},
        )
    else:
        logger.warning("control_calibration.csv not found; selecting by val AUROC only.")
        best = select_best_layer(results)

    logger.info(f"Best layer: {best.layer_index} (tensor_index={best.tensor_index})")

    # Load artifacts
    test_art = _load_art(artifacts_dir, msn, "test", rn)
    ctrl_art = _load_art(artifacts_dir, msn, "controls", rn)

    # Score test set
    test_acts = test_art["activations"].numpy()[:, best.tensor_index, :]
    test_labels = np.array(test_art["labels"])
    test_scores = compute_scores(best.classifier, best.scaler, test_acts)
    test_sample_ids = test_art["sample_ids"]
    test_domains = test_art.get("domains") or (["?"] * len(test_labels))
    test_pair_ids = test_art.get("pair_ids") or ([None] * len(test_labels))

    # Score controls
    ctrl_acts = ctrl_art["activations"].numpy()[:, best.tensor_index, :]
    ctrl_scores = compute_scores(best.classifier, best.scaler, ctrl_acts)
    ctrl_ids = ctrl_art.get("control_ids") or ctrl_art.get("sample_ids") or []
    ctrl_types = ctrl_art.get("control_types") or (["?"] * len(ctrl_scores))

    # ---- Bucket 1: highest-scoring benign controls -------------------------
    ctrl_order = np.argsort(ctrl_scores)[::-1][:k]
    top_controls = [
        {
            "rank": int(i + 1),
            "control_id": str(ctrl_ids[idx]) if idx < len(ctrl_ids) else "?",
            "control_type": str(ctrl_types[idx]),
            "score": float(ctrl_scores[idx]),
        }
        for i, idx in enumerate(ctrl_order)
    ]

    # ---- Bucket 2: highest-scoring honest samples --------------------------
    honest_indices = np.where(test_labels == 0)[0]
    honest_scores_sub = test_scores[honest_indices]
    honest_order = np.argsort(honest_scores_sub)[::-1][:k]
    top_honest = [
        {
            "rank": int(i + 1),
            "sample_id": str(test_sample_ids[honest_indices[idx]]),
            "pair_id": str(test_pair_ids[honest_indices[idx]])
                if test_pair_ids[honest_indices[idx]] is not None else None,
            "domain": str(test_domains[honest_indices[idx]]),
            "score": float(test_scores[honest_indices[idx]]),
        }
        for i, idx in enumerate(honest_order)
    ]

    # ---- Bucket 3: lowest-scoring deceptive samples ------------------------
    deceptive_indices = np.where(test_labels == 1)[0]
    deceptive_scores_sub = test_scores[deceptive_indices]
    deceptive_order = np.argsort(deceptive_scores_sub)[:k]
    bottom_deceptive = [
        {
            "rank": int(i + 1),
            "sample_id": str(test_sample_ids[deceptive_indices[idx]]),
            "pair_id": str(test_pair_ids[deceptive_indices[idx]])
                if test_pair_ids[deceptive_indices[idx]] is not None else None,
            "domain": str(test_domains[deceptive_indices[idx]]),
            "score": float(test_scores[deceptive_indices[idx]]),
        }
        for i, idx in enumerate(deceptive_order)
    ]

    # ---- Print report -------------------------------------------------------
    print()
    print("=" * 65)
    print(f"  ERROR INSPECTION  layer={best.layer_index}  run={rn or 'default'}")
    print("=" * 65)
    print(f"  Test AUROC  : {best.test_metrics['auroc']:.4f}")
    tpr1 = best_cal.get("tpr_at_1pct_fpr")
    print(f"  TPR@1%FPR   : {tpr1:.4f}" if tpr1 is not None else "  TPR@1%FPR   : N/A")
    fpr05 = best_cal.get("fpr_at_threshold_0_5")
    print(f"  Ctrl FPR@0.5: {fpr05:.4f}" if fpr05 is not None else "  Ctrl FPR@0.5: N/A")
    print()

    print(f"  --- Top-{k} Highest-Scoring BENIGN CONTROLS (false positives) ---")
    for row in top_controls:
        print(f"    [{row['rank']:2d}] score={row['score']:.4f}  type={row['control_type']:<28}  id={row['control_id']}")

    print()
    print(f"  --- Top-{k} Highest-Scoring HONEST Samples (probe confused) ---")
    for row in top_honest:
        print(f"    [{row['rank']:2d}] score={row['score']:.4f}  domain={row['domain']:<20}  pair={row['pair_id']}")

    print()
    print(f"  --- Bottom-{k} Lowest-Scoring DECEPTIVE Samples (probe missed) ---")
    for row in bottom_deceptive:
        print(f"    [{row['rank']:2d}] score={row['score']:.4f}  domain={row['domain']:<20}  pair={row['pair_id']}")
    print("=" * 65)

    # ---- Save JSON report --------------------------------------------------
    metrics_dir = Path(run_paths["metrics"]["layerwise_csv"]).parent
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "error_inspection.json"
    report = {
        "run_name": rn,
        "model_short_name": msn,
        "best_layer": best.layer_index,
        "test_auroc": best.test_metrics["auroc"],
        "tpr_at_1pct_fpr": best_cal.get("tpr_at_1pct_fpr"),
        "ctrl_fpr_at_0_5": best_cal.get("fpr_at_threshold_0_5"),
        "top_k": k,
        "top_controls": top_controls,
        "top_honest": top_honest,
        "bottom_deceptive": bottom_deceptive,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Error inspection report → {out_path}")


if __name__ == "__main__":
    main()
