#!/usr/bin/env python3
"""
Score benign controls and compute calibrated TPR at fixed FPR.

Usage:
    python scripts/05_evaluate_controls.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.analysis.summaries import (
    save_best_layer_json,
    save_calibration_csv,
)
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.calibration import calibrate_all_layers
from deception_guardrail.probes.train import load_probes, select_best_layer
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control calibration and TPR at fixed FPR")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)

    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])

    probes_path = artifacts_dir / "probes" / model_cfg.model_short_name / "layerwise_probes.pkl"
    results = load_probes(probes_path)

    control_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "controls"))
    test_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "test"))

    calibration_rows = calibrate_all_layers(results, control_art, test_art)

    cal_path = Path(exp_cfg.metrics_output_paths["control_calibration_csv"])
    save_calibration_csv(calibration_rows, cal_path)

    # Update best_layer_json with calibration data
    control_fpr = {row["layer_index"]: row["fpr_at_threshold_0_5"] for row in calibration_rows}
    best = select_best_layer(results, control_fpr=control_fpr)
    best_cal = next(r for r in calibration_rows if r["layer_index"] == best.layer_index)

    best_path = Path(exp_cfg.metrics_output_paths["best_layer_json"])
    save_best_layer_json(best, best_cal, best_path)

    logger.info(
        f"Best layer (after calibration): {best.layer_index} | "
        f"ctrl_fpr@0.5={best_cal['fpr_at_threshold_0_5']:.4f} | "
        f"tpr@1%fpr={best_cal['tpr_at_1pct_fpr']:.4f}"
    )


if __name__ == "__main__":
    main()
