#!/usr/bin/env python3
"""
Score benign controls and compute calibrated TPR at fixed FPR.

Usage:
    python scripts/05_evaluate_controls.py \\
        --model-config configs/qwen2_5_7b.yaml \\
        --experiment-config configs/experiment.yaml \\
        --run-name qwen_pilot_32
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.analysis.summaries import save_best_layer_json, save_calibration_csv
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.calibration import calibrate_all_layers
from deception_guardrail.probes.train import load_probes, select_best_layer
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.paths import resolve_run_paths

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control calibration and TPR at fixed FPR")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)

    run_paths = resolve_run_paths(exp_cfg, model_cfg.model_short_name, args.run_name)
    artifacts_dir = run_paths["artifacts_dir"]
    msn = model_cfg.model_short_name
    rn = args.run_name

    probes_path = Path(run_paths["probes_pkl"])
    if not probes_path.exists():
        raise FileNotFoundError(
            f"Probes file not found: {probes_path}\n"
            "Run 04_train_layerwise_probes.py first."
        )
    results = load_probes(probes_path)

    def _load(split: str) -> dict:
        path = activation_path(artifacts_dir, msn, split, rn)
        if not path.exists():
            raise FileNotFoundError(
                f"Activation file not found: {path}\n"
                f"Run 03_capture_activations.py for split='{split}'"
                + (f" --run-name {rn}" if rn else "") + " first."
            )
        art = load_activations(path)
        validate_activation_artifact(art, context=split)
        return art

    control_art = _load("controls")
    test_art = _load("test")

    calibration_rows = calibrate_all_layers(results, control_art, test_art)

    save_calibration_csv(calibration_rows, Path(run_paths["metrics"]["control_calibration_csv"]))

    control_fpr = {row["layer_index"]: row["fpr_at_threshold_0_5"] for row in calibration_rows}
    best = select_best_layer(results, control_fpr=control_fpr)
    best_cal = next(r for r in calibration_rows if r["layer_index"] == best.layer_index)

    save_best_layer_json(best, best_cal, Path(run_paths["metrics"]["best_layer_json"]))

    logger.info(
        f"Best layer (post-calibration): {best.layer_index} | "
        f"ctrl_fpr@0.5={best_cal['fpr_at_threshold_0_5']:.4f} | "
        f"tpr@1%fpr={best_cal['tpr_at_1pct_fpr']:.4f} | "
        f"tpr@5%fpr={best_cal['tpr_at_5pct_fpr']:.4f}"
    )


if __name__ == "__main__":
    main()
