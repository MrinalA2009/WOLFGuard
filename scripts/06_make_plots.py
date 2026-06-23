#!/usr/bin/env python3
"""
Generate all figures for the experiment.

Usage:
    python scripts/06_make_plots.py \\
        --model-config configs/qwen2_5_7b.yaml \\
        --experiment-config configs/experiment.yaml \\
        --run-name qwen_pilot_32
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.analysis.plots import make_all_plots
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.train import load_probes, select_best_layer
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.paths import resolve_run_paths

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate experiment figures")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def _load_calibration_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed = {}
            for k, v in row.items():
                try:
                    typed[k] = int(v) if k == "layer_index" else float(v)
                except ValueError:
                    typed[k] = v
            rows.append(typed)
    return rows


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
        raise FileNotFoundError(f"Probes not found: {probes_path}. Run script 04 first.")
    results = load_probes(probes_path)

    cal_path = Path(run_paths["metrics"]["control_calibration_csv"])
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {cal_path}. Run script 05 first.")
    calibration_rows = _load_calibration_rows(cal_path)

    control_fpr = {row["layer_index"]: row["fpr_at_threshold_0_5"] for row in calibration_rows}
    best = select_best_layer(results, control_fpr=control_fpr)

    test_path = activation_path(artifacts_dir, msn, "test", rn)
    if not test_path.exists():
        raise FileNotFoundError(f"Test activations not found: {test_path}. Run script 03 first.")
    test_art = load_activations(test_path)
    validate_activation_artifact(test_art, context="test")

    make_all_plots(
        results=results,
        calibration_rows=calibration_rows,
        best_result=best,
        test_artifact=test_art,
        plot_paths=run_paths["plots"],
    )

    logger.info(f"All plots generated → {Path(list(run_paths['plots'].values())[0]).parent}")


if __name__ == "__main__":
    main()
