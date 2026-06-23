#!/usr/bin/env python3
"""
Generate all figures for the experiment.

Usage:
    python scripts/06_make_plots.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.analysis.plots import make_all_plots
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.train import load_probes, select_best_layer
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate experiment figures")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
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

    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])

    probes_path = artifacts_dir / "probes" / model_cfg.model_short_name / "layerwise_probes.pkl"
    results = load_probes(probes_path)

    cal_path = Path(exp_cfg.metrics_output_paths["control_calibration_csv"])
    calibration_rows = _load_calibration_rows(cal_path)

    control_fpr = {row["layer_index"]: row["fpr_at_threshold_0_5"] for row in calibration_rows}
    best = select_best_layer(results, control_fpr=control_fpr)

    test_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "test"))

    make_all_plots(
        results=results,
        calibration_rows=calibration_rows,
        best_result=best,
        test_artifact=test_art,
        plot_paths=exp_cfg.plot_output_paths,
    )

    logger.info("All plots generated.")


if __name__ == "__main__":
    main()
