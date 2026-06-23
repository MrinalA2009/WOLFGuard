#!/usr/bin/env python3
"""
Print and save a complete run summary.

Usage:
    python scripts/07_summarize_run.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml \
        --run-id run_001
"""

import argparse
import csv
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.analysis.summaries import (
    build_run_summary,
    print_run_summary,
    save_run_summary,
)
from deception_guardrail.config import config_hash, load_experiment_config, load_model_config
from deception_guardrail.probes.train import load_probes, select_best_layer
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize and save run metadata")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--run-id", default=None, help="Run ID (auto-generated if not provided)")
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

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"
    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])

    probes_path = artifacts_dir / "probes" / model_cfg.model_short_name / "layerwise_probes.pkl"
    results = load_probes(probes_path)

    cal_path = Path(exp_cfg.metrics_output_paths["control_calibration_csv"])
    calibration_rows = _load_calibration_rows(cal_path)

    control_fpr = {row["layer_index"]: row["fpr_at_threshold_0_5"] for row in calibration_rows}
    best = select_best_layer(results, control_fpr=control_fpr)

    train_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "train"))
    val_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "validation"))
    test_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "test"))
    ctrl_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "controls"))

    summary = build_run_summary(
        run_id=run_id,
        model_name=model_cfg.model_name,
        model_short_name=model_cfg.model_short_name,
        exp_config_path=args.experiment_config,
        model_config_path=args.model_config,
        exp_config_hash=config_hash(args.experiment_config),
        model_config_hash=config_hash(args.model_config),
        best_result=best,
        calibration_rows=calibration_rows,
        train_artifact=train_art,
        val_artifact=val_art,
        test_artifact=test_art,
        control_artifact=ctrl_art,
    )

    metadata_dir = Path(exp_cfg.paths["metadata_dir"])
    save_run_summary(summary, metadata_dir)
    print_run_summary(summary)


if __name__ == "__main__":
    main()
