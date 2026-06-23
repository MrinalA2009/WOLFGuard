#!/usr/bin/env python3
"""
Train layer-wise logistic regression probes on captured activations.

Usage:
    python scripts/04_train_layerwise_probes.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.analysis.summaries import save_best_layer_json, save_layerwise_csv
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.train import (
    save_probes,
    select_best_layer,
    train_all_layers,
)
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train layer-wise probes")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)
    set_seed(exp_cfg.seed)

    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])

    train_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "train"))
    val_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "validation"))
    test_art = load_activations(activation_path(artifacts_dir, model_cfg.model_short_name, "test"))

    results = train_all_layers(
        train_art, val_art, test_art,
        c_grid=exp_cfg.c_grid,
        seed=exp_cfg.seed,
    )

    probes_path = artifacts_dir / "probes" / model_cfg.model_short_name / "layerwise_probes.pkl"
    save_probes(results, probes_path)

    best = select_best_layer(results)
    logger.info(
        f"Best layer: {best.layer_index} | "
        f"val_auroc={best.val_metrics['auroc']:.4f} | "
        f"test_auroc={best.test_metrics['auroc']:.4f}"
    )

    csv_path = Path(exp_cfg.metrics_output_paths["layerwise_csv"])
    save_layerwise_csv(results, csv_path)

    best_path = Path(exp_cfg.metrics_output_paths["best_layer_json"])
    save_best_layer_json(best, {}, best_path)


if __name__ == "__main__":
    main()
