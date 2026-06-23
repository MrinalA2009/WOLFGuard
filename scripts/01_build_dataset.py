#!/usr/bin/env python3
"""
Build the factual lie pair dataset and benign controls.

Usage:
    python scripts/01_build_dataset.py \
        --experiment-config configs/experiment.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.config import load_experiment_config
from deception_guardrail.data.build_dataset import run_build
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the deception probe dataset")
    p.add_argument("--experiment-config", default="configs/experiment.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)
    set_seed(cfg.seed)

    output_dir = Path(cfg.paths["processed_data_dir"])
    summary = run_build(cfg, output_dir)

    logger.info(f"Dataset built successfully.")
    logger.info(f"  Pairs: {summary['n_pairs']}")
    logger.info(f"  Probe samples: {summary['n_probe_samples']}")
    logger.info(f"  Controls: {summary['n_controls']}")
    logger.info(f"  Splits: {summary['split_counts']}")
    logger.info(f"  Domains: {summary['domain_counts']}")


if __name__ == "__main__":
    main()
