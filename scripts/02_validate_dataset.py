#!/usr/bin/env python3
"""
Validate the dataset for structural integrity, split hygiene, and field correctness.
Exits with code 1 on failure.

Usage:
    python scripts/02_validate_dataset.py \
        --experiment-config configs/experiment.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.config import load_experiment_config
from deception_guardrail.data.validate_dataset import ValidationError, run_validation
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the deception probe dataset")
    p.add_argument("--experiment-config", default="configs/experiment.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)
    processed_dir = Path(cfg.paths["processed_data_dir"])

    try:
        run_validation(processed_dir, cfg.split_sizes)
        logger.info("Dataset validation PASSED.")
    except ValidationError as e:
        logger.error(f"Dataset validation FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
