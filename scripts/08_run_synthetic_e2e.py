#!/usr/bin/env python3
"""
Synthetic end-to-end integration test.

Generates synthetic activation artifacts with a known deception signal injected
at layers 3, 4, and 5 (0-indexed), then runs the full pipeline:
    activation capture (synthetic) → probe training → calibration → plots → summary

Purpose: validate the entire pipeline (probe training, calibration, plotting,
summaries, file I/O, metric calculations) without running any HuggingFace model.

Expected results:
  - Best validation AUROC > 0.95
  - Best test AUROC > 0.95
  - Best layer is in [3, 4, 5]
  - TPR at 5% benign-control FPR > 0.90

Usage:
    python scripts/08_run_synthetic_e2e.py --experiment-config configs/experiment.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from deception_guardrail.activations.store import activation_path, load_activations, save_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.analysis.plots import make_all_plots
from deception_guardrail.analysis.summaries import (
    build_run_summary,
    print_run_summary,
    save_best_layer_json,
    save_calibration_csv,
    save_layerwise_csv,
    save_run_summary,
)
from deception_guardrail.config import load_experiment_config
from deception_guardrail.probes.calibration import calibrate_all_layers
from deception_guardrail.probes.train import (
    load_probes,
    save_probes,
    select_best_layer,
    train_all_layers,
)
from deception_guardrail.utils.io import read_jsonl
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.seed import set_seed

logger = get_logger(__name__)

# Synthetic model identifier — all artifacts go under this name
MODEL_SHORT_NAME = "synthetic_debug"
RUN_ID = "synthetic_e2e_run"
NUM_LAYERS = 8
HIDDEN_DIM = 64
SIGNAL_LAYERS = [3, 4, 5]      # 0-indexed transformer block indices
SIGNAL_STRENGTH = 6.0          # Must be strong enough for AUROC > 0.95
SEED = 42


def _build_synthetic_artifact(
    labels: list[int],
    sample_ids: list[str],
    pair_ids: list[str] | None,
    domains: list[str] | None,
    split: str,
    rng: np.random.Generator,
    signal_dir: np.ndarray,
) -> dict:
    """
    Build a synthetic activation artifact.

    Deceptive samples (label=1) receive an additive signal vector at SIGNAL_LAYERS.
    Honest samples (label=0) receive only Gaussian noise.

    signal_dir must be the same unit vector for all splits so the probe trained
    on train data generalises to val/test.
    """
    n = len(labels)
    acts = rng.standard_normal((n, NUM_LAYERS, HIDDEN_DIM)).astype(np.float32)

    for i, label in enumerate(labels):
        if label == 1:
            for layer_idx in SIGNAL_LAYERS:
                acts[i, layer_idx, :] += signal_dir * SIGNAL_STRENGTH

    artifact = {
        "activations": torch.from_numpy(acts),
        "labels": labels,
        "sample_ids": sample_ids,
        "pair_ids": pair_ids,
        "domains": domains,
        "split": split,
        "model_name": MODEL_SHORT_NAME,
        "tokenizer_name": MODEL_SHORT_NAME,
        "layer_indices": list(range(NUM_LAYERS)),
        "token_position": "final_prompt_token",
    }
    validate_activation_artifact(artifact, context=split)
    return artifact


def _build_control_artifact(
    control_ids: list[str],
    control_types: list[str],
    rng: np.random.Generator,
) -> dict:
    """Build a synthetic control artifact with no deception signal."""
    n = len(control_ids)
    acts = rng.standard_normal((n, NUM_LAYERS, HIDDEN_DIM)).astype(np.float32)
    artifact = {
        "activations": torch.from_numpy(acts),
        "labels": None,
        "sample_ids": control_ids,
        "pair_ids": None,
        "split": "controls",
        "model_name": MODEL_SHORT_NAME,
        "tokenizer_name": MODEL_SHORT_NAME,
        "layer_indices": list(range(NUM_LAYERS)),
        "token_position": "final_prompt_token",
        "control_ids": control_ids,
        "control_types": control_types,
    }
    validate_activation_artifact(artifact, context="controls")
    return artifact


def run_synthetic_e2e(exp_cfg_path: str, run_assertions: bool = True) -> dict:
    """
    Run the full synthetic E2E pipeline.

    Returns a results dict with keys:
        best_layer, best_val_auroc, best_test_auroc, tpr_5pct_fpr, all_files_exist
    """
    exp_cfg = load_experiment_config(exp_cfg_path)
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])
    processed_dir = Path(exp_cfg.paths["processed_data_dir"])

    # Output paths — namespaced under synthetic_debug to avoid polluting main outputs
    syn_acts_dir = artifacts_dir / "activations" / MODEL_SHORT_NAME
    syn_probes_path = artifacts_dir / "probes" / MODEL_SHORT_NAME / "layerwise_probes.pkl"
    syn_metrics_dir = Path(exp_cfg.paths["results_dir"]) / "metrics" / MODEL_SHORT_NAME
    syn_figures_dir = Path(exp_cfg.paths["results_dir"]) / "figures" / MODEL_SHORT_NAME
    syn_metadata_dir = Path(exp_cfg.paths["metadata_dir"])

    syn_metrics_dir.mkdir(parents=True, exist_ok=True)
    syn_figures_dir.mkdir(parents=True, exist_ok=True)

    layerwise_csv_path = syn_metrics_dir / "layerwise_probe_metrics.csv"
    best_layer_json_path = syn_metrics_dir / "best_layer_summary.json"
    calibration_csv_path = syn_metrics_dir / "control_calibration.csv"
    plot_paths = {
        "layer_vs_test_auroc": str(syn_figures_dir / "layer_vs_test_auroc.png"),
        "layer_vs_validation_auroc": str(syn_figures_dir / "layer_vs_validation_auroc.png"),
        "layer_vs_test_auprc": str(syn_figures_dir / "layer_vs_test_auprc.png"),
        "layer_vs_control_fpr": str(syn_figures_dir / "layer_vs_control_fpr_at_0_5.png"),
        "tpr_at_fixed_fpr": str(syn_figures_dir / "tpr_at_fixed_fpr_by_layer.png"),
        "score_distributions": str(syn_figures_dir / "score_distributions_best_layer.png"),
    }

    # ---- Load labels/ids from the real processed dataset -------------------
    logger.info("Loading probe_samples.jsonl for labels and splits...")
    samples_jsonl = processed_dir / "probe_samples.jsonl"
    controls_jsonl = processed_dir / "benign_controls.jsonl"

    if not samples_jsonl.exists():
        raise FileNotFoundError(
            f"{samples_jsonl} not found. Run 01_build_dataset.py first."
        )

    all_samples = read_jsonl(samples_jsonl)
    controls_raw = read_jsonl(controls_jsonl)

    def _get_split(split_name: str) -> tuple[list, list, list, list]:
        rows = [r for r in all_samples if r["split"] == split_name]
        return (
            [r["label"] for r in rows],
            [r["sample_id"] for r in rows],
            [r["pair_id"] for r in rows],
            [r["domain"] for r in rows],
        )

    logger.info("Building synthetic activation artifacts...")

    # Single signal direction shared across all splits — probe trained on train
    # must generalise to val/test using the same direction.
    signal_dir = rng.standard_normal(HIDDEN_DIM).astype(np.float32)
    signal_dir /= np.linalg.norm(signal_dir)

    train_labels, train_ids, train_pair_ids, train_domains = _get_split("train")
    val_labels, val_ids, val_pair_ids, val_domains = _get_split("validation")
    test_labels, test_ids, test_pair_ids, test_domains = _get_split("test")

    train_art = _build_synthetic_artifact(train_labels, train_ids, train_pair_ids, train_domains, "train", rng, signal_dir)
    val_art = _build_synthetic_artifact(val_labels, val_ids, val_pair_ids, val_domains, "validation", rng, signal_dir)
    test_art = _build_synthetic_artifact(test_labels, test_ids, test_pair_ids, test_domains, "test", rng, signal_dir)

    ctrl_ids = [r["control_id"] for r in controls_raw]
    ctrl_types = [r["control_type"] for r in controls_raw]
    ctrl_art = _build_control_artifact(ctrl_ids, ctrl_types, rng)

    # ---- Save and reload (exercises file I/O code path) -------------------
    logger.info("Saving synthetic artifacts...")
    save_activations(train_art, activation_path(artifacts_dir, MODEL_SHORT_NAME, "train"))
    save_activations(val_art, activation_path(artifacts_dir, MODEL_SHORT_NAME, "validation"))
    save_activations(test_art, activation_path(artifacts_dir, MODEL_SHORT_NAME, "test"))
    save_activations(ctrl_art, activation_path(artifacts_dir, MODEL_SHORT_NAME, "controls"))

    logger.info("Reloading artifacts to exercise load path...")
    train_art = load_activations(activation_path(artifacts_dir, MODEL_SHORT_NAME, "train"))
    val_art = load_activations(activation_path(artifacts_dir, MODEL_SHORT_NAME, "validation"))
    test_art = load_activations(activation_path(artifacts_dir, MODEL_SHORT_NAME, "test"))
    ctrl_art = load_activations(activation_path(artifacts_dir, MODEL_SHORT_NAME, "controls"))

    # Validate after loading
    for art, ctx in [(train_art, "train"), (val_art, "validation"), (test_art, "test"), (ctrl_art, "controls")]:
        validate_activation_artifact(art, context=ctx)

    # ---- Train probes ------------------------------------------------------
    logger.info("Training layer-wise probes...")
    results = train_all_layers(
        train_art, val_art, test_art,
        c_grid=exp_cfg.c_grid,
        seed=SEED,
    )

    save_probes(results, syn_probes_path)
    # Reload to exercise serialization
    results = load_probes(syn_probes_path)

    save_layerwise_csv(results, layerwise_csv_path)

    # ---- Calibration -------------------------------------------------------
    logger.info("Running control calibration...")
    calibration_rows = calibrate_all_layers(results, ctrl_art, test_art)
    save_calibration_csv(calibration_rows, calibration_csv_path)

    # ---- Select best layer -------------------------------------------------
    ctrl_fpr_by_layer = {row["layer_index"]: row["fpr_at_threshold_0_5"] for row in calibration_rows}
    best = select_best_layer(results, control_fpr=ctrl_fpr_by_layer)
    best_cal = next(r for r in calibration_rows if r["layer_index"] == best.layer_index)

    save_best_layer_json(best, best_cal, best_layer_json_path)

    # ---- Plots -------------------------------------------------------------
    logger.info("Generating plots...")
    make_all_plots(results, calibration_rows, best, test_art, plot_paths)

    # ---- Run summary -------------------------------------------------------
    logger.info("Building run summary...")
    summary = build_run_summary(
        run_id=RUN_ID,
        model_name=MODEL_SHORT_NAME,
        model_short_name=MODEL_SHORT_NAME,
        exp_config_path=exp_cfg_path,
        model_config_path="N/A (synthetic)",
        exp_config_hash="synthetic",
        model_config_hash="synthetic",
        best_result=best,
        calibration_rows=calibration_rows,
        train_artifact=train_art,
        val_artifact=val_art,
        test_artifact=test_art,
        control_artifact=ctrl_art,
    )
    save_run_summary(summary, syn_metadata_dir)
    print_run_summary(summary)

    # ---- Collect expected file paths for existence check -------------------
    expected_files = [
        activation_path(artifacts_dir, MODEL_SHORT_NAME, "train"),
        activation_path(artifacts_dir, MODEL_SHORT_NAME, "validation"),
        activation_path(artifacts_dir, MODEL_SHORT_NAME, "test"),
        activation_path(artifacts_dir, MODEL_SHORT_NAME, "controls"),
        syn_probes_path,
        layerwise_csv_path,
        best_layer_json_path,
        calibration_csv_path,
        *[Path(p) for p in plot_paths.values()],
        syn_metadata_dir / f"{RUN_ID}.json",
    ]

    all_files_exist = all(f.exists() for f in expected_files)
    missing = [f for f in expected_files if not f.exists()]
    if missing:
        logger.warning(f"Missing output files: {[str(f) for f in missing]}")

    results_summary = {
        "best_layer": best.layer_index,
        "best_val_auroc": best.val_metrics["auroc"],
        "best_test_auroc": best.test_metrics["auroc"],
        "tpr_5pct_fpr": best_cal.get("tpr_at_5pct_fpr"),
        "all_files_exist": all_files_exist,
        "missing_files": [str(f) for f in missing],
    }

    # ---- Assertions --------------------------------------------------------
    if run_assertions:
        failures = []

        if results_summary["best_val_auroc"] <= 0.95:
            failures.append(
                f"Best val AUROC={results_summary['best_val_auroc']:.4f} <= 0.95"
            )
        if results_summary["best_test_auroc"] <= 0.95:
            failures.append(
                f"Best test AUROC={results_summary['best_test_auroc']:.4f} <= 0.95"
            )
        if results_summary["best_layer"] not in SIGNAL_LAYERS:
            failures.append(
                f"Best layer={results_summary['best_layer']} not in {SIGNAL_LAYERS}"
            )
        if results_summary["tpr_5pct_fpr"] is not None and results_summary["tpr_5pct_fpr"] <= 0.90:
            failures.append(
                f"TPR@5%FPR={results_summary['tpr_5pct_fpr']:.4f} <= 0.90"
            )
        if not results_summary["all_files_exist"]:
            failures.append(
                f"Missing output files: {results_summary['missing_files']}"
            )

        if failures:
            msg = "Synthetic E2E assertions FAILED:\n" + "\n".join(f"  - {f}" for f in failures)
            logger.error(msg)
            print(msg, file=sys.stderr)
            return {**results_summary, "passed": False, "failures": failures}
        else:
            logger.info("All synthetic E2E assertions PASSED.")
            print(
                f"\n[PASS] Synthetic E2E: best_layer={results_summary['best_layer']}, "
                f"val_auroc={results_summary['best_val_auroc']:.4f}, "
                f"test_auroc={results_summary['best_test_auroc']:.4f}, "
                f"tpr@5%fpr={results_summary['tpr_5pct_fpr']:.4f}"
            )

    return {**results_summary, "passed": not bool(missing)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic end-to-end pipeline test")
    p.add_argument("--experiment-config", required=True)
    p.add_argument(
        "--no-assert", action="store_true",
        help="Run without assertions (useful when probing signal-free behavior)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_synthetic_e2e(
        args.experiment_config,
        run_assertions=not args.no_assert,
    )
    if not result.get("passed", True):
        sys.exit(1)
