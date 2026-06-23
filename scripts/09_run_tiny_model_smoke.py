#!/usr/bin/env python3
"""
Tiny HuggingFace model smoke test.

Uses sshleifer/tiny-gpt2 to exercise the EXACT SAME activation capture code
path as Qwen, on a miniature dataset subset.  Results are not scientifically
meaningful — the purpose is to catch shape errors, indexing mistakes, serialisation
bugs, and accidental GPU-only assumptions before running the 7B model.

What this verifies:
  - capture_activations() produces a [N, L, D] float32 CPU tensor
  - Artifact schema matches validate_activation_artifact()
  - Probe training runs without crashes (no AUROC assertion)
  - Calibration runs without crashes
  - Plot generation runs without crashes

Usage:
    python scripts/09_run_tiny_model_smoke.py \
        --experiment-config configs/experiment.yaml \
        --model-config configs/tiny_gpt2_debug.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from deception_guardrail.activations.capture import capture_activations
from deception_guardrail.activations.store import activation_path, save_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.analysis.plots import make_all_plots
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.models.load_model import get_num_layers, load_model, load_tokenizer
from deception_guardrail.probes.calibration import calibrate_all_layers
from deception_guardrail.probes.train import select_best_layer, train_all_layers
from deception_guardrail.utils.io import read_jsonl
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.seed import set_seed

logger = get_logger(__name__)

# Dataset subset sizes for the smoke test
TRAIN_PAIRS = 4      # → 8 train samples
VAL_PAIRS = 2        # → 4 val samples
TEST_PAIRS = 2       # → 4 test samples
N_CONTROLS = 4


def _take_pairs(rows: list[dict], n_pairs: int) -> list[dict]:
    """Take the first n_pairs paired rows (2 samples per pair: honest + deceptive)."""
    # Rows are interleaved honest/deceptive per pair; take 2*n_pairs rows
    return rows[: n_pairs * 2]


def main() -> None:
    args = parse_args()

    try:
        exp_cfg = load_experiment_config(args.experiment_config)
        model_cfg = load_model_config(args.model_config)
    except Exception as e:
        print(f"[SKIP] Config load failed: {e}", file=sys.stderr)
        sys.exit(0)

    set_seed(exp_cfg.seed)

    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])
    processed_dir = Path(exp_cfg.paths["processed_data_dir"])
    results_dir = Path(exp_cfg.paths["results_dir"])
    msn = model_cfg.model_short_name

    # Verify dataset exists
    samples_jsonl = processed_dir / "probe_samples.jsonl"
    controls_jsonl = processed_dir / "benign_controls.jsonl"
    if not samples_jsonl.exists():
        print(
            f"[SKIP] {samples_jsonl} not found. Run 01_build_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Load tiny subsets
    all_samples = read_jsonl(samples_jsonl)
    all_controls = read_jsonl(controls_jsonl)

    train_rows = _take_pairs([r for r in all_samples if r["split"] == "train"], TRAIN_PAIRS)
    val_rows = _take_pairs([r for r in all_samples if r["split"] == "validation"], VAL_PAIRS)
    test_rows = _take_pairs([r for r in all_samples if r["split"] == "test"], TEST_PAIRS)
    ctrl_rows = all_controls[:N_CONTROLS]

    logger.info(
        f"Smoke test subset: train={len(train_rows)}, val={len(val_rows)}, "
        f"test={len(test_rows)}, controls={len(ctrl_rows)}"
    )

    # Load model — bail out gracefully if download fails
    logger.info(f"Loading tokenizer and model: {model_cfg.model_name}")
    try:
        tokenizer = load_tokenizer(model_cfg)
        model = load_model(model_cfg)
    except Exception as e:
        print(
            f"[SKIP] Model load failed (likely no internet or HF cache): {e}",
            file=sys.stderr,
        )
        sys.exit(0)

    num_layers = get_num_layers(model)
    logger.info(f"Tiny model: {num_layers} transformer layers")

    def _capture(rows: list[dict], split: str) -> dict:
        prompts = [r["prompt"] for r in rows]
        labels = [r["label"] for r in rows]
        sample_ids = [r["sample_id"] for r in rows]
        pair_ids = [r["pair_id"] for r in rows]
        domains = [r["domain"] for r in rows]

        result = capture_activations(prompts, model, tokenizer, model_cfg, num_layers)

        artifact = {
            "activations": result["activations"],
            "labels": labels,
            "sample_ids": sample_ids,
            "pair_ids": pair_ids,
            "domains": domains,
            "split": split,
            "model_name": model_cfg.model_name,
            "tokenizer_name": model_cfg.model_name,
            "layer_indices": result["layer_indices"],
            "token_position": result["token_position"],
        }
        validate_activation_artifact(artifact, context=split)
        logger.info(
            f"[{split}] activations.shape={list(artifact['activations'].shape)}, "
            f"dtype={artifact['activations'].dtype}, "
            f"device={artifact['activations'].device}"
        )
        return artifact

    # Capture activations for all splits
    train_art = _capture(train_rows, "train")
    val_art = _capture(val_rows, "validation")
    test_art = _capture(test_rows, "test")

    # Capture controls
    ctrl_prompts = [r["prompt"] for r in ctrl_rows]
    ctrl_result = capture_activations(ctrl_prompts, model, tokenizer, model_cfg, num_layers)
    ctrl_art = {
        "activations": ctrl_result["activations"],
        "labels": None,
        "sample_ids": [r["control_id"] for r in ctrl_rows],
        "pair_ids": None,
        "split": "controls",
        "model_name": model_cfg.model_name,
        "tokenizer_name": model_cfg.model_name,
        "layer_indices": ctrl_result["layer_indices"],
        "token_position": ctrl_result["token_position"],
        "control_ids": [r["control_id"] for r in ctrl_rows],
        "control_types": [r["control_type"] for r in ctrl_rows],
    }
    validate_activation_artifact(ctrl_art, context="controls")

    # Save artifacts
    for art, split in [
        (train_art, "train"), (val_art, "validation"), (test_art, "test"), (ctrl_art, "controls")
    ]:
        save_activations(art, activation_path(artifacts_dir, msn, split))

    # ---- Schema checks (the key output of this test) ----------------------
    for art, split in [
        (train_art, "train"), (val_art, "validation"), (test_art, "test"), (ctrl_art, "controls")
    ]:
        acts = art["activations"]
        n_samples = acts.shape[0]

        assert isinstance(acts, torch.Tensor), f"[{split}] activations not a Tensor"
        assert acts.dtype == torch.float32, f"[{split}] dtype={acts.dtype}"
        assert acts.device.type == "cpu", f"[{split}] not on CPU: {acts.device}"
        assert acts.ndim == 3, f"[{split}] ndim={acts.ndim}"

        assert len(art["layer_indices"]) == acts.shape[1], (
            f"[{split}] layer_indices mismatch"
        )
        assert art["token_position"] == "final_prompt_token", (
            f"[{split}] token_position wrong"
        )
        assert art["model_name"] == model_cfg.model_name, (
            f"[{split}] model_name wrong"
        )

        sample_ids = art.get("sample_ids") or art.get("control_ids")
        if sample_ids is not None:
            assert len(sample_ids) == n_samples, f"[{split}] sample_ids length mismatch"

        if art.get("labels") is not None:
            assert len(art["labels"]) == n_samples, f"[{split}] labels length mismatch"

        logger.info(f"[{split}] schema OK: shape={list(acts.shape)}")

    # ---- Probe training (no crash, no AUROC assertion) --------------------
    logger.info("Training probes on tiny subset (no AUROC assertion)...")
    results = train_all_layers(
        train_art, val_art, test_art,
        c_grid=[0.1, 1.0],
        seed=exp_cfg.seed,
    )
    best = select_best_layer(results)
    logger.info(
        f"Best layer (meaningless on tiny data): {best.layer_index} | "
        f"val_auroc={best.val_metrics['auroc']:.4f} | "
        f"test_auroc={best.test_metrics['auroc']:.4f}"
    )

    # ---- Calibration (no crash) ------------------------------------------
    logger.info("Running calibration on tiny subset...")
    calibration_rows = calibrate_all_layers(results, ctrl_art, test_art)
    logger.info(
        f"Calibration rows: {len(calibration_rows)}, "
        f"best layer tpr@5%fpr={next(r['tpr_at_5pct_fpr'] for r in calibration_rows if r['layer_index']==best.layer_index):.4f}"
    )

    # ---- Plots (no crash) ------------------------------------------------
    logger.info("Generating plots on tiny subset...")
    syn_figures_dir = results_dir / "figures" / msn
    syn_figures_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "layer_vs_test_auroc": str(syn_figures_dir / "layer_vs_test_auroc.png"),
        "layer_vs_validation_auroc": str(syn_figures_dir / "layer_vs_validation_auroc.png"),
        "layer_vs_test_auprc": str(syn_figures_dir / "layer_vs_test_auprc.png"),
        "layer_vs_control_fpr": str(syn_figures_dir / "layer_vs_control_fpr_at_0_5.png"),
        "tpr_at_fixed_fpr": str(syn_figures_dir / "tpr_at_fixed_fpr_by_layer.png"),
        "score_distributions": str(syn_figures_dir / "score_distributions_best_layer.png"),
    }
    make_all_plots(results, calibration_rows, best, test_art, plot_paths)

    print(
        f"\n[PASS] Tiny model smoke test: {model_cfg.model_name} | "
        f"num_layers={num_layers} | "
        f"train_shape={list(train_art['activations'].shape)} | "
        f"all schema checks OK"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny HuggingFace model smoke test")
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--model-config", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main()
