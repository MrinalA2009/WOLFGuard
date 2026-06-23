#!/usr/bin/env python3
"""
Capture hidden-state activations for one split or the benign controls.

Resumable at the split level: if the output file already exists, the script
skips that split. Delete the .pt file to re-capture.

Usage:
    # Full capture for a probe split:
    python scripts/03_capture_activations.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml \
        --split train

    # Benign controls:
    python scripts/03_capture_activations.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml \
        --controls

    # Dry-run: load tokenizer + format prompts, no model load, no file write:
    python scripts/03_capture_activations.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml \
        --split train --dry-run

    # Capture only first N samples (useful for debugging before full run):
    python scripts/03_capture_activations.py \
        --model-config configs/qwen2_5_7b.yaml \
        --experiment-config configs/experiment.yaml \
        --split train --limit 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.capture import capture_activations
from deception_guardrail.activations.store import activation_path, save_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.config import config_hash, load_experiment_config, load_model_config
from deception_guardrail.models.chat_format import format_prompts_batch
from deception_guardrail.models.load_model import get_num_layers, load_model, load_tokenizer
from deception_guardrail.utils.io import read_jsonl
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture activations for a split or controls")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=["train", "validation", "test"])
    g.add_argument("--controls", action="store_true")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Capture only the first N samples (useful for smoke tests before full run).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Load tokenizer, format prompts, and print tokenized shapes. "
            "Does NOT load the model or write any artifact. "
            "Use to verify prompt formatting and token lengths before the full capture."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)
    set_seed(exp_cfg.seed)

    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])
    processed_dir = Path(exp_cfg.paths["processed_data_dir"])

    split_key = "controls" if args.controls else args.split
    out_path = activation_path(artifacts_dir, model_cfg.model_short_name, split_key)

    if not args.dry_run and out_path.exists():
        logger.info(f"Output already exists, skipping: {out_path}")
        return

    # --- Load dataset rows --------------------------------------------------
    if args.controls:
        raw = read_jsonl(processed_dir / "benign_controls.jsonl")
        prompts = [r["prompt"] for r in raw]
        ids = [r["control_id"] for r in raw]
        control_types = [r["control_type"] for r in raw]
        labels = None
        pair_ids = None
        metadata_extra: dict = {
            "control_ids": ids,
            "control_types": control_types,
        }
    else:
        raw = read_jsonl(processed_dir / "probe_samples.jsonl")
        split_rows = [r for r in raw if r["split"] == args.split]
        prompts = [r["prompt"] for r in split_rows]
        labels = [r["label"] for r in split_rows]
        pair_ids = [r["pair_id"] for r in split_rows]
        ids = [r["sample_id"] for r in split_rows]
        metadata_extra = {
            "sample_ids": ids,
            "pair_ids": pair_ids,
            "domains": [r["domain"] for r in split_rows],
        }

    # --- Apply limit --------------------------------------------------------
    if args.limit is not None:
        n = args.limit
        logger.info(f"--limit {n}: restricting to first {n} of {len(prompts)} samples")
        prompts = prompts[:n]
        ids = ids[:n]
        if labels is not None:
            labels = labels[:n]
        if pair_ids is not None:
            pair_ids = pair_ids[:n]
        for k in list(metadata_extra.keys()):
            if isinstance(metadata_extra[k], list):
                metadata_extra[k] = metadata_extra[k][:n]

    logger.info(f"Dataset rows loaded: {len(prompts)} prompts ({split_key})")

    # --- Dry-run: tokenizer only, no model, no write ------------------------
    if args.dry_run:
        logger.info("[DRY RUN] Loading tokenizer only — skipping model load")
        tokenizer = load_tokenizer(model_cfg)
        sample_n = min(3, len(prompts))
        formatted = format_prompts_batch(prompts[:sample_n], tokenizer, model_cfg)
        logger.info(f"[DRY RUN] Formatting {sample_n} prompts with use_chat_template={model_cfg.use_chat_template}")
        for i, fp in enumerate(formatted):
            enc = tokenizer(
                fp,
                return_tensors="pt",
                truncation=True,
                max_length=model_cfg.max_length,
            )
            shape = list(enc["input_ids"].shape)
            logger.info(f"[DRY RUN] Prompt {i}: input_ids.shape={shape}, "
                        f"last 60 chars: ...{fp[-60:]!r}")
        logger.info("[DRY RUN] Done. No model loaded; no artifact written.")
        return

    # --- Full capture -------------------------------------------------------
    logger.info(f"Loading model: {model_cfg.model_name}")
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg)
    num_layers = get_num_layers(model)
    logger.info(f"Model has {num_layers} transformer layers (layer_indices 0..{num_layers-1})")

    result = capture_activations(prompts, model, tokenizer, model_cfg, num_layers)

    artifact = {
        "activations": result["activations"],
        "labels": labels,
        "sample_ids": ids,
        "pair_ids": pair_ids,
        "split": split_key,
        "model_name": model_cfg.model_name,
        "tokenizer_name": model_cfg.model_name,
        "layer_indices": result["layer_indices"],
        "token_position": result["token_position"],
        "experiment_config_hash": config_hash(args.experiment_config),
        "model_config_hash": config_hash(args.model_config),
        **metadata_extra,
    }

    validate_activation_artifact(artifact, context=split_key)
    save_activations(artifact, out_path)
    logger.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
