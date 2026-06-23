#!/usr/bin/env python3
"""
Capture hidden-state activations for one split or the benign controls.

Resumable: if the output file already exists, the script skips it.
Delete the .pt file to re-capture.

Usage examples
--------------
# Full train split:
python scripts/03_capture_activations.py \\
    --model-config configs/qwen2_5_7b.yaml \\
    --experiment-config configs/experiment.yaml \\
    --split train

# Dry-run (tokenizer only, no model load, no file write):
python scripts/03_capture_activations.py \\
    --model-config configs/qwen2_5_7b.yaml \\
    --experiment-config configs/experiment.yaml \\
    --split train --dry-run

# Pilot with 4 pairs (8 samples: 4 honest + 4 deceptive), isolated namespace:
python scripts/03_capture_activations.py \\
    --model-config configs/qwen2_5_7b.yaml \\
    --experiment-config configs/experiment.yaml \\
    --split train --limit-pairs 4 --run-name qwen_pilot_4

# Pilot controls:
python scripts/03_capture_activations.py \\
    --model-config configs/qwen2_5_7b.yaml \\
    --experiment-config configs/experiment.yaml \\
    --controls --limit-controls 4 --run-name qwen_pilot_4
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
from deception_guardrail.utils.paths import resolve_run_paths
from deception_guardrail.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture activations for a probe split or benign controls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-config", required=True, metavar="PATH")
    p.add_argument("--experiment-config", required=True, metavar="PATH")

    split_group = p.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--split", choices=["train", "validation", "test"])
    split_group.add_argument("--controls", action="store_true")

    p.add_argument(
        "--limit-pairs", type=int, default=None, metavar="N",
        help=(
            "Capture only the first N *pairs* from the selected probe split "
            "(2N samples: N honest + N deceptive). "
            "Use for pilots before the full run. Ignored when --controls is set."
        ),
    )
    p.add_argument(
        "--limit-controls", type=int, default=None, metavar="N",
        help=(
            "Capture only the first N control examples. "
            "Ignored when --split is set."
        ),
    )
    p.add_argument(
        "--run-name", default=None, metavar="NAME",
        help=(
            "Optional run namespace. Isolates artifacts under "
            "artifacts/activations/{model}/{run_name}/, "
            "results/metrics/{run_name}/, etc. "
            "Use different names for pilots vs. full runs to avoid overwriting."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Load tokenizer and format prompts — print tokenized shapes. "
            "Does NOT load the model or write any artifact. "
            "Use to verify prompt formatting before the full capture."
        ),
    )
    return p.parse_args()


def _limit_to_pairs(rows: list[dict], n_pairs: int) -> list[dict]:
    """
    Return rows for the first n_pairs unique pair_ids (preserving both labels).
    Result always contains exactly 2*n_pairs rows (n_pairs honest + n_pairs deceptive).
    """
    seen: dict[str, int] = {}
    for r in rows:
        pid = r["pair_id"]
        if pid not in seen:
            if len(seen) >= n_pairs:
                break
            seen[pid] = 1
    selected_pairs = set(seen.keys())
    return [r for r in rows if r["pair_id"] in selected_pairs]


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)
    set_seed(exp_cfg.seed)

    run_paths = resolve_run_paths(exp_cfg, model_cfg.model_short_name, args.run_name)
    artifacts_dir = run_paths["artifacts_dir"]
    processed_dir = Path(exp_cfg.paths["processed_data_dir"])

    split_key = "controls" if args.controls else args.split
    out_path = activation_path(artifacts_dir, model_cfg.model_short_name, split_key, args.run_name)

    if not args.dry_run and out_path.exists():
        logger.info(f"Output already exists, skipping: {out_path}")
        return

    # ---- Load dataset rows --------------------------------------------------
    if args.controls:
        raw = read_jsonl(processed_dir / "benign_controls.jsonl")
        if args.limit_controls is not None:
            raw = raw[: args.limit_controls]
            logger.info(f"--limit-controls {args.limit_controls}: using {len(raw)} controls")
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

        if args.limit_pairs is not None:
            before = len(split_rows)
            split_rows = _limit_to_pairs(split_rows, args.limit_pairs)
            n_honest = sum(1 for r in split_rows if r["label"] == 0)
            n_deceptive = sum(1 for r in split_rows if r["label"] == 1)
            logger.info(
                f"--limit-pairs {args.limit_pairs}: {before} → {len(split_rows)} samples "
                f"({n_honest} honest, {n_deceptive} deceptive)"
            )

        prompts = [r["prompt"] for r in split_rows]
        labels = [r["label"] for r in split_rows]
        pair_ids = [r["pair_id"] for r in split_rows]
        ids = [r["sample_id"] for r in split_rows]
        metadata_extra = {
            "sample_ids": ids,
            "pair_ids": pair_ids,
            "domains": [r["domain"] for r in split_rows],
        }

    logger.info(
        f"Dataset loaded: {len(prompts)} prompts ({split_key})"
        + (f" | run_name={args.run_name}" if args.run_name else "")
    )

    # ---- Dry-run: tokenizer only, no model, no write ------------------------
    if args.dry_run:
        logger.info("[DRY RUN] Loading tokenizer only — skipping model load")
        try:
            tokenizer = load_tokenizer(model_cfg)
        except OSError as e:
            logger.error(
                f"[DRY RUN] Tokenizer load failed: {e}\n"
                "Check that the model name is correct and the HF cache is populated "
                "(or internet is available). Run: huggingface-cli login if access-gated."
            )
            sys.exit(1)

        sample_n = min(3, len(prompts))
        try:
            formatted = format_prompts_batch(prompts[:sample_n], tokenizer, model_cfg)
        except Exception as e:
            logger.error(
                f"[DRY RUN] Chat template failed: {e}\n"
                f"Set use_chat_template: false in {args.model_config} if the model "
                "has no chat template, or check the tokenizer's chat_template field."
            )
            sys.exit(1)

        logger.info(
            f"[DRY RUN] Formatting {sample_n} prompts "
            f"(use_chat_template={model_cfg.use_chat_template})"
        )
        for i, fp in enumerate(formatted):
            enc = tokenizer(fp, return_tensors="pt", truncation=True, max_length=model_cfg.max_length)
            shape = list(enc["input_ids"].shape)
            logger.info(
                f"[DRY RUN] Prompt {i}: input_ids.shape={shape}, "
                f"tail: ...{fp[-80:]!r}"
            )
        logger.info("[DRY RUN] Done. No model loaded; no artifact written.")
        return

    # ---- Full capture -------------------------------------------------------
    logger.info(f"Loading model: {model_cfg.model_name}")
    try:
        tokenizer = load_tokenizer(model_cfg)
        model = load_model(model_cfg)
    except OSError as e:
        logger.error(
            f"Model/tokenizer load failed: {e}\n"
            "Possible causes:\n"
            "  - Model weights not downloaded (run with --dry-run first to check tokenizer)\n"
            "  - No internet access to HuggingFace Hub\n"
            "  - Access-gated model: run 'huggingface-cli login'\n"
            "  - Corrupted cache: delete ~/.cache/huggingface/hub/{model_dir}"
        )
        sys.exit(1)

    num_layers = get_num_layers(model)
    logger.info(f"Model loaded: {num_layers} transformer layers, layer_indices=[0..{num_layers-1}]")

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
        "run_name": args.run_name,
        "experiment_config_hash": config_hash(args.experiment_config),
        "model_config_hash": config_hash(args.model_config),
        **metadata_extra,
    }

    validate_activation_artifact(artifact, context=split_key)
    save_activations(artifact, out_path)
    logger.info(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
