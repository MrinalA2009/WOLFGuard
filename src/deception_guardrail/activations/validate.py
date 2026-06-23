"""
Validate activation artifact dictionaries before training or calibration.

Convention:
    layer_indices = [0, 1, ..., num_transformer_layers - 1]  (0-indexed)
    activations[:, i, :] corresponds to transformer block i output
    embedding (hidden_states[0]) is excluded from all artifacts
"""

import torch

from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)

VALID_SPLITS = frozenset({"train", "validation", "test", "controls"})
VALID_TOKEN_POSITIONS = frozenset({"final_prompt_token"})


def validate_activation_artifact(artifact: dict, context: str = "") -> None:
    """
    Validate an activation artifact dictionary.

    Works for both probe-split artifacts (train/val/test) and control artifacts.
    Optional fields (labels, sample_ids, pair_ids, domains, control_types) are
    checked for length consistency only when they are present and non-None.

    Raises:
        ValueError if any check fails.
    """
    prefix = f"[{context}] " if context else ""

    # required keys present in every artifact
    for key in ("activations", "layer_indices", "token_position", "model_name"):
        if key not in artifact:
            raise ValueError(f"{prefix}Missing required key: '{key}'")

    # activations tensor checks
    acts = artifact["activations"]
    if not isinstance(acts, torch.Tensor):
        raise ValueError(
            f"{prefix}activations must be a torch.Tensor, got {type(acts).__name__}"
        )
    if acts.ndim != 3:
        raise ValueError(
            f"{prefix}activations.ndim={acts.ndim}, expected 3 [N, L, D]"
        )
    if acts.dtype != torch.float32:
        raise ValueError(
            f"{prefix}activations.dtype={acts.dtype}, expected torch.float32"
        )
    if acts.device.type != "cpu":
        raise ValueError(
            f"{prefix}activations must be on CPU, got device={acts.device}"
        )

    n_samples, n_layers, _ = acts.shape

    if acts.numel() > 0 and torch.isnan(acts).any().item():
        raise ValueError(f"{prefix}activations contain NaN values")
    if acts.numel() > 0 and torch.isinf(acts).any().item():
        raise ValueError(f"{prefix}activations contain Inf values")

    # layer_indices length must match activations second dimension
    layer_indices = artifact["layer_indices"]
    if len(layer_indices) != n_layers:
        raise ValueError(
            f"{prefix}len(layer_indices)={len(layer_indices)} != "
            f"n_layers (activations.shape[1])={n_layers}"
        )

    # per-sample length checks for optional fields
    for meta_key in ("labels", "sample_ids", "pair_ids", "domains", "control_types"):
        val = artifact.get(meta_key)
        if val is not None:
            if len(val) != n_samples:
                raise ValueError(
                    f"{prefix}len({meta_key})={len(val)} != n_samples={n_samples}"
                )

    # split field
    if "split" in artifact:
        split = artifact["split"]
        if split not in VALID_SPLITS:
            raise ValueError(
                f"{prefix}split='{split}' not in {set(VALID_SPLITS)}"
            )

    # token_position
    token_position = artifact["token_position"]
    if token_position not in VALID_TOKEN_POSITIONS:
        raise ValueError(
            f"{prefix}token_position='{token_position}' not in "
            f"{set(VALID_TOKEN_POSITIONS)}"
        )

    # model_name non-empty
    model_name = artifact.get("model_name") or ""
    if not model_name.strip():
        raise ValueError(f"{prefix}model_name is empty or missing")

    logger.debug(
        f"{prefix}Artifact valid: shape={list(acts.shape)}, "
        f"split={artifact.get('split', 'N/A')}, model={model_name}"
    )
