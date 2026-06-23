"""
Activation capture at the final non-padding prompt token.

Hidden-state indexing convention (HuggingFace):
    hidden_states[0]  = embedding output           (excluded from all artifacts)
    hidden_states[1]  = transformer block 0 output
    hidden_states[k]  = transformer block k-1 output
    hidden_states[L]  = transformer block L-1 output (last transformer layer)

Saved tensor convention:
    activations[:, 0, :] = output of transformer block 0  (layer_index=0)
    activations[:, i, :] = output of transformer block i  (layer_index=i)
    activations[:, L-1, :] = output of transformer block L-1 (layer_index=L-1)

The metadata field `layer_indices` records [0, 1, ..., L-1] (0-indexed transformer
block numbers).  tensor_index == layer_index for all artifacts produced here.
"""

import math
from typing import Optional

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from deception_guardrail.config import ModelConfig
from deception_guardrail.models.chat_format import format_prompts_batch
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


def _final_non_padding_indices(attention_mask: Tensor) -> Tensor:
    """Return the index of the last non-padding token for each sample in the batch."""
    indices = attention_mask.sum(dim=1) - 1
    if (indices < 0).any().item():
        raise ValueError(
            "Attention mask contains all-zero rows — at least one prompt is fully masked."
        )
    return indices


def _capture_batch(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    num_layers: int,
) -> Tensor:
    """
    Run one forward pass and extract activations at the final non-padding token.

    Returns:
        Tensor of shape [batch_size, num_layers, hidden_dim] in float32 on CPU.
        Layer axis i corresponds to transformer block i (excludes embedding).
    """
    if input_ids.numel() == 0:
        raise ValueError("input_ids is empty — tokenizer produced no tokens.")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError(
            "Model returned hidden_states=None. "
            "Ensure output_hidden_states=True is supported by this model family."
        )

    # embedding (index 0) + L transformer blocks = L+1 total
    expected_hs_len = num_layers + 1
    if len(hidden_states) != expected_hs_len:
        raise RuntimeError(
            f"Expected {expected_hs_len} hidden states (1 embedding + {num_layers} "
            f"transformer blocks), got {len(hidden_states)}. "
            f"Check that get_num_layers() returned the correct value."
        )

    token_indices = _final_non_padding_indices(attention_mask)  # [B]
    batch_size = input_ids.size(0)
    batch_range = torch.arange(batch_size, device=device)

    # Collect transformer block outputs — skip index 0 (embedding)
    layer_acts: list[Tensor] = []
    for block_idx in range(num_layers):
        hs = hidden_states[block_idx + 1]  # [B, T, D], skip embedding at [0]
        act = hs[batch_range, token_indices, :]  # [B, D]
        layer_acts.append(act.float().cpu())

    stacked = torch.stack(layer_acts, dim=1)  # [B, num_layers, D]
    assert stacked.shape == (batch_size, num_layers, hidden_states[1].size(-1)), (
        f"Unexpected stacked shape: {stacked.shape}"
    )
    return stacked


def capture_activations(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: ModelConfig,
    num_layers: int,
) -> dict:
    """
    Capture final-token activations for all prompts.

    Prompts are processed in batches. Left-padding ensures the final non-padding
    token is always at the last attended position of each sequence.

    Returns a dict with keys:
        activations: float32 CPU tensor [N, num_layers, hidden_dim]
            activations[:, i, :] = transformer block i output
        formatted_prompts: list of formatted prompt strings
        token_position: "final_prompt_token"
        layer_indices: [0, 1, ..., num_layers-1]
    """
    if not prompts:
        raise ValueError("prompts list is empty — nothing to capture.")
    if any(not p.strip() for p in prompts):
        empty_idx = next(i for i, p in enumerate(prompts) if not p.strip())
        raise ValueError(f"Empty prompt at index {empty_idx}.")

    formatted = format_prompts_batch(prompts, tokenizer, cfg)
    n = len(formatted)
    batch_size = cfg.batch_size
    n_batches = math.ceil(n / batch_size)

    all_acts: list[Tensor] = []
    for batch_i in tqdm(range(n_batches), desc="Capturing activations", leave=False):
        batch_prompts = formatted[batch_i * batch_size : (batch_i + 1) * batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        )
        if encoded["input_ids"].numel() == 0:
            raise RuntimeError(
                f"Tokenizer returned empty input_ids for batch {batch_i}."
            )
        batch_acts = _capture_batch(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            num_layers,
        )
        all_acts.append(batch_acts)

    activations = torch.cat(all_acts, dim=0)  # [N, num_layers, D]
    assert activations.shape[0] == n, (
        f"Sample count mismatch: expected {n}, got {activations.shape[0]}. "
        "Batch concatenation may have dropped samples."
    )
    assert activations.dtype == torch.float32
    assert activations.device.type == "cpu"

    # 0-indexed: layer_index i corresponds to transformer block i (hidden_states[i+1])
    layer_indices = list(range(num_layers))

    logger.info(
        f"Captured activations: shape={list(activations.shape)}, "
        f"layer_indices=[0..{num_layers-1}]"
    )
    return {
        "activations": activations,
        "formatted_prompts": formatted,
        "token_position": cfg.token_position,
        "layer_indices": layer_indices,
    }
