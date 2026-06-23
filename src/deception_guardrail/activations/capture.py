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
block numbers).  tensor_index == layer_index for every saved artifact.
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

_OOM_HELP = (
    "\n\nCUDA out of memory during activation capture. Suggestions:"
    "\n  - Reduce batch_size in model config (try 2 or 4)."
    "\n  - Reduce max_length in model config."
    "\n  - Use --limit-pairs to capture fewer samples first."
    "\n  - Ensure no other GPU process holds VRAM (nvidia-smi)."
    "\n  - Verify model dtype is bfloat16 or float16 (set dtype: auto in config)."
    "\n  - For debugging without a GPU, use configs/tiny_gpt2_debug.yaml."
)


def _final_non_padding_indices(attention_mask: Tensor) -> Tensor:
    """Return the index of the last non-padding token for each sample in the batch."""
    indices = attention_mask.sum(dim=1) - 1
    if (indices < 0).any().item():
        raise ValueError(
            "Attention mask contains all-zero rows — at least one prompt is fully masked. "
            "Check tokenizer pad token configuration."
        )
    return indices


def _log_gpu_memory(device: torch.device, label: str) -> None:
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.info(
            f"GPU memory [{label}]: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB"
        )
    elif device.type == "mps":
        try:
            alloc = torch.mps.current_allocated_memory() / 1e9
            logger.info(f"MPS memory [{label}]: allocated={alloc:.2f} GB")
        except AttributeError:
            pass


def _capture_batch(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    num_layers: int,
    log_first: bool = False,
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

    if log_first:
        _log_gpu_memory(device, "before first batch forward pass")

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        is_oom = (
            isinstance(e, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(e).lower()
        )
        if is_oom:
            device_label = device.type.upper()
            raise RuntimeError(
                f"{device_label} out of memory (batch_size={input_ids.shape[0]}, "
                f"seq_len={input_ids.shape[1]}).{_OOM_HELP}"
            ) from e
        raise

    if log_first:
        _log_gpu_memory(device, "after first batch forward pass")

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError(
            "Model returned hidden_states=None. "
            "Ensure output_hidden_states=True is honoured by this model family. "
            "Check that the model was loaded via AutoModelForCausalLM."
        )

    expected_hs_len = num_layers + 1  # embedding + N transformer blocks
    if len(hidden_states) != expected_hs_len:
        raise RuntimeError(
            f"Expected {expected_hs_len} hidden states "
            f"(1 embedding + {num_layers} transformer blocks), "
            f"got {len(hidden_states)}. "
            f"Check that get_num_layers() returned the correct value for this model family."
        )

    token_indices = _final_non_padding_indices(attention_mask)  # [B]
    batch_size = input_ids.size(0)
    batch_range = torch.arange(batch_size, device=device)

    # Collect transformer block outputs — skip index 0 (embedding)
    layer_acts: list[Tensor] = []
    for block_idx in range(num_layers):
        hs = hidden_states[block_idx + 1]  # [B, T, D]
        act = hs[batch_range, token_indices, :]  # [B, D]
        layer_acts.append(act.float().cpu())

    stacked = torch.stack(layer_acts, dim=1)  # [B, num_layers, D]
    assert stacked.shape == (batch_size, num_layers, hidden_states[1].size(-1)), (
        f"Unexpected stacked activation shape: {stacked.shape}"
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
    token is always at the last attended position in each sequence.

    Returns a dict with keys:
        activations      : float32 CPU tensor [N, num_layers, hidden_dim]
                           activations[:, i, :] = transformer block i output
        formatted_prompts: list of formatted prompt strings
        token_position   : "final_prompt_token"
        layer_indices    : [0, 1, ..., num_layers-1]
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

    device = next(model.parameters()).device

    logger.info(
        f"Activation capture: prompts={n}, batch_size={batch_size}, "
        f"max_length={cfg.max_length}, n_batches={n_batches}, "
        f"num_layers={num_layers}, device={device}"
    )
    if device.type == "cuda":
        logger.info(
            f"Model dtype: {next(model.parameters()).dtype} | "
            f"CUDA device: {torch.cuda.get_device_name(device)}"
        )

    all_acts: list[Tensor] = []
    log_first_batch = True

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
                f"Tokenizer returned empty input_ids for batch {batch_i}. "
                "Check that the tokenizer is correctly configured and prompts are non-empty."
            )

        if log_first_batch:
            seq_len = encoded["input_ids"].shape[1]
            logger.info(
                f"First batch: actual_batch_size={encoded['input_ids'].shape[0]}, "
                f"seq_len={seq_len}, tokens_per_batch={encoded['input_ids'].shape[0] * seq_len}"
            )

        batch_acts = _capture_batch(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            num_layers,
            log_first=log_first_batch,
        )

        if log_first_batch:
            logger.info(
                f"First batch activations: shape={list(batch_acts.shape)}, "
                f"dtype={batch_acts.dtype}, device={batch_acts.device}"
            )
            log_first_batch = False

        all_acts.append(batch_acts)

        # Periodic progress log at 25% intervals (useful when tqdm goes to file)
        pct = (batch_i + 1) / n_batches
        if pct in (0.25, 0.5, 0.75) or batch_i + 1 == n_batches:
            logger.info(
                f"Progress: {batch_i+1}/{n_batches} batches "
                f"({100*pct:.0f}%), samples captured so far: {(batch_i+1)*batch_size}"
            )

    activations = torch.cat(all_acts, dim=0)  # [N, num_layers, D]
    assert activations.shape[0] == n, (
        f"Sample count mismatch after concatenation: expected {n}, "
        f"got {activations.shape[0]}. Batch processing may have dropped samples."
    )
    assert activations.dtype == torch.float32
    assert activations.device.type == "cpu"

    # 0-indexed: layer_index i corresponds to transformer block i (hidden_states[i+1])
    layer_indices = list(range(num_layers))

    logger.info(
        f"Captured activations: final shape={list(activations.shape)}, "
        f"layer_indices=[0..{num_layers-1}]"
    )
    return {
        "activations": activations,
        "formatted_prompts": formatted,
        "token_position": cfg.token_position,
        "layer_indices": layer_indices,
    }
