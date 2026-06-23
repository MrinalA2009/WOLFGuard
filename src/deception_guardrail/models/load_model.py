import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from deception_guardrail.config import ModelConfig
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)

_MPS_AVAILABLE = torch.backends.mps.is_available()
_CUDA_AVAILABLE = torch.cuda.is_available()


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "auto":
        if _CUDA_AVAILABLE:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # MPS does not support bfloat16; default to float16
        return torch.float16
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Pad token was None; set to eos_token")
    logger.info(f"Loaded tokenizer: {cfg.model_name}")
    return tokenizer


def load_model(cfg: ModelConfig) -> PreTrainedModel:
    dtype = resolve_dtype(cfg.dtype)
    device_map = cfg.device_map

    # MPS: Accelerate's device_map="auto" is CUDA-only; load on CPU then move to MPS
    use_mps = _MPS_AVAILABLE and not _CUDA_AVAILABLE and device_map in ("auto", "mps")
    if use_mps:
        logger.info(
            f"MPS detected (no CUDA). Loading model on CPU then moving to MPS. "
            f"dtype override: {dtype} (bfloat16 not supported on MPS)."
        )
        if dtype == torch.bfloat16:
            dtype = torch.float16
            logger.info("bfloat16 → float16 for MPS compatibility")

        # low_cpu_mem_usage=True loads weights directly into target dtype buffers,
        # avoiding a full extra CPU copy (~14 GB peak → ~7 GB peak for a 7B model).
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=cfg.trust_remote_code,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=cfg.trust_remote_code,
        )

    model.eval()
    device_str = "mps" if use_mps else str(device_map)
    logger.info(
        f"Loaded model: {cfg.model_name} | dtype={dtype} | device={device_str}"
    )
    return model


def get_num_layers(model: PreTrainedModel) -> int:
    """Return the number of transformer blocks (excluding the embedding layer)."""
    cfg = model.config
    # Works for Qwen2, LLaMA, Mistral families
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise AttributeError(
        f"Cannot determine num_hidden_layers from config: {type(cfg)}"
    )
