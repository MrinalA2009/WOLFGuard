from transformers import PreTrainedTokenizerBase

from deception_guardrail.config import ModelConfig
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)

_LOGGED_TEMPLATE_ONCE = False

def format_prompt(prompt: str, tokenizer: PreTrainedTokenizerBase, cfg: ModelConfig) -> str:
    global _LOGGED_TEMPLATE_ONCE
    if cfg.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not _LOGGED_TEMPLATE_ONCE:
                logger.info(f"Using chat template. Example formatted prompt (truncated):\n{formatted[:300]}")
                _LOGGED_TEMPLATE_ONCE = True
            return formatted
        except Exception as e:
            logger.warning(f"apply_chat_template failed ({e}); falling back to plain format")

    return f"User: {prompt}\nAssistant:"

def format_prompts_batch(
    prompts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    cfg: ModelConfig,
) -> list[str]:
    return [format_prompt(p, tokenizer, cfg) for p in prompts]
