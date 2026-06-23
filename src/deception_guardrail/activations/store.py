"""
Save and load activation artifacts.

Each artifact is a dict saved via torch.save / torch.load.
All tensors are float32 on CPU.

File layout:
    artifacts/activations/{model_short_name}/{split}_activations.pt
    artifacts/activations/{model_short_name}/control_activations.pt
"""

from pathlib import Path
from typing import Optional

import torch

from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


def save_activations(
    artifact: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure tensor is float32 CPU
    if "activations" in artifact:
        artifact["activations"] = artifact["activations"].float().cpu()
    torch.save(artifact, path)
    shape = list(artifact["activations"].shape) if "activations" in artifact else "?"
    logger.info(f"Saved activations: shape={shape} → {path}")


def load_activations(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    shape = list(artifact["activations"].shape) if "activations" in artifact else "?"
    logger.info(f"Loaded activations: shape={shape} ← {path}")
    return artifact


def activation_path(artifacts_dir: Path, model_short_name: str, split: str) -> Path:
    fname = f"{split}_activations.pt" if split != "controls" else "control_activations.pt"
    return artifacts_dir / "activations" / model_short_name / fname
