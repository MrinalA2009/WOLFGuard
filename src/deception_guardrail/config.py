import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class ExperimentConfig:
    seed: int
    paths: dict
    split_sizes: dict
    allowed_domains: list[str]
    control_types: list[str]
    c_grid: list[float]
    metrics_output_paths: dict
    plot_output_paths: dict

@dataclass
class ModelConfig:
    model_name: str
    model_short_name: str
    batch_size: int
    max_length: int
    dtype: str
    device_map: str
    token_position: str
    use_chat_template: bool
    trust_remote_code: bool

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(
        seed=data["seed"],
        paths=data["paths"],
        split_sizes=data["split_sizes"],
        allowed_domains=data["allowed_domains"],
        control_types=data["control_types"],
        c_grid=data["c_grid"],
        metrics_output_paths=data["metrics_output_paths"],
        plot_output_paths=data["plot_output_paths"],
    )

def load_model_config(path: str | Path) -> ModelConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ModelConfig(
        model_name=data["model_name"],
        model_short_name=data["model_short_name"],
        batch_size=data["batch_size"],
        max_length=data["max_length"],
        dtype=data["dtype"],
        device_map=data["device_map"],
        token_position=data["token_position"],
        use_chat_template=data["use_chat_template"],
        trust_remote_code=data.get("trust_remote_code", False),
    )

def config_hash(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]
