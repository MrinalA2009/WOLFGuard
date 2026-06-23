"""
Tests for activation tensor shape logic using synthetic data.
No GPU or model download required.
"""

import numpy as np
import pytest
import torch


def make_synthetic_artifact(n_samples: int, n_layers: int, hidden_dim: int, labels=None):
    acts = torch.randn(n_samples, n_layers, hidden_dim)
    if labels is None:
        labels = [i % 2 for i in range(n_samples)]
    return {
        "activations": acts.float(),
        "labels": labels,
        "sample_ids": [f"sample_{i}" for i in range(n_samples)],
        "pair_ids": [f"pair_{i // 2}" for i in range(n_samples)],
        "layer_indices": list(range(n_layers)),   # 0-indexed: [0, 1, ..., n_layers-1]
        "token_position": "final_prompt_token",
        "split": "train",
        "model_name": "synthetic_test",
    }


def test_activation_shape():
    art = make_synthetic_artifact(100, 28, 3584)
    acts = art["activations"]
    assert acts.shape == (100, 28, 3584)


def test_activation_dtype_float32():
    art = make_synthetic_artifact(50, 10, 64)
    assert art["activations"].dtype == torch.float32


def test_layer_indices_match_n_layers():
    n_layers = 12
    art = make_synthetic_artifact(10, n_layers, 64)
    assert len(art["layer_indices"]) == n_layers
    # 0-indexed: first layer is 0, last layer is n_layers-1
    assert art["layer_indices"][0] == 0
    assert art["layer_indices"][-1] == n_layers - 1


def test_labels_length_matches_samples():
    n = 80
    art = make_synthetic_artifact(n, 8, 32)
    assert len(art["labels"]) == n


def test_extract_single_layer():
    art = make_synthetic_artifact(100, 28, 3584)
    acts = art["activations"].numpy()
    layer_0 = acts[:, 0, :]
    assert layer_0.shape == (100, 3584)


def test_store_save_load(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from deception_guardrail.activations.store import load_activations, save_activations

    art = make_synthetic_artifact(20, 4, 16)
    path = tmp_path / "test_acts.pt"
    save_activations(art, path)
    loaded = load_activations(path)

    assert loaded["activations"].shape == art["activations"].shape
    assert torch.allclose(loaded["activations"], art["activations"])
    assert loaded["labels"] == art["labels"]
    assert loaded["layer_indices"] == art["layer_indices"]


def test_activation_cpu_after_save(tmp_path):
    from deception_guardrail.activations.store import load_activations, save_activations

    art = make_synthetic_artifact(10, 4, 16)
    path = tmp_path / "acts.pt"
    save_activations(art, path)
    loaded = load_activations(path)
    assert loaded["activations"].device == torch.device("cpu")
