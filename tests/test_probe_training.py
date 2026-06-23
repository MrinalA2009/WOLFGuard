"""
Tests for probe training on synthetic separable activations.
No GPU or model download required.
"""

import numpy as np
import pytest
import torch

from deception_guardrail.probes.train import (
    LayerProbeResult,
    select_best_layer,
    train_all_layers,
    train_layer_probe,
)


def _make_separable_artifact(
    n_train: int,
    n_val: int,
    n_test: int,
    n_layers: int,
    hidden_dim: int,
    rng: np.random.Generator,
    signal_strength: float = 3.0,
) -> tuple[dict, dict, dict]:
    """Create linearly separable synthetic activations for probe training tests."""

    def make_split(n: int) -> tuple[torch.Tensor, list[int]]:
        labels = [i % 2 for i in range(n)]
        acts = rng.standard_normal((n, n_layers, hidden_dim)).astype(np.float32)
        for i, label in enumerate(labels):
            # Add signal only to last layer
            if label == 1:
                acts[i, -1, 0] += signal_strength
            else:
                acts[i, -1, 0] -= signal_strength
        return torch.from_numpy(acts), labels

    train_acts, train_labels = make_split(n_train)
    val_acts, val_labels = make_split(n_val)
    test_acts, test_labels = make_split(n_test)

    def make_artifact(acts, labels, split):
        return {
            "activations": acts,
            "labels": labels,
            "layer_indices": list(range(n_layers)),  # 0-indexed
            "split": split,
            "model_name": "synthetic_test",
            "token_position": "final_prompt_token",
        }

    return (
        make_artifact(train_acts, train_labels, "train"),
        make_artifact(val_acts, val_labels, "validation"),
        make_artifact(test_acts, test_labels, "test"),
    )


@pytest.fixture
def separable_artifacts():
    rng = np.random.default_rng(42)
    return _make_separable_artifact(
        n_train=200, n_val=60, n_test=60,
        n_layers=4, hidden_dim=32,
        rng=rng,
        signal_strength=5.0,
    )


def test_probe_trains_on_separable_data(separable_artifacts):
    train_art, val_art, test_art = separable_artifacts
    n_layers = len(train_art["layer_indices"])

    X_train = train_art["activations"].numpy()[:, -1, :]
    X_val = val_art["activations"].numpy()[:, -1, :]
    X_test = test_art["activations"].numpy()[:, -1, :]
    y_train = np.array(train_art["labels"])
    y_val = np.array(val_art["labels"])
    y_test = np.array(test_art["labels"])

    result = train_layer_probe(
        X_train, y_train, X_val, y_val, X_test, y_test,
        c_grid=[0.1, 1.0, 10.0],
        seed=42,
        layer_index=n_layers,
        tensor_index=n_layers - 1,
    )
    assert result.test_metrics["auroc"] > 0.9, (
        f"Expected high AUROC on separable data, got {result.test_metrics['auroc']}"
    )
    assert result.best_c in [0.1, 1.0, 10.0]


def test_all_layers_trained(separable_artifacts):
    train_art, val_art, test_art = separable_artifacts
    results = train_all_layers(
        train_art, val_art, test_art,
        c_grid=[0.1, 1.0],
        seed=42,
    )
    assert len(results) == len(train_art["layer_indices"])


def test_best_layer_selected_by_val_auroc(separable_artifacts):
    train_art, val_art, test_art = separable_artifacts
    results = train_all_layers(
        train_art, val_art, test_art,
        c_grid=[0.1, 1.0],
        seed=42,
    )
    best = select_best_layer(results)
    best_val = best.val_metrics["auroc"]
    for r in results:
        assert r.val_metrics["auroc"] <= best_val + 1e-6


def test_probe_result_fields(separable_artifacts):
    train_art, val_art, test_art = separable_artifacts
    X = train_art["activations"].numpy()[:, 0, :]
    y = np.array(train_art["labels"])
    result = train_layer_probe(
        X, y, X, y, X, y,
        c_grid=[1.0],
        seed=0,
        layer_index=1,
        tensor_index=0,
    )
    for key in ["auroc", "auprc", "accuracy", "f1", "precision", "recall",
                "confusion_matrix", "mean_score_deceptive", "mean_score_honest",
                "score_separation"]:
        assert key in result.test_metrics, f"Missing key '{key}' in test_metrics"


def test_scaler_fit_only_on_train():
    """The scaler's mean should match train data mean, not val/test."""
    rng = np.random.default_rng(99)
    n, d = 100, 16

    X_train = rng.standard_normal((n, d)).astype(np.float32) + 10.0
    X_val = rng.standard_normal((n // 2, d)).astype(np.float32) + 100.0  # very different
    y = np.array([i % 2 for i in range(n)])
    y_half = y[:n // 2]

    result = train_layer_probe(
        X_train, y, X_val, y_half, X_val, y_half,
        c_grid=[1.0],
        seed=0,
        layer_index=1,
        tensor_index=0,
    )
    # The scaler's mean should be close to 10 (train), not 100 (val/test)
    assert abs(float(result.scaler.mean_[0]) - 10.0) < 5.0, (
        f"Scaler mean={result.scaler.mean_[0]:.2f} suggests scaler was fit on non-train data"
    )
