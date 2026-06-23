"""
Tests for validate_activation_artifact().
No GPU or model download required.
"""

import pytest
import torch

from deception_guardrail.activations.validate import validate_activation_artifact


def _valid_artifact(n=20, n_layers=8, d=64, split="train") -> dict:
    """Return a minimal valid artifact."""
    labels = [i % 2 for i in range(n)]
    return {
        "activations": torch.randn(n, n_layers, d, dtype=torch.float32),
        "labels": labels,
        "sample_ids": [f"s_{i}" for i in range(n)],
        "pair_ids": [f"p_{i//2}" for i in range(n)],
        "layer_indices": list(range(n_layers)),
        "token_position": "final_prompt_token",
        "split": split,
        "model_name": "test_model",
    }


def _valid_control_artifact(n=10, n_layers=8, d=64) -> dict:
    """Return a minimal valid control artifact (no labels, no pair_ids)."""
    return {
        "activations": torch.randn(n, n_layers, d, dtype=torch.float32),
        "labels": None,
        "sample_ids": [f"ctrl_{i}" for i in range(n)],
        "pair_ids": None,
        "layer_indices": list(range(n_layers)),
        "token_position": "final_prompt_token",
        "split": "controls",
        "model_name": "test_model",
        "control_ids": [f"ctrl_{i}" for i in range(n)],
        "control_types": ["normal_factual_qa"] * n,
    }


# ---- passing cases ----------------------------------------------------------

def test_valid_probe_artifact_passes():
    validate_activation_artifact(_valid_artifact(), context="test_probe")


def test_valid_control_artifact_passes():
    validate_activation_artifact(_valid_control_artifact(), context="test_ctrl")


def test_all_valid_splits_accepted():
    for split in ("train", "validation", "test", "controls"):
        art = _valid_artifact(split=split)
        validate_activation_artifact(art)  # should not raise


def test_optional_fields_absent_passes():
    """Artifact with only required fields should pass."""
    art = {
        "activations": torch.randn(10, 4, 32, dtype=torch.float32),
        "layer_indices": list(range(4)),
        "token_position": "final_prompt_token",
        "model_name": "tiny",
    }
    validate_activation_artifact(art)


# ---- failing cases — missing required keys ----------------------------------

@pytest.mark.parametrize("missing_key", ["activations", "layer_indices", "token_position", "model_name"])
def test_missing_required_key_raises(missing_key):
    art = _valid_artifact()
    del art[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        validate_activation_artifact(art)


# ---- failing cases — activations tensor properties -------------------------

def test_wrong_tensor_type_raises():
    art = _valid_artifact()
    art["activations"] = art["activations"].numpy()  # ndarray, not Tensor
    with pytest.raises(ValueError, match="torch.Tensor"):
        validate_activation_artifact(art)


def test_wrong_dtype_raises():
    art = _valid_artifact()
    art["activations"] = art["activations"].to(torch.float16)
    with pytest.raises(ValueError, match="float32"):
        validate_activation_artifact(art)


def test_wrong_ndim_2d_raises():
    art = _valid_artifact()
    art["activations"] = torch.randn(20, 64, dtype=torch.float32)
    with pytest.raises(ValueError, match="ndim"):
        validate_activation_artifact(art)


def test_wrong_ndim_4d_raises():
    art = _valid_artifact()
    art["activations"] = torch.randn(20, 8, 64, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="ndim"):
        validate_activation_artifact(art)


def test_nan_in_activations_raises():
    art = _valid_artifact()
    art["activations"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_activation_artifact(art)


def test_inf_in_activations_raises():
    art = _valid_artifact()
    art["activations"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="Inf"):
        validate_activation_artifact(art)


# ---- failing cases — metadata length mismatches ----------------------------

def test_labels_wrong_length_raises():
    art = _valid_artifact(n=20)
    art["labels"] = art["labels"][:10]  # too short
    with pytest.raises(ValueError, match="labels"):
        validate_activation_artifact(art)


def test_sample_ids_wrong_length_raises():
    art = _valid_artifact(n=20)
    art["sample_ids"] = art["sample_ids"][:5]
    with pytest.raises(ValueError, match="sample_ids"):
        validate_activation_artifact(art)


def test_layer_indices_wrong_length_raises():
    art = _valid_artifact(n_layers=8)
    art["layer_indices"] = list(range(4))  # should be 8
    with pytest.raises(ValueError, match="layer_indices"):
        validate_activation_artifact(art)


# ---- failing cases — invalid enum values ------------------------------------

def test_invalid_split_raises():
    art = _valid_artifact()
    art["split"] = "holdout"
    with pytest.raises(ValueError, match="split"):
        validate_activation_artifact(art)


def test_invalid_token_position_raises():
    art = _valid_artifact()
    art["token_position"] = "first_token"
    with pytest.raises(ValueError, match="token_position"):
        validate_activation_artifact(art)


def test_empty_model_name_raises():
    art = _valid_artifact()
    art["model_name"] = ""
    with pytest.raises(ValueError, match="model_name"):
        validate_activation_artifact(art)


def test_whitespace_model_name_raises():
    art = _valid_artifact()
    art["model_name"] = "   "
    with pytest.raises(ValueError, match="model_name"):
        validate_activation_artifact(art)


# ---- integration: synthetic E2E pipeline ----------------------------------

@pytest.mark.integration
def test_synthetic_e2e_pipeline():
    """
    Run the full synthetic E2E pipeline inline (no subprocess, no model download).
    Checks that AUROC > 0.95, best layer is one of the signal-injected layers,
    and TPR@5%FPR > 0.90.

    Marked 'integration': trains 8 × 4 logistic probes on 1400 samples (~5s on CPU).
    Run with: pytest -m integration
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    if not (processed_dir / "probe_samples.jsonl").exists():
        pytest.skip("Dataset not built — run 01_build_dataset.py first")

    exp_cfg_path = str(Path(__file__).parent.parent / "configs" / "experiment.yaml")

    # Import run_synthetic_e2e via importlib so the numeric filename is not a problem
    import importlib.util
    script_path = Path(__file__).parent.parent / "scripts" / "08_run_synthetic_e2e.py"
    spec = importlib.util.spec_from_file_location("synth_e2e", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod.run_synthetic_e2e(exp_cfg_path, run_assertions=True)

    assert result.get("passed"), f"Synthetic E2E failed: {result.get('failures', [])}"
    assert result["best_layer"] in [3, 4, 5], (
        f"Best layer {result['best_layer']} not in signal layers [3,4,5]"
    )
    assert result["best_val_auroc"] > 0.95, (
        f"Val AUROC {result['best_val_auroc']:.4f} <= 0.95"
    )
    assert result["best_test_auroc"] > 0.95, (
        f"Test AUROC {result['best_test_auroc']:.4f} <= 0.95"
    )
    assert result["tpr_5pct_fpr"] > 0.90, (
        f"TPR@5%FPR {result['tpr_5pct_fpr']:.4f} <= 0.90"
    )
    assert result["all_files_exist"], (
        f"Missing output files: {result.get('missing_files')}"
    )
