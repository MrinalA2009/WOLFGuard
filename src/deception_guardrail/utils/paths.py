"""
Resolve output paths for a given run, with optional namespacing via run_name.

Without run_name (None), default config paths are used unchanged (backward-compatible).
With run_name='qwen_pilot_32', all outputs are isolated under that name:

    artifacts/activations/{model_short_name}/{run_name}/{split}_activations.pt
    artifacts/probes/{model_short_name}/{run_name}/layerwise_probes.pkl
    results/metrics/{run_name}/
    results/figures/{run_name}/
    artifacts/metadata/{run_name}.json
"""

from pathlib import Path

from deception_guardrail.config import ExperimentConfig


def resolve_run_paths(
    exp_cfg: ExperimentConfig,
    model_short_name: str,
    run_name: str | None,
) -> dict:
    """
    Return a dict of all resolved output paths for a given run.

    Keys:
        artifacts_dir      : base artifacts directory (Path)
        activations_dir    : directory where split .pt files live (Path)
        probes_pkl         : path to layerwise_probes.pkl (Path)
        metrics            : dict with layerwise_csv, best_layer_json, control_calibration_csv (str paths)
        plots              : dict with all 6 plot file paths (str paths)
        metadata_dir       : directory for run summary JSON (Path)
    """
    artifacts_dir = Path(exp_cfg.paths["artifacts_dir"])
    results_dir = Path(exp_cfg.paths["results_dir"])
    metadata_dir = Path(exp_cfg.paths["metadata_dir"])

    if run_name is None:
        return {
            "artifacts_dir": artifacts_dir,
            "activations_dir": artifacts_dir / "activations" / model_short_name,
            "probes_pkl": artifacts_dir / "probes" / model_short_name / "layerwise_probes.pkl",
            "metrics": {
                "layerwise_csv": exp_cfg.metrics_output_paths["layerwise_csv"],
                "best_layer_json": exp_cfg.metrics_output_paths["best_layer_json"],
                "control_calibration_csv": exp_cfg.metrics_output_paths["control_calibration_csv"],
            },
            "plots": exp_cfg.plot_output_paths,
            "metadata_dir": metadata_dir,
        }

    metrics_dir = results_dir / "metrics" / run_name
    figures_dir = results_dir / "figures" / run_name
    return {
        "artifacts_dir": artifacts_dir,
        "activations_dir": artifacts_dir / "activations" / model_short_name / run_name,
        "probes_pkl": artifacts_dir / "probes" / model_short_name / run_name / "layerwise_probes.pkl",
        "metrics": {
            "layerwise_csv": str(metrics_dir / "layerwise_probe_metrics.csv"),
            "best_layer_json": str(metrics_dir / "best_layer_summary.json"),
            "control_calibration_csv": str(metrics_dir / "control_calibration.csv"),
        },
        "plots": {
            "layer_vs_test_auroc": str(figures_dir / "layer_vs_test_auroc.png"),
            "layer_vs_validation_auroc": str(figures_dir / "layer_vs_validation_auroc.png"),
            "layer_vs_test_auprc": str(figures_dir / "layer_vs_test_auprc.png"),
            "layer_vs_control_fpr": str(figures_dir / "layer_vs_control_fpr_at_0_5.png"),
            "tpr_at_fixed_fpr": str(figures_dir / "tpr_at_fixed_fpr_by_layer.png"),
            "score_distributions": str(figures_dir / "score_distributions_best_layer.png"),
            "auroc_tpr_panel": str(figures_dir / "auroc_tpr_panel.png"),
        },
        "metadata_dir": metadata_dir,
    }


def activation_split_path(activations_dir: Path, split: str) -> Path:
    """Map split name → .pt file inside activations_dir."""
    fname = "control_activations.pt" if split == "controls" else f"{split}_activations.pt"
    return Path(activations_dir) / fname
