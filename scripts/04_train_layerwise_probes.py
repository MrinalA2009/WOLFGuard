import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.analysis.summaries import save_best_layer_json, save_layerwise_csv
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.train import (
    save_probes,
    select_best_layer,
    train_all_layers,
)
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.paths import resolve_run_paths
from deception_guardrail.utils.seed import set_seed

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train layer-wise probes on captured activations")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    p.add_argument(
        "--run-name", default=None,
        help="Run namespace (must match the --run-name used in 03_capture_activations.py)",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)
    set_seed(exp_cfg.seed)

    run_paths = resolve_run_paths(exp_cfg, model_cfg.model_short_name, args.run_name)
    artifacts_dir = run_paths["artifacts_dir"]
    msn = model_cfg.model_short_name
    rn = args.run_name

    def _load(split: str) -> dict:
        path = activation_path(artifacts_dir, msn, split, rn)
        if not path.exists():
            raise FileNotFoundError(
                f"Activation file not found: {path}\n"
                f"Run 03_capture_activations.py --split {split}"
                + (f" --run-name {rn}" if rn else "")
                + " first."
            )
        art = load_activations(path)
        validate_activation_artifact(art, context=split)
        return art

    train_art = _load("train")
    val_art = _load("validation")
    test_art = _load("test")

    logger.info(
        f"Activation shapes — train: {list(train_art['activations'].shape)}, "
        f"val: {list(val_art['activations'].shape)}, "
        f"test: {list(test_art['activations'].shape)}"
    )

    results = train_all_layers(
        train_art, val_art, test_art,
        c_grid=exp_cfg.c_grid,
        seed=exp_cfg.seed,
    )

    probes_path = Path(run_paths["probes_pkl"])
    save_probes(results, probes_path)

    best = select_best_layer(results)
    logger.info(
        f"Best layer: {best.layer_index} | "
        f"val_auroc={best.val_metrics['auroc']:.4f} | "
        f"test_auroc={best.test_metrics['auroc']:.4f}"
    )

    save_layerwise_csv(results, Path(run_paths["metrics"]["layerwise_csv"]))
    save_best_layer_json(best, {}, Path(run_paths["metrics"]["best_layer_json"]))

if __name__ == "__main__":
    main()
