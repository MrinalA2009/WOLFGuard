#!/usr/bin/env python3
"""
Domain held-out evaluation: test cross-domain generalisation.

For each domain in the test set:
  1. Filter training activations to exclude that domain.
  2. Retrain a probe on the filtered train set (same C-grid, same seed).
  3. Evaluate on test-set samples from the held-out domain only.
  4. Report per-domain AUROC.

CPU-only — uses stored activations, no model inference required.

Usage:
    python scripts/16_domain_held_out.py \\
        --model-config configs/qwen2_5_1b5.yaml \\
        --experiment-config configs/experiment.yaml \\
        --run-name qwen1_5b_full_v1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from deception_guardrail.activations.store import activation_path, load_activations
from deception_guardrail.activations.validate import validate_activation_artifact
from deception_guardrail.config import load_experiment_config, load_model_config
from deception_guardrail.probes.evaluate import compute_metrics, compute_scores
from deception_guardrail.probes.train import load_probes, select_best_layer, train_layer_probe
from deception_guardrail.utils.logging import get_logger
from deception_guardrail.utils.paths import resolve_run_paths

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Domain held-out evaluation")
    p.add_argument("--model-config", required=True)
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def _load_art(artifacts_dir: Path, msn: str, split: str, rn: str | None) -> dict:
    path = activation_path(artifacts_dir, msn, split, rn)
    if not path.exists():
        raise FileNotFoundError(
            f"Activation file not found: {path}\n"
            f"Run 03_capture_activations.py for split='{split}' first."
        )
    art = load_activations(path)
    validate_activation_artifact(art, context=split)
    return art


def main() -> None:
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)
    model_cfg = load_model_config(args.model_config)

    run_paths = resolve_run_paths(exp_cfg, model_cfg.model_short_name, args.run_name)
    artifacts_dir = run_paths["artifacts_dir"]
    msn = model_cfg.model_short_name
    rn = args.run_name

    # Load probes and identify best layer
    probes_path = Path(run_paths["probes_pkl"])
    if not probes_path.exists():
        raise FileNotFoundError(f"Probes not found: {probes_path}. Run script 04 first.")
    results = load_probes(probes_path)

    cal_csv = Path(run_paths["metrics"]["control_calibration_csv"])
    if cal_csv.exists():
        with open(cal_csv) as f:
            calibration_rows = list(csv.DictReader(f))
        calibration_rows_typed = [
            {k: (int(v) if k == "layer_index" else float(v)) for k, v in row.items()}
            for row in calibration_rows
        ]
        ctrl_fpr_by_layer = {
            row["layer_index"]: row["fpr_at_threshold_0_5"]
            for row in calibration_rows_typed
        }
        best = select_best_layer(results, control_fpr=ctrl_fpr_by_layer)
    else:
        best = select_best_layer(results)

    ti = best.tensor_index
    logger.info(f"Best layer: {best.layer_index} (tensor_index={ti})")

    # Load all three splits
    train_art = _load_art(artifacts_dir, msn, "train", rn)
    val_art = _load_art(artifacts_dir, msn, "validation", rn)
    test_art = _load_art(artifacts_dir, msn, "test", rn)

    train_acts_full = train_art["activations"].numpy()  # [N, L, D]
    val_acts_full = val_art["activations"].numpy()
    test_acts_full = test_art["activations"].numpy()

    train_labels = np.array(train_art["labels"])
    val_labels = np.array(val_art["labels"])
    test_labels = np.array(test_art["labels"])

    train_domains = train_art.get("domains") or (["unknown"] * len(train_labels))
    val_domains = val_art.get("domains") or (["unknown"] * len(val_labels))
    test_domains = test_art.get("domains") or (["unknown"] * len(test_labels))

    unique_domains = sorted(set(test_domains))
    logger.info(f"Domains in test set: {unique_domains}")

    # Full-model baseline AUROC for reference
    full_test_scores = compute_scores(
        best.classifier, best.scaler, test_acts_full[:, ti, :]
    )
    full_test_auroc = compute_metrics(test_labels, full_test_scores)["auroc"]

    # ---- Per-domain leave-one-domain-out -----------------------------------
    domain_rows: list[dict] = []

    for domain in unique_domains:
        # Training mask: exclude held-out domain
        train_mask = np.array([d != domain for d in train_domains])
        val_mask = np.array([d != domain for d in val_domains])

        n_train_kept = int(train_mask.sum())
        if n_train_kept < 10:
            logger.warning(
                f"Domain '{domain}': only {n_train_kept} train samples after exclusion — skipping."
            )
            domain_rows.append({
                "domain": domain,
                "n_train_kept": n_train_kept,
                "n_test_domain": int(np.sum([d == domain for d in test_domains])),
                "held_out_auroc": None,
                "held_out_f1": None,
                "held_out_accuracy": None,
                "note": "too few train samples",
            })
            continue

        y_train_filt = train_labels[train_mask]
        if len(np.unique(y_train_filt)) < 2:
            logger.warning(
                f"Domain '{domain}': filtered train has only one class — skipping."
            )
            domain_rows.append({
                "domain": domain,
                "n_train_kept": n_train_kept,
                "n_test_domain": int(np.sum([d == domain for d in test_domains])),
                "held_out_auroc": None,
                "held_out_f1": None,
                "held_out_accuracy": None,
                "note": "single class in train",
            })
            continue

        X_train_filt = train_acts_full[train_mask, ti, :]

        # Validation: use filtered val if it has both classes, else fall back to full val
        y_val_filt = val_labels[val_mask]
        if len(np.unique(y_val_filt)) >= 2:
            X_val_use = val_acts_full[val_mask, ti, :]
            y_val_use = y_val_filt
        else:
            logger.warning(
                f"Domain '{domain}': filtered val has one class; using full val set."
            )
            X_val_use = val_acts_full[:, ti, :]
            y_val_use = val_labels

        # Test: held-out domain only
        test_domain_mask = np.array([d == domain for d in test_domains])
        n_test_domain = int(test_domain_mask.sum())

        if n_test_domain < 2:
            logger.warning(
                f"Domain '{domain}': only {n_test_domain} test samples — skipping."
            )
            domain_rows.append({
                "domain": domain,
                "n_train_kept": n_train_kept,
                "n_test_domain": n_test_domain,
                "held_out_auroc": None,
                "held_out_f1": None,
                "held_out_accuracy": None,
                "note": "too few test samples",
            })
            continue

        X_test_domain = test_acts_full[test_domain_mask, ti, :]
        y_test_domain = test_labels[test_domain_mask]

        if len(np.unique(y_test_domain)) < 2:
            logger.warning(
                f"Domain '{domain}': test domain has only one class — AUROC undefined."
            )
            domain_rows.append({
                "domain": domain,
                "n_train_kept": n_train_kept,
                "n_test_domain": n_test_domain,
                "held_out_auroc": None,
                "held_out_f1": None,
                "held_out_accuracy": None,
                "note": "single class in test",
            })
            continue

        # Retrain probe at best layer, excluding this domain
        retrained = train_layer_probe(
            X_train=X_train_filt,
            y_train=y_train_filt,
            X_val=X_val_use,
            y_val=y_val_use,
            X_test=X_test_domain,
            y_test=y_test_domain,
            c_grid=exp_cfg.c_grid,
            seed=exp_cfg.seed,
            layer_index=best.layer_index,
            tensor_index=ti,
        )

        held_scores = compute_scores(
            retrained.classifier, retrained.scaler, X_test_domain
        )
        held_metrics = compute_metrics(y_test_domain, held_scores)

        logger.info(
            f"Domain '{domain}' | n_train_kept={n_train_kept} | n_test={n_test_domain} "
            f"| AUROC={held_metrics['auroc']:.4f}"
        )

        domain_rows.append({
            "domain": domain,
            "n_train_kept": n_train_kept,
            "n_test_domain": n_test_domain,
            "held_out_auroc": held_metrics["auroc"],
            "held_out_f1": held_metrics["f1"],
            "held_out_accuracy": held_metrics["accuracy"],
            "note": "ok",
        })

    # ---- Print report -------------------------------------------------------
    valid_aurocs = [r["held_out_auroc"] for r in domain_rows if r["held_out_auroc"] is not None]

    print()
    print("=" * 70)
    print(f"  DOMAIN HELD-OUT EVALUATION  layer={best.layer_index}  run={rn or 'default'}")
    print("=" * 70)
    print(f"  Full-model test AUROC (all domains, v1 probe): {full_test_auroc:.4f}")
    print()
    print(f"  {'Domain':<30}  {'N_train':>7}  {'N_test':>6}  {'AUROC':>7}  {'Note'}")
    print("  " + "-" * 62)
    for row in domain_rows:
        auroc_str = f"{row['held_out_auroc']:.4f}" if row["held_out_auroc"] is not None else "   N/A"
        print(
            f"  {row['domain']:<30}  {row['n_train_kept']:>7}  "
            f"{row['n_test_domain']:>6}  {auroc_str:>7}  {row['note']}"
        )

    if valid_aurocs:
        print()
        print(f"  Min held-out AUROC : {min(valid_aurocs):.4f}")
        print(f"  Max held-out AUROC : {max(valid_aurocs):.4f}")
        print(f"  Mean held-out AUROC: {np.mean(valid_aurocs):.4f}")
        n_weak = sum(a < 0.70 for a in valid_aurocs)
        print(f"  Domains with AUROC < 0.70: {n_weak}/{len(valid_aurocs)}")
        if n_weak == 0:
            print("  Verdict: probe generalises across all domains.")
        else:
            print(f"  Verdict: {n_weak} domain(s) show weak generalisation — inspect.")
    print("=" * 70)

    # ---- Save CSV and JSON --------------------------------------------------
    metrics_dir = Path(run_paths["metrics"]["layerwise_csv"]).parent
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_out = metrics_dir / "domain_held_out.csv"
    fieldnames = [
        "domain", "n_train_kept", "n_test_domain",
        "held_out_auroc", "held_out_f1", "held_out_accuracy", "note",
    ]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(domain_rows)
    logger.info(f"Domain held-out CSV → {csv_out}")

    json_out = metrics_dir / "domain_held_out.json"
    report = {
        "run_name": rn,
        "model_short_name": msn,
        "best_layer": best.layer_index,
        "full_test_auroc": full_test_auroc,
        "domain_results": domain_rows,
        "summary": {
            "n_domains_evaluated": len(valid_aurocs),
            "min_auroc": float(min(valid_aurocs)) if valid_aurocs else None,
            "max_auroc": float(max(valid_aurocs)) if valid_aurocs else None,
            "mean_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
            "n_below_0_70": int(sum(a < 0.70 for a in valid_aurocs)),
        },
    }
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Domain held-out JSON → {json_out}")


if __name__ == "__main__":
    main()
