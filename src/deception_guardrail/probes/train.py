"""
Layer-wise logistic regression probe training.

For each layer:
1. Fit StandardScaler on train activations only.
2. Search C on validation AUROC.
3. Report test metrics with the best C.

Scaler is fit once per layer on train data; the same fitted scaler is used
for validation and test. Test data is never used for C selection.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from deception_guardrail.probes.evaluate import compute_metrics, compute_scores
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LayerProbeResult:
    layer_index: int       # 1-indexed transformer block number
    tensor_index: int      # 0-indexed position in the activations tensor
    best_c: float
    val_metrics: dict
    test_metrics: dict
    scaler: StandardScaler
    classifier: LogisticRegression


def _extract_layer(activations: np.ndarray, tensor_index: int) -> np.ndarray:
    """Extract activations for one layer: [N, D]."""
    return activations[:, tensor_index, :]


def train_layer_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: list[float],
    seed: int,
    layer_index: int,
    tensor_index: int,
) -> LayerProbeResult:
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    best_c = c_grid[0]
    best_val_auroc = -1.0
    best_clf: Optional[LogisticRegression] = None

    for c in c_grid:
        clf = LogisticRegression(
            C=c,
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_train_sc, y_train)
        val_scores = clf.predict_proba(X_val_sc)[:, 1]
        from sklearn.metrics import roc_auc_score
        val_auroc = float(roc_auc_score(y_val, val_scores))
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_c = c
            best_clf = clf

    # Refit with best_c to get a clean model (already done above but be explicit)
    final_clf = LogisticRegression(
        C=best_c,
        max_iter=5000,
        solver="lbfgs",
        random_state=seed,
    )
    final_clf.fit(X_train_sc, y_train)

    val_scores = compute_scores(final_clf, scaler, X_val)
    val_metrics = compute_metrics(y_val, val_scores)

    test_scores = compute_scores(final_clf, scaler, X_test)
    test_metrics = compute_metrics(y_test, test_scores)

    return LayerProbeResult(
        layer_index=layer_index,
        tensor_index=tensor_index,
        best_c=best_c,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        scaler=scaler,
        classifier=final_clf,
    )


def train_all_layers(
    train_artifact: dict,
    val_artifact: dict,
    test_artifact: dict,
    c_grid: list[float],
    seed: int,
) -> list[LayerProbeResult]:
    """Train one probe per layer across all transformer blocks."""
    acts_train = train_artifact["activations"].numpy()  # [N_train, L, D]
    acts_val = val_artifact["activations"].numpy()
    acts_test = test_artifact["activations"].numpy()

    y_train = np.array(train_artifact["labels"])
    y_val = np.array(val_artifact["labels"])
    y_test = np.array(test_artifact["labels"])

    layer_indices = train_artifact["layer_indices"]  # [1, 2, ..., L]
    n_layers = len(layer_indices)

    results: list[LayerProbeResult] = []
    for tensor_idx in tqdm(range(n_layers), desc="Training layer probes"):
        layer_idx = layer_indices[tensor_idx]
        X_train = _extract_layer(acts_train, tensor_idx)
        X_val = _extract_layer(acts_val, tensor_idx)
        X_test = _extract_layer(acts_test, tensor_idx)

        result = train_layer_probe(
            X_train, y_train, X_val, y_val, X_test, y_test,
            c_grid=c_grid,
            seed=seed,
            layer_index=layer_idx,
            tensor_index=tensor_idx,
        )
        results.append(result)
        logger.info(
            f"Layer {layer_idx:02d} | C={result.best_c} | "
            f"val_auroc={result.val_metrics['auroc']:.4f} | "
            f"test_auroc={result.test_metrics['auroc']:.4f}"
        )

    return results


def select_best_layer(
    results: list[LayerProbeResult],
    control_fpr: Optional[dict] = None,
) -> LayerProbeResult:
    """
    Select best layer by validation AUROC.
    Tiebreak: lower control FPR > earlier layer > first encountered.
    """
    best = results[0]
    for r in results[1:]:
        if r.val_metrics["auroc"] > best.val_metrics["auroc"]:
            best = r
        elif r.val_metrics["auroc"] == best.val_metrics["auroc"]:
            # Tiebreak: lower control FPR
            if control_fpr is not None:
                r_fpr = control_fpr.get(r.layer_index, float("inf"))
                best_fpr = control_fpr.get(best.layer_index, float("inf"))
                if r_fpr < best_fpr:
                    best = r
                elif r_fpr == best_fpr and r.layer_index < best.layer_index:
                    best = r
            elif r.layer_index < best.layer_index:
                best = r
    return best


def save_probes(results: list[LayerProbeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probes": [
            {
                "layer_index": r.layer_index,
                "tensor_index": r.tensor_index,
                "best_c": r.best_c,
                "val_metrics": r.val_metrics,
                "test_metrics": r.test_metrics,
                "scaler": r.scaler,
                "classifier": r.classifier,
            }
            for r in results
        ]
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Saved {len(results)} probes → {path}")


def load_probes(path: Path) -> list[LayerProbeResult]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    results = [
        LayerProbeResult(
            layer_index=p["layer_index"],
            tensor_index=p["tensor_index"],
            best_c=p["best_c"],
            val_metrics=p["val_metrics"],
            test_metrics=p["test_metrics"],
            scaler=p["scaler"],
            classifier=p["classifier"],
        )
        for p in payload["probes"]
    ]
    logger.info(f"Loaded {len(results)} probes ← {path}")
    return results
