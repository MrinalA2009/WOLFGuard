# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

WOLFGuard (Phase 1) is a mechanistic interpretability experiment that tests whether direct factual deception can be linearly detected from the internal residual-stream activations of an instruction-tuned LLM. The pipeline: paired honest/deceptive prompts → model hidden states → layer-wise logistic regression probes → metrics/calibration/plots. Phase 1 is detection-only (no causal intervention, no activation steering).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+. Activation capture for Qwen2.5-7B requires a CUDA GPU with ≥16 GB VRAM and bfloat16 support. Apple MPS does not support bfloat16; use `configs/qwen2_5_1b5_mps.yaml` (float32) for pipeline validation on Apple Silicon.

## Commands

### Run tests (no GPU required)

```bash
pytest tests/ -v
```

### Full pipeline (Steps 1–7)

```bash
python scripts/01_build_dataset.py --experiment-config configs/experiment.yaml
python scripts/02_validate_dataset.py --experiment-config configs/experiment.yaml

for split in train validation test; do
  python scripts/03_capture_activations.py \
      --model-config configs/qwen2_5_7b.yaml \
      --experiment-config configs/experiment.yaml \
      --split $split
done
python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --controls

python scripts/04_train_layerwise_probes.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml
python scripts/05_evaluate_controls.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml
python scripts/06_make_plots.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml
python scripts/07_summarize_run.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --run-id phase1_qwen2_5_7b
```

### Pre-Qwen validation (run in this order before any GPU capture)

```bash
# 1. Synthetic E2E — no model required (~10 s)
python scripts/08_run_synthetic_e2e.py --experiment-config configs/experiment.yaml

# 2. Tiny HF model smoke test — requires internet (~30 s)
python scripts/09_run_tiny_model_smoke.py \
    --experiment-config configs/experiment.yaml \
    --model-config configs/tiny_gpt2_debug.yaml

# 3. Qwen dry run — tokenizer only, no GPU
python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --split train --dry-run
```

### Named-run (output isolation)

Pass `--run-name <name>` to scripts 03–07 to write all artifacts under an isolated subdirectory (e.g. `qwen_pilot_32`). Always use this for pilots and experiments; omit for the default full run.

## Architecture

### Two config files drive everything

- `configs/experiment.yaml` — dataset sizes (700/150/150 train/val/test), domain list, control types, C-grid for logistic regression, all output paths.
- `configs/qwen2_5_7b.yaml` (or another model config) — `model_name`, `model_short_name`, `batch_size`, `max_length`, `dtype`, `device_map`, `token_position`, `use_chat_template`.

Both are loaded into typed dataclasses (`ExperimentConfig`, `ModelConfig`) by `src/deception_guardrail/config.py`.

### Data layer (`src/deception_guardrail/data/`)

- `facts.py` — static bank of (fact, question, truth, lie_target, domain) tuples; the raw material for the dataset.
- `build_dataset.py` — samples from the fact bank, creates paired prompts, assigns pair-level splits (both honest and deceptive variants always land in the same split to prevent pair leakage), writes three JSONL files to `data/processed/`.
- `schema.py` — Pydantic models: `FactualLiePair`, `ProbeSample`, `BenignControl`. Enforces domain allowlist, binary label, and truth ≠ lie_target constraint.
- `controls.py` — generates benign control prompts across six categories.
- `validate_dataset.py` — strict checks run by script 02; exits nonzero on any violation.

### Activation layer (`src/deception_guardrail/activations/`)

- `capture.py` — `capture_activations()`: batched forward passes with `output_hidden_states=True, use_cache=False`; extracts activations at the **final non-padding prompt token** (left-padded batches); stacks into `[N, num_layers, hidden_dim]` float32 CPU tensors. Excludes `hidden_states[0]` (embedding); layer index 0 = transformer block 0 output.
- `store.py` — `save_activations()` / `load_activations()` / `activation_path()`; file layout: `artifacts/activations/{model_short_name}/[{run_name}/]{split}_activations.pt`.
- `validate.py` — checks artifact keys, tensor shape/dtype/device, NaN/inf, label count consistency.

### Model utilities (`src/deception_guardrail/models/`)

- `load_model.py` — loads `AutoModelForCausalLM` + `AutoTokenizer`; sets pad token to eos if missing; configures left-padding; calls `model.eval()`.
- `chat_format.py` — applies the HF chat template (or falls back to plain text) to format raw prompts for the tokenizer.

### Probe layer (`src/deception_guardrail/probes/`)

- `train.py` — for each layer: fits `StandardScaler` on train activations only, then fits `LogisticRegression` over C-grid, selects best C on validation AUROC. Returns list of `(scaler, clf)` pairs saved as `artifacts/probes/{model_short_name}/layerwise_probes.pkl`.
- `evaluate.py` — computes per-layer AUROC, AUPRC, accuracy, F1, score separation; writes `results/metrics/layerwise_probe_metrics.csv` and `best_layer_summary.json`.
- `calibration.py` — scores benign controls with each layer's probe; computes FPR at 0.5 threshold and TPR@1%/5%/10% benign-control FPR; writes `results/metrics/control_calibration.csv`.

### Analysis layer (`src/deception_guardrail/analysis/`)

- `plots.py` — six standard figures: layer vs test AUROC, layer vs val AUROC, layer vs test AUPRC, layer vs control FPR, TPR-at-fixed-FPR by layer, score distributions at best layer.
- `summaries.py` — compiles run metadata JSON written to `artifacts/metadata/{run_id}.json`.

### Utility layer (`src/deception_guardrail/utils/`)

- `paths.py` — path helpers for resolving artifact and result directories.
- `logging.py` — standard logger factory.
- `seed.py` — seeds Python, NumPy, and PyTorch.
- `io.py` — JSONL read/write helpers.

## Key conventions

**Layer indexing:** All artifacts use 0-indexed transformer block numbers. `layer_indices[i] == i`; `tensor[:, i, :]` = output of transformer block `i` = `hidden_states[i+1]` in HF. The embedding output (`hidden_states[0]`) is never saved.

**Pair leakage rule:** Both the honest and deceptive variants of every fact pair must always be in the same split. `validate_dataset.py` enforces this and exits nonzero if violated. Never split pairs across train/test.

**Scaler fit rule:** `StandardScaler` is always fit on training activations only. Validation and test activations are transformed with the already-fitted scaler.

**Test set is final:** The C regularization parameter is selected on validation AUROC; test metrics are computed once after hyperparameter selection.

**Output isolation:** Use `--run-name` for pilots and one-off experiments. The default (no `--run-name`) paths are for the canonical full run.

## Adding a new model

Create a new YAML in `configs/` following the pattern in `configs/qwen2_5_7b.yaml`, then re-run scripts 03–07 with `--model-config configs/<new_model>.yaml`. Each model's artifacts live under `artifacts/activations/{model_short_name}/`.

## Debugging

| Symptom | Likely cause |
|---|---|
| `RuntimeError: Fact bank has only N entries` | Duplicates removed too aggressively in `facts.py` |
| `ValidationError: Expected 1000 pairs` | Re-run script 01; add more facts if fact bank is too small |
| `FileNotFoundError: Activation file not found` | Run script 03 for all splits before script 04 |
| OOM during capture | Reduce `batch_size` in model config |
| AUROC ~0.5 on all layers | Check model loaded correctly, chat template applied, `output_hidden_states=True` passed, correct token position indexed |
| High control FPR | Probe may be detecting prompt style, not deception intent (expected in Phase 1) |
| Val AUROC high, test AUROC drops | Possible overfitting or split leakage |
