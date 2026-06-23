# Deception Guardrail — Phase 1: Linear Probe Experiment

## Project overview

This repository implements Phase 1 of a mechanistic interpretability experiment designed to test whether direct factual deception can be detected from the internal activations of an instruction-tuned open-source language model.

**Research question:**  
Can direct factual deception be linearly detected from internal activations of an instruction-tuned LLM, and does the deception signal concentrate in particular transformer layers?

**Relationship to `research.md`:**  
`research.md` contains the literature review and design brief. The key findings driving this design are: (1) truth-related information is often linearly readable from residual stream activations; (2) robustness is the central bottleneck — probes can detect prompt style or roleplay mode rather than deception intent; (3) monitor evasion is real and this should be treated as a defense-in-depth component, not a guarantee. This Phase 1 implements the simplest scientifically valid version of the probe pipeline before any causal intervention.

---

## What Phase 1 tests

- Whether logistic regression probes trained on residual-stream activations at the final prompt token can classify honest vs. deceptive prompts above chance.
- Which transformer layers carry the strongest linear deception signal.
- Whether that signal survives calibration against benign controls (i.e., is it detecting deception intent, or just prompt style?).
- TPR at fixed FPR thresholds (1%, 5%, 10%) calibrated on benign controls.

## What Phase 1 does NOT test

- Robustness to stylistic shifts, adversarial suffixes, or distribution shift.
- Multi-turn or strategic deception (only direct factual lying under explicit instruction).
- Causal intervention (no activation steering or neuron disabling).
- Generalization across models or domains not in the training set.
- Whether a deployed model would fail to evade this probe.

**This experiment does not prove that a model cannot deceive. It tests whether direct factual deception, under controlled prompt-pair conditions, is linearly detectable from internal activations.**

---

## Pre-Qwen validation checklist

Before running full activation capture on Qwen/Qwen2.5-7B-Instruct, execute these
steps in order.  Each step is designed to catch a distinct class of bugs without
requiring a GPU or a large model download.

### 1 — Synthetic end-to-end (no model required, ~10 seconds)

Generates synthetic activations with a known deception signal at layers 3–5
(0-indexed), then runs probe training, calibration, plots, and run summary.

```bash
python scripts/08_run_synthetic_e2e.py --experiment-config configs/experiment.yaml
```

**Expected output:**
- Best layer: one of [3, 4, 5]
- Best validation AUROC > 0.95
- Best test AUROC > 0.95
- TPR at 5% benign-control FPR > 0.90
- All artifact, metric, plot, and metadata files written

If any assertion fails, the script exits nonzero with a clear error message.

### 2 — Tiny HuggingFace model smoke test (requires internet, ~30 seconds)

Uses `sshleifer/tiny-gpt2` to exercise the **exact same** `capture_activations()`
code path as Qwen, on a 8-sample subset.  Results are not scientifically meaningful.

```bash
python scripts/09_run_tiny_model_smoke.py \
    --experiment-config configs/experiment.yaml \
    --model-config configs/tiny_gpt2_debug.yaml
```

**Verifies:**
- `activations` is a CPU float32 tensor of shape `[N, L, D]`
- `layer_indices = [0, 1, ..., L-1]` (0-indexed)
- `token_position = "final_prompt_token"`
- `model_name` is saved correctly
- `sample_ids` length matches N
- Probe training, calibration, and plots do not crash

If the model download fails, the script exits 0 with a `[SKIP]` message.

### 3 — Qwen dry run (requires Qwen model in HF cache or network access)

Loads only the tokenizer (not the 7B weights), formats a few prompts, and
prints the tokenized shapes.  No GPU required.  No artifact written.

```bash
python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --split train --dry-run
```

Check the logged `input_ids.shape` and the last 60 characters of each formatted
prompt to verify the chat template is applied correctly.

### 4 — Qwen pilot capture (requires GPU with CUDA/bfloat16 support)

Captures 4 pairs (8 samples) per split under an isolated run namespace to avoid
touching full-run artifacts.

> **MPS (Apple Silicon) limitation:** Qwen/Qwen2.5-7B-Instruct requires bfloat16,
> which is not supported on MPS.  Float16 causes NaN overflow in deep layers (visible
> from layer 19 onward).  The 7B model requires a CUDA GPU with at least 16 GB VRAM.
> For pipeline validation on Apple Silicon, use `configs/qwen2_5_1b5_mps.yaml`
> (Qwen2.5-1.5B-Instruct in float32; see pilot results below).

```bash
# 4-pair limit pilot (all splits, isolated namespace)
for split in train validation test; do
  python scripts/03_capture_activations.py \
      --model-config configs/qwen2_5_7b.yaml \
      --experiment-config configs/experiment.yaml \
      --split $split --limit-pairs 4 --run-name qwen_pilot_4
done
python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --controls --limit-controls 4 --run-name qwen_pilot_4
```

Verify train artifact has shape `[8, 28, 3584]` (4 pairs × 2 labels, 28 layers, 3584 hidden dim).

After all four pre-Qwen steps pass, proceed to the pilot pipeline.

### 5 — Qwen 32-pair pilot (requires GPU)

Run the full pipeline over 32 pairs per split under a named namespace:

```bash
# Capture
for split in train validation test; do
  python scripts/03_capture_activations.py \
      --model-config configs/qwen2_5_7b.yaml \
      --experiment-config configs/experiment.yaml \
      --split $split --limit-pairs 32 --run-name qwen_pilot_32
done
python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --controls --limit-controls 32 --run-name qwen_pilot_32

# Train + evaluate + plot + summarize
python scripts/04_train_layerwise_probes.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --run-name qwen_pilot_32
python scripts/05_evaluate_controls.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --run-name qwen_pilot_32
python scripts/06_make_plots.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --run-name qwen_pilot_32
python scripts/07_summarize_run.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --run-name qwen_pilot_32 --run-id qwen_pilot_32
```

> **Important:** Pilot AUROC is not scientific evidence.
> N=32 pairs per split gives ~±0.08 confidence intervals on AUROC.
> The pilot validates pipeline correctness only.

After all steps pass, proceed to the full pipeline (Steps 3–7) without `--limit-pairs` or `--run-name`.

---

### Layer indexing convention

All activation artifacts produced by this codebase use **0-indexed** layer numbers:

| `layer_indices` value | Corresponds to | HF `hidden_states` index |
|---|---|---|
| 0 | Transformer block 0 output | `hidden_states[1]` |
| 1 | Transformer block 1 output | `hidden_states[2]` |
| L-1 | Transformer block L-1 output (last) | `hidden_states[L]` |

`hidden_states[0]` (embedding output) is **excluded** from all artifacts.
`tensor_index == layer_index` for every saved artifact.

---

## Installation

```bash
# Clone and enter repo
git clone <repo-url>
cd WOLFGuard

# Create environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -e ".[dev]"
```

**Hardware:**
- Activation capture (Qwen2.5-7B): CUDA GPU with ≥16 GB VRAM, bfloat16 support required.  Apple MPS does not support bfloat16; float16 causes NaN overflow in Qwen2.5-7B from layer ~19.
- Pipeline validation (MPS, Apple Silicon): Use `configs/qwen2_5_1b5_mps.yaml` (Qwen2.5-1.5B-Instruct, float32, ~6 GB MPS memory).
- Probe training, calibration, plots, and all tests: CPU only.

---

## Exact run commands

### Step 1 — Build dataset

```bash
python scripts/01_build_dataset.py --experiment-config configs/experiment.yaml
```

Outputs: `data/processed/factual_lie_pairs.jsonl`, `probe_samples.jsonl`, `benign_controls.jsonl`, `dataset_summary.json`

### Step 2 — Validate dataset

```bash
python scripts/02_validate_dataset.py --experiment-config configs/experiment.yaml
```

Exits with code 1 if any validation check fails. Run this before capturing activations.

### Step 3 — Capture activations (requires GPU + model download)

```bash
python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --split train

python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --split validation

python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --split test

python scripts/03_capture_activations.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --controls
```

Each split is resumable: if the `.pt` file already exists, it is skipped.

### Step 4 — Train layer-wise probes

```bash
python scripts/04_train_layerwise_probes.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml
```

### Step 5 — Evaluate controls and calibrate

```bash
python scripts/05_evaluate_controls.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml
```

### Step 6 — Generate plots

```bash
python scripts/06_make_plots.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml
```

### Step 7 — Print and save run summary

```bash
python scripts/07_summarize_run.py \
    --model-config configs/qwen2_5_7b.yaml \
    --experiment-config configs/experiment.yaml \
    --run-id phase1_qwen2_5_7b
```

### Run tests (no GPU required)

```bash
pytest tests/ -v
```

---

## Expected output files

After a complete run:

```
data/processed/
    factual_lie_pairs.jsonl      # 1000 pair-level rows
    probe_samples.jsonl          # 2000 sample-level rows
    benign_controls.jsonl        # 300+ control rows
    dataset_summary.json

artifacts/activations/qwen2_5_7b/
    train_activations.pt         # [1400, 28, 3584]
    validation_activations.pt    # [300, 28, 3584]
    test_activations.pt          # [300, 28, 3584]
    control_activations.pt       # [310+, 28, 3584]

artifacts/probes/qwen2_5_7b/
    layerwise_probes.pkl         # 28 fitted (scaler, LogisticRegression) pairs

artifacts/metadata/
    {run_id}.json                # full run metadata

results/metrics/
    layerwise_probe_metrics.csv  # per-layer metrics
    control_calibration.csv      # per-layer FPR and TPR@fixed FPR
    best_layer_summary.json      # metrics for the best layer

results/figures/
    layer_vs_test_auroc.png
    layer_vs_validation_auroc.png
    layer_vs_test_auprc.png
    layer_vs_control_fpr_at_0_5.png
    tpr_at_fixed_fpr_by_layer.png
    score_distributions_best_layer.png
```

---

## Metric interpretation

| Condition | Interpretation |
|---|---|
| Best test AUROC > 0.95 | Very strong signal under this controlled setup |
| Best test AUROC > 0.85 | Promising — investigate robustness next |
| Best test AUROC 0.65–0.85 | Moderate — check prompt design, token position, model loading |
| Best test AUROC < 0.65 | Weak — likely an implementation or data issue |
| High control FPR (>20%) | Probe may be detecting prompt style, not deception intent |
| Val AUROC high, test AUROC drops | Possible overfitting or split leakage |
| Train perfect, val/test weak | Probe is not robust |

**TPR at fixed FPR** is the operational metric: even with high AUROC, a probe that fires on 30% of benign chat is not useful as a monitor.

---

## Validity risks

**Pair leakage:** Both honest and deceptive variants of each pair are always assigned to the same split. If they were split across train/test, the probe could exploit shared factual content rather than learning the deception instruction. The validator checks this explicitly and will exit with code 1 if violated.

**Lexical leakage:** The deceptive prompt contains `lie_target` as a string; the honest prompt does not. Because we capture activations at the final *prompt* token (before any generation), the probe is reading the model's internal representation of the task instruction, not its output text. However, the probe could still detect lexical differences in the instruction text rather than a semantic "deception intent" representation. This is a genuine limitation of Phase 1 that Phase 2 should address with style-augmentation and ablation.

**High AUROC alone is not enough:** A probe with AUROC=0.99 that fires on 40% of benign roleplay prompts is not a useful safety monitor. Always check `control_calibration.csv` and `tpr_at_fixed_fpr_by_layer.png` before drawing conclusions.

**Benign-control FPR:** Controls include normal factual QA, creative writing, harmless roleplay, uncertainty explanations, non-deceptive persuasion, and instruction-following. High FPR on any of these categories signals that the probe is detecting that category's style, not deception.

---

## Why pair-level split integrity matters

The honest and deceptive prompts for a given fact share the fact sentence verbatim. If they ended up in different splits, a probe trained on the train split could learn to recognize specific facts and use that signal to predict the label on the test split — not because it learned to detect deception, but because it saw the matching fact in a deceptive context during training. Splitting at the pair level prevents this.

## Why this is a foundation for causal intervention but does not implement it

A linear probe identifies a *direction* in activation space that correlates with deception. Phase 2 would: (1) extract this direction as a mean-difference vector; (2) attempt to subtract it from activations during inference; (3) measure whether the model still produces deceptive outputs; (4) measure utility degradation on benign tasks. Phase 1 only establishes that the linear signal exists — which is a prerequisite for any intervention to make sense.

---

## Switching models

Edit or create a new config file following the pattern in `configs/qwen2_5_7b.yaml`:

```yaml
model_name: mistralai/Mistral-7B-Instruct-v0.3
model_short_name: mistral_7b
batch_size: 8
max_length: 512
dtype: auto
device_map: auto
token_position: final_prompt_token
use_chat_template: true
trust_remote_code: false
```

Then re-run steps 3–7 with `--model-config configs/mistral_7b.yaml`. Each model's activations are stored under `artifacts/activations/{model_short_name}/`.

Supported model families (chat template compatible): `Qwen/Qwen2.5-*-Instruct`, `mistralai/Mistral-*-Instruct-*`, `meta-llama/Llama-3.*-*-Instruct`.

---

## Debugging common failures

**`RuntimeError: Fact bank has only N entries`** → The deduplication in `facts.py` removed too many entries. Check for accidentally identical `(fact, question)` pairs.

**`ValidationError: Expected 1000 pairs, got N`** → Re-run `01_build_dataset.py`. If the fact bank is below 1000 unique entries after deduplication, add more facts to `src/deception_guardrail/data/facts.py`.

**`FileNotFoundError: Activation file not found`** → Run step 3 for all splits and controls before step 4. Order matters.

**OOM during activation capture** → Reduce `batch_size` in `configs/qwen2_5_7b.yaml` (try 2 or 4). Activation tensors are moved to CPU after each batch, so VRAM usage is bounded by the batch size.

**AUROC ~0.5 on all layers** → Check (1) that the model loaded correctly; (2) that the chat template is being applied; (3) that the tokenizer pad token is set; (4) that `output_hidden_states=True` is passed; (5) that you are indexing the correct token position (final non-padding token with left padding).

**High control FPR** → The probe may be detecting the style of the deceptive prompt instruction rather than a deep internal representation of deception intent. This is expected in Phase 1 and motivates Phase 2 style-augmentation.

---

## What result justifies moving to Phase 2

**Minimum bar:** Best layer test AUROC > 0.80 AND control FPR at threshold 0.5 < 0.15 on at least one layer.

**Comfortable bar:** Best layer test AUROC > 0.90, TPR@5%FPR > 0.70, control FPR < 0.10.

If these are met, Phase 2 should:
1. Extract the mean-difference direction at the best layer.
2. Apply inference-time intervention (add/subtract direction during generation).
3. Measure deception rate reduction and utility regression on benign tasks.
4. Add style-shift augmentation to test robustness.
5. Test on a second model family.

If AUROC is weak or control FPR is high, investigate prompt design, token position, and whether a different model family gives cleaner separation before moving to intervention.

---

## License and legal reuse

This codebase is written from scratch. It does not contain code from repositories with unclear licensing (Apollo deception-detection, geometry-of-truth, SafeSwitch). Conceptual design draws on publicly available research papers. Model weights are subject to their respective upstream licenses (Qwen: Tongyi Qianwen License; Mistral/LLaMA: their respective model licenses).

---

## Related Work

The closest prior work is Goldowsky-Dill et al. (2025), "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407, ApolloResearch). That paper trains linear probes on the residual-stream activations of instruction-tuned LLMs and reports AUROC of 0.96–0.999 for detecting honest versus deceptive behavior across a range of scenarios. The authors caution that high AUROC does not imply robustness: probes can be evaded by adversarial prompt reformulations, and separation may reflect prompt style rather than a deep internal representation of deception intent. WOLFGuard Phase 1 implements an independently designed version of this core experiment — paired factual prompts, logistic regression probes, calibration against benign controls — on open-weight models under an MIT license, with pair-leakage prevention and TPR@fixed-FPR as the primary operational metric.
