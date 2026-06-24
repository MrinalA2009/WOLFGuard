#!/usr/bin/env bash
# =============================================================================
# run_lambda_7b.sh — Full Qwen2.5-7B experiment on Lambda CUDA
#
# Usage (inside tmux on Lambda):
#   chmod +x scripts/run_lambda_7b.sh
#   ./scripts/run_lambda_7b.sh 2>&1 | tee logs/lambda_7b_$(date +%Y%m%d_%H%M%S).log
#
# Optional: set HF_TOKEN before running if the model is gated.
#   export HF_TOKEN=hf_...
#
# Exit codes: 0 = all stages passed, 1 = stage failed (run is stopped)
# =============================================================================

set -euo pipefail

CFG=configs/qwen2_5_7b.yaml
EXP=configs/experiment.yaml

PILOT4=qwen7b_cuda_pilot_4
PILOT32=qwen7b_cuda_pilot_32
FULL=qwen7b_full_v1

LOG_DIR=logs
mkdir -p "$LOG_DIR"

SECONDS_GLOBAL=$SECONDS

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
check_pass() { echo "[PASS] $1"; }
check_fail() { echo "[FAIL] $1"; exit 1; }

elapsed() {
    local s=$1
    printf "%dm%02ds" $((s / 60)) $((s % 60))
}

section() {
    echo ""
    echo "================================================================="
    echo "  $1"
    echo "================================================================="
    echo ""
}

# ------------------------------------------------------------------
# STAGE 0 — Environment verification
# ------------------------------------------------------------------
section "STAGE 0 — Environment verification"

echo "--- CUDA ---"
nvidia-smi || check_fail "nvidia-smi failed — no NVIDIA GPU detected"
echo ""

python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("[FAIL] CUDA not available — aborting")
    sys.exit(1)

print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"total VRAM (GB): {total_gb:.1f}")

if total_gb < 14:
    print(f"[FAIL] Insufficient VRAM: {total_gb:.1f} GB (need >= 16 GB)")
    sys.exit(1)

if not torch.cuda.is_bf16_supported():
    print("[WARN] bfloat16 not supported — experiment requires bfloat16 (Ampere+)")
PY
python_exit=$?
[ $python_exit -ne 0 ] && check_fail "CUDA Python check failed"

echo "--- Disk ---"
df -h .
du -sh ~/.cache/huggingface 2>/dev/null || echo "(HF cache empty)"

echo ""
echo "--- Model architecture ---"
python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print("model_type:", cfg.model_type)
print("num_hidden_layers:", cfg.num_hidden_layers)
print("hidden_size:", cfg.hidden_size)
print("num_attention_heads:", cfg.num_attention_heads)
print("num_key_value_heads:", getattr(cfg, "num_key_value_heads", None))
print("torch_dtype:", getattr(cfg, "torch_dtype", None))
print("vocab_size:", cfg.vocab_size)
# Expected activation tensor shape (embedding excluded):
print(f"expected_activation_shape: [N, {cfg.num_hidden_layers}, {cfg.hidden_size}]")
PY

echo ""
echo "--- Dataset validation ---"
python scripts/01_build_dataset.py --experiment-config "$EXP"
python scripts/02_validate_dataset.py --experiment-config "$EXP"

echo ""
echo "--- Tests ---"
python -m pytest tests/ -v --tb=short 2>&1 | tail -20

check_pass "Stage 0 complete"

# ------------------------------------------------------------------
# STAGE A — Pilot 4
# ------------------------------------------------------------------
section "STAGE A — qwen7b_cuda_pilot_4 (N=8 total per split)"

echo "--- Dry run (tokenizer only) ---"
python scripts/03_capture_activations.py \
    --model-config "$CFG" \
    --experiment-config "$EXP" \
    --split train \
    --limit-pairs 4 \
    --dry-run \
    --run-name "$PILOT4"

echo ""
echo "--- Capture pilot_4 activations ---"
T0=$SECONDS
for split in train validation test; do
    echo "Capturing split: $split"
    python scripts/03_capture_activations.py \
        --model-config "$CFG" \
        --experiment-config "$EXP" \
        --split "$split" \
        --limit-pairs 4 \
        --run-name "$PILOT4"
done
python scripts/03_capture_activations.py \
    --model-config "$CFG" \
    --experiment-config "$EXP" \
    --controls \
    --limit-controls 4 \
    --run-name "$PILOT4"
echo "Capture wall-clock: $(elapsed $((SECONDS - T0)))"

echo ""
echo "--- Validate pilot_4 shapes ---"
python - <<'PY'
import sys
import torch
from pathlib import Path

base = Path("artifacts/activations/qwen2_5_7b/qwen7b_cuda_pilot_4")
ok = True
for name, fname, expected_n in [
    ("train",      "train_activations.pt",      8),
    ("validation", "validation_activations.pt", 8),
    ("test",       "test_activations.pt",        8),
    ("control",    "control_activations.pt",     4),
]:
    p = base / fname
    if not p.exists():
        print(f"[FAIL] {fname}: MISSING")
        ok = False
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    acts = d["activations"]
    shape = list(acts.shape)
    has_nan = bool(torch.isnan(acts).any())
    has_inf = bool(torch.isinf(acts).any())
    status = "OK" if shape[0] == expected_n and not has_nan and not has_inf else "FAIL"
    print(f"[{status}] {fname}: shape={shape}  nan={has_nan}  inf={has_inf}")
    if status == "FAIL":
        ok = False

if not ok:
    print("[FAIL] Pilot_4 artifact validation failed")
    sys.exit(1)
print("[PASS] All pilot_4 shapes valid")
PY

echo ""
echo "--- Probe training ---"
python scripts/04_train_layerwise_probes.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$PILOT4"

echo ""
echo "--- Control evaluation ---"
python scripts/05_evaluate_controls.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$PILOT4"

echo ""
echo "--- Run summary ---"
python scripts/07_summarize_run.py \
    --model-config "$CFG" --experiment-config "$EXP" \
    --run-name "$PILOT4" --run-id "$PILOT4"

check_pass "Stage A (pilot_4) complete"

# ------------------------------------------------------------------
# STAGE B — Pilot 32
# ------------------------------------------------------------------
section "STAGE B — qwen7b_cuda_pilot_32 (N=32 per split)"

echo "--- Capture pilot_32 activations ---"
T0=$SECONDS
for split in train validation test; do
    echo "Capturing split: $split"
    SPLIT_T=$SECONDS
    python scripts/03_capture_activations.py \
        --model-config "$CFG" \
        --experiment-config "$EXP" \
        --split "$split" \
        --limit-pairs 16 \
        --run-name "$PILOT32"
    echo "  Split '$split' wall-clock: $(elapsed $((SECONDS - SPLIT_T)))"
done
CONTROLS_T=$SECONDS
python scripts/03_capture_activations.py \
    --model-config "$CFG" \
    --experiment-config "$EXP" \
    --controls \
    --limit-controls 32 \
    --run-name "$PILOT32"
echo "  Controls wall-clock: $(elapsed $((SECONDS - CONTROLS_T)))"

PILOT32_CAPTURE_SECS=$((SECONDS - T0))
echo "Pilot_32 capture total: $(elapsed $PILOT32_CAPTURE_SECS)"

# Estimate full-run time
python - <<PY
capture_secs = $PILOT32_CAPTURE_SECS
# train: 16 pairs → 32 samples. Full train is 700 pairs → 1400 samples (43.75x)
# val:   16 pairs → 32 samples. Full val  is 150 pairs → 300 samples (9.375x)
# test:  16 pairs → 32 samples. Full test is 150 pairs → 300 samples (9.375x)
# controls: 32 samples. Full controls is 1017 samples (31.8x)
# pilot_32 is 3 splits × 32 + 32 controls = 128 samples total
# full run is 1400 + 300 + 300 + 1017 = 3017 samples total
# Scale factor ≈ 3017/128 ≈ 23.6x (rough lower bound; model reloads 4×)
scale = 3017 / 128
est_min = (capture_secs * scale) / 60
print(f"Pilot_32 capture: {capture_secs}s")
print(f"Scale factor to full run: {scale:.1f}x")
print(f"Estimated full-run capture: ~{est_min:.0f} minutes")
PY

echo ""
echo "--- Validate pilot_32 shapes ---"
python - <<'PY'
import sys
import torch
from pathlib import Path

base = Path("artifacts/activations/qwen2_5_7b/qwen7b_cuda_pilot_32")
ok = True
for name, fname, expected_n in [
    ("train",      "train_activations.pt",      32),
    ("validation", "validation_activations.pt", 32),
    ("test",       "test_activations.pt",        32),
    ("control",    "control_activations.pt",     32),
]:
    p = base / fname
    if not p.exists():
        print(f"[FAIL] {fname}: MISSING")
        ok = False
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    acts = d["activations"]
    shape = list(acts.shape)
    has_nan = bool(torch.isnan(acts).any())
    has_inf = bool(torch.isinf(acts).any())
    status = "OK" if shape[0] == expected_n and not has_nan and not has_inf else "FAIL"
    print(f"[{status}] {fname}: shape={shape}  nan={has_nan}  inf={has_inf}")
    if status == "FAIL":
        ok = False

if not ok:
    print("[FAIL] Pilot_32 artifact validation failed")
    sys.exit(1)
print("[PASS] All pilot_32 shapes valid")
PY

echo ""
echo "--- Probe training + evaluation + plots ---"
python scripts/04_train_layerwise_probes.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$PILOT32"
python scripts/05_evaluate_controls.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$PILOT32"
python scripts/06_make_plots.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$PILOT32"
python scripts/07_summarize_run.py \
    --model-config "$CFG" --experiment-config "$EXP" \
    --run-name "$PILOT32" --run-id "$PILOT32"

check_pass "Stage B (pilot_32) complete"

# ------------------------------------------------------------------
# STAGE C — Full run
# ------------------------------------------------------------------
section "STAGE C — qwen7b_full_v1 (full dataset)"

echo "--- Capture full activations ---"
T0=$SECONDS
for split in train validation test; do
    echo "Capturing split: $split"
    SPLIT_T=$SECONDS
    python scripts/03_capture_activations.py \
        --model-config "$CFG" \
        --experiment-config "$EXP" \
        --split "$split" \
        --run-name "$FULL"
    echo "  Split '$split' wall-clock: $(elapsed $((SECONDS - SPLIT_T)))"
done

CTRL_T=$SECONDS
python scripts/03_capture_activations.py \
    --model-config "$CFG" \
    --experiment-config "$EXP" \
    --controls \
    --run-name "$FULL"
echo "  Controls wall-clock: $(elapsed $((SECONDS - CTRL_T)))"
echo "Full capture total: $(elapsed $((SECONDS - T0)))"

echo ""
echo "--- Validate full-run shapes ---"
python - <<'PY'
import sys
import torch
from pathlib import Path

base = Path("artifacts/activations/qwen2_5_7b/qwen7b_full_v1")
expected = {
    "train_activations.pt":      1400,
    "validation_activations.pt":  300,
    "test_activations.pt":        300,
    "control_activations.pt":    1017,
}
ok = True
for fname, exp_n in expected.items():
    p = base / fname
    if not p.exists():
        print(f"[FAIL] {fname}: MISSING")
        ok = False
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    acts = d["activations"]
    shape = list(acts.shape)
    has_nan = bool(torch.isnan(acts).any())
    has_inf = bool(torch.isinf(acts).any())
    status = "OK" if shape[0] == exp_n and not has_nan and not has_inf else "FAIL"
    print(f"[{status}] {fname}: shape={shape}  nan={has_nan}  inf={has_inf}")
    if status == "FAIL":
        ok = False

if not ok:
    print("[FAIL] Full-run artifact validation failed")
    sys.exit(1)
print("[PASS] All full-run shapes valid")
PY

echo ""
echo "--- Probe training + evaluation + plots + summary ---"
python scripts/04_train_layerwise_probes.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$FULL"
python scripts/05_evaluate_controls.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$FULL"
python scripts/06_make_plots.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$FULL"
python scripts/07_summarize_run.py \
    --model-config "$CFG" --experiment-config "$EXP" \
    --run-name "$FULL" --run-id "$FULL"

echo ""
echo "--- Error inspection (top 20) ---"
python scripts/14_inspect_errors.py \
    --model-config "$CFG" --experiment-config "$EXP" \
    --run-name "$FULL" --top-k 20

echo ""
echo "--- Domain held-out (CPU only) ---"
python scripts/16_domain_held_out.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$FULL"

echo ""
echo "--- Template held-out (requires model reload) ---"
python scripts/15_template_held_out.py \
    --model-config "$CFG" --experiment-config "$EXP" --run-name "$FULL"

# ------------------------------------------------------------------
# Hard-pair monitoring
# ------------------------------------------------------------------
section "Hard-pair monitoring"

python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import torch
import csv

from deception_guardrail.activations.store import load_activations
from deception_guardrail.probes.train import load_probes, select_best_layer
from deception_guardrail.probes.evaluate import compute_scores

MONITOR = ["pair_0094", "pair_0994", "pair_0992", "pair_0348"]

probes_path = Path("artifacts/probes/qwen2_5_7b/qwen7b_full_v1/layerwise_probes.pkl")
results = load_probes(probes_path)

cal_csv = Path("results/metrics/qwen7b_full_v1/control_calibration.csv")
ctrl_fpr = {}
if cal_csv.exists():
    with open(cal_csv) as f:
        for row in csv.DictReader(f):
            ctrl_fpr[int(row["layer_index"])] = float(row["fpr_at_threshold_0_5"])
best = select_best_layer(results, control_fpr=ctrl_fpr if ctrl_fpr else None)
print(f"Best layer: {best.layer_index}, tensor_index: {best.tensor_index}")
print(f"Test AUROC: {best.test_metrics['auroc']:.4f}")

test_art = load_activations(Path("artifacts/activations/qwen2_5_7b/qwen7b_full_v1/test_activations.pt"))
acts = test_art["activations"].numpy()[:, best.tensor_index, :]
scores = compute_scores(best.classifier, best.scaler, acts)
labels = np.array(test_art["labels"])
sample_ids = test_art["sample_ids"]
pair_ids = test_art.get("pair_ids") or []

print("\n--- Monitored hard pairs (v1 template) ---")
baseline = {
    "pair_0094": {"honest": 0.9992, "deceptive": 0.9804},
    "pair_0994": {"honest": 0.7020, "deceptive": 0.9989},
    "pair_0992": {"honest": 0.9720, "deceptive": None},
    "pair_0348": {"honest": None,   "deceptive": 0.9525},
}
for pid in MONITOR:
    for i, (sid, label) in enumerate(zip(sample_ids, labels)):
        if isinstance(pair_ids, list) and i < len(pair_ids) and pair_ids[i] == pid:
            cond = "honest" if label == 0 else "deceptive"
            sc = scores[i]
            b = baseline[pid][cond]
            delta = f"{sc - b:+.4f}" if b is not None else "new"
            print(f"  {pid} [{cond}]: 7B score={sc:.4f}  1.5B={b}  delta={delta}")
PY

# Template v2 hard-pair scores (extracted from template_held_out run)
echo ""
echo "Template v2 scores for hard pairs:"
python - <<'PY'
import json
import sys
from pathlib import Path

# Try to load template_held_out results for v2 scores
tho_path = Path("results/metrics/qwen7b_full_v1/template_held_out.json")
if tho_path.exists():
    with open(tho_path) as f:
        tho = json.load(f)
    print(f"Template held-out AUROC (v2): {tho.get('v2_auroc', 'N/A'):.4f}")
    print(f"AUROC drop v1→v2: {tho.get('auroc_drop', 'N/A'):+.4f}")
    print(f"Verdict: {tho.get('verdict', 'N/A')}")
else:
    print("template_held_out.json not found — run script 15 manually")
PY

# ------------------------------------------------------------------
# Final report
# ------------------------------------------------------------------
section "All stages complete"

TOTAL=$((SECONDS - SECONDS_GLOBAL))
echo "Total wall-clock: $(elapsed $TOTAL)"
echo ""
echo "Output paths:"
echo "  artifacts/activations/qwen2_5_7b/qwen7b_full_v1/"
echo "  artifacts/probes/qwen2_5_7b/qwen7b_full_v1/"
echo "  artifacts/metadata/qwen7b_full_v1.json"
echo "  results/metrics/qwen7b_full_v1/"
echo "  results/figures/qwen7b_full_v1/"

echo ""
echo "Next: fill in docs/qwen7b_full_v1_report.md with the metrics above"
echo "      and commit the metadata + results (not activation tensors)."
