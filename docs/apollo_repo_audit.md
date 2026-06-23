# Apollo Research Repository Audit

**Repository:** https://github.com/ApolloResearch/deception-detection  
**Paper:** "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407)  
**Audit date:** 2026-06-23  
**Auditor:** Mechanistic interpretability pipeline analysis  

---

## 1. Repository Overview

### 1.1 Main Purpose

The ApolloResearch repository implements code from the paper "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407). Its central claim is that linear probes monitoring model activations can detect **strategic deception** — scenarios where a model is instructed (or incentivized) to conceal information from a user or evaluator. The deception is not simple factual lying; it is goal-directed, multi-turn, and contextually embedded.

The paper focuses on **Llama-3.3-70B-Instruct** (a 70-billion parameter model requiring substantial GPU infrastructure) and reports AUROC 0.96–0.999 and TPR@1%FPR of 95–99% on held-out evaluation scenarios distinct from training scenarios.

### 1.2 Paper Relationship

The repository is the official code release for the paper. Probe weights, configurations, and result artifacts are committed to `example_results/` to enable reproduction. The paper is the primary scientific output; the repository is a research-grade implementation, not a library.

### 1.3 Main Datasets

| Dataset | Type | Description |
|---|---|---|
| `roleplaying` | Training | 1,894 scenarios where a person faces pressure to lie (student grade appeals, employment reviews, legal testimony, financial fraud, etc.). Each has honest/deceptive completion pairs. |
| `descriptive_qa__pair` | Training | Contrasting honest/deceptive instructed pairs (similar to Zou et al. 2023). Pre-written Q&A format with explicit truthfulness instruction. |
| `insider_trading` | Evaluation | Multi-turn transcripts where an AI assistant conceals a stock trade conflict of interest. Model-generated rollouts. |
| `sandbagging_v2__wmdp_mmlu` | Evaluation | Scenarios where the model deliberately underperforms on safety evaluations. Requires rollout generation. |
| `ai_liar__original_without_answers` | Evaluation | "How to Catch an AI Liar" benchmark questions. |
| `alpaca__plain` | Control (negative) | 10,000 benign Alpaca instruction-following examples. Used to compute FPR at fixed thresholds. |

### 1.4 Main Experiment Files

```
deception_detection/
  experiment.py          # ExperimentConfig + main training/eval orchestration
  detectors.py           # 7 probe types: MMS, LAT, LR, MLP, CMMS, MLR, BestLatents
  activations.py         # Activation extraction (layers, token positions, batching)
  metrics.py             # AUROC, TPR@FPR computation
  tokenized_data.py      # Tokenization, detection masks, paired dataset alignment
  prompt_scorer.py       # Token-level → sample-level score reduction (mean/max/relu/topk/final)
  repository.py          # Dataset registry and loading
  types.py               # DialogueDataType, Label enum, Message/Dialogue types
  scores.py              # Score container class

scripts/
  experiment.py          # Entrypoint script
  by_layer.py            # Layer-by-layer AUROC and TPR@1%FPR evaluation
  paper_plots.py         # Publication-quality plots (violin, ROC, bar, grid)
  configs/
    roleplaying.yaml     # Llama-70B, lr probe, layer 22, control=alpaca 10k
    descriptive.yaml     # Llama-70B, mms probe, layer 22, control=alpaca 10k
    repe.yaml            # ...
    sae_rp.yaml          # SAE-based probe variant
    sae_repe.yaml        # ...

example_results/
  roleplaying/           # cfg.yaml, detector.pt, scores.json, control_scores.json, results_table.csv
  descriptive/
  instructed_pairs/
  followup/
  sae_rp/
```

### 1.5 Main Package Structure

The `deception_detection` package is a research package, not a clean library. It is tightly coupled to:
- Specific model families (Gemma, Mistral, Llama via HuggingFace)
- External APIs (Anthropic, Together, Goodfire, OpenAI) for rollout generation
- A specific naming scheme for datasets (`BASE__VARIANT__MODEL`)
- A non-standard `DialogueDataset` abstraction with a `detect` field per message

### 1.6 Main Dependencies

**Notable** (beyond standard ML stack):
- `anthropic`, `openai`, `together`: LLM API access for rollout generation
- `goodfire`: SAE feature inspection
- `sae-lens`: Sparse autoencoder features
- `inspect-ai`: AI evaluation framework
- `wandb`: Experiment tracking
- `peft`: LoRA fine-tuning (used for sandbagging setup)
- `streamlit`: Interactive demo
- `circuitsvis`: Mechanistic interpretability visualization

Dependencies that differ from ours: their stack has no package version pinning on transformers, requires Python 3.11+, and has heavy API-key dependencies.

### 1.7 Reproducibility

**Partially reproducible** without API keys:
- Pre-trained probe weights in `example_results/` allow re-running evaluation
- Configs are committed and exact
- Rollout generation (insider trading, sandbagging) requires API keys and is expensive
- `roleplaying` dataset is committed (YAML) and doesn't need API calls

**Not reproducible without API keys:**
- Insider trading transcript generation
- Sandbagging rollout scoring
- Goodfire SAE experiments

### 1.8 External API Keys Required

Yes. `.env` file requires:
- `ANTHROPIC_API_KEY`
- `TOGETHER_API_KEY`
- `HUGGING_FACE_API_KEY`
- `GOODFIRE_API_KEY`
- `OPENAI_API_KEY`

Our project requires **none** of these. This is a meaningful differentiator.

### 1.9 Parts Relevant to Our Phase 1

**Highly relevant:**
- `metrics.py`: AUROC, TPR@1%FPR implementation (conceptually)
- `detectors.py`: Logistic regression with StandardScaler (conceptually)
- `by_layer.py`: Layer-by-layer AUROC + TPR@1%FPR evaluation and plotting pattern
- Roleplaying dataset design: paired honest/deceptive completions with scenario context
- Control dataset approach (10,000 Alpaca; we need to scale up ours)
- `example_results/` structure: committing probe weights + configs for reproducibility

**Partially relevant (adaptation required):**
- Token position strategy: they use "start of assistant turn" or "last token before EOT" in *responses*; we use final prompt token. The comparison is relevant for future work.
- `PromptScorer` reduction methods: applicable if we extend to multi-position probing later.

### 1.10 Parts That Are Later-Stage Only

**Phase 2 or beyond for us:**
- Rollout-based datasets (insider trading, sandbagging): require model generation pipelines
- SAE feature analysis (`sae.py`, `goodfire_client.py`, `BestLatentsDetector`)
- Cross-scenario generalization tests (train on roleplaying, test on insider trading)
- MLP probe (`MLPDetector`): non-linear, requires early stopping
- Multi-layer ensemble scoring (their `MeanLogisticRegressionDetector` flattens across layers)
- `streamlit` interactive probe explorer

---

## 2. License and Reuse Assessment

### 2.1 License Findings

**No open-source license was found.**

- No `LICENSE` or `COPYING` file is present in the repository root
- `pyproject.toml` contains no `license` field, `classifiers`, or `license-files` entry
- `README.md` contains no license badge or license text
- `ACCESS.md` states only "Disclosure Level - Public" — this indicates the repository is publicly visible on GitHub, **not** that its code is freely reusable
- No SPDX identifier was found anywhere in the codebase

Under the GitHub Terms of Service (section D.7), forking and *viewing* a public repository is permitted. However, in the absence of a license, **all code and data remain All Rights Reserved** under the Berne Convention. Public ≠ open source.

**Legal status:** Source-available, all rights reserved.

### 2.2 Component Classification

| Component | License Status | Recommended Treatment |
|---|---|---|
| Python source code (`*.py`) | No license — All Rights Reserved | **Conceptually useful only.** Do not copy. Reimplement independently. |
| Dataset YAML files (`data/roleplaying/dataset.yaml`) | No license — All Rights Reserved | **Conceptually useful only.** Do not copy scenarios. Our factual deception dataset is structurally different anyway. |
| Generated rollouts (`data/insider_trading/`, `data/sandbagging_v2/`) | No license + API-generated content (ownership unclear) | **Not reusable.** Do not use. |
| Pre-trained probe weights (`example_results/*.pt`) | No license | **Not reusable.** Our models differ. |
| Config files (`scripts/configs/*.yaml`) | No license | **Conceptually useful only.** Note their hyperparameter choices (C=10, layer=22, control_size=10000) as inspiration. |
| Metrics code (`metrics.py`) | No license | **Conceptually useful only.** AUROC and TPR@FPR are standard scikit-learn calls. Reimplement independently. |
| Plotting code (`paper_plots.py`, `by_layer.py`) | No license | **Conceptually useful only.** Note their visualization choices (violin plots, ROC with AUROC in legend, layer-vs-metric two-panel plots). |
| README and paper | Paper is citable; README has no license | **Cite the paper**. Do not reproduce README text. |

### 2.3 Recommendation

**Do not copy any code, data, or configuration verbatim from the ApolloResearch repository.**

Instead:
1. Cite the paper: "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407)
2. Add an attribution note to `research.md` and our `README.md`
3. Reimplement any desired methodological ideas independently (they are mostly standard ML)
4. Our metrics code (AUROC, TPR@FPR via scikit-learn) is already independently implemented

---

## 3. Experimental Design Extraction

### 3.1 Their Experimental Design

| Aspect | ApolloResearch |
|---|---|
| Model | Llama-3.3-70B-Instruct |
| Training scenario | Roleplaying (human social lying) OR Descriptive Q&A pairs |
| Evaluation scenarios | Insider trading, sandbagging, AI liar benchmark |
| Dataset type | Model-generated responses + pre-written pairs |
| Honest/deceptive labels | Explicitly instructed: system prompt tells model to lie or tell truth |
| Token positions probed | "Start of turn" (first token of assistant response) OR "last token before EOT" |
| Layer selected | Layer 22 (single layer, ~27% depth in Llama-70B's 80 layers) |
| Layer selection method | Empirical sweep (`by_layer.py`) → pick best |
| Probe type | Logistic regression (primary), also MMS, LAT, MLP |
| StandardScaler | Yes, optional per-detector flag |
| Regularization | C=10 (roleplaying config), searched in by_layer.py |
| Val fraction | 0.2 |
| Negative controls | Alpaca 10,000 samples (unrelated instruction-following) |
| Primary metric | TPR@1%FPR on control data (recall at 1% false positive rate) |
| Secondary metric | AUROC (deceptive vs. honest) |
| Key result | AUROC 0.96–0.999; TPR@1%FPR = 95–99% |
| Multi-position reduction | `PromptScorer`: mean/max/relu/topk/final |
| Probe serialization | `detector.pt` (torch.save of state dict) + `cfg.yaml` |
| Result serialization | `scores.json`, `control_scores.json`, `results_table.csv` |
| Generalization test | Train on roleplaying → test on insider trading + sandbagging |
| Confidence intervals | Not mentioned; no bootstrapping visible |

### 3.2 Adoption Recommendations Per Item

| Aspect | Adopt Now | Adapt Later | Reject | Future Work |
|---|---|---|---|---|
| TPR@1%FPR as **primary** headline metric | ✅ **Adopt** | | | |
| Control dataset size ~1000+ diverse samples | ✅ **Adopt** | | | |
| Layer-by-layer evaluation plot (2 panels: AUROC + TPR@FPR by layer) | ✅ **Adopt** | | | |
| Exact configs committed to repo (hyperparameters, C=10) | ✅ **Adopt** (we do hash; consider explicit best-C doc) | | | |
| Probe + cfg saved together in example_results-style folder | ✅ **Adopt** | | | |
| Multi-position token probing (start-of-turn, EOT-minus-1) | | ✅ **Adapt** | | |
| Roleplaying dataset (social lying scenarios) | | | | ✅ **Future** (Phase 2) |
| Insider trading / sandbagging datasets | | | | ✅ **Future** (Phase 2+) |
| Model-generated rollouts | | | | ✅ **Future** (Phase 2+) |
| MMS / LAT / MLP probes | | | | ✅ **Future** (Phase 2) |
| Goodfire SAE feature probing | | | | ✅ **Future** (Phase 3) |
| `inspect-ai` evaluation framework | | | ✅ Reject for now | |
| wandb experiment tracking | | ✅ Adapt | | |
| Multi-layer score averaging | | | ✅ Reject for now | ✅ Future |
| Alpaca as control set | | ✅ Adapt | | |
| API-based rollout generation | | | ✅ Reject | |

---

## 4. Methodological Ideas to Borrow Conceptually

All of the following are **ideas only** — independently reimplemented using public ML knowledge:

### 4.1 TPR@1%FPR as Headline Metric

Their strongest result is not AUROC but "95–99% of deceptive responses caught at 1% false positive rate on benign controls." This framing is more operationally meaningful than AUROC for a monitoring system. Our calibration code already computes this; it should be promoted to the primary reported metric in our run summary and plots.

**Action:** Reorder our `print_run_summary()` to show TPR@1%FPR before AUROC. Add it to plot titles.

### 4.2 Control Dataset Scale

They use 10,000 Alpaca samples as their benign control. This gives tight TPR@FPR estimates (the 1st percentile of 10,000 samples is well-estimated). Our pilot used 32 controls, which gives extremely noisy FPR estimates. For a rigorous evaluation:
- Control N should be ≥ 500 for reasonable FPR estimates at the 1% level (≥50 samples above threshold)
- Control N = 1000+ is the standard for publishable FPR calibration

**Action:** Expand our benign controls dataset to 500–1000 samples before the full Qwen run. Current dataset appears to have more available; check.

### 4.3 Contrastive Honest/Deceptive Dataset Design

Their roleplaying dataset pairs a scenario with explicit honest and deceptive completions, exactly as we do. Key difference: their scenarios are realistic social situations (grade appeals, employment reviews) where deception has plausible motivation. Our factual deception is simpler but more controlled (we know the exact fact being lied about).

Neither is strictly better — they're different threat models. Our setup has stricter experimental control (the lie is precisely specified); theirs is more ecologically valid.

**Action:** No change needed for Phase 1. Document this distinction explicitly in research.md.

### 4.4 Cross-Scenario Generalization as Primary Scientific Claim

Their most scientifically interesting result is training on roleplaying → testing on insider trading and sandbagging. The probe generalizes across scenarios. This generalization is more meaningful than within-scenario performance.

For our project: the analogue is training on one domain (e.g., biology) and testing on another (e.g., history). We currently train and test on facts from all 7 domains mixed together. Domain-held-out evaluation would be a cleaner test of generalization.

**Action:** Add domain-stratified evaluation as a Phase 1+ extension. Not required for the initial full run, but should be planned.

### 4.5 Token Position Strategy

They probe two positions:
- **"Start of turn"**: First token of each assistant turn in multi-turn dialogue
- **"Last token before EOT"**: The token just before `</s>` or `<|eot_id|>` in the response

We probe the **"final prompt token"**: last non-padding token before the response begins.

These are genuinely different and complementary. Their approach captures what the model is "thinking" as it *generates* the response; ours captures the "decision state" just before generation begins.

**Action:** Probe both positions in the full Qwen run (add `--token-position` flag to script 03). Report comparison. This is a concrete novelty contribution relative to ApolloResearch.

### 4.6 Example Results Folder Structure

They commit `cfg.yaml + detector.pt + scores.json + control_scores.json + results_table.csv` for each experiment. This allows anyone to reproduce results without rerunning the full pipeline.

We commit `layerwise_probes.pkl` but could add a cleaner `example_results/` style folder with a frozen snapshot of results.

**Action:** After the full Qwen run, create `example_results/qwen2_5_7b_full/` with: `cfg.yaml`, `probes.pkl`, `best_layer.json`, `layerwise_metrics.csv`, `calibration.csv`, and one or two PNGs. This is good for the eventual paper/open release.

### 4.7 Score Reduction Method Comparison

Their `PromptScorer` applies different reductions (mean, max, relu, topk, final) to multi-position scores. We use only the single final-prompt-token score. If we add multi-position probing, we should test at minimum `mean` and `final` reductions and compare.

### 4.8 Violin Plot Visualization

Their paper uses violin plots to show score distributions (honest vs. deceptive) across datasets. This is more informative than our current score distributions histogram for comparison across multiple conditions. Our `plots.py` currently produces a histogram; a violin/box plot is worth adding.

---

## 5. Comparison Table

| Component | ApolloResearch | Our Implementation | Gap | Recommended Action | Priority |
|---|---|---|---|---|---|
| **Dataset type** | Roleplaying (social), instructed Q&A pairs, model-generated rollouts | Paired factual Q&A (1,005 facts, 7 domains) | Different threat models (strategic vs. factual) | No change. Document distinction. | None |
| **Threat model** | Strategic deception: model conceals info to achieve goal | Direct factual deception: model explicitly told to lie | Our model is simpler, more controlled | No change. Explicit mention in README. | None |
| **Model choice** | Llama-3.3-70B-Instruct | Qwen/Qwen2.5-7B-Instruct (planned) | Different family; comparable size class | Keep Qwen. Note family difference as novelty. | None |
| **Activation token position** | Response tokens (start-of-turn, last-before-EOT) | Final prompt token (pre-response) | Fundamental difference in what is probed | Add response-token position as comparison in full run | High |
| **Probe type** | 7 types (MMS, LAT, LR, MLP, CMMS, MLR, BestLatents) | 1 type (LogisticRegression) | Fewer probe types | Add MMS and LAT comparison in Phase 2 | Medium |
| **Layer selection** | Single layer (22 of 80); empirical sweep | All layers compared; best by val AUROC | We compare more layers | No change — our approach is more thorough | None |
| **Calibration data** | Alpaca 10,000 samples | Benign controls (current pilot: 32; full dataset unknown) | **Critical gap**: far fewer controls | Expand controls to ≥500 before full run | **Critical** |
| **Primary metric** | TPR@1%FPR on control (headline) | AUROC (headline), TPR@FPR reported | We underemphasize TPR@1%FPR | Promote TPR@1%FPR to headline in summaries and plots | High |
| **Fixed-FPR reporting** | 1% FPR on Alpaca 10k | 1%, 5%, 10% FPR computed | Comparable but 1% is noisier without 10k controls | More controls + emphasize 1% | High |
| **Controls type** | Alpaca (instruction-following, unrelated to deception) | Benign controls (factual Q&A but not involving deception) | Similar concept; our controls are topically closer to probes | Check controls for topic leakage; consider adding truly unrelated samples | Medium |
| **Generalization test** | Cross-scenario (train roleplaying → test insider trading) | None yet; all domains mixed in train | Gap in scientific rigor | Add domain-held-out split evaluation | Medium |
| **Result artifacts** | cfg.yaml + detector.pt + scores.json + CSV in `example_results/` | probes.pkl + metrics CSVs + PNGs in run-named dirs | Less structured for external consumption | Add `example_results/` frozen snapshot after full run | Medium |
| **Run metadata** | wandb integration | JSON with git hash, config hash, timestamps | Comparable; we have more provenance info | No change | None |
| **Reproducibility** | Requires API keys for rollout datasets; probe weights committed | No API keys required; all data generated locally | Our project is more reproducible | Emphasize as a key differentiator | None |
| **Tests** | Not visible / appears minimal | 74 unit tests + integration test + synthetic E2E | We have stronger test infrastructure | No change | None |
| **Synthetic E2E** | None found | `08_run_synthetic_e2e.py` with injected signal at layers 3-5 | We have something they don't | Emphasize in paper as methodology contribution | None |
| **License status** | No license (All Rights Reserved) | Not yet specified | We should add a license to our repo | Add MIT or Apache-2.0 license to our repo | High |
| **Confidence intervals** | None visible | None | Both lack this | Add bootstrap CIs to layer plots in post-Qwen analysis | Low/Phase 2 |
| **Pair leakage prevention** | Not mentioned explicitly | Pair-level splitting enforced | We are stricter | Emphasize in methods | None |
| **Artifact validation** | None visible | `validate_activation_artifact()` + schema tests | We are more rigorous | Keep; mention as methodological contribution | None |

---

## 6. Concrete Implementation Recommendations

All recommendations below are independent reimplementations — no ApolloResearch code should be copied.

### 6.1 Immediate (Before Full Qwen Run)

#### R1: Expand benign controls to ≥500 samples (Critical)

The 1% FPR calibration threshold requires at least 100 controls above the threshold to be well-estimated. At 32 controls, the estimate has standard error of ~0.05. At 500 controls, SE drops to ~0.014. At 1000, SE drops to ~0.01.

- **Action**: Expand `src/deception_guardrail/data/controls.py` and the benign_controls dataset to 500–1000 samples.
- **Approach**: Add more diverse benign-control categories (factual questions the model answers truthfully, math problems, coding questions, creative writing prompts) to reduce the risk that our controls are too topically similar to the probe prompts.

#### R2: Promote TPR@1%FPR to Primary Headline (High Priority)

Currently our summary prints AUROC first and TPR@FPR after. Operationally, TPR@1%FPR is more meaningful.

- **Action**: Reorder `print_run_summary()` to print `TPR@1%FPR` before AUROC.
- **Action**: Add `TPR@1%FPR` to the title of the layer-vs-metric plot in `analysis/plots.py`.
- **Action**: In `save_layerwise_csv()`, ensure TPR@1%FPR is adjacent to AUROC.

#### R3: Add Second Plot Panel: Layer vs. TPR@1%FPR

`by_layer.py` in the Apollo repo generates a two-panel figure: AUROC by layer (top) and TPR@1%FPR by layer (bottom). We have separate plots. Combining them into one figure makes the relationship between layer quality and calibrated performance visible.

- **Action**: Add a `plot_layer_auroc_and_tpr` function to `analysis/plots.py` that produces a 2-panel figure (top: val/test AUROC by layer; bottom: TPR@1%FPR by layer). Mark the best layer with a vertical line.

#### R4: Add Response-Token Position Capture

This is our most concrete novelty opportunity relative to Apollo. They capture from response tokens; we capture from the final prompt token. Running both and comparing is new.

- **Action**: Add `--token-position last_response_token` to `scripts/03_capture_activations.py` and update `capture.py` to support extracting from the *first token of the assistant response turn* (or the last response token).
- **Note**: This requires knowing where the assistant turn starts. For Qwen, the assistant turn begins after `<|im_start|>assistant\n`. The token ID for `<|im_start|>` can be located in the tokenized sequence.

#### R5: Cite ApolloResearch in README and research.md

- **Action**: Add to `README.md` a section "Related Work" citing the paper.
- **Action**: Add to `research.md` a note: "Methodology influenced by ApolloResearch's approach to linear probe deception detection (arXiv:2502.03407), which demonstrated 95–99% TPR@1%FPR in strategic deception scenarios using Llama-3.3-70B. Our project differs in threat model (direct factual deception vs. strategic), model family (Qwen vs. Llama), token position (final prompt token vs. response tokens), and dataset design (controlled factual pairs vs. social roleplaying and rollouts)."

#### R6: Add License to Our Repository

The ApolloResearch repo is all rights reserved due to absent license. We should not make the same mistake.

- **Action**: Add `LICENSE` file (MIT or Apache-2.0) to the WOLFGuard repository root.
- **Action**: Add `license = "MIT"` (or chosen license) to `pyproject.toml`.

### 6.2 Post-Qwen Analysis Changes

#### R7: Domain-Held-Out Generalization Evaluation

Add a cross-domain evaluation: train probe on 6 of 7 domains, test on the held-out domain. Repeat for all 7 domains and report mean AUROC and TPR@1%FPR per held-out domain. This is a within-our-framework generalization test analogous to their cross-scenario generalization.

#### R8: Bootstrap Confidence Intervals on Layer Plots

Add ±1 SD bootstrap intervals (200 resamples, stratum-preserving for class balance) to the layer-vs-AUROC plot. At N=64 test samples (pilot), intervals will be wide (~±0.05). At N=600 (full run), they narrow to ~±0.015.

#### R9: Example Results Folder

After full Qwen run, commit a frozen `example_results/qwen2_5_7b_v1/` directory containing:
- `cfg.yaml` (exact model and experiment config)
- `probes.pkl` (trained probes)
- `best_layer.json` (layer index, AUROC, TPR@FPR)
- `layerwise_metrics.csv`
- `calibration.csv`
- `layer_auroc_and_tpr.png`
- `score_distributions_best_layer.png`

This enables third-party reproduction of our analysis results without rerunning capture.

#### R10: Score Separation Plot

Report not just AUROC but the raw score distributions (mean deceptive score, mean honest score, their separation). We compute `score_separation` in our metrics — add a horizontal bar chart showing this per layer.

### 6.3 Phase 2 or Later

#### R11: Roleplaying Deception Subset

Construct a small roleplaying-style deception dataset (50–100 scenarios) similar in structure to theirs — social situations with honest and deceptive completions. Use this to test whether our prompt-token probe generalizes to social deception (not just factual deception).

#### R12: Response-Token vs. Prompt-Token Systematic Comparison

Run both `final_prompt_token` and `first_response_token` (and possibly `mean_response_tokens`) on the same Qwen model. Report whether probing where the model *starts to answer* differs from probing where it *decides to answer*.

#### R13: MMS and LAT Probe Comparison

Reimplement Mass Mean Shift (MMS) — the direction from mean deceptive to mean honest activations — and Linear Artificial Tomography (LAT) — PCA on the difference vectors. Compare these to logistic regression. These are simpler probes that their paper also tests; MMS in particular is a useful sanity check.

#### R14: SAE Feature Analysis

After probe training, identify which residual-stream directions the best-layer probe weights point toward, and whether those directions overlap with SAE features. Requires `sae-lens` and a compatible SAE for Qwen. This is entirely absent from Phase 1.

---

## 7. Codebase Integration Plan

### 7.1 Immediate Changes for Phase 1 (Before Full Qwen Run)

| # | Change | File(s) | Effort |
|---|---|---|---|
| R1 | Expand benign controls to ≥500 samples | `data/controls.py`, `scripts/01_build_dataset.py` | Medium |
| R2 | Promote TPR@1%FPR to headline in summaries | `analysis/summaries.py`, `scripts/07_summarize_run.py` | Small |
| R3 | Two-panel layer plot (AUROC + TPR@1%FPR) | `analysis/plots.py` | Small |
| R5 | Cite ApolloResearch in README and research.md | `README.md`, `research.md` | Trivial |
| R6 | Add MIT/Apache-2.0 license | `LICENSE`, `pyproject.toml` | Trivial |

**Defer R4 (response-token position) to post-Qwen** — requires significant changes to `capture.py` and is a dedicated experiment, not a Phase 1 prerequisite.

### 7.2 Post-Qwen Analysis Changes

| # | Change | Effort |
|---|---|---|
| R7 | Domain-held-out evaluation | Medium |
| R8 | Bootstrap confidence intervals on layer plots | Small |
| R9 | `example_results/` frozen snapshot | Small |
| R10 | Score separation horizontal bar chart | Small |
| R4 | Response-token vs. prompt-token comparison | Medium |

### 7.3 Phase 2 or Later

| # | Change | Notes |
|---|---|---|
| R11 | Roleplaying deception subset | Requires scenario writing or generation |
| R12 | Systematic response vs. prompt token study | Requires re-capture with new token position |
| R13 | MMS and LAT probe comparison | Clean-room reimplementation, ~2 days |
| R14 | SAE feature analysis | Requires compatible Qwen SAE weights |

---

## 8. Risk Assessment

### 8.1 Redundancy Risk

**Risk**: Our project replicates Apollo's findings on a different model without adding novelty.

**Mitigation**:
- Our factual deception threat model is strictly different (not strategic, not roleplaying)
- Our pair-level leakage prevention is more rigorous than anything visible in their design
- Our synthetic E2E test is novel infrastructure
- Our token position (final prompt token vs. response tokens) is a genuine comparison
- Our Qwen family focus adds cross-model coverage
- Later: causal activation suppression (not in their paper), SAE analysis (they have it but with different model)

### 8.2 Lack of Novelty

**Risk**: Linear probes on deception activations are already demonstrated; we add little.

**Mitigation**:
- Our scientific question is different: *direct factual deception* under *known ground truth*, with strict pair control. Their question is strategic deception in realistic agent scenarios.
- Our Phase 2 (causal activation steering) is a distinct contribution not in their paper.
- Cross-model comparison (Qwen vs. Llama) is a contribution.
- Domain-held-out evaluation within factual deception is not in their paper.

### 8.3 License Risk

**Risk**: We inadvertently copy code or data from an unlicensed repository and face copyright issues.

**Mitigation**: This audit explicitly classifies all their components as "conceptually useful only." No code is to be copied. Our implementation is independent. Any methodological overlap (AUROC, StandardScaler, TPR@FPR) uses standard scikit-learn, which is separately licensed (BSD-3).

### 8.4 Benchmark Overfitting

**Risk**: Knowing their results (0.96–0.999 AUROC) biases our hyperparameter choices and evaluation.

**Mitigation**:
- Our C grid search is validation-based (no test leakage)
- Our test set is never used for model selection
- We report all layer results, not just best-layer cherry-picked results
- Our synthetic E2E test shows the pipeline works regardless of results

### 8.5 Copying Their Threat Model Too Closely

**Risk**: We add roleplaying scenarios (Phase 2) that are too similar to their dataset.

**Mitigation**:
- Keep our Phase 1 strictly factual deception (no roleplaying)
- If we add roleplaying in Phase 2, design scenarios independently (different domains: scientific misconduct, medical, financial — not the exact scenarios they published)
- Do not use their dataset YAML even as inspiration for specific scenarios

### 8.6 Prompt Artifacts

**Risk**: Our probe detects prompt *style* (instruction tokens like "Answer deceptively") rather than internal deception state.

**Mitigation**:
- Our paired design uses *identical* question text — only the instruction differs
- Both honest and deceptive prompts contain "You know the following fact:" — the deception instruction is the only difference
- Benign control calibration catches prompt-style artifacts (their high control FPR ~78% in our pilot reflects this risk)
- Document this limitation explicitly; our Phase 1 may be detecting instruction-following rather than deception state

### 8.7 Insufficient Distinction from ApolloResearch

**Risk**: Reviewers or the community view our work as an extension of theirs rather than a distinct contribution.

**Mitigation**:
- Emphasize our distinct angles: factual deception, pair control, no API dependencies, causal intervention (Phase 2), SAE analysis (Phase 3)
- Make our methodology section explicitly compare and contrast with their paper
- Demonstrate domain-held-out generalization (missing from their work)
- Publish our probe weights and datasets under a permissive license (they have no license)

---

## 9. Novelty Recommendations

### 9.1 How Our Project Remains Non-Redundant

**Distinct from ApolloResearch in every major design decision:**

| Dimension | ApolloResearch | Ours | Novelty |
|---|---|---|---|
| Threat model | Strategic (multi-turn, goal-directed) | Direct factual (one-turn, ground-truth known) | Cleaner scientific control |
| Dataset source | Human-written + model rollouts + API calls | 1,005 curated factual pairs, no API | Fully reproducible, no API cost |
| Model family | Llama-3.3-70B | Qwen2.5-7B | Cross-family coverage |
| Token position | Response tokens | Final prompt token | Different mechanism hypothesis |
| Layer selection | Single layer (22) | All layers compared | More thorough mapping |
| Pair leakage | Not mentioned | Enforced at split level | Stricter experimental design |
| Artifact validation | Not present | `validate_activation_artifact()` | Novel pipeline safety |
| Synthetic E2E test | Not present | `08_run_synthetic_e2e.py` | Novel sanity check methodology |
| License | None (All Rights Reserved) | MIT/Apache-2.0 (proposed) | Better for research community |
| API dependency | Required (5 external APIs) | None | Higher reproducibility |

### 9.2 Proposed Novel Contributions

1. **Strict factual pair control**: The lie target is precisely specified; any detected signal is attributable to the deception instruction, not to topic, length, or vocabulary.

2. **Final-prompt-token vs. response-token comparison** (Phase 1+): Testing whether the probe's power comes from "what the model is about to say" vs. "what the model has decided" is a concrete mechanistic question not answered by Apollo's paper.

3. **Domain-held-out generalization**: Train on biology facts, test on history facts. Measures whether the probe learns "lying as a pattern" vs. "biology vocabulary."

4. **Causal activation suppression** (Phase 2): If we find a deception direction in the residual stream, can suppressing it reduce deceptive outputs? This is the causal question; Apollo's paper is purely observational.

5. **SAE feature-level analysis** (Phase 3): Identify which SAE features activate differentially in deceptive vs. honest conditions. Apollo does this with Goodfire for Llama-70B; we can do it for Qwen with open SAE weights.

6. **Cross-model generalization** (Phase 2): Train probe on Qwen2.5-7B, transfer weights to Llama-3.2-3B or Mistral-7B. Does the deception direction generalize across model families?

7. **Misleading truth and omission categories** (Phase 2): Extend beyond binary lies to technically-true-but-misleading and answer-omission deception types. These are systematically absent from both our current dataset and Apollo's.

---

## 10. Final Deliverables

### 10.1 Most Relevant Files in ApolloResearch (for conceptual reference only)

```
deception_detection/metrics.py         # AUROC, TPR@FPR implementations (concept)
deception_detection/detectors.py       # MMS, LAT, LR probe types (concept)
deception_detection/activations.py     # Token position strategy (concept)
deception_detection/prompt_scorer.py   # Multi-position reduction methods (concept)
deception_detection/scripts/by_layer.py  # Two-panel layer evaluation plot (concept)
deception_detection/scripts/configs/roleplaying.yaml  # C=10, layer 22, alpaca 10k (reference)
data/roleplaying/dataset.yaml          # Social deception scenario structure (concept only)
example_results/roleplaying/           # Artifact organization pattern (concept)
```

### 10.2 Methods to Reimplement Independently

| Method | Independent Reimplementation Notes |
|---|---|
| TPR@1%FPR as primary headline | Already implemented in our `calibration.py` — just reorder display |
| Two-panel layer plot (AUROC + TPR@FPR) | Pure matplotlib; no code to copy |
| Bootstrap CI on layer AUROC | `sklearn.utils.resample` + `roc_auc_score`; standard pattern |
| Domain-held-out split | Subset our existing dataset by domain field |
| Response-token position capture | New `capture.py` mode; find EOT token and extract `hidden_states[-1]` |
| MMS probe | Mean deceptive activations minus mean honest activations; one line |
| `example_results/` frozen snapshot | Copy our existing artifacts to a new committed directory |

### 10.3 Methods NOT to Use

- Do not use their `DialogueDataset` / `TokenizedDataset` abstractions
- Do not use their `repository.py` dataset naming system
- Do not use their `PairedActivations` data structure (ours is simpler and sufficient)
- Do not use their rollout generation scripts (API-dependent)
- Do not use their Goodfire SAE interface (Phase 3 at earliest, Qwen SAE needed anyway)
- Do not use the `inspect-ai` framework they reference
- Do not use their dataset YAML files

### 10.4 Immediate Code Changes Recommended

Listed in order of priority:

1. **Expand controls to ≥500 samples** — most scientifically critical
2. **Add MIT license** — legal hygiene
3. **Promote TPR@1%FPR to headline in `print_run_summary()`** — one reorder
4. **Add two-panel layer plot** — adds methodological clarity
5. **Cite ApolloResearch in README and research.md** — academic integrity

### 10.5 Proposed Experimental Roadmap

```
Phase 1 (Current) — Direct Factual Deception, Controlled Setup
  ├── Full Qwen2.5-7B activation capture (1,005 fact pairs)
  ├── Layer-wise LR probe training + calibration (all 28 layers)
  ├── TPR@1%FPR as primary metric (≥500 controls)
  ├── Domain-held-out generalization evaluation
  └── Frozen example_results/ snapshot

Phase 1+ — Token Position Comparison (Qwen2.5-7B, same dataset)
  ├── Add final-response-token capture mode
  ├── Compare: final-prompt-token vs. first-response-token probe
  └── Report: where in generation is deception most linearly readable?

Phase 2 — Causal Intervention + Probe Extension
  ├── Activation steering to suppress deception direction
  ├── MMS and LAT probe comparison
  ├── Roleplaying deception subset (independently authored)
  ├── Cross-model probe transfer (Qwen → Llama, Qwen → Mistral)
  └── Bootstrap CI on all layer plots

Phase 3 — SAE and Multi-Model Analysis
  ├── SAE feature identification for deception direction (Qwen)
  ├── Misleading-truth and omission deception categories
  ├── Larger model comparison (Qwen2.5-72B)
  └── Publication
```

**This roadmap is clearly differentiated from ApolloResearch:**
- Phase 1: simpler threat model, stricter control, Qwen, final-prompt-token
- Phase 1+: token position comparison (their result + our result, new comparison)
- Phase 2: causal (their paper is observational only)
- Phase 3: SAE on Qwen (they do SAE on Llama with commercial API)

---

*This audit was produced using public information only. No ApolloResearch code was copied or adapted. All recommendations are for independent reimplementation.*
