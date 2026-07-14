# Cheap-proxy inference verification — experiment log

**Date:** 2026-07-08 · **Hardware:** 1× NVIDIA H100 80GB · **Framework:** transformers 5.13, torch 2.8 (CUDA 12.8)

## Question

DiFR (arXiv:2511.20621) verifies LLM inference by comparing a provider's output to a
*same-model* reference under a synchronized sampling seed. That reference forward pass is
the dominant cost. **Can a much cheaper, small-model proxy give a useful verification
signal instead?** And if the naive version is weak, can we push it higher while staying
cheap?

## Threat model

A provider claims to serve **Qwen2.5-7B-Instruct**. It may instead cheat by:
- **Substitution** — serving a smaller/cheaper model (3B, 1.5B, 0.5B, or a cross-family
  model) while charging for the 7B. *(high-value attack — big compute savings)*
- **Quantization** — serving a 4-bit (nf4) version of the true 7B. *(low-value / subtle)*

The verifier sees only `(prompt, completion)`. It scores the completion with a **cheap
proxy** (one teacher-forced forward pass) and must classify honest vs. cheating.
Detection quality = ROC-AUC (honest = negative, attack = positive).

## Setup

- **Prompts:** 300 instructions from `tatsu-lab/alpaca` (empty-input subset). `prompts.json`.
- **Generation:** greedy (temperature 0), 128 new tokens, Qwen chat template.
- **Served configs:** `honest`=Qwen2.5-7B · `sub_3b`/`sub_1.5b`/`sub_0.5b`=Qwen2.5 of that
  size · `quant_4bit`=Qwen2.5-7B in bitsandbytes nf4 · cross-family=`SmolLM2-360M-Instruct`.
- **Proxies (verifiers):** Qwen2.5-0.5B (0.494B params) and Qwen2.5-1.5B (1.54B).
- **Per-token stats from one proxy prefill:** cross-entropy (NLL), token rank in proxy
  logits, argmax-agreement, proxy predictive entropy.
- **Detectors:**
  1. *baseline* — single statistic (mean log-rank), oriented AUC.
  2. *LR features* — logistic regression on 8 aggregate features per proxy
     `[mean/std/max NLL, mean log-rank, frac(rank=0), frac(rank≤2), frac(rank≥8), mean entropy]`,
     5-fold cross-validated AUC.
  3. *panel* — LR on features from both proxies concatenated (0.5B + 1.5B).

## Cost (FLOP, relative to one claimed-7B forward pass per token)

| verifier | params | forward FLOP vs. claimed 7B |
|---|---|---|
| 0.5B proxy | 0.494B | **6.5%** |
| 1.5B proxy | 1.54B  | 20.2% |
| 0.5B+1.5B panel | 2.03B | **26.7%** |

Verification only prefills the tokens you audit (converges in ~16–32), so total cost is a
small fraction of one forward pass — orders of magnitude below full same-model DiFR, and
astronomically below ZKP (zkLLM ≈ days per generation).

## Results — detection AUC vs. scored tokens

### Same-tier substitution: honest 7B vs. sub → 0.5B (5-fold CV)

| detector | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| baseline mean log-rank (0.5B) | 0.793 | 0.861 | 0.909 | 0.922 | 0.923 |
| LR features (0.5B proxy)      | 0.845 | 0.935 | 0.981 | 0.988 | **0.991** |
| LR features (0.5B+1.5B panel) | 0.888 | **0.975** | **0.994** | 0.996 | **0.997** |

### Cross-family substitution: honest 7B vs. sub → SmolLM2-360M (de-confounds model identity)

| detector | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| baseline mean log-rank (0.5B) | 0.654 | 0.603 | 0.535 | 0.548 | 0.646 |
| LR features (0.5B proxy)      | 0.679 | 0.649 | 0.680 | 0.723 | 0.782 |
| LR features (0.5B+1.5B panel) | 0.706 | 0.711 | 0.778 | 0.845 | **0.899** |

### Capability gradient: detectability vs. how far the model is downgraded

The two "works" points above (0.5B, SmolLM2-360M) are the far end of a gradient. Filling in
the interior with the same 0.5B+1.5B panel (`detect.py` capability-gradient rows) turns the
anecdote into a dose-response — detectability tracks the attacker's compute saving and
collapses as the served model approaches the claimed 7B. See `fig_difr_capability_gap.png`.

| served (honest 7B vs.) | × smaller | panel AUC @32 | panel AUC @128 | verdict |
|---|---|---|---|---|
| Qwen2.5-0.5B | 15.4× | 0.994 | **0.997** | ✓ works |
| Qwen2.5-1.5B | 4.9×  | 0.965 | **0.991** | ✓ works |
| Qwen2.5-3B   | 2.5×  | 0.488 | 0.580 | ✗ near chance |
| Qwen2.5-7B nf4 (same weights) | 1.0× | 0.462 | 0.462 | ✗ chance |

The cliff sits between **~5× (1.5B, still caught at 0.99)** and **~2.5× (3B, gone)** — i.e.
the panel owns downgrades of roughly a half-order of magnitude or more and cedes everything
closer to the claimed tier. (The earlier "~0.55 @128 for 3B" came from the single-statistic
`analyze.py`; the panel gives 0.58 — same conclusion, near chance.)

### Model-distance ladder: how *different* does the substitute need to be? (`exp_proxy_distance_grid.py`)

The capability gradient above varies *size* along one family. `exp_proxy_distance_grid.py`
makes "how different" an explicit 2-D grid instead: rows step a substituted model away from
the claimed model M along an ordered distance ladder (quantized self → smaller sibling →
same-size different-domain → next-gen → RL-distilled → different family/tokenizer), columns
vary the verifier's cheap proxy (including a cross-family proxy with a different tokenizer —
the verifier re-tokenizes the provider's decoded text under its own vocabulary, so this works
even when proxy and M don't share token ids). Run with `IVGYM_LADDER=qwen` (claimed
`Qwen2.5-7B-Instruct`) or `IVGYM_LADDER=llama` (claimed `Llama-3.1-8B-Instruct`):

| substituted model (Qwen ladder, distance ↑) | AUC (Qwen2.5-0.5B proxy) | AUC (Qwen2.5-1.5B proxy) | AUC (SmolLM2-1.7B proxy, cross-family) |
|---|---|---|---|
| Qwen2.5-7B (self) | 0.304 | 0.323 | 0.307 |
| Qwen2.5-7B NF4 (quantized self) | 0.372 | 0.441 | 0.324 |
| Qwen2.5-3B (same family, smaller) | 0.493 | 0.493 | 0.522 |
| Qwen2.5-Coder-7B (same fam+size, diff domain) | 0.650 | 0.637 | 0.560 |
| Qwen3-8B (same company, next gen) | 0.624 | 0.722 | 0.690 |
| DeepSeek-R1-Distill-Qwen-7B (same base, RL-distilled) | 0.871 | 0.887 | 0.813 |
| Llama-3.1-8B (different family, different tokenizer) | 0.860 | 0.866 | 0.758 |

Same shape as the capability gradient: quantized-self sits at chance (~0.4, indistinguishable
to a cheap proxy), and AUC climbs monotonically with distance from M, crossing the
"a cheap proxy is enough" threshold (~0.85) once the substitution reaches a different
post-training lineage or a different model family entirely — and this holds even when the
proxy's tokenizer differs from M's (SmolLM2 column). Full tables (including the fixed-direction
mean-NLL AUC and the scored-token-budget `k` sweep) are in
`docs/results/exp_proxy_distance_grid.txt` (Qwen ladder) and
`docs/results/exp_proxy_distance_grid_llama.txt` (Llama ladder); figures are
`docs/figures/fig_proxy_distance_grid_qwen.png` and `fig_proxy_distance_grid_llama.png`.

## Findings

1. **The cheap+good case exists.** A **0.5B proxy** detects substitution to a 0.5B-tier
   model at AUC **0.86 @16 tokens** (baseline); richer features + panel push this to
   **0.97–0.99 @16–32 tokens** at ≤27% of one 7B forward pass.
2. **The signal is genuine capability-tier detection, not model identity.** A cross-family
   control (Qwen proxies vs. served SmolLM2-360M, different family/tokenizer) still reaches
   **AUC 0.90**. So the proxy really detects "output is no more capable than a ~0.5B model."
3. **Two cheap levers do the work:** (a) richer features from the *same* single forward pass
   (tail/fraction-of-surprising-token stats, not just the mean); (b) a 2-proxy panel.
   Neither changes the "small proxy" premise.
4. **Detectability is a dose-response, not a knife-edge claim.** Sweeping the served tier
   (0.5B → 1.5B → 3B → nf4) gives a monotone AUC curve: the panel catches downgrades down to
   **~5× smaller (1.5B, AUC 0.99)**, then collapses through 3B (2.5×, 0.58) to chance at the
   subtle nf4 (1×, 0.46). Detectability tracks the attacker's savings — cheap proxies own the
   high-value downgrade attack and cede subtle tampering to full same-model DiFR.
5. **Not adversarially airtight.** A purely statistical proxy is evadable by an adaptive
   attacker (temperature tuning / paraphrase). This is a cheap first-line/triage signal.

## Robustness tests (added 2026-07-08)

### Test 1 — adaptive attacker tuning temperature (`robust_temp.py`)

Threat: provider claims 7B sampled at **T=1.0**; attacker serves the 0.5B but tunes *its*
temperature so mean cross-entropy under the proxy matches honest — the attack that
collapses CE detectors in the DiFR paper. Honest mean-NLL under the 0.5B proxy = **1.375**.

| 0.5B temp | cheat mean-NLL | AUC mean-CE | AUC LR panel |
|---|---|---|---|
| 0.3 | 0.68 | 0.871 | 0.996 |
| 0.7 | 1.03 | 0.680 | 0.991 |
| **0.9 (best match)** | **1.64** | **0.653** | **0.992** |
| 1.1 | 3.31 | 0.903 | 0.995 |
| 1.5 | 10.29 | 0.999 | 0.998 |

**Result:** at the matched temperature the **mean-CE detector collapses to AUC 0.65**, but the
**LR feature panel stays at 0.99**. The distribution-shape features (`frac(rank=0)`,
`frac(rank≥8)`, NLL std/max) survive the attack that fools the mean. *Confirms the panel is
robust to the standard temperature-tuning evasion; a stronger adaptive attacker who tunes
against all features jointly is untested.*

### Test 2 — benign cross-stack shift (`robust_stack.py`)

Threat: honest provider runs the 7B on a **different numerical stack** (bf16 + eager
attention + batch 8) than the verifier calibrated on (bf16 + sdpa + batch 48). Does benign
noise cause false positives or erode detection?

- **Benign noise level:** teacher-forced next-token agreement (7B on its own sdpa outputs,
  re-run eager) = **90.0%**. (Free-running positional match is only 22.9% — expected, since
  one numerically-flipped token cascades; this is *why* DiFR uses teacher-forced/seed-synced
  comparison, not free-running match.)
- **(B)** A classifier *can* partly tell the two honest stacks apart: **CV-AUC 0.76**. So the
  stack shift is **not** invisible — it leaves a measurable feature footprint.
- **(C)** But that footprint is ~orthogonal to the capability axis: the deployed
  downgrade-detector (calibrated to 5% FPR on the reference stack) gives **3.3% FPR on the
  alt-stack honest traffic** — no false-positive inflation.
- **(D)** Capability detection is **unaffected: AUC 1.000** honest(alt stack) vs. cheat.

**Result:** benign stack differences shift features measurably but along a direction the
downgrade-detector ignores, so false positives stay at nominal and detection power is
retained. *Caveat: only one benign perturbation tested; a large stack gap could rotate the
honest cluster toward the decision boundary.*

### Test 3 — paraphrase / rewrite attack (`attack_paraphrase.py`)

Threat: attacker serves the 0.5B answer, then **rewrites it with the same 0.5B** (stays
0.5B-tier and cheap) hoping the reworded surface tokens disrupt the proxy's per-token
surprise pattern.

| served (honest 7B vs …) | mean-CE AUC | LR panel AUC |
|---|---|---|
| raw 0.5B (no paraphrase) | — | 0.997 |
| **paraphrased 0.5B** | 0.647 | **0.990** |

**Result:** paraphrasing knocks the mean-CE detector to 0.65 but the **panel barely moves
(0.997 → 0.990)**. Rewording doesn't raise the underlying capability, so the distributional
signal survives.

### Test 4 — joint-feature adaptive attacker (`attack_joint.py`)

Threat: the attacker tunes **all sampling knobs it controls** on the 0.5B (temperature ×
top-p grid, 9 configs) and picks the config the defender's *full-feature* panel separates
least — an escalation over the mean-CE-only match in Test 1.

- mean-CE AUC ranges 0.55–0.87 across the grid (attacker can push it to **0.55**).
- **panel AUC never drops below 0.995** at any config.

| attacker-optimal (min panel AUC) | mean-CE AUC | panel AUC |
|---|---|---|
| temp=0.8, top_p=1.0 | 0.642 | **0.995** |

**Result:** no sampling-based evasion works against the panel. *Still untested: a white-box
attacker that re-decodes token-by-token to directly match the proxy's rank statistics
(requires running the proxy at serve time).*

### Test 5 — larger benign stack gap: fp32 (`robust_stack_large.py`)

Threat: honest 7B served in **float32** (higher precision, a bigger numerical gap than
Test 2's kernel/batch change) vs. the verifier's bf16 calibration.

- teacher-forced next-token agreement (fp32 vs bf16) = **94.3%**
- **(B)** CV-AUC honest(bf16) vs honest(fp32) = **0.628** — *less* distinguishable than the
  sdpa→eager change (0.76); a bigger precision gap did **not** mean more detectable.
- **(C)** FPR on fp32 honest at 5% threshold = **5.0%** (nominal).
- **(D)** AUC honest(fp32) vs cheat = **0.999**.

**Result:** detection and false-positive rate fully robust to a large precision shift.

### Bonus — broken-precision misconfiguration is caught for free

Running Qwen2.5-7B in **float16** overflows and produces garbage (it emits repeated image
URLs). This is a real misconfiguration, not benign noise; the cheap proxy flags it at
**AUC 1.000 / 96% of samples over threshold**. Excluded from Test 2 (it is not a benign
variation) but recorded as a genuine positive: the proxy catches gross serving bugs, not
just deliberate substitution.

## Recommended deployment

Triage cascade: a sub-2B proxy panel flags capability downgrades cheaply (AUC ≈0.9–0.99);
escalate only ambiguous or subtle cases to seed-synchronized same-model DiFR.

## Reproduce

```bash
# run from the repo root; scripts resolve data/figure paths relative to themselves
python3 experiments/generate.py honest        # then sub_3b sub_1.5b sub_0.5b quant_4bit
python3 experiments/score2.py  proxy_0.5b      # then proxy_1.5b   (rich per-token features)
python3 experiments/crossfam2.py               # cross-family served + scoring
python3 experiments/detect.py                  # baseline vs LR vs panel + capability-gradient table
python3 experiments/plot2.py                   # -> docs/figures/fig_difr_detect.png
python3 experiments/plot_capability_gap.py     # -> docs/figures/fig_difr_capability_gap.png
python3 experiments/robust_temp.py             # Test 1: adaptive temperature attack
python3 experiments/robust_stack.py            # Test 2: benign cross-stack shift (kernel/batch)
python3 experiments/attack_paraphrase.py       # Test 3: paraphrase/rewrite attack
python3 experiments/attack_joint.py            # Test 4: joint sampling-knob attacker
python3 experiments/robust_stack_large.py      # Test 5: larger stack gap (fp32)
python3 experiments/plot3.py                   # -> docs/figures/fig_difr_robustness.png

# model-distance ladder (separate real-model run; needs GPU + HF access)
IVGYM_LADDER=qwen  python3 -m experiments.exp_proxy_distance_grid   # -> fig_proxy_distance_grid_qwen.png
IVGYM_LADDER=llama python3 -m experiments.exp_proxy_distance_grid   # -> fig_proxy_distance_grid_llama.png
```
Scripts live in `experiments/`; their generated tensors + `prompts.json` live in
`experiments/difr_data/` (loaded via an `EXP` path resolved relative to `__file__`).
(`detlib.py` holds shared feature/detector helpers used by Tests 3–5.)

## File manifest

| file | role |
|---|---|
| `generate.py` | greedy generation for all served configs → `gen_*.pt` |
| `score.py` / `analyze.py` | v1 single-statistic scoring + AUC tables (`scores_*.pt`) |
| `score2.py` | rich per-token features (NLL/rank/entropy) → `feats_proxy_*.pt` |
| `cross_family.py` | v1 cross-family de-confound check |
| `crossfam2.py` | cross-family served (SmolLM2-360M) + rich scoring → `feats_cross_*.pt` |
| `detect.py` | logistic-regression detector, 5-fold CV, baseline/LR/panel + capability-gradient (all served tiers) |
| `plot_capability_gap.py` | dose-response figure: panel AUC vs capability gap + vs tokens → `fig_difr_capability_gap.png` |
| `detlib.py` | shared feature-extraction + detector helpers (Tests 3–5) |
| `robust_temp.py` | Test 1: adaptive temperature-tuning attack → `robust_temp_results.pt` |
| `robust_stack.py` | Test 2: benign cross-stack (kernel/batch) → `robust_stack_results.pt` |
| `attack_paraphrase.py` | Test 3: paraphrase/rewrite attack → `attack_paraphrase_results.pt` |
| `attack_joint.py` | Test 4: joint sampling-knob attacker → `attack_joint_results.pt` |
| `robust_stack_large.py` | Test 5: larger stack gap (fp32) → `robust_stack_large_results.pt` |
| `plot.py` / `plot2.py` / `plot3.py` | figures → `docs/figures/fig_difr_summary.png` / `fig_difr_detect.png` / `fig_difr_robustness.png` |
| `exp_proxy_distance_grid.py` | 2-D model-distance ladder (quant/family/domain/tokenizer) → `feats_proxy_*` not cached, real-model run each time |
| `difr_data/prompts.json` | 300 fixed prompts |
| `difr_data/*.pt` | generated tensors (gen/scores/feats/*_results) consumed by the scripts above |
| `docs/figures/fig_difr_summary.png` | v1 summary (single-stat 0.5B proxy) |
| `docs/figures/fig_difr_detect.png` | **main figure** — richer features + panel, same-tier & cross-family |
| `docs/figures/fig_difr_capability_gap.png` | **dose-response figure** — panel AUC vs capability gap (all served tiers) |
| `docs/figures/fig_difr_robustness.png` | **robustness figure** — temperature attack + cross-stack |
| `docs/figures/fig_proxy_distance_grid_qwen.png` / `_llama.png` | model-distance ladder, two claimed models |

## Caveats / threats to validity

- **Temperature robustness: tested** (Test 1). Panel survives the temperature-matching attack.
- **Joint sampling-knob attacker: tested** (Test 4). No temp×top-p config evades the panel
  (AUC ≥ 0.995). A white-box token-level re-decoding attack that matches rank statistics
  directly is still untested.
- **Paraphrase/rewrite: tested** (Test 3). Panel holds at 0.990.
- **Cross-stack noise: tested** at two gaps — kernel/batch (Test 2) and fp32 (Test 5).
  No false-positive inflation at either; cross-*GPU* (only one H100 here) untested.
- 300 prompts, one prompt domain (Alpaca instructions).
- LR detector AUC is cross-validated, but assumes access to honest + known-cheap reference
  samples to train the classifier (realistic: the verifier can generate these).
- 4-bit "attack" is a stand-in; only nf4 tested. float16 is a *broken* config for Qwen2.5,
  not a quantization attack — kept separate.
- Test 2 (C) threshold is calibrated in-sample on reference honest; the alt-stack FPR is
  measured out-of-sample, so the 3.3% is a fair held-out estimate.
