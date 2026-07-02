# Speculative decoding *is* the proxy detector's ceiling — and it's family-graded

*Answering the Slack thread: "How is it recomputation if it's with a different
model? Does this work because smaller and bigger Qwens are trained on the same
data (low perplexity correlated within the family)? Are you just looking for low
perplexity, or is this something else? … Looks like it's speculative decoding —
the acceptance rate is a measurement of proxy↔M distributional agreement, high
within a family and collapses across families. Look at vLLM/SGLang for reference."*

Short version: **yes, it's the speculative-decoding acceptance rate, not raw
perplexity, and not a recomputation of M.** Below is the mechanism (grounded in
vLLM/SGLang source), the exact quantity our black-box detectors are bounded by,
and the new cross-family measurements that were missing.

---

## 1. Resolving "how is it recomputation if it's with a different model?"

It isn't. Two different things were being conflated:

* **Speculative decoding (SD).** The *large* model M is fully run — it is the
  *target/verifier*. A small *draft* model only proposes tokens; M then accepts
  or rejects them and **still emits M's own distribution exactly**. You run M;
  you just run it cheaper. (This is the "large model is using the small model but
  still sampling from the large model's distribution" point.)
* **Black-box proxy detection** (`ivgym/io_detectors.py`: `surface_stat`,
  `surface_rank`, `logit_judge`). Here M is **never** run. The verifier only has
  a cheap proxy, and the proxy's *agreement with M* is the entire signal.

The link between them: **the SD acceptance rate is exactly the quantity the proxy
detector lives or dies on.** For a drafted token `x ~ q` (proxy) verified against
`p` (M), SD accepts with probability `min(1, p(x)/q(x))`, and the expected
single-token acceptance rate is

```
E_{x~q}[min(1, p(x)/q(x))]  =  Σ_x min(p(x), q(x))  =  1 − TV(p, q)
```

That is precisely `accept_rate = 1 − TV(M, proxy)` computed in
`experiments/exp_family_correlation.py` and `exp_cross_family_accept.py`. So the
"acceptance rate is a measurement of proxy↔M distributional agreement" statement
is exactly right, and it's the same number an SD engine realizes at runtime.

### Grounded in the engines

* **vLLM** (`vllm/v1/sample/rejection_sampler.py`): the kernel accepts iff
  `target_prob / draft_prob >= uniform_prob` — i.e. `min(1, p_target/p_draft)`.
  On rejection it samples a **recovered token** from the residual
  `max(target_prob − draft_prob, 0)` (normalized), and appends a **bonus token**
  when all drafts are accepted. That residual construction is the classic
  Leviathan/Chen speculative-sampling correction that makes the marginal output
  **identical to the target's distribution** — confirming "still sampling from
  the large model's distribution."
* **SGLang** (`python/sglang/srt/speculative/eagle_utils.py` → `eagle_sample`):
  two paths — a **greedy** path (`argmax(target_logits)` match, i.e. our
  `top1_agree` metric) and a **probabilistic rejection-sampling** path
  (`tree_speculative_sampling_target_only` / `chain_speculative_sampling_triton`)
  taking `target_probs`, `draft_probs`, and uniform "coins" — same
  `min(1, p_target/p_draft)` + residual mechanism, in CUDA/Triton kernels.

---

## 2. Is it "just low perplexity from shared training data"? No — it's *conditional* agreement.

Two controls separate "generic fluency / shared-corpus perplexity" from genuine
conditional-distribution agreement:

**(a) Shuffled-position null.** Pair each proxy distribution with M's distribution
from a *different, shuffled* position (same marginals, conditional relationship
destroyed). Every agreement metric collapses to chance:

| metric | Qwen3-1.7B proxy | shuffled null |
|---|---|---|
| accept rate (1−TV) | **0.767** | 0.019 |
| top-1 argmax agree | **0.749** | 0.019 |
| top-8 Jaccard | 0.539 | 0.022 |

If it were merely "both models find English text typical," the shuffled pairing
would score high too. It scores ~0.02. The signal is *this proxy tracks M at this
position*, i.e. conditional agreement.

**(b) Matched-size, cross-family (the new measurement).** Every model below shares
Qwen's **exact** token ids (verified: Qwen3 / Qwen2.5 / Qwen2.5-Coder /
DeepSeek-R1-Distill-Qwen all encode to identical ids; all have `vocab_size`
151936), so accept-rate/top1/KL are token-aligned *across families*. This is what
`exp_family_correlation.py` had to punt on ("different family → different
tokenizer → undefined"). Reference **M = Qwen3-4B**, honest samples from M,
proxies at a matched ~1.5B size:

| proxy (~1.5B) | family distance | accept (1−TV) | top-1 | KL(M‖proxy) |
|---|---|---|---|---|
| **Qwen3-1.7B** | same family (sibling) | **0.767** | **0.749** | **0.421** |
| Qwen2.5-1.5B | cross generation | 0.687 (−0.080) | 0.723 | 0.563 |
| Qwen2.5-Coder-1.5B | cross domain (code post-train) | 0.673 (−0.093) | 0.700 | 0.612 |
| DeepSeek-R1-Distill-Qwen-1.5B | cross post-train (RL/reasoning) | 0.625 (−0.141) | 0.644 | 1.030 |

The decisive point: **Qwen2.5-Coder-1.5B and DeepSeek-R1-Distill-Qwen-1.5B are
Qwen bases — they share M's pretraining data — yet they agree with M *less* than
the same-size Qwen3 sibling.** Size, tokenizer, and pretraining corpus are held
fixed; only the model family/training differs, and the accept rate falls
monotonically with family distance (0.767 → 0.687 → 0.673 → 0.625; KL rises
0.42 → 1.03). So it is **not** shared-data perplexity — it's conditional
distribution agreement, graded by how close the proxy's *trained* distribution is
to M's.

**Honest caveat on "collapses."** Because all four still share the Qwen tokenizer
and much pretraining data, the cross-family drop is a **graded decline, not a
cliff** — every proxy stays far above the shuffled null (0.02). A genuinely
different family (different tokenizer, e.g. Llama/Gemma) would fall further, but
token-aligned TV/KL are undefined there (Llama/Gemma are also gated on this host).
The shared-tokenizer axis is the strongest *clean* test available: it isolates
training/family from every confound.

---

## 3. Why this bounds the verifier — the "so what" for the game

`E[surface_stat | honest] = H(M) + KL(M‖proxy)`, so **KL(M‖proxy) is the proxy
detector's entire discriminative budget.** Within a family that budget is small
(0.42 nats for the 1.7B sibling), which is exactly why the black-box detectors
can't close the recompute-dominant gap. `experiments/exp_detectability_vs_kl.py`
(M=Qwen3-4B, proxy=Qwen3-0.6B) shows `surface_stat` detectability tracks
`|ΔCE| / KL`: an attack is catchable by the cheap proxy only to the extent it
spends that budget. An output-preserving deviation moves ~0 of it — e.g. `bug_k2`
(|ΔCE|/KL = 0.017) leaves `surface_stat` at its floor (0.51) while the M-recompute
`token_difr` still catches it (0.74). The wrong-seed attack redraws the *same*
distribution, so no proxy — however same-family — can see it; only recomputing M
can. (These small-pool AUCs are noisier than the large-pool README headline; the
*direction* and the KL-budget relationship are the robust part.)

---

## Reproduce

```bash
V=/home/ubuntu/inference-verification/.venv/bin/python
cd /home/ubuntu/inference-verification

# cross-family accept-rate ladder (the new result; ~2 min, ~23GB pulled+pruned)
IVGYM_M=Qwen/Qwen3-4B IVGYM_PROMPTS=16 IVGYM_TOKENS=64 \
  $V -m experiments.exp_cross_family_accept

# within-family ladder + surprisal scatter
IVGYM_M=Qwen/Qwen3-4B IVGYM_PROXIES=Qwen/Qwen3-1.7B,Qwen/Qwen3-0.6B \
  IVGYM_PROMPTS=16 IVGYM_TOKENS=64 $V -m experiments.exp_family_correlation

# accept-rate/KL budget -> detector AUC (the "so what")
IVGYM_M=Qwen/Qwen3-4B IVGYM_PROXY=Qwen/Qwen3-0.6B IVGYM_PROMPTS=16 IVGYM_TOKENS=96 \
  IVGYM_BATCH=200 $V -m experiments.exp_detectability_vs_kl
```

Results: `docs/results/exp_cross_family_accept_qwen3-4b.txt`,
`exp_family_correlation_qwen3-4b.txt`,
`exp_detectability_vs_kl_qwen3-4b_proxy0.6b.txt`.
Figures: `docs/figures/fig_cross_family_accept.png`,
`fig_family_correlation.png`, `fig_detectability_vs_kl.png`.
