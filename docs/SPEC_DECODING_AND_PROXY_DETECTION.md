# Speculative decoding and client-side proxy detection

*Why a cheap proxy the client runs itself can stand in for recomputing the
reference model — and exactly where it stops working. Ties together
`ivgym/spec_decode.py`, `ivgym/io_detectors.py`, and the real-model measurements
in `experiments/exp_family_correlation.py` / `exp_cross_family_accept.py` /
`exp_detectability_vs_kl.py`.*

## The setup

The strong inference verifier (`defenses.token_difr`) recomputes the reference
model `M` for every claimed token. It is sound but expensive — as expensive as
re-running inference. We want a **cheap** verifier that runs **entirely on the
client** and asks for nothing from the provider (the provider is the untrusted
party; a verifier that needs its cooperation is one it can stub out).

The client holds its own small **proxy** model `q` — a genuinely different,
smaller model that shares `M`'s tokenizer (e.g. Qwen3-0.6B for Qwen3-4B). Two
equivalent readings of the same signal:

- **Speculative decoding.** A drafted token `x ~ q` is accepted with probability
  `min(1, p(x)/q(x))`. The expected acceptance rate is
  `E_{x~q}[min(1, p/q)] = Σ_x min(p, q) = 1 − TV(p, q)`.
- **Proxy surprisal.** The per-token NLL of a served token under the proxy,
  `−log q(x)` (`io_detectors.proxy_nll` / the `surface_stat` detector). Its
  expectation over honest tokens is the cross-entropy `H(M, q) = H(M) + KL(M‖q)`.

Both reduce to a single quantity: **how well the cheap proxy `q` tracks the target
`p`.** The client owns `q`, so `q` is a **trusted anchor** even though `p` (the
served distribution) is not.

## Why it detects a corrupted target

When a provider serves under a corrupted target — quantized / fp8 / fewer-layer /
stolen weights, `p* → p̂` — the acceptance rate moves from `1 − TV(p*, q)` to
`1 − TV(p̂, q)`. The client sees the shift **without recomputing `M`**, by
comparing the realized rate to a **one-time offline honest reference** for this
`(M, proxy, spec)` (`spec_decode.ProxyReference`). `spec_decode.ProxySpecVerifier`
calibrates a threshold on honest samples and flags a served sample whose
standardized acceptance-rate distance exceeds it.

Crucially, the acceptance rate is **draft-anchored**, so it survives the classic
evasion: a provider that quantizes and then **retunes temperature** to match the
honest *entropy* blinds a generic entropy / cross-entropy fingerprint, but does
**not** restore `TV(p̂, q)` — so the acceptance rate still separates it
(`adv_quant_temp` in `experiments/exp_proxy_spec_verify.py`).

## Where it stops — the detection budget

The signal is bounded by the proxy↔`M` disagreement:

```
E[surface_stat | honest] = H(M, q) = H(M) + KL(M ‖ q)
```

so `KL(M ‖ q)` is the **entire budget** of any proxy detector. `exp_family_correlation`
measures this on a real Qwen3 ladder (M = Qwen3-4B; proxies 1.7B, 0.6B): within a
family the conditional agreement is high (`accept_rate = 1 − TV(M, proxy)`, top-1
argmax agreement, top-8 Jaccard all far above their shuffled-context null) and
`KL(M‖proxy)` shrinks as the proxy approaches `M`. `exp_cross_family_accept` shows
the acceptance rate falling monotonically with family distance across the Qwen
tokenizer-sharing families (Qwen3 / Qwen2.5 / Qwen2.5-Coder / DeepSeek-R1-Distill-Qwen).

The consequences (the same recompute-dominant boundary the DiFR detectors hit):

- **Small quantization is invisible** — it moves `TV(p̂, q)` less than honest
  run-to-run variance.
- An **output-preserving** deviation (a wrong sampling *seed* redraws the *same*
  distribution) moves **zero** of the budget → the proxy sits at its honest floor
  while `token_difr` (recompute) separates it (`exp_io_detector_gpu`).
- The cheap proxy therefore **shrinks how often the exact recompute must fire**; it
  does not replace it.

## Measured on real models

Two GPU experiments put numbers on both sides of that boundary (H100, Qwen3
family; full logs in `docs/results/`):

- **The win case — model substitution** (`exp_spec_substitution_gpu`). A provider
  paid to serve Qwen3-4B secretly serves Qwen3-0.6B and bills for the 4B. Against
  a trusted Qwen3-1.7B draft the honest accept rate is 0.776 vs 0.700 under the
  substitute — token-batched **AUC 0.998** (per-sequence 0.819), and the verifier
  never runs `M`. Swapping the whole model changes the served conditional
  distribution wholesale, so `TV(p, q)` shifts far past honest variance.
- **The stop case — realistic quant** (`exp_spec_verifier_cost`). The
  `ProxySpecVerifier` run end-to-end against real forward-pass cheats costs
  `params(q)/params(M)` of the recompute (6.7× fewer FLOPs for 4B/0.6B), but at
  realistic strength (`quant_4bit`, `adv_quant_temp`) its measured AUC sits near
  chance while `token_difr` separates every attack (0.87–1.0). Real quant noise
  moves `TV(p, q)` less than a real model's honest run-to-run variance — the
  recompute-dominant regime, exactly as the budget above predicts.

## Files

- `ivgym/spec_decode.py` — `accept_rate` / `per_token_accept_prob`,
  `ProxyReference`, `ProxySpecVerifier`, `recompute_divergence` (the full-`M`
  baseline the proxy approximates); pure numpy, CPU-runnable.
- `ivgym/io_detectors.py` — `proxy_nll` / `proxy_rank` and `surface_stat`: the
  per-token client-side proxy signals on a real backend.
- `ivgym/backends/hf_gpu.py` — `proxy_model_name=` loads a real second model;
  `proxy_logits` is its genuine forward pass (must share `M`'s tokenizer/vocab).
- `experiments/exp_proxy_spec_verify.py` — the acceptance-rate verifier, CPU sweep
  plus an optional real-proxy run vs `token_difr`.
- `experiments/exp_spec_substitution_gpu.py` — the real-model win case: model
  substitution caught from the accept rate alone, never running `M`.
- `experiments/exp_spec_verifier_cost.py` — `ProxySpecVerifier` end-to-end on GPU:
  measured cost saving vs detection AUC (generates
  `docs/figures/fig_spec_verifier_cost.png`).
- `experiments/exp_family_correlation.py` / `exp_cross_family_accept.py` /
  `exp_detectability_vs_kl.py` — the real-model measurements of the budget above.

## Sources

- DiFR: Inference Verification Despite Nondeterminism — https://arxiv.org/pdf/2511.20621
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding — https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating LLM Decoding with Speculative Sampling — https://arxiv.org/abs/2302.01318
