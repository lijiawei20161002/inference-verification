"""ONE clean real-model GPU example that speculative verification is useful.

The scenario is the fraud speculative verification is actually good at: **model
substitution**. A provider is paid to serve the reference model ``M`` (Qwen3-4B)
but quietly serves a much cheaper model ``S`` (Qwen3-0.6B, ~6.7x cheaper) and
bills for M. The client cannot afford to recompute M on every token (that is as
expensive as inference itself -- the ``token_difr`` recompute baseline). Instead
it holds a small *trusted* proxy ``q`` (Qwen3-1.7B) and scores the
speculative-decoding acceptance rate

    accept_rate = 1 - TV(p, q)

between the provider-SERVED distribution ``p`` (free from a logprob API) and its
own proxy ``q`` (one cheap forward pass). It NEVER runs M.

Why this is the regime where the speculative verifier wins (unlike the
forward-pass-noise attacks in exp_spec_verifier_cost, where quant barely moves
TV(p,q)): swapping the whole model changes the served *conditional distribution*
wholesale, so TV(p, q) shifts well past the honest anchor's run-to-run variance
-- exactly the "acceptance rate collapses across models" intuition that makes
speculative decoding work, used here as a cheap detector.

We report, side by side:
  performance : detection AUC (honest-served vs substitute-served), batched and
                averaged over batch-composition seeds like the rest of the repo.
  cost        : MEASURED GPU wall-clock of the verifier's only forward pass (the
                1.7B proxy prefill) vs the full-M recompute a DiFR verifier needs,
                plus the analytic FLOP ratio.

Run:
    .venv/bin/python -m experiments.exp_spec_substitution_gpu
Env:
    IVGYM_M (claimed model, default Qwen/Qwen3-4B)
    IVGYM_PROXY (client trusted draft, default Qwen/Qwen3-1.7B)
    IVGYM_SUB (what the cheat actually serves, default Qwen/Qwen3-0.6B)
    IVGYM_PROMPTS (default 40), IVGYM_TOKENS (default 96), IVGYM_BATCH (default 128)
    IVGYM_AUC_SEEDS (default 16), IVGYM_GPU_USD_PER_HR (default 2.50)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import harness, spec_decode as sd
from ivgym.backends.hf_gpu import DEFAULT_PROMPTS
from ivgym.metrics import roc_auc

M_NAME = os.environ.get("IVGYM_M", "Qwen/Qwen3-4B")
PROXY_NAME = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-1.7B")
SUB_NAME = os.environ.get("IVGYM_SUB", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 40))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 128))
AUC_SEEDS = int(os.environ.get("IVGYM_AUC_SEEDS", 16))
USD_PER_HR = float(os.environ.get("IVGYM_GPU_USD_PER_HR", 2.50))
MAX_PROMPT_TOK = 32
TEMP = 1.0


def load(name, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = (AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
             attn_implementation="eager").to("cuda").eval())
    n = sum(p.numel() for p in model.parameters())
    return tok, model, n


def serve(model, tok, torch, prompt_text):
    """Autoregressively serve N_TOKENS from `model`; return (token_ids, served
    logits per position [T, V]). This is what a provider running `model` emits
    and hands back through a logprob API."""
    ids = tok(prompt_text, return_tensors="pt").input_ids[:, :MAX_PROMPT_TOK].to("cuda")
    served, toks = [], []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past, last = out.past_key_values, out.logits[0, -1]
        for _ in range(N_TOKENS):
            lg = last.float().cpu().numpy()
            served.append(lg)
            p = sd.softmax(lg / TEMP)
            # sample from the served distribution (rng varied by position count)
            t = int(np.argmax(np.log(p + 1e-12) + _gumbel(len(p), len(toks))))
            toks.append(t)
            step = torch.tensor([[t]], device="cuda", dtype=ids.dtype)
            out = model(step, past_key_values=past, use_cache=True)
            past, last = out.past_key_values, out.logits[0, -1]
    return toks, np.stack(served)


_G = {}
def _gumbel(v, pos):
    # deterministic per (vocab,pos) so honest/cheat share sampling noise structure
    if (v, pos) not in _G:
        rng = np.random.default_rng((v, pos, 12345))
        _G[(v, pos)] = rng.gumbel(size=v)
    return _G[(v, pos)]


def proxy_prefill(proxy, tok, torch, prompt_text, served_toks, timer=None):
    """One cheap proxy forward pass over [prompt + served_toks]; return per-position
    proxy logits [T, V] aligned to the served tokens."""
    ids = tok(prompt_text, return_tensors="pt").input_ids[:, :MAX_PROMPT_TOK].to("cuda")
    L = ids.shape[1]
    full = torch.cat([ids, torch.tensor([served_toks], device="cuda", dtype=ids.dtype)], dim=1)
    if timer is not None:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = proxy(full)
    if timer is not None:
        torch.cuda.synchronize(); timer.append(time.perf_counter() - t0)
    idx = slice(L - 1, L - 1 + N_TOKENS)
    return out.logits[0, idx].float().cpu().numpy()


def spec_tv_scores(served_logits_list, proxy_logits_list):
    """Per-token TV(served p, proxy q) = 1 - accept_rate, flattened over tokens."""
    out = []
    for sv, px in zip(served_logits_list, proxy_logits_list):
        for lp, lq in zip(sv, px):
            out.append(sd.tv(sd.softmax(lp / TEMP), sd.softmax(lq)))
    return np.asarray(out, float)


def batched_auc(h_scores, a_scores, name):
    """One-sided batched AUC: substitution only ever LOWERS the accept rate (raises
    TV), so the raw per-token TV is the detector directly -- higher = more likely a
    cheaper substitute. No two-sided |x-mu| anchor (that is for forward-pass noise
    that can move TV either way) and no winsorization (which would clip the cheat's
    higher tail back toward the honest range and erase the signal)."""
    ts_h = harness.TokenScores("honest", {name: h_scores})
    ts_a = harness.TokenScores("attack", {name: a_scores})
    shim = type("D", (), {"name": name})()
    aucs, tprs = [], []
    for seed in range(AUC_SEEDS):
        r = harness.evaluate(ts_h, ts_a, [shim], [BATCH], seed=seed, winsor_pct=None)[0]
        aucs.append(r.auc); tprs.append(r.tpr_at_1pct)
    return float(np.mean(aucs)), float(np.mean(tprs))


def split_floor(scores, name):
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(scores)); half = len(scores) // 2
    return batched_auc(scores[perm[:half]], scores[perm[half:]], name)


def per_seq_auc(h_served, h_proxy, a_served, a_proxy):
    """The provider-level verdict: one mean accept rate per served SEQUENCE, then
    AUC of separating honest sequences from substitute sequences (score = mean TV)."""
    def seq_tv(served, proxy):
        return np.array([np.mean([sd.tv(sd.softmax(lp / TEMP), sd.softmax(lq))
                                  for lp, lq in zip(sv[1], px)])
                         for sv, px in zip(served, proxy)])
    h = seq_tv(h_served, h_proxy); a = seq_tv(a_served, a_proxy)
    return roc_auc(h, a), h, a


def main():
    import torch
    print("=" * 84)
    print("Speculative verification catches MODEL SUBSTITUTION (real models, GPU)")
    print("=" * 84)
    t0 = time.time()
    tok_m, M, n_m = load(M_NAME, torch)
    tok_s, S, n_s = load(SUB_NAME, torch)
    tok_q, Q, n_q = load(PROXY_NAME, torch)
    assert M.config.vocab_size == Q.config.vocab_size == S.config.vocab_size, "shared tokenizer required"
    print(f"loaded ({time.time()-t0:.1f}s)")
    print(f"  claimed model  M = {M_NAME} ({n_m/1e9:.2f}B)   -- what the provider is PAID to serve")
    print(f"  substitute     S = {SUB_NAME} ({n_s/1e9:.2f}B)  -- what the cheat ACTUALLY serves "
          f"({n_m/n_s:.1f}x cheaper)")
    print(f"  client proxy   q = {PROXY_NAME} ({n_q/1e9:.2f}B)  -- the trusted draft; verifier runs ONLY this")
    print(f"config: {N_PROMPTS} prompts x {N_TOKENS} tokens, batch={BATCH}, temp={TEMP}\n")

    prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(N_PROMPTS)]

    # --- provider serves (honest = real M ; cheat = substitute S) ---
    print("serving honest sequences (provider runs M) ...")
    h_served = [serve(M, tok_m, torch, p) for p in prompts]
    print("serving cheat sequences (provider secretly runs S) ...")
    a_served = [serve(S, tok_s, torch, p) for p in prompts]

    # --- client speculative verifier: proxy prefill over each served sequence ---
    print("client verifier: proxy prefill over served sequences (the ONLY forward pass it runs) ...")
    proxy_ms = []
    h_proxy = [proxy_prefill(Q, tok_q, torch, prompts[i], h_served[i][0], proxy_ms)
               for i in range(N_PROMPTS)]
    a_proxy = [proxy_prefill(Q, tok_q, torch, prompts[i], a_served[i][0], proxy_ms)
               for i in range(N_PROMPTS)]

    h_raw = spec_tv_scores([s[1] for s in h_served], h_proxy)
    a_raw = spec_tv_scores([s[1] for s in a_served], a_proxy)
    mu, sig = h_raw.mean(), h_raw.std() + 1e-12
    print(f"\nhonest anchor:  accept_rate = 1 - TV(M-served, proxy)  = {1-mu:.3f}")
    print(f"under cheat:    accept_rate = 1 - TV(S-served, proxy)  = {1-a_raw.mean():.3f}  "
          f"(shift {a_raw.mean()-mu:+.3f} in TV)\n")

    auc_floor, tpr_floor = split_floor(h_raw, "spec_accept")
    auc, tpr = batched_auc(h_raw, a_raw, "spec_accept")
    seq_auc, h_seq, a_seq = per_seq_auc(h_served, h_proxy, a_served, a_proxy)

    print("DETECTION PERFORMANCE  (spec_accept = 1 - TV(served p, proxy q); "
          "substitute lowers it)")
    print(f"  honest-null floor      (token-batched, avg {AUC_SEEDS} seeds) : "
          f"AUC {auc_floor:.3f}  (TPR {tpr_floor:.2f})")
    print(f"  spec_accept  honest vs substitute, token-batched (b={BATCH})  : "
          f"AUC {auc:.3f}  (TPR {tpr:.2f})")
    print(f"  spec_accept  per-SEQUENCE verdict (N={N_PROMPTS} seqs each)    : "
          f"AUC {seq_auc:.3f}")
    print(f"    honest seq mean-accept {1-h_seq.mean():.3f}+-{h_seq.std():.3f}   "
          f"substitute {1-a_seq.mean():.3f}+-{a_seq.std():.3f}\n")

    # --- cost: proxy prefill (what verifier runs) vs full-M recompute (DiFR) ---
    print("=" * 84)
    print("COST  (measured GPU wall-clock, one prefill over [prompt+served] per seq)")
    print("=" * 84)
    # time a full-M prefill on the same sequences for the recompute baseline
    ref_ms = []
    for i in range(N_PROMPTS):
        proxy_prefill(M, tok_m, torch, prompts[i], h_served[i][0], ref_ms)  # M as "recompute"
    prx = float(np.mean(proxy_ms)) * 1e3
    ref = float(np.mean(ref_ms)) * 1e3

    def per_1m_usd(ms):
        return (ms / 1e3 / N_TOKENS) * 1e6 / 3600.0 * USD_PER_HR

    print(f"  {'verifier':>28} | {'ms / seq':>9} | {'GFLOP / tok':>11} | {'$ / 1M tok':>10}")
    print("  " + "-" * 70)
    print(f"  {'token_difr (recompute M 4B)':>28} | {ref:>9.1f} | {2*n_m/1e9:>11.1f} | "
          f"{per_1m_usd(ref):>10.3f}")
    print(f"  {'spec_accept (proxy q 1.7B)':>28} | {prx:>9.1f} | {2*n_q/1e9:>11.1f} | "
          f"{per_1m_usd(prx):>10.3f}")
    print("  " + "-" * 70)
    print(f"  cost saving (recompute / proxy):  {ref/max(prx,1e-9):.1f}x wall-clock   "
          f"|   {n_m/n_q:.1f}x FLOPs")
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
