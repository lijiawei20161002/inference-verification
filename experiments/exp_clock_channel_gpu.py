"""The clock channel, measured. This is the experiment the four clock figures propose.

`fig_clock_basic.png` / `fig_clock_simple.png` / `fig_clock_vs_difr.png` /
`fig_clock_channel_principle.png` all say the same thing in their headers: *nothing in
this repo has ever timed a provider, so read this as the design of
`exp_clock_channel_gpu` and not its result.* This file is that experiment, and it
tests the two assumptions the whole channel rests on:

  1. **THE FLOOR IS THE ROOFLINE.** "time per token >= bytes read / bandwidth", and the
     honest floor is that quantity, so a provider reading 1/k of the bytes lands at
     1/k of the time. The figures compute every floor as arithmetic off a spec sheet.
  2. **THE NUISANCE IS ADDITIVE, POSITIVE AND SMALL.** One-sided jitter sigma = 0.50 ms,
     so the minimum gap is clean and one gap left of the floor is a physical
     impossibility rather than a p-value.

Neither is a claim about a provider; both are claims about a *serving stack*, and a
stack can be measured. So: measure the whole observable, decompose it, and price what
is left on the same axis (`signal.batch_for_pauc`, delta* = 3.767) that the token
channel is priced on in `cost_of_a_verdict.json`.

WHAT IS MEASURED   One primitive, `itl_samples()`: per-token inter-token latency of a
    real batch-1..64 decode on this GPU, sampled `steps` times per rep, `reps` reps,
    with a `torch.cuda.synchronize()` after every step. Every claim below is
    post-processing on that one grid, so no arm can pick its own timing method.

      ARM floor        ITL vs (model, dtype, context) at B=1, in two stack modes --
                       eager HF, and a captured CUDA graph (what an optimized server
                       does). Fits ITL = c_stack + bytes/BW_eff and compares BW_eff
                       to this card's measured copy bandwidth. Tests assumption 1.
      ARM jitter       the honest null: sd, skew and min of device-side ITL. This is a
                       LOWER BOUND on a real client's nuisance -- there is no network
                       here -- so every d' it implies is an UPPER bound on the clock.
      ARM deviations   the byte-ratio prediction, per deviation, measured:
                         quant_nf4     real bitsandbytes NF4 weights (1/3.5 the weight
                                       bytes) vs bf16, same model, same stack.
                         substitution  0.6B served as 1.7B, and 1.7B as 4B.
                         kv_ctx        the KV term, read off the ctx slope.
                       Prediction: measured speedup == byte ratio. Reported as the
                       ratio of ratios, so a 1.0 is a pass.
      ARM batching     ITL vs B at two contexts. The figures predict the weight term
                       decays as 1/B and the KV term does not. Measured crossover.
      ARM slope        THE NEW VERIFIER. d(ITL)/d(ctx) is the KV bytes per token over
                       BW_eff -- a DIFFERENTIAL observable, so the unknown additive
                       constant c_stack (stack, queueing, network offset) cancels
                       exactly, and it is per-request, so it does not amortize over B.
                       Measured slope, its noise, and the d' / tokens-per-verdict it
                       gives against an fp8 KV cache (half the slope) and a
                       context-truncating provider (zero slope past W).
      ARM specdec      real greedy speculative decoding, 0.6B drafting for 1.7B. It is
                       distribution-EXACT, so it is an honest provider, and it emits
                       tokens in accepted blocks. Measures how many honest gaps land
                       in the "IMPOSSIBLE" region of fig_clock_basic panel 3.

MODELLED INPUTS   none. Bandwidth is measured on this card, not read off a spec sheet;
    every byte count is computed from the loaded model's own parameter count and
    config (layers, KV heads, head_dim). The only inherited constant is delta* =
    3.767, taken from `cost_of_a_verdict.json` so the two channels share one axis.

    python -m experiments.exp_clock_channel_gpu
Env: IVGYM_CLOCK_ARMS(all), IVGYM_CLOCK_STEPS(96), IVGYM_CLOCK_REPS(3),
     IVGYM_CLOCK_TAG(""), IVGYM_CLOCK_MODES(graph,eager)

Writes `docs/results/clock_channel[_TAG].json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import signal

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

STEPS = int(os.environ.get("IVGYM_CLOCK_STEPS", 96))
REPS = int(os.environ.get("IVGYM_CLOCK_REPS", 3))
TAG = os.environ.get("IVGYM_CLOCK_TAG", "")
ARMS = os.environ.get("IVGYM_CLOCK_ARMS", "all")
MODES = os.environ.get("IVGYM_CLOCK_MODES", "graph,eager").split(",")

DELTA_STAR = json.load(open(RES / "cost_of_a_verdict.json"))["delta_star"]
MEM_BUDGET_GB = 62.0            # skip a cell whose KV cache would not fit comfortably

# The grid. Contexts are powers of four-ish so the KV slope is read over two decades;
# batch sizes span the range a real server actually runs at.
CTXS = [256, 1024, 4096, 8192, 16384, 32768]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
BATCH_CTXS = [1024, 8192]
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
QUANT_MODEL = "Qwen/Qwen3-1.7B"
# The out-of-sample arm. Same card, same stack, deliberately DIFFERENT KV geometry:
# Qwen2.5 uses 2 KV heads where Qwen3-1.7B uses 8, and SmolLM2 is full MHA with 32.
# If the measured slope tracks each model's own config, a client can predict the
# honest slope for a model it has never timed -- which is what turns the differential
# verifier from a relative test into an absolute one.
ARCH_MODELS = os.environ.get(
    "IVGYM_CLOCK_ARCH",
    "Qwen/Qwen2.5-1.5B,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen3-0.6B").split(",")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- the card
def measure_bandwidth() -> dict:
    """Achieved HBM bandwidth and single-kernel launch cost on THIS card. The clock's
    floor is bytes/bandwidth, so a measured bandwidth is the honest denominator."""
    n = int(1e9)
    a = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        b.copy_(a)
    torch.cuda.synchronize()
    copy_s = (time.perf_counter() - t) / 20
    s = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(2000):
        s.add_(1)
    torch.cuda.synchronize()
    launch_us = (time.perf_counter() - t) / 2000 * 1e6
    del a, b
    torch.cuda.empty_cache()
    return {"bw_copy_tb_s": 4e9 / copy_s / 1e12, "launch_us": launch_us,
            "device": torch.cuda.get_device_name(0)}


# ------------------------------------------------------------------- the models
_LOADED: dict[str, tuple] = {}


def load(name: str, quant: str = "bf16"):
    """(model, config, bytes_of_weights). Cached, because the grid revisits models."""
    key = f"{name}|{quant}"
    if key in _LOADED:
        return _LOADED[key]
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if quant != "nf4":
        model = model.cuda()
    model.eval()
    # Weight bytes as the loaded object actually reads them: for NF4 the packed
    # 4-bit blocks plus their scales, i.e. element_size() per parameter tensor.
    wb = sum(p.numel() * p.element_size() for p in model.parameters())
    _LOADED[key] = (model, model.config, float(wb))
    return _LOADED[key]


_TEXT_IDS: list[int] = []


def real_ids(n: int, offset: int = 0) -> torch.Tensor:
    """[1, n] of REAL text token ids (the repo's own long-text fixture), tiled if the
    requested context exceeds it. Timing is content-independent, but the acceptance
    rate of the speculation arm is not -- random ids make both models degenerate into
    repetition and accept ~everything, which would flatter the arm."""
    global _TEXT_IDS
    if not _TEXT_IDS:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(MODELS[0])
        txt = (ROOT / "experiments" / "data" / "specdec_longtext.txt").read_text()
        _TEXT_IDS = tk(txt).input_ids
    need = n + offset
    reps = need // len(_TEXT_IDS) + 1
    ids = (_TEXT_IDS * reps)[offset: offset + n]
    return torch.tensor([ids], device="cuda")


def kv_bytes_per_token(cfg, width: int = 2) -> float:
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # GQA models expose num_key_value_heads; a full-MHA config (GPT-NeoX) does not.
    kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    return 2.0 * cfg.num_hidden_layers * kvh * hd * width


# ------------------------------------------------------- the measurement primitive
def itl_samples(model, cfg, B: int, ctx: int, steps: int, reps: int,
                mode: str = "graph") -> np.ndarray | None:
    """[reps, steps] per-token inter-token latency in ms, batch-B decode at context
    `ctx`. `mode='graph'` captures one decode step into a CUDA graph and replays it,
    which is what a production server's overhead looks like; `mode='eager'` is the
    plain HF loop. Returns None if the cell does not fit."""
    from transformers import DynamicCache, StaticCache

    total = ctx + steps + 8
    kv_gb = kv_bytes_per_token(cfg) * total * B / 1e9
    if kv_gb > MEM_BUDGET_GB:
        return None
    out = np.empty((reps, steps))
    try:
        with torch.inference_mode():
            for r in range(reps):
                # ids are clamped into this model's vocab: the fixture is tokenized
                # once with one tokenizer, and decode latency is content-independent.
                ids = (real_ids(ctx, offset=1024 * r) % cfg.vocab_size)
                ids = ids.expand(B, -1).contiguous()
                if mode == "graph":
                    cache = StaticCache(config=cfg, max_batch_size=B,
                                        max_cache_len=total, device="cuda",
                                        dtype=torch.bfloat16)
                    o = model(input_ids=ids, past_key_values=cache, use_cache=True,
                              cache_position=torch.arange(ctx, device="cuda"))
                    tok = o.logits[:, -1:].argmax(-1)
                    pos = torch.tensor([ctx], device="cuda")
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):          # warm up before capture
                        for _ in range(3):
                            o = model(input_ids=tok, past_key_values=cache,
                                      cache_position=pos, use_cache=True)
                            tok.copy_(o.logits[:, -1:].argmax(-1))
                            pos.add_(1)
                    torch.cuda.current_stream().wait_stream(stream)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        o = model(input_ids=tok, past_key_values=cache,
                                  cache_position=pos, use_cache=True)
                        tok.copy_(o.logits[:, -1:].argmax(-1))
                        pos.add_(1)
                    torch.cuda.synchronize()
                    for i in range(steps):
                        t0 = time.perf_counter()
                        g.replay()
                        torch.cuda.synchronize()
                        out[r, i] = (time.perf_counter() - t0) * 1e3
                    del g, cache
                else:
                    if mode.startswith("kvq"):
                        from transformers import QuantizedCache
                        cache = QuantizedCache(backend="quanto", config=cfg,
                                               nbits=int(mode[3:]), q_group_size=64,
                                               residual_length=64)
                    else:
                        cache = DynamicCache()
                    o = model(input_ids=ids, past_key_values=cache, use_cache=True)
                    tok = o.logits[:, -1:].argmax(-1)
                    torch.cuda.synchronize()
                    for i in range(steps):
                        t0 = time.perf_counter()
                        o = model(input_ids=tok, past_key_values=cache, use_cache=True)
                        tok = o.logits[:, -1:].argmax(-1)
                        torch.cuda.synchronize()
                        out[r, i] = (time.perf_counter() - t0) * 1e3
                    del cache
                del ids
                torch.cuda.empty_cache()
    except Exception as e:                  # OOM, or a stack that will not graph-capture
        torch.cuda.empty_cache()
        log(f"    !! cell failed (B={B} ctx={ctx} {mode}): {type(e).__name__}: "
            f"{str(e)[:120]}")
        return None
    return out


def cell(model, cfg, wb, B, ctx, mode, label) -> dict | None:
    s = itl_samples(model, cfg, B, ctx, STEPS, REPS, mode)
    if s is None:
        log(f"    {label:34s} B={B:<3d} ctx={ctx:<6d} {mode:5s}  SKIPPED (memory)")
        return None
    flat = s.reshape(-1)
    bytes_read = wb + kv_bytes_per_token(cfg) * (ctx + STEPS / 2) * B
    d = {"label": label, "B": B, "ctx": ctx, "mode": mode,
         "mean": float(flat.mean()), "min": float(flat.min()),
         "p50": float(np.median(flat)), "sd": float(flat.std()),
         "skew": float(((flat - flat.mean()) ** 3).mean() / (flat.std() ** 3 + 1e-12)),
         "rep_means": [float(x) for x in s.mean(axis=1)],
         "bytes_read": bytes_read, "weight_bytes": wb,
         "kv_bytes_per_token": kv_bytes_per_token(cfg),
         "itl_ms": [round(float(x), 4) for x in flat]}
    log(f"    {label:34s} B={B:<3d} ctx={ctx:<6d} {mode:5s}  "
        f"mean {d['mean']:7.3f}  min {d['min']:7.3f}  sd {d['sd']:6.3f} ms")
    return d


# ------------------------------------------------------------------ ARM specdec
def specdec_gaps(target_name: str, draft_name: str, k: int = 4, blocks: int = 64,
                 ctx: int = 512, reps: int = 3) -> dict:
    """Real greedy speculative decoding, timed. Lossless: with greedy acceptance the
    emitted sequence is EXACTLY the target's own greedy continuation, so this is an
    honest provider by construction. Tokens become visible to a client in accepted
    blocks, so the per-token gap is (block time) for the first token of a block and
    ~0 for the rest -- which is the thing the minimum statistic cannot survive."""
    from transformers import DynamicCache
    tgt, tcfg, _ = load(target_name)
    drf, dcfg, _ = load(draft_name)
    gaps, accepted, block_ms = [], [], []
    with torch.inference_mode():
        for r in range(reps):
            ids = real_ids(ctx, offset=4096 * r)
            tc, dc = DynamicCache(), DynamicCache()
            # Invariant: both caches hold the committed prefix EXCLUDING its last
            # token `x0`, which is fed at the head of the next verification pass.
            o = tgt(input_ids=ids, past_key_values=tc, use_cache=True)
            x0 = o.logits[:, -1:].argmax(-1)              # first committed output token
            drf(input_ids=ids, past_key_values=dc, use_cache=True)
            keep = ctx                                    # cache length = committed - 1
            torch.cuda.synchronize()
            for _ in range(blocks):
                t0 = time.perf_counter()
                q, nxt = [], x0
                for _ in range(k):                        # k drafts, k drafter passes
                    o = drf(input_ids=nxt, past_key_values=dc, use_cache=True)
                    nxt = o.logits[:, -1:].argmax(-1)
                    q.append(nxt)
                blk = torch.cat([x0] + q, dim=1)          # [1, k+1] = x0, q_1..q_k
                o = tgt(input_ids=blk, past_key_values=tc, use_cache=True)
                t = o.logits.argmax(-1)                   # t[j] = target greedy after blk[j]
                n = 0
                while n < k and int(q[n]) == int(t[0, n]):
                    n += 1
                emitted = n + 1                           # n accepted drafts + 1 bonus
                x0 = t[:, n: n + 1]                       # new last-committed token
                keep = keep + 1 + n                       # ids + x0 + q_1..q_n, minus x0'
                tc.crop(-(tc.get_seq_length() - keep))
                if n == k:               # every draft accepted: the drafter never saw
                    drf(input_ids=q[k - 1], past_key_values=dc, use_cache=True)
                else:                    # its own last draft, so feed it before moving on
                    dc.crop(-(dc.get_seq_length() - keep))
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e3
                block_ms.append(dt)
                accepted.append(emitted)
                gaps.extend([dt] + [0.0] * (emitted - 1))  # client-visible arrivals
            del tc, dc, ids
            torch.cuda.empty_cache()
    g = np.asarray(gaps)
    return {"k": k, "ctx": ctx, "blocks": blocks * reps,
            "emitted_mean": float(np.mean(accepted)), "emitted_max": k + 1,
            "emitted_hist": [int((np.asarray(accepted) == i).sum()) for i in range(1, k + 2)],
            "amortized_ms_per_token": float(np.sum(block_ms) / np.sum(accepted)),
            "block_ms_mean": float(np.mean(block_ms)),
            "gap_min": float(g.min()), "gap_mean": float(g.mean()),
            "frac_gaps_zero": float((g == 0).mean()),
            "gaps_ms": [round(float(x), 4) for x in g[:2000]]}


# ---------------------------------------------------------------------- analysis
def fit_line(x, y):
    """Least squares y = a + b x, with the sd of the residuals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(coef[0]), float(coef[1]), float(resid.std())


def price(d_prime: float) -> float:
    """Tokens of stream per verdict at the repo's own operating point."""
    if d_prime <= 0:
        return float("inf")
    return float((DELTA_STAR / d_prime) ** 2)


def main() -> None:
    t0 = time.time()
    arms = ARMS.split(",") if ARMS != "all" else \
        ["bandwidth", "floor", "batching", "arch", "kvquant", "deviations", "slope",
         "specdec"]
    log("=" * 96)
    log("The clock channel, measured.  Does 'time per token = bytes / bandwidth' "
        "survive a real stack?")
    log(f"  steps={STEPS} reps={REPS} modes={MODES} arms={arms}")
    log("=" * 96)

    out: dict = {"steps": STEPS, "reps": REPS, "modes": MODES, "arms": arms,
                 "delta_star": DELTA_STAR, "cells": [], "specdec": None}

    out["card"] = measure_bandwidth()
    log(f"  card: {out['card']['device']}   copy bandwidth "
        f"{out['card']['bw_copy_tb_s']:.2f} TB/s   launch {out['card']['launch_us']:.1f} us")

    # ---- ARM floor: ITL vs (model, dtype, ctx) at B=1, both stack modes
    if "floor" in arms:
        log("\n  ARM floor / jitter / deviations: ITL vs (model, dtype, context), B=1")
        for name in MODELS:
            model, cfg, wb = load(name)
            for mode in MODES:
                for ctx in CTXS:
                    c = cell(model, cfg, wb, 1, ctx, mode, name.split("/")[-1])
                    if c:
                        c["model"], c["quant"] = name, "bf16"
                        out["cells"].append(c)
        try:
            model, cfg, wb = load(QUANT_MODEL, "nf4")
            for mode in MODES:
                for ctx in CTXS:
                    c = cell(model, cfg, wb, 1, ctx, mode, "Qwen3-1.7B-NF4")
                    if c:
                        c["model"], c["quant"] = QUANT_MODEL, "nf4"
                        out["cells"].append(c)
        except Exception as e:                             # bnb absent / not capturable
            log(f"    NF4 arm unavailable: {type(e).__name__}: {e}")

    # ---- ARM batching: ITL vs B, honest and deviating, at two contexts
    if "batching" in arms:
        log("\n  ARM batching: ITL vs concurrent requests B")
        for name, quant in ((QUANT_MODEL, "bf16"), ("Qwen/Qwen3-0.6B", "bf16"),
                            (QUANT_MODEL, "nf4")):
            try:
                model, cfg, wb = load(name, quant)
            except Exception as e:
                log(f"    {name}/{quant} unavailable: {type(e).__name__}")
                continue
            lab = name.split("/")[-1] + ("-NF4" if quant == "nf4" else "")
            for mode in MODES:
                for ctx in BATCH_CTXS:
                    for B in BATCHES:
                        c = cell(model, cfg, wb, B, ctx, mode, lab)
                        if c:
                            c["model"], c["quant"] = name, quant
                            out["cells"].append(c)

    # ---- ARM arch: out-of-sample KV geometry, graph mode only
    if "arch" in arms:
        log("\n  ARM arch: does the slope track a model's own KV config, "
            "out of sample?")
        for name in ARCH_MODELS:
            try:
                model, cfg, wb = load(name)
            except Exception as e:
                log(f"    {name} unavailable: {type(e).__name__}: {str(e)[:90]}")
                continue
            for ctx in CTXS:
                c = cell(model, cfg, wb, 1, ctx, "graph", name.split("/")[-1])
                if c:
                    c["model"], c["quant"], c["arm"] = name, "bf16", "arch"
                    out["cells"].append(c)

    # ---- ARM kvquant: a genuinely quantized KV cache, same positions, fewer bytes
    if "kvquant" in arms:
        log("\n  ARM kvquant: does a real quantized KV cache flatten the context "
            "slope?")
        model, cfg, wb = load(QUANT_MODEL)
        for mode in ("eager", "kvq8", "kvq4"):
            for ctx in [c for c in CTXS if c <= 8192]:
                c = cell(model, cfg, wb, 1, ctx, mode, "Qwen3-1.7B-kv")
                if c:
                    c["model"], c["quant"], c["arm"] = QUANT_MODEL, "bf16", "kvquant"
                    out["cells"].append(c)

    # ---- ARM specdec: honest, distribution-exact, and bursty
    if "specdec" in arms:
        log("\n  ARM specdec: real greedy speculation, 0.6B drafting for 1.7B")
        out["specdec"] = specdec_gaps(QUANT_MODEL, "Qwen/Qwen3-0.6B")
        s = out["specdec"]
        log(f"    emitted {s['emitted_mean']:.2f}/{s['k'] + 1} tokens per block, "
            f"amortized {s['amortized_ms_per_token']:.3f} ms/token, "
            f"{s['frac_gaps_zero']:.0%} of client-visible gaps are 0 ms")

    out["elapsed_s"] = time.time() - t0
    path = RES / f"clock_channel{('_' + TAG) if TAG else ''}.json"
    path.write_text(json.dumps(out))
    log(f"\n  wrote {path}  ({out['elapsed_s']:.0f}s, {len(out['cells'])} cells)")


if __name__ == "__main__":
    main()
