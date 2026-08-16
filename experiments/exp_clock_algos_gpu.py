"""Three more clocks, and a better way to read them. Same premise, different observables.

`fig_clock_basic.png` says one thing: *a token cannot arrive before its bytes have
moved*, so a provider that reads fewer bytes arrives early and the saving IS the
signal. Everything measured so far read that premise in one place -- the inter-token
gap of a batch-1 decode -- and `docs/CLOCK_MEASURED.md` found that place to be the
worst one available: an unknown 4.15 ms stack constant sits on it, continuous batching
buries the weight term in noise, and 73% of an honest speculative stream lands in the
region the figure calls impossible.

The premise is not the problem; the observable was. A decode step is the ONE part of
serving that is neither compute-bound nor proportional to what the client asked for.
This experiment reads the same premise in three other places, all still in the time
dimension, and asks a fourth question about the estimator.

  ARM ttft       THE PREFILL CLOCK -- time to the FIRST token. Prefill is
                 compute-bound and its work is 2 x params x prompt_tokens plus
                 attention, so it reads PARAMETERS directly, where a batch-1 decode
                 step reads them through a launch-bound floor. This is the arm that
                 could give back the row the decode clock lost: `0.6B served as 1.7B`
                 was a 1.20x decode gap, and prefill should see the full parameter
                 ratio. One probe, one token, no stream.

  ARM intra      THE INTRA-STREAM CLOCK -- no probe requests at all. During YOUR OWN
                 generation the KV cache grows by one position per token, so an honest
                 ITL must RISE with the output index at a rate the client can measure
                 in the same stream. A provider windowing or evicting your context is
                 flat. This needs no second request, no pairing, no absolute floor, and
                 no traffic the client was not already buying -- which is exactly what
                 the figure claims for the channel and what no arm has delivered yet.
                 Honest growth is measured against a REAL sliding-window provider that
                 crops its KV cache to W every step.

  ARM estimator  (analysis, in plot_clock_algos) HOW TO READ IT. The figure's panel 3
                 is right that the wire's nuisance is additive and positive, and wrong
                 that this makes the MINIMUM the statistic -- the minimum is what a
                 speculative honest server destroys. But one-sidedness does mean a LOW
                 QUANTILE of the differential statistic beats its mean. Mean, trimmed
                 mean, low quantiles and the intra-stream regression are compared at
                 matched cost over the measured samples.

  ARM lockin     (analysis) MODULATE, THEN CORRELATE. Slow load drift is the one
                 nuisance a paired probe does not remove. If the client MODULATES its
                 own context length on a known pseudorandom schedule and correlates,
                 drift lands out of band. Classic lock-in detection, tested against a
                 drifting load simulated over the measured ITL(B) grid.

    python -m experiments.exp_clock_algos_gpu
Env: IVGYM_ALGO_ARMS(all), IVGYM_ALGO_REPS(5), IVGYM_ALGO_OUT(3072)

Writes `docs/results/clock_algos.json`.
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

from experiments.exp_clock_channel_gpu import (  # noqa: E402
    RES, kv_bytes_per_token, load, log, measure_bandwidth, real_ids,
)

REPS = int(os.environ.get("IVGYM_ALGO_REPS", 5))
WARMUP = int(os.environ.get("IVGYM_ALGO_WARMUP", 128))
N_OUT = int(os.environ.get("IVGYM_ALGO_OUT", 3072))
ARMS = os.environ.get("IVGYM_ALGO_ARMS", "all")

TTFT_MODELS = [("Qwen/Qwen3-0.6B", "bf16"), ("Qwen/Qwen3-1.7B", "bf16"),
               ("Qwen/Qwen3-4B", "bf16"), ("Qwen/Qwen3-1.7B", "nf4")]
TTFT_CTXS = [256, 1024, 4096, 8192, 16384, 32768]
INTRA_MODEL = "Qwen/Qwen3-1.7B"
INTRA_CTX0 = 512                      # the prompt the client actually sent
INTRA_WINDOWS = [None, 1024, 2048]    # None = honest, else a sliding window of W


# --------------------------------------------------------------- ARM ttft
def ttft_samples(model, cfg, ctx: int, reps: int) -> np.ndarray:
    """Seconds to the first token: one prefill forward over `ctx` real tokens.

    This is the whole of TTFT that the provider controls on the GPU. It is
    compute-bound -- 2 x params x ctx FLOPs of weight math plus attention -- so unlike
    a decode step it is not sitting on a launch-latency floor."""
    out = np.empty(reps)
    with torch.inference_mode():
        for r in range(reps):
            ids = real_ids(ctx, offset=97 * r) % cfg.vocab_size
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o = model(input_ids=ids, use_cache=False)
            tok = o.logits[:, -1:].argmax(-1)          # the first token is now known
            torch.cuda.synchronize()
            out[r] = (time.perf_counter() - t0) * 1e3
            del o, tok, ids
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------- ARM intra
def crop_window(cache, W: int) -> None:
    """Keep only the last W positions of every layer -- a real sliding-window server.

    Slicing the key/value tensors is a view, so the crop itself costs nothing; the
    saving is real, because the next attention reads W positions instead of all of
    them. Positions (and therefore RoPE) are not re-based, which is wrong for text
    and irrelevant for timing: the kernels, shapes and byte counts are those of a
    windowed server."""
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None and layer.keys.shape[2] > W:
            layer.keys = layer.keys[:, :, -W:, :].contiguous()
            layer.values = layer.values[:, :, -W:, :].contiguous()


def intra_stream(model, cfg, ctx0: int, n_out: int, W: int | None,
                 reps: int) -> np.ndarray:
    """[reps, n_out] inter-token latency against the OUTPUT INDEX of one generation.

    Eager, because the point is that the KV cache grows during the stream and a
    captured graph has a fixed shape by construction. Eager is also the stack where
    `CLOCK_MEASURED` measured the context term to be 53x flatter than a graph stack,
    so every number this arm produces is a conservative floor for the channel."""
    from transformers import DynamicCache
    out = np.empty((reps, n_out))
    with torch.inference_mode():
        for r in range(reps):
            ids = real_ids(ctx0, offset=1024 * r) % cfg.vocab_size
            cache = DynamicCache()
            o = model(input_ids=ids, past_key_values=cache, use_cache=True)
            tok = o.logits[:, -1:].argmax(-1)
            if W:
                crop_window(cache, W)
            # Burn in before timing: the first tens of steps of any stream carry an
            # allocator / clock-ramp transient that is not the KV cache growing, and
            # a client reading the growth in its own stream should skip it too.
            for _ in range(WARMUP):
                o = model(input_ids=tok, past_key_values=cache, use_cache=True)
                tok = o.logits[:, -1:].argmax(-1)
                if W:
                    crop_window(cache, W)
            torch.cuda.synchronize()
            for i in range(n_out):
                t0 = time.perf_counter()
                o = model(input_ids=tok, past_key_values=cache, use_cache=True)
                tok = o.logits[:, -1:].argmax(-1)
                if W:
                    crop_window(cache, W)
                torch.cuda.synchronize()
                out[r, i] = (time.perf_counter() - t0) * 1e3
            del cache, ids
            torch.cuda.empty_cache()
    return out


def main() -> None:
    t0 = time.time()
    arms = ARMS.split(",") if ARMS != "all" else ["ttft", "intra"]
    log("=" * 96)
    log("Three more clocks: the prefill clock, the intra-stream clock, and how to "
        "read them")
    log(f"  reps={REPS} n_out={N_OUT} arms={arms}")
    log("=" * 96)
    out: dict = {"reps": REPS, "n_out": N_OUT, "warmup": WARMUP,
                 "intra_ctx0": INTRA_CTX0,
                 "ttft": [], "intra": []}
    out["card"] = measure_bandwidth()
    log(f"  card: {out['card']['device']}  {out['card']['bw_copy_tb_s']:.2f} TB/s")

    if "ttft" in arms:
        log("\n  ARM ttft: time to the first token vs prompt length")
        for name, quant in TTFT_MODELS:
            try:
                model, cfg, wb = load(name, quant)
            except Exception as e:
                log(f"    {name}/{quant}: {type(e).__name__}: {str(e)[:80]}")
                continue
            lab = name.split("/")[-1] + ("-NF4" if quant == "nf4" else "")
            params = sum(p.numel() for p in model.parameters())
            for ctx in TTFT_CTXS:
                try:
                    s = ttft_samples(model, cfg, ctx, REPS)
                except Exception as e:
                    log(f"    {lab} ctx={ctx}: {type(e).__name__}")
                    torch.cuda.empty_cache()
                    continue
                rec = {"label": lab, "model": name, "quant": quant, "ctx": ctx,
                       "mean": float(s.mean()), "min": float(s.min()),
                       "sd": float(s.std()), "params": int(params),
                       "weight_bytes": wb, "ttft_ms": [round(float(x), 4) for x in s]}
                out["ttft"].append(rec)
                log(f"    {lab:18s} ctx={ctx:<6d} TTFT mean {s.mean():9.2f} ms  "
                    f"min {s.min():9.2f}  sd {s.std():7.3f}   "
                    f"{ctx / s.mean() * 1e3:8.0f} prompt tok/s")

    if "intra" in arms:
        log(f"\n  ARM intra: ITL vs output index over one {N_OUT}-token generation")
        model, cfg, wb = load(INTRA_MODEL)
        for W in INTRA_WINDOWS:
            s = intra_stream(model, cfg, INTRA_CTX0, N_OUT, W, max(2, REPS // 2))
            x = np.arange(N_OUT, dtype=float)
            slopes = []
            for r in range(s.shape[0]):
                A = np.vstack([np.ones_like(x), x]).T
                co, *_ = np.linalg.lstsq(A, s[r], rcond=None)
                slopes.append(co[1] * 1e3)                   # us per output token
            rec = {"label": "honest" if W is None else f"window_{W}", "window": W,
                   "ctx0": INTRA_CTX0, "n_out": N_OUT,
                   "slope_us_per_token": float(np.mean(slopes)),
                   "slope_reps": [float(v) for v in slopes],
                   "mean": float(s.mean()), "sd": float(s.std()),
                   "kv_bytes_per_token": kv_bytes_per_token(cfg),
                   "itl_ms": [[round(float(v), 4) for v in row] for row in s]}
            out["intra"].append(rec)
            log(f"    {rec['label']:12s} growth {rec['slope_us_per_token']:+7.3f} "
                f"us per output token   (reps: "
                f"{', '.join(f'{v:+.3f}' for v in slopes)})   mean ITL "
                f"{s.mean():.2f} ms")

    out["elapsed_s"] = time.time() - t0
    path = RES / "clock_algos.json"
    path.write_text(json.dumps(out))
    log(f"\n  wrote {path}  ({out['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
