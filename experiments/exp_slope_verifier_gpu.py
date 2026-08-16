"""The context-slope verifier, examined. The one channel that survived the timing run.

`docs/CLOCK_MEASURED.md` killed three quarters of the clock: the floor is a stack
property, real NF4 weights deliver 14% of their predicted time saving and invert under
batching, and 73% of an honest speculative stream lands inside the region the figures
call impossible. One thing survived -- the DIFFERENTIAL statistic

    D  =  ITL(ctx_hi)  -  ITL(ctx_lo)

measured on two probe requests. The stack constant, any fixed network offset and any
constant padding cancel; co-tenancy only ever raises it; and it reads the one
deviation family the returned-token channel is worst at, a provider that does not
attend the context it charges for.

That was a proposal with a measured slope behind it. This experiment asks the four
questions that decide whether it is a verifier:

  ARM arch2      CAN A CLIENT PREDICT THE HONEST FLOOR WITHOUT A TRUSTED BASELINE?
                 Five architectures never used to fit anything -- Llama-3.2, TinyLlama,
                 Pythia (full MHA, GPT-NeoX), OLMo-2, Qwen2.5-0.5B -- spanning 16-24
                 layers and 12 kB to 197 kB of KV per token. If the slope is predicted
                 by a model's own layer count, a client computes the floor off the
                 model card and the test is ABSOLUTE. If it needs a fitted constant,
                 the test is relative and needs a run it is willing to call honest.

  ARM window     DOES IT DETECT A REAL TRUNCATING PROVIDER, ON THE HOUSE PROTOCOL?
                 A provider that claims 32 768 tokens of context and holds W. This is
                 not a simulation: the cell prefills W positions and decodes with W in
                 cache, which is byte-for-byte and position-for-position what a
                 sliding-window or evicting server does. Scored through
                 `harness.evaluate` -- standardized pAUC at FPR <= 0.5%, honest
                 calibration split, winsorization, batch/pool ceiling -- so the clock
                 lands on the same scoreboard as `token_difr` instead of next to it.

  ARM stability  HOW MUCH DOES THE HONEST FLOOR MOVE ON ITS OWN? The same cells, a
                 different process, an hour later. Whatever this drift is, it is the
                 calibration tolerance a deployed floor test has to leave, and every
                 deviation smaller than it is undetectable by construction.

  ARM loadvary   (analysis, in plot_slope_verifier) PAIRED OR UNPAIRED PROBES? The
                 measured ITL(B) distributions say what fluctuating co-tenancy does to
                 an unpaired difference, and how much of it a paired probe removes.

    python -m experiments.exp_slope_verifier_gpu
Env: IVGYM_SLOPE_ARMS(all), IVGYM_SLOPE_STEPS(192), IVGYM_SLOPE_REPS(4)

Writes `docs/results/slope_verifier.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_clock_channel_gpu as C  # noqa: E402
from experiments.exp_clock_channel_gpu import (  # noqa: E402  (shared primitives)
    CTXS, RES, cell, load, log, measure_bandwidth,
)

STEPS = int(os.environ.get("IVGYM_SLOPE_STEPS", 192))
REPS = int(os.environ.get("IVGYM_SLOPE_REPS", 4))
ARMS = os.environ.get("IVGYM_SLOPE_ARMS", "all")
TAG = os.environ.get("IVGYM_SLOPE_TAG", "")
# `cell()` reads the sample counts off the module it lives in, so point them here.
# The detection arm needs a pool >= 10x its largest batch to stay inside the repo's
# batch/pool ceiling, which is why this file sets them at all.
C.STEPS, C.REPS = STEPS, REPS

# Never used to fit anything in CLOCK_MEASURED: three families the repo has not
# touched, one of them (Pythia) full multi-head attention rather than GQA.
ARCH2 = ["unsloth/Llama-3.2-1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
         "EleutherAI/pythia-1.4b", "allenai/OLMo-2-0425-1B", "Qwen/Qwen2.5-0.5B"]

PROBE_MODEL = "Qwen/Qwen3-1.7B"
CLAIMED_CTX = 32768                       # what the provider bills for
WINDOWS = [512, 2048, 8192, 16384, 32768]  # what it actually holds; last = honest
PROBE_LO = 256                            # the short probe of the pair


def main() -> None:
    t0 = time.time()
    arms = ARMS.split(",") if ARMS != "all" else ["arch2", "window", "stability"]
    log("=" * 96)
    log("The context-slope verifier: predictable floor? real detection? stable null?")
    log(f"  steps={STEPS} reps={REPS} arms={arms}")
    log("=" * 96)
    out = {"steps": STEPS, "reps": REPS, "claimed_ctx": CLAIMED_CTX,
           "probe_lo": PROBE_LO, "windows": WINDOWS, "cells": []}
    out["card"] = measure_bandwidth()
    log(f"  card: {out['card']['device']}  {out['card']['bw_copy_tb_s']:.2f} TB/s")

    # ---- ARM arch2: out-of-sample slope law
    if "arch2" in arms:
        log("\n  ARM arch2: five architectures that fitted nothing")
        for name in ARCH2:
            try:
                model, cfg, wb = load(name)
            except Exception as e:
                log(f"    {name} unavailable: {type(e).__name__}: {str(e)[:90]}")
                continue
            for ctx in CTXS:
                c = cell(model, cfg, wb, 1, ctx, "graph", name.split("/")[-1])
                if c:
                    c.update(model=name, quant="bf16", arm="arch2",
                             layers=cfg.num_hidden_layers,
                             kv_heads=getattr(cfg, "num_key_value_heads", None)
                             or cfg.num_attention_heads)
                    out["cells"].append(c)
            del model
            _drop(name)

    # ---- ARM window: a provider that claims CLAIMED_CTX and holds W
    if "window" in arms:
        log(f"\n  ARM window: claims {CLAIMED_CTX} tokens of context, holds W")
        model, cfg, wb = load(PROBE_MODEL)
        c = cell(model, cfg, wb, 1, PROBE_LO, "graph", "probe_lo")
        if c:
            c.update(model=PROBE_MODEL, arm="window", effective_ctx=PROBE_LO,
                     claimed_ctx=PROBE_LO, role="lo")
            out["cells"].append(c)
        for W in WINDOWS:
            c = cell(model, cfg, wb, 1, W, "graph",
                     "honest_hi" if W == CLAIMED_CTX else f"window_{W}")
            if c:
                c.update(model=PROBE_MODEL, arm="window", effective_ctx=W,
                         claimed_ctx=CLAIMED_CTX, role="hi")
                out["cells"].append(c)

    # ---- ARM stability: the honest null, re-measured in a fresh process
    if "stability" in arms:
        log("\n  ARM stability: the same honest cells, a different process")
        model, cfg, wb = load(PROBE_MODEL)
        for ctx in CTXS:
            c = cell(model, cfg, wb, 1, ctx, "graph", "Qwen3-1.7B-rerun")
            if c:
                c.update(model=PROBE_MODEL, quant="bf16", arm="stability")
                out["cells"].append(c)

    out["elapsed_s"] = time.time() - t0
    path = RES / f"slope_verifier{('_' + TAG) if TAG else ''}.json"
    path.write_text(json.dumps(out))
    log(f"\n  wrote {path}  ({out['elapsed_s']:.0f}s, {len(out['cells'])} cells)")


def _drop(name: str) -> None:
    """Free the weights of a model the grid is done with -- five extra checkpoints do
    not fit on this box alongside the four the main run already cached."""
    import gc
    import shutil
    import torch
    from experiments.exp_clock_channel_gpu import _LOADED
    _LOADED.pop(f"{name}|bf16", None)
    gc.collect()
    torch.cuda.empty_cache()
    if os.environ.get("IVGYM_SLOPE_KEEP", "0") == "1":
        return
    d = (Path.home() / ".cache" / "huggingface" / "hub"
         / ("models--" + name.replace("/", "--")))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
