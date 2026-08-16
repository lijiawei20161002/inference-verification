"""Is the price of a token a constant? -- the audit-batch sweep behind c(tier).

`exp_cost_of_a_verdict_gpu` prices a verdict as `b*(d') x c(tier)` and measures
`c(tier)` the way a verifier auditing one sequence at a time pays it: one
GPU-synchronised prefill per [prompt + claimed] row, batch 1. At that batch the
1.7B reference and the 0.6B proxy price within 1.05x of each other even though the
proxy is 3.2x fewer FLOPs per token -- which is the poster's headline, and is also
exactly the regime where a reader should not believe a price. A single 256-token
prefill on an H100 is memory-bandwidth bound: it reads every weight to do a few
hundred token-rows of work, so the price tracks PARAMETER BYTES, not FLOPs, and
the cheap tier's arithmetic advantage is invisible.

A real auditor batches. This sweep measures the same two prices across audit
batch B (rows verified per forward pass), so `c(tier)` becomes a curve and the
claim "the cheap tier buys 1.05x" gets its regime attached:

    c_ref(B), c_proxy(B)   GPU-synchronised seconds / (B x T) tokens
    ratio(B) = c_ref / c_proxy   ->  measured tier gap, vs the 3.2x FLOP bound

The verdict-cost consequence is the point, not the throughput: a tier's price is
only worth paying for if `b*` does not eat it. The sweep is reported next to the
measured token penalty of the proxy tier from `cost_of_a_verdict.json`, so the
break-even batch (if any) is a number and not a hope.

Protocol, matched to `measure_prices` so the two artifacts are comparable:
bfloat16, `attn_implementation="eager"`, real tokenized prompt+continuation rows
of exactly T tokens, full logits kept (a token-level detector scores every
position, so `num_logits_to_keep` would price a detector nobody deployed),
warmup passes discarded, `torch.cuda.synchronize()` on both sides of the timed
region, R repeats reported as median [min, max].

    python -m experiments.exp_verdict_price_batch_gpu

Env: IVGYM_M(Qwen/Qwen3-1.7B), IVGYM_PROXY(Qwen/Qwen3-0.6B), IVGYM_TOKENS(256),
     IVGYM_BATCHES(1,2,4,8,16,32,64), IVGYM_REPS(9), IVGYM_WARMUP(3),
     IVGYM_GPU_USD_PER_HR(2.50).

Writes `docs/results/verdict_price_batch.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
T = int(os.environ.get("IVGYM_TOKENS", 256))
BATCHES = [int(b) for b in os.environ.get("IVGYM_BATCHES", "1,2,4,8,16,32,64").split(",")]
REPS = int(os.environ.get("IVGYM_REPS", 9))
WARMUP = int(os.environ.get("IVGYM_WARMUP", 3))
USD_PER_HR = float(os.environ.get("IVGYM_GPU_USD_PER_HR", 2.50))


def rows_of_length(tok, prompts: list[str], n: int, t: int, seed: int = 0):
    """`n` token rows of EXACTLY `t` ids, built from the same prompt pool the
    detectors were scored on.

    A prompt is ~10 ids and a scored row is [prompt + 256 claimed], so the tail
    has to come from somewhere. It comes from the pool itself, cycled -- the ids
    are real text, the shape is the shape a verifier prefills, and timing does
    not depend on which token sits in a slot.
    """
    ids = [i for p in prompts for i in tok(p).input_ids]
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        start = int(rng.integers(0, len(ids)))
        row = [ids[(start + j) % len(ids)] for j in range(t)]
        out.append(row)
    return out


def time_prefill(torch, model, batch_ids, reps: int, warmup: int) -> list[float]:
    """GPU-synchronised seconds for one full-logit prefill of `batch_ids`."""
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=batch_ids)
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=batch_ids)
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
    return out


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ivgym.backends.hf_gpu import DEFAULT_PROMPTS

    t_start = time.time()
    print("=" * 84)
    print(f"Price of a token vs audit batch   M={M}  proxy={PROXY}  T={T}")
    print(f"  {REPS} timed repeats after {WARMUP} warmups, full logits kept, eager attn")
    print("=" * 84, flush=True)

    tok = AutoTokenizer.from_pretrained(M)
    models = {}
    for tier, name in (("reference", M), ("proxy", PROXY)):
        m = (AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                  attn_implementation="eager")
             .to("cuda").eval())
        n_embed = int(m.config.vocab_size) * int(m.config.hidden_size)
        n_params = sum(p.numel() for p in m.parameters())
        models[tier] = {"name": name, "model": m, "n_params": n_params,
                        "n_nonembed": max(n_params - n_embed, 0)}
        print(f"  {tier:>9} {name:<20} {n_params/1e9:.3f}B params "
              f"({models[tier]['n_nonembed']/1e9:.3f}B non-embedding)", flush=True)

    flop_ratio = models["reference"]["n_nonembed"] / models["proxy"]["n_nonembed"]
    byte_ratio = models["reference"]["n_params"] / models["proxy"]["n_params"]
    print(f"\n  FLOP bound on the tier gap (non-embedding params): {flop_ratio:.2f}x")
    print(f"  bandwidth bound (all params, batch-1 weight read):  {byte_ratio:.2f}x",
          flush=True)

    rows = {}
    print(f"\n{'B':>5} {'tokens':>8} | {'ref ms':>9} {'us/tok':>8} | "
          f"{'proxy ms':>9} {'us/tok':>8} | {'gap':>6} {'ref Mtok/s':>11}")
    for B in BATCHES:
        ids = torch.tensor(rows_of_length(tok, DEFAULT_PROMPTS, B, T),
                           device="cuda", dtype=torch.long)
        rec = {"batch": B, "tokens": B * T}
        try:
            for tier in ("reference", "proxy"):
                s = time_prefill(torch, models[tier]["model"], ids, REPS, WARMUP)
                rec[tier] = {
                    "sec_per_pass": float(np.median(s)),
                    "sec_per_pass_min": float(np.min(s)),
                    "sec_per_pass_max": float(np.max(s)),
                    "sec_per_token": float(np.median(s)) / (B * T),
                    "usd_per_1m_tokens": float(np.median(s)) / (B * T) * 1e6 / 3600 * USD_PER_HR,
                }
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>5} out of memory -- an audit this wide does not fit; stopping")
            torch.cuda.empty_cache()
            break
        rec["gap"] = rec["reference"]["sec_per_token"] / rec["proxy"]["sec_per_token"]
        rec["gap_frac_of_flop_bound"] = rec["gap"] / flop_ratio
        rows[B] = rec
        print(f"{B:>5} {B*T:>8} | {rec['reference']['sec_per_pass']*1e3:>9.2f} "
              f"{rec['reference']['sec_per_token']*1e6:>8.2f} | "
              f"{rec['proxy']['sec_per_pass']*1e3:>9.2f} "
              f"{rec['proxy']['sec_per_token']*1e6:>8.2f} | {rec['gap']:>5.2f}x "
              f"{B*T/rec['reference']['sec_per_pass']/1e6:>11.3f}", flush=True)
        del ids
        torch.cuda.empty_cache()

    # ---- what the sweep does to a verdict -----------------------------------
    # The proxy tier is only worth its token penalty if the price gap covers it.
    # Both numbers are measured: the penalty from cost_of_a_verdict.json (same
    # model pair, same protocol), the gap here.
    cv_path = RES / "cost_of_a_verdict.json"
    verdict = None
    if cv_path.exists():
        CV = json.loads(cv_path.read_text())
        both = [(a, CV["cells"][a]["accept_rate"], CV["cells"][a]["token_difr"])
                for a in CV["cells"]
                if CV["cells"][a]["accept_rate"]["reachable"]
                and CV["cells"][a]["token_difr"]["reachable"]]
        if both:
            a, pr, ref = min(both, key=lambda t: t[1]["tokens_per_verdict"]
                             / t[2]["tokens_per_verdict"])
            penalty = pr["tokens_per_verdict"] / ref["tokens_per_verdict"]
            verdict = {
                "attack": a,
                "proxy_token_penalty": penalty,
                "note": "smallest token penalty among cells both tiers can price",
                "flop_bound_covers_it": bool(flop_ratio >= penalty),
                "best_measured_gap": max((r["gap"] for r in rows.values()), default=None),
            }
            print(f"\n  cheapest proxy cell is {a}: the proxy needs {penalty:.1f}x more "
                  f"tokens.")
            print(f"  best measured price gap over the sweep is "
                  f"{verdict['best_measured_gap']:.2f}x, and even the FLOP bound "
                  f"{flop_ratio:.2f}x does not reach {penalty:.1f}x --> the proxy tier "
                  f"is more expensive per VERDICT at every batch in this sweep."
                  if not verdict["flop_bound_covers_it"] else
                  f"  the FLOP bound {flop_ratio:.2f}x does cover the {penalty:.1f}x "
                  f"penalty; break-even is inside this sweep.")

    out = {
        "M": M, "proxy": PROXY, "tokens_per_row": T, "reps": REPS, "warmup": WARMUP,
        "dtype": "bfloat16", "attn": "eager", "keep_full_logits": True,
        "usd_per_gpu_hour": USD_PER_HR,
        "m_params": models["reference"]["n_params"],
        "proxy_params": models["proxy"]["n_params"],
        "m_nonembed": models["reference"]["n_nonembed"],
        "proxy_nonembed": models["proxy"]["n_nonembed"],
        "flop_ratio_bound": flop_ratio,
        "param_byte_ratio": byte_ratio,
        "batches": sorted(rows),
        "rows": {str(k): v for k, v in rows.items()},
        "verdict_consequence": verdict,
        "elapsed_s": time.time() - t_start,
    }
    (RES / "verdict_price_batch.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RES / 'verdict_price_batch.json'}  [{time.time()-t_start:.0f}s]")


if __name__ == "__main__":
    main()
