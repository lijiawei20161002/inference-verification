"""The second mechanism: a speculative server's logits come from a different-shaped
forward pass, so they are different floats.

`exp_spec_decode_difr_gpu` measures the *randomness-path* break -- an honest
speculative server samples from `p` without using the verifier's Gumbel vector, so
the seeded replay disagrees. That break is bookkeeping: in principle a provider
could disclose its draft tokens and acceptance uniforms and a spec-aware verifier
could replay the decision (`exp_spec_aware_verifier_gpu` tries exactly that).

This experiment isolates a mechanism that no bookkeeping can fix. Reading the
target's logits at generated position `j` can be done three ways:

  prefill  one forward over `[prompt + claimed]`, read row `L-1+j`  <- THE VERIFIER
  decode   feed tokens one at a time against a KV cache            <- an ordinary server
  verify   feed `gamma` drafted tokens at once against a KV cache   <- a SPECULATIVE server

All three are mathematically the same number. None of them is the same float: they
are different reduction orders over the same arithmetic, dispatched to different
kernels because the sequence-length dimension differs (`1`, `gamma`, `L+n`).

The question this experiment answers is quantitative and it has two possible
answers, both worth having. If the speculative path's divergence is the same order
as the batch-composition nondeterminism DiFR already tolerates, then speculation
adds nothing numerically and the false positive is purely about randomness. If it
is larger, a spec-aware verifier inherits an irreducible disagreement floor.

Everything simulated is switched off: `verifier_sigma = act_benign_sigma = 0` and
the provider generates with `benign_sigma = 0`. Every number below is real
bf16-on-H100 arithmetic.

The comparison arms, all measured against the `prefill` reference at identical
positions on identical token prefixes:

  prefill_repeat   the same call again, same shapes -- the determinism control
  batch2 / batch8  the same request prefilled in a batch of 2 / 8 identical rows.
                   Pure batch-dimension change; this is the canonical benign
                   nondeterminism a verifier must tolerate, and the yardstick.
  decode           sequential single-token decoding
  verify_g4/g8     batched speculative verify at gamma = 4 / 8

Run:  .venv/bin/python -m experiments.exp_spec_batch_numerics_gpu
Env:  IVGYM_MODEL, IVGYM_DRAFT, IVGYM_PROMPTS, IVGYM_TOKENS, IVGYM_OUT
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks
from ivgym.backends.hf_gpu_spec import SpecDecodeHFBackend
from ivgym.core import SamplingSpec
from ivgym.sampling import filtered_logits, gumbel_noise, position_seed

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-1.7B")
DRAFT = os.environ.get("IVGYM_DRAFT", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 24))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 64))
OUT = Path(os.environ.get(
    "IVGYM_OUT", Path(__file__).resolve().parents[1] / "docs/results/spec_batch_numerics.json"))


def _prefill(be, ids, n, batch: int = 1):
    """Rows `L-1 .. L-1+n-1` of one forward over the whole sequence, optionally
    replicated into a batch of `batch` identical rows. Replication changes only the
    batch dimension -- no padding, no mask, nothing else to confound it."""
    torch = be._torch
    L = ids.shape[1] - n
    x = ids.expand(batch, -1) if batch > 1 else ids
    with torch.no_grad():
        out = be.model(x)
    return out.logits[0, L - 1: L - 1 + n].float().cpu().numpy()


def _chunked(be, prompt_ids, claimed, chunk: int):
    """Rows produced by feeding `claimed` against a KV cache in chunks of `chunk`.

    `chunk=1` is ordinary sequential decoding. `chunk=gamma` is exactly the
    speculative verify pass: `gamma` tokens scored in one forward. Row `j` of the
    result is the target's prediction for generated position `j`, the same quantity
    `_prefill` returns -- reached by a different sequence of kernel launches.
    """
    torch = be._torch
    rows = []
    with torch.no_grad():
        out = be.model(prompt_ids, use_cache=True)
        past = out.past_key_values
        rows.append(out.logits[0, -1].float().cpu().numpy())     # position 0
        i = 0
        while len(rows) < len(claimed):
            step = claimed[i: i + chunk]
            t = torch.tensor([step], device=be.device, dtype=torch.long)
            out = be.model(t, past_key_values=past, use_cache=True)
            past = out.past_key_values
            for j in range(len(step)):
                if len(rows) < len(claimed):
                    rows.append(out.logits[0, j].float().cpu().numpy())
            i += chunk
    return np.stack(rows)


def _sampled(logits: np.ndarray, spec: SamplingSpec, prompt_id: int) -> np.ndarray:
    """The token seeded Gumbel-Max would draw at each position from these logits.
    Comparing this between two forward paths is exactly the disagreement a
    seed-replaying verifier reports."""
    out = np.empty(len(logits), int)
    for j, row in enumerate(logits):
        g = gumbel_noise(len(row), position_seed(spec.seed, prompt_id, j))
        filt = filtered_logits(row, spec.top_k, spec.top_p)
        out[j] = int(np.argmax(filt + spec.temperature * g))
    return out


def _margin(ref: np.ndarray, claimed: np.ndarray, spec: SamplingSpec,
            prompt_id: int) -> np.ndarray:
    """`token_difr`'s per-token score for `claimed` against reference logits `ref`
    -- the post-Gumbel margin, computed exactly as `verifiers.TokenDiFR` does."""
    out = np.empty(len(ref))
    for j, row in enumerate(ref):
        g = gumbel_noise(len(row), position_seed(spec.seed, prompt_id, j))
        filt = filtered_logits(row, spec.top_k, spec.top_p)
        z = filt + spec.temperature * g
        out[j] = (30.0 if filt[claimed[j]] <= -1e29
                  else min(float(z[int(np.argmax(z))] - z[claimed[j]]), 30.0))
    return out


def main():
    t0 = time.time()
    print(f"loading target={MODEL} draft={DRAFT} (all simulated noise OFF) ...", flush=True)
    be = SpecDecodeHFBackend(model_name=MODEL, draft_model_name=DRAFT,
                             verifier_sigma=0.0, act_benign_sigma=0.0)
    print(f"loaded in {time.time()-t0:.1f}s | {N_PROMPTS} prompts x {N_TOKENS} tokens",
          flush=True)
    spec = SamplingSpec()
    torch = be._torch
    # A provider with zero simulated benign noise: any divergence below is real.
    honest = attacks.Attack(name="honest", benign_sigma=0.0)

    arms = ["prefill_repeat", "batch2", "batch8", "decode", "verify_g4", "verify_g8"]
    acc = {a: {"absdiff": [], "maxdiff": [], "argmax_flip": [], "token_flip": [],
               "margin": []} for a in arms}
    n_pos = 0

    for p in range(N_PROMPTS):
        seq = be.generate(p, N_TOKENS, spec, honest, False)
        claimed = [st.claimed_token for st in seq.steps]
        n = len(claimed)
        n_pos += n
        prompt_ids = be._prompt_ids(p)
        full = torch.cat([prompt_ids,
                          torch.tensor([claimed], device=be.device,
                                       dtype=prompt_ids.dtype)], dim=1)

        ref = _prefill(be, full, n)                     # what the verifier computes
        ref_tok = _sampled(ref, spec, p)
        variants = {
            "prefill_repeat": _prefill(be, full, n),
            "batch2": _prefill(be, full, n, batch=2),
            "batch8": _prefill(be, full, n, batch=8),
            "decode": _chunked(be, prompt_ids, claimed, 1),
            "verify_g4": _chunked(be, prompt_ids, claimed, 4),
            "verify_g8": _chunked(be, prompt_ids, claimed, 8),
        }
        for name, alt in variants.items():
            d = np.abs(alt - ref)
            acc[name]["absdiff"].append(d.mean(axis=1))
            acc[name]["maxdiff"].append(d.max(axis=1))
            acc[name]["argmax_flip"].append(
                (alt.argmax(axis=1) != ref.argmax(axis=1)).astype(float))
            # A server that produced `alt` emits these tokens; the verifier replays
            # against `ref`. The flip rate is the irreducible disagreement the
            # forward-pass SHAPE alone creates, with randomness held fixed.
            alt_tok = _sampled(alt, spec, p)
            acc[name]["token_flip"].append((alt_tok != ref_tok).astype(float))
            acc[name]["margin"].append(_margin(ref, alt_tok, spec, p))
        if (p + 1) % 6 == 0:
            print(f"  {p+1}/{N_PROMPTS} prompts ({n_pos} positions) "
                  f"{time.time()-t0:.0f}s", flush=True)

    per_pos = {a: {k: np.concatenate(v) for k, v in acc[a].items()} for a in arms}
    stats = {}
    for a in arms:
        m = per_pos[a]
        stats[a] = {
            "mean_abs_logit_diff": float(m["absdiff"].mean()),
            "max_abs_logit_diff": float(m["maxdiff"].max()),
            "p99_max_logit_diff": float(np.percentile(m["maxdiff"], 99)),
            "argmax_flip_rate": float(m["argmax_flip"].mean()),
            "token_flip_rate": float(m["token_flip"].mean()),
            "mean_margin": float(m["margin"].mean()),
            "bit_identical": bool(m["maxdiff"].max() == 0.0),
        }

    print("\n" + "=" * 96)
    print(f"Forward-pass shape vs the verifier's prefill, same prefixes, "
          f"{n_pos} positions, {MODEL} bf16")
    print("=" * 96)
    hdr = (f"{'path':>15} | {'mean|dlogit|':>12} {'max|dlogit|':>12} "
           f"{'argmax flip':>12} {'sampled-token flip':>19} {'mean margin':>12}")
    print(hdr)
    print("-" * len(hdr))
    for a in arms:
        s = stats[a]
        print(f"{a:>15} | {s['mean_abs_logit_diff']:>12.3e} {s['max_abs_logit_diff']:>12.3e} "
              f"{s['argmax_flip_rate']:>12.2%} {s['token_flip_rate']:>19.2%} "
              f"{s['mean_margin']:>12.4f}")

    ctl = stats["prefill_repeat"]
    print(f"\n  Control: the same call twice is "
          f"{'bit-identical' if ctl['bit_identical'] else 'NOT bit-identical'} "
          f"(max|dlogit| = {ctl['max_abs_logit_diff']:.3e}). Every other row is "
          f"therefore\n  attributable to the shape of the forward pass, not to "
          f"run-to-run jitter.")
    bench = max(stats["batch2"]["token_flip_rate"], stats["batch8"]["token_flip_rate"])
    for g in ("verify_g4", "verify_g8"):
        r = stats[g]["token_flip_rate"]
        rel = (r / bench) if bench > 0 else float("inf")
        print(f"  {g}: the shape alone flips the sampled token at {r:.2%}, "
              f"{'vs' if bench else 'against'} {bench:.2%} for the benign "
              f"batch-composition\n    yardstick"
              + (f" -- {rel:.1f}x." if bench > 0 else " -- which is exactly 0."))

    # The comparison the framing of this experiment could easily miss: `decode` is
    # what an ORDINARY, non-speculative server runs. If speculative verify is no
    # worse than that, the numerical floor is not speculation's -- it belongs to
    # every server that generates incrementally while the verifier prefills, and
    # attributing it to speculation would be wrong.
    dec = stats["decode"]["token_flip_rate"]
    spec_worst = max(stats["verify_g4"]["token_flip_rate"],
                     stats["verify_g8"]["token_flip_rate"])
    print(f"\n  Against the right control: an ORDINARY sequential server (`decode`) "
          f"already flips\n  {dec:.2%} of sampled tokens against the verifier's prefill; "
          f"the speculative verify pass flips\n  {spec_worst:.2%}. ", end="")
    if spec_worst <= 1.25 * dec:
        print(f"Speculation therefore adds no measurable numerical penalty of its\n"
              f"  own ({spec_worst/dec:.2f}x decode) -- the floor is the "
              f"PREFILL-vs-INCREMENTAL mismatch that every\n"
              f"  serving stack has, and a spec-aware verifier inherits it rather than "
              f"creating it.\n"
              f"  It is {spec_worst/bench:.1f}x the benign batch-composition noise a "
              f"verifier already tolerates,\n  which is the size of the problem: real, "
              f"bounded, and not fixable by disclosure.")
    else:
        print(f"Speculation adds {spec_worst/dec:.2f}x on top of ordinary\n"
              f"  decoding, so a spec-aware verifier inherits a floor strictly worse "
              f"than the one\n  plain recomputation already lives with.")

    # ---------------------------------------- the divergence DISTRIBUTION
    # The scalars above answer "how big on average". What a verifier is actually
    # exposed to is the tail: a threshold is crossed by the worst positions, not by
    # the mean one. Keep enough of the shape to plot without re-running -- a shared
    # log-spaced histogram, a quantile grid, and the sampled-token flip rate as a
    # function of divergence, which is the mechanism linking the two.
    QS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
    EDGES = np.concatenate([[0.0], np.logspace(-4, 1, 51)])
    dist = {}
    for a in arms:
        m = per_pos[a]
        counts, _ = np.histogram(m["absdiff"], bins=EDGES)
        # flip rate per divergence bucket, on the SAME edges, so the two align.
        # Buckets with too few positions to estimate a rate are dropped rather
        # than plotted as noise.
        idx = np.digitize(m["absdiff"], EDGES) - 1
        flip = {}
        for b in range(len(EDGES) - 1):
            sel = idx == b
            if int(sel.sum()) >= 30:
                flip[str(b)] = [float(m["token_flip"][sel].mean()), int(sel.sum())]
        dist[a] = {
            "absdiff_q": [float(np.percentile(m["absdiff"], q)) for q in QS],
            "maxdiff_q": [float(np.percentile(m["maxdiff"], q)) for q in QS],
            "margin_q": [float(np.percentile(m["margin"], q)) for q in QS],
            "absdiff_hist": [int(c) for c in counts],
            "flip_by_absdiff": flip,
        }

    print("\n" + "=" * 96)
    print("The same divergence as a distribution: per-position mean |dlogit| vs "
          "the verifier's prefill")
    print("=" * 96)
    hdr2 = (f"{'path':>15} | {'p50':>10} {'p90':>10} {'p99':>10} {'p99.9':>10} "
            f"{'max':>10} | {'p99 of max|dlogit|':>19}")
    print(hdr2)
    print("-" * len(hdr2))
    qi = {q: i for i, q in enumerate(QS)}
    for a in arms:
        d = dist[a]
        print(f"{a:>15} | " + " ".join(f"{d['absdiff_q'][qi[q]]:>10.3e}"
                                      for q in (50, 90, 99, 99.9, 100))
              + f" | {d['maxdiff_q'][qi[99]]:>19.3e}")
    print("\n  The mean is not the exposure. Compare p50 to p99.9 within a row: the\n"
          "  divergence is concentrated in a tail, which is why a small mean still\n"
          "  flips whole tokens.")

    art = {"model": MODEL, "draft": DRAFT, "n_prompts": N_PROMPTS,
           "n_tokens": N_TOKENS, "n_positions": n_pos, "stats": stats,
           "dist": {"quantiles": QS, "edges": [float(e) for e in EDGES],
                    "by_path": dist},
           "seconds": time.time() - t0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
