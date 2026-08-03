"""Can a verifier be made spec-aware? A ladder of four, and what each one buys.

`exp_spec_decode_difr_gpu` establishes the problem: `token_difr` flags an honest
speculative server at a rate that makes it unusable. The obvious response is to
teach the verifier about speculative decoding. This experiment builds that
verifier and prices it.

The ladder, each rung a strictly stronger disclosure requirement on the provider:

  V0  token_difr          replay the Gumbel draw. Needs nothing from the provider
                          beyond the public seed -- and false-positives on honest
                          speculation. The baseline being replaced.

  V1  spec_accept         Verifier reruns the ATTESTED draft model and asks a
                          seed-free question: how hard would it have been to accept
                          the claimed token, `-log min(1, p(t)/q(t))`? One prefill of
                          each model, no round structure, no randomness. Cannot
                          false-positive on honest speculation by construction.
                          The question is whether it can see anything.

  V1b accept_test         The same, as the protocol's own binary test: would the
                          claimed token have passed the acceptance test at the
                          seeded uniform? Reported as a pass rate, because a test
                          that honest and cheating traffic both pass is not a test.

  V2  spec_replay         Verifier re-runs the whole speculative loop -- draft
                          rollout, batched target verify, accept/reject at the
                          PUBLIC per-position uniforms -- and checks it reproduces
                          the claimed tokens. Sound. Requires the provider to (a)
                          derive all its randomness from the public seed, (b) attest
                          the draft model, (c) disclose gamma. And it is not a
                          prefill: it is a re-generation.

  V2' spec_replay vs an ordinary speculative server (`honest_spec`, own RNG) --
      the same verifier against a server that did NOT adopt (a). If this
      false-positives too, then V2's soundness is bought entirely with a change to
      the SERVER, not with cleverness in the verifier.

  V3  trust_declared      The cheap shortcut: skip rerunning the draft and believe
                          the provider's assertion about what it proposed. Reported
                          for completeness -- it certifies everything, because a
                          provider may always assert `draft = claimed` with a draft
                          probability low enough to force acceptance.

Arms: honest (non-speculative null), honest_spec_seeded (the honest server V2 is
built for), honest_spec (honest, own RNG), seed_43 (the output-distribution-
preserving cheat only a seeded replay can see), spec_lenient (a speculative cheat).

Run:  .venv/bin/python -m experiments.exp_spec_aware_verifier_gpu
Env:  IVGYM_MODEL, IVGYM_DRAFT, IVGYM_PROMPTS, IVGYM_TOKENS, IVGYM_GAMMA,
      IVGYM_SEEDS, IVGYM_OUT
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, signal, verifiers
from ivgym.backends.hf_gpu_spec import SpecDecodeHFBackend
from ivgym.core import SamplingSpec
from ivgym.sampling import position_seed
from ivgym.spec_server import position_uniforms, spec_probs

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-1.7B")
DRAFT = os.environ.get("IVGYM_DRAFT", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 24))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 64))
GAMMA = int(os.environ.get("IVGYM_GAMMA", 4))
N_SEEDS = int(os.environ.get("IVGYM_SEEDS", 5))
OUT = Path(os.environ.get(
    "IVGYM_OUT", Path(__file__).resolve().parents[1] / "docs/results/spec_aware_verifier.json"))

ARMS = ["honest_spec_seeded", "honest_spec", "seed_43", "spec_lenient"]
HONEST_ARMS = {"honest_spec_seeded", "honest_spec"}
SCORES = ["token_difr", "spec_accept", "spec_replay"]
CAP = 30.0                      # same clip token_difr uses for an impossible token


class _Named(verifiers.Verifier):
    """A name for `harness.evaluate` to key on. The scores are computed here rather
    than through the `Verifier` interface because V1/V2 need two models and, for V2,
    the sequential round structure -- neither fits the per-token `VContext` contract,
    and pretending otherwise would hide the cost that is half the result."""

    def __init__(self, name):
        self.name = name


def spec_accept_evidence(backend, seq, spec) -> tuple[np.ndarray, np.ndarray]:
    """V1 / V1b for one sequence.

    Returns `(difficulty, passed)`: the nats by which the claimed token falls short
    of certain acceptance, `-log min(1, p(t)/q(t))`, and whether it would have
    passed the acceptance test at the seeded uniform. `p` is the verifier's
    recompute of `M`; `q` is a prefill of the attested draft model.
    """
    steps = seq.steps
    claimed = [st.claimed_token for st in steps]
    ref = np.stack([backend.reference_logits(seq.prompt_id, st.position) for st in steps])
    dq = backend.draft_prefill(seq.prompt_id, claimed)
    diff = np.empty(len(steps))
    passed = np.empty(len(steps))
    for j, t in enumerate(claimed):
        p = spec_probs(ref[j], spec)
        q = spec_probs(dq[j], spec)
        if p[t] <= 0.0:                       # outside M's own filtered support
            diff[j], passed[j] = CAP, 0.0
            continue
        if q[t] <= 0.0:                       # draft could not propose it; the
            diff[j], passed[j] = 0.0, 1.0     # residual path takes it unconditionally
            continue
        r = min(1.0, float(p[t] / q[t]))
        diff[j] = min(-np.log(max(r, 1e-13)), CAP)
        u = position_uniforms(position_seed(spec.seed, seq.prompt_id, j), 3)[1]
        passed[j] = float(u <= r)
    return diff, passed


def _timed(backend, fn):
    """GPU-synchronised wall clock for one audit, so the cost column is device time
    and not kernel-launch latency."""
    torch = backend._torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    t0 = time.time()
    print(f"loading target={MODEL} draft={DRAFT} gamma={GAMMA} ...", flush=True)
    be = SpecDecodeHFBackend(model_name=MODEL, draft_model_name=DRAFT, gamma=GAMMA)
    print(f"loaded in {time.time()-t0:.1f}s | {N_PROMPTS} prompts x {N_TOKENS} tokens "
          f"per arm", flush=True)
    spec = SamplingSpec()
    difr = [verifiers.get("token_difr")]
    stubs = [_Named(s) for s in SCORES]
    cfg = harness.EvalConfig()
    pool = N_PROMPTS * N_TOKENS
    eval_split = int(pool * (1.0 - cfg.calib_frac))
    batch = max(1, int(eval_split * cfg.max_pool_ratio))

    cost = {"prefill_difr": [], "spec_replay": []}

    def score_arm(name):
        """Generate one arm and score it with V0, V1, V1b and V2."""
        atk = attacks.get(name)
        seqs = [be.generate(p, N_TOKENS, spec, atk, True) for p in range(N_PROMPTS)]
        ts = harness.verify(be, seqs, spec, difr)
        acc, pas, rep = [], [], []
        for seq in seqs:
            a, p_ = spec_accept_evidence(be, seq, spec)
            acc.append(a)
            pas.append(p_)
            claimed = [st.claimed_token for st in seq.steps]
            r, dt = _timed(be, lambda: be.spec_replay(seq.prompt_id, claimed, spec, GAMMA))
            rep.append(r)
            cost["spec_replay"].append(dt)
            # What V0 pays on the same sequence: one prefill over [prompt+claimed].
            _, dt2 = _timed(be, lambda: be._populate_ref_cache(
                seq.prompt_id, be._prompt_ids(seq.prompt_id), claimed, len(claimed)))
            cost["prefill_difr"].append(dt2)
        return harness.TokenScores(name, {
            "token_difr": ts.scores["token_difr"],
            "spec_accept": np.concatenate(acc),
            "spec_replay": np.concatenate(rep),
        }), np.concatenate(pas), (be.spec_summary() if be.spec_stats else {})

    print("\ngenerating + auditing honest (non-speculative) null ...", flush=True)
    be.spec_stats.clear()
    honest, honest_pass, _ = score_arm("honest")
    print(f"  {len(honest.scores['token_difr'])} tokens, {time.time()-t0:.0f}s", flush=True)

    scored, passrate, runs = {}, {"honest": float(honest_pass.mean())}, {}
    for name in ARMS:
        t = time.time()
        be.spec_stats.clear()
        scored[name], pas, run = score_arm(name)
        passrate[name] = float(pas.mean())
        if run:
            runs[name] = run
        print(f"  {name:19s} {len(scored[name].scores['token_difr']):5d} tokens "
              f"in {time.time()-t:5.1f}s", flush=True)

    # -------------------------------------------------------- 1. the two replays
    # A replay verifier reproduces the server whose ALGORITHM it implements, and
    # nothing else. Stating that first, as raw disagreement rates, makes the rest of
    # the table readable -- and it is symmetric in a way that matters: V0 cannot
    # reproduce a speculative server and V2 cannot reproduce a sequential one.
    allts = {"honest": honest, **scored}
    print("\n" + "=" * 92)
    print("1. Share of a server's tokens each replay verifier CANNOT reproduce")
    print("=" * 92)
    print(f"{'server':>19} | {'V0 Gumbel rerun':>17} {'V2 spec replay':>16}")
    print("-" * 58)
    dis, dis0 = {}, {}
    for name, ts in allts.items():
        dis0[name] = float(np.mean(ts.scores["token_difr"] > 0))
        dis[name] = float(np.mean(ts.scores["spec_replay"] > 0))
        print(f"{name:>19} | {dis0[name]:>17.1%} {dis[name]:>16.1%}")
    print("\n  Read the diagonal. V0 reproduces the sequential server and fails on the\n"
          "  speculative one; V2 reproduces the seeded speculative server and fails on\n"
          "  the sequential one. Each verifier is sound only against the decoding\n"
          "  algorithm it assumes -- and which algorithm a provider runs is exactly what\n"
          "  a client cannot check.")

    # ---------------------------------------- 2. AUC, each verifier against ITS null
    # V0/V1 are calibrated on the sequential honest server; V2 on the seeded
    # speculative one. Scoring V2 against the sequential null would report its own
    # mismatched assumption as a detection, which is the error the block above rules
    # out.
    NULL_OF = {"token_difr": "honest", "spec_accept": "honest",
               "spec_replay": "honest_spec_seeded"}
    print("\n" + "=" * 92)
    print(f"2. Detection AUC @ FPR<=0.5%, each verifier calibrated on ITS OWN honest\n"
          f"   null   (batch={batch}, eval pool={eval_split}, ratio={batch/eval_split:.1%}, "
          f"{N_SEEDS} protocol seeds)")
    print("=" * 92)
    print(f"   V0/V1 null = honest (sequential)   |   V2 null = honest_spec_seeded")
    hdr = (f"{'server':>19} | {'V0 token_difr':>16} {'V1 spec_accept':>16} "
           f"{'V2 spec_replay':>16}")
    print(hdr)
    print("-" * len(hdr))
    auc, tpr, dpr = {}, {}, {}
    for name in ["honest"] + ARMS:
        auc[name], tpr[name], dpr[name] = {}, {}, {}
        cells = []
        for s in SCORES:
            null = NULL_OF[s]
            if name == null:
                cells.append(f"{'-- (the null)':>16}")
                continue
            rows = [harness.evaluate(allts[null], allts[name], stubs, [batch],
                                     config=replace(cfg, seed=sd)) for sd in range(N_SEEDS)]
            a = np.array([[r.auc for r in row if r.defense == s][0] for row in rows])
            t_ = np.array([[r.tpr for r in row if r.defense == s][0] for row in rows])
            auc[name][s] = (float(a.mean()), float(a.std()))
            tpr[name][s] = (float(t_.mean()), float(t_.std()))
            dpr[name][s] = float(signal.per_token_stats(
                allts[null].scores[s], allts[name].scores[s], cfg.winsor_pct)["d_prime"])
            cells.append(f"{a.mean():>10.3f}+-{a.std():.3f}")
        print(f"{name:>19} | " + " ".join(cells))

    print("\nFlag rate at a threshold calibrated to 0.5% FPR on that verifier's own null")
    print(hdr)
    print("-" * len(hdr))
    for name in ["honest"] + ARMS:
        cells = [(f"{'-- (the null)':>16}" if s not in tpr[name]
                  else f"{tpr[name][s][0]:>16.1%}") for s in SCORES]
        tag = ("  <- HONEST server" if name in HONEST_ARMS or name == "honest" else "")
        print(f"{name:>19} | " + " ".join(cells) + tag)
    print(f"\n  V2's honest floor is {dis['honest_spec_seeded']:.1%} disagreement, against "
          f"{dis0['honest']:.1%} for V0 on its own\n  null -- the replay has to rerun the "
          f"draft rollout and the batched verify pass\n  itself, and its forward passes do "
          f"not have the shapes the server's did, so a\n  share of acceptance decisions "
          f"flip on numerics alone. The fix is sound, but it\n  is noisier than the "
          f"detector it replaces.")

    print("\n" + "=" * 92)
    print("3. V1b: share of claimed tokens that PASS the acceptance test")
    print("   (attested draft model, uniform from the public position seed)")
    print("=" * 92)
    print(f"{'server':>19} | {'pass rate':>10}")
    print("-" * 33)
    for name in ["honest"] + ARMS:
        print(f"{name:>19} | {passrate[name]:>10.1%}")
    print(f"\n  The acceptance test passes {passrate['seed_43']:.1%} of the wrong-seed "
          f"ATTACK's tokens and\n  {passrate['honest']:.1%} of honest ones. It is a "
          f"DISTRIBUTIONAL check: it asks whether the\n  claimed token was a plausible "
          f"draw, which is the one thing an output-preserving\n  deviation preserves. The "
          f"cheap spec-aware rung buys nothing against exactly the\n  attack that motivated "
          f"seeded verification in the first place.")
    print("\n  V3 trust_declared: certifies 100% of every server's tokens, without a\n"
          "  forward pass. A provider asserting `draft = claimed` may name any draft\n"
          "  probability q_hat <= p(t)/u, so acceptance is forced for every token with\n"
          "  p(t) > 0. Not measured because it is arithmetic: the disclosure has to be\n"
          "  RECOMPUTED, not believed.")

    # ------------------------------------------------------------------ the cost
    pf = float(np.mean(cost["prefill_difr"]))
    rp = float(np.mean(cost["spec_replay"]))
    print("\n" + "=" * 92)
    print("What the sound rung costs, per audited sequence (GPU-synchronised)")
    print("=" * 92)
    print(f"  V0 token_difr   one prefill over [prompt+claimed] : {pf*1000:8.1f} ms")
    print(f"  V2 spec_replay  re-run the speculative loop       : {rp*1000:8.1f} ms"
          f"   ({rp/pf:.1f}x)")
    print(f"\n  V2 is not a prefill. It replays ~{N_TOKENS/GAMMA:.0f} sequential rounds, each a\n"
          f"  {GAMMA}-step draft rollout plus a batched target pass, because the round\n"
          f"  structure is only knowable by walking it. The sound spec-aware audit\n"
          f"  therefore costs more than the generation it is checking -- where plain\n"
          f"  recomputation costs one prefill.")

    art = {"model": MODEL, "draft": DRAFT, "gamma": GAMMA, "n_prompts": N_PROMPTS,
           "n_tokens": N_TOKENS, "n_seeds": N_SEEDS, "batch": batch,
           "eval_pool": eval_split, "pool_ratio": batch / eval_split,
           "scores": SCORES, "spec_runs": runs,
           "auc": {a: {s: list(v) for s, v in m.items()} for a, m in auc.items()},
           "tpr": {a: {s: list(v) for s, v in m.items()} for a, m in tpr.items()},
           "d_prime": dpr, "replay_disagreement": dis, "accept_pass_rate": passrate,
           "cost_seconds": {"prefill_difr": pf, "spec_replay": rp, "ratio": rp / pf},
           "seconds": time.time() - t0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
