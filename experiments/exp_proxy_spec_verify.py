"""Client-side proxy verification: can a cheap proxy approximate full recompute?

The client holds its own small **proxy** ``q`` and uses the speculative-decoding
acceptance rate ``accept_rate = 1 − TV(p, q)`` (``ivgym.spec_decode``) as a cheap
stand-in for recomputing the reference model ``M``. Nothing here trusts the
provider or asks vLLM to emit anything -- the client runs the proxy itself.

Two parts:

  (1) **CPU (always runs, pure numpy).** Over synthetic correlated (target,
      proxy) pairs, sweep the quantization strength and report the detection AUC
      of the draft-anchored **acceptance-rate** signal, the generic **entropy**
      fingerprint, and the exact **recompute** baseline. Then the temperature-
      retune evasion: retune ``T`` to match honest entropy -- it blinds the
      entropy fingerprint but NOT the acceptance rate (matching entropy does not
      restore ``TV(p̂, q)``), while the recompute baseline stays at ~1.0.

  (2) **REAL proxy (GPU, opt-in).** With ``IVGYM_M`` + ``IVGYM_PROXY`` set, load a
      real reference and a real cheap proxy (same Qwen family / shared tokenizer)
      via ``ivgym.backends.hf_gpu`` and measure, on tokens ``M`` actually sampled,
      the honest ``accept_rate = 1 − TV(M, proxy)`` anchor, then the detection AUC
      of the cheap proxy signal (``surface_stat`` = proxy NLL) vs the full
      recompute (``token_difr``) on ``quant_4bit`` / ``kv_fp8``.

Run:
    python -m experiments.exp_proxy_spec_verify                       # CPU only
    IVGYM_M=Qwen/Qwen3-4B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_proxy_spec_verify                   # + real proxy (GPU)

Env (CPU):  IVGYM_TRACES, IVGYM_POS, IVGYM_VOCAB, IVGYM_AGREE
Env (GPU):  IVGYM_M, IVGYM_PROXY, IVGYM_PROMPTS, IVGYM_TOKENS, IVGYM_BATCH
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import spec_decode as sd
from ivgym.core import SamplingSpec
from ivgym.metrics import roc_auc

N_SAMPLES = int(os.environ.get("IVGYM_TRACES", 120))
N_POS = int(os.environ.get("IVGYM_POS", 200))
VOCAB = int(os.environ.get("IVGYM_VOCAB", 64))
AGREE = float(os.environ.get("IVGYM_AGREE", 0.8))
SIGMAS = [0.1, 0.25, 0.5, 1.0]


# ---------------------------------------------------------------------------
# (1) CPU: synthetic acceptance-rate fingerprint sweep
# ---------------------------------------------------------------------------
def build(cheat, n_samples, seed0):
    spec = SamplingSpec()
    out = []
    for t in range(n_samples):
        rng = np.random.default_rng(1_000_003 * seed0 + t)
        pos = sd.synthetic_positions(rng, N_POS, vocab=VOCAB, agreement=AGREE)
        out.append(sd.generate_sample(rng, cheat, pos, spec))
    return out


def feature_auc(fp, feature, honest_null, cheat):
    h = np.array([fp.score(s)[feature] for s in honest_null])
    c = np.array([fp.score(s)[feature] for s in cheat])
    return roc_auc(h, c)


def recompute_auc(honest_null, cheat):
    h = np.array([sd.recompute_divergence(s) for s in honest_null])
    c = np.array([sd.recompute_divergence(s) for s in cheat])
    return roc_auc(h, c)


def mean_feat(samples, feat):
    return float(np.mean([sd.sequence_features(s)[feat] for s in samples]))


def cpu_sweep():
    print("=" * 78)
    print(f"(1) CPU synthetic sweep  samples={N_SAMPLES} positions/sample={N_POS} "
          f"vocab={VOCAB} agreement={AGREE}")
    print("=" * 78)

    honest_fit = build(sd.get_cheat("honest"), N_SAMPLES, seed0=1)
    honest_null = build(sd.get_cheat("honest"), N_SAMPLES, seed0=2)
    fp = sd.ProxyReference().fit(honest_fit)
    h_ent, h_acc = mean_feat(honest_null, "mean_entropy"), mean_feat(honest_null, "accept_rate")
    print(f"honest reference: accept_rate={h_acc:.3f}  mean_entropy={h_ent:.3f}\n")

    honest_floor = build(sd.get_cheat("honest"), N_SAMPLES, seed0=3)
    print("(honest-vs-honest floor, should be ~0.5)")
    print(f"  accept_rate AUC={feature_auc(fp,'accept_rate',honest_null,honest_floor):.3f}"
          f"   mean_entropy AUC={feature_auc(fp,'mean_entropy',honest_null,honest_floor):.3f}\n")

    print("quantization-strength sweep")
    hdr = (f"  {'sigma':>6} {'Δaccept':>8} {'Δentropy':>9} | "
           f"{'accept_fp':>10} {'entropy_fp':>11} {'recompute':>10}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for sig in SIGMAS:
        tr = build(sd.QuantTarget(name=f"q{sig}", sigma=sig), N_SAMPLES, seed0=4)
        print(f"  {sig:>6} {mean_feat(tr,'accept_rate')-h_acc:>8.3f} "
              f"{mean_feat(tr,'mean_entropy')-h_ent:>9.3f} | "
              f"{feature_auc(fp,'accept_rate',honest_null,tr):>10.3f} "
              f"{feature_auc(fp,'mean_entropy',honest_null,tr):>11.3f} "
              f"{recompute_auc(honest_null,tr):>10.3f}")

    # temperature-retune evasion at sigma=1.0
    print("\ntemperature-retune evasion (sigma=1.0): tune T to match honest entropy")
    best_T, best_gap = 1.0, 1e9
    for T in np.linspace(0.6, 2.0, 29):
        tr = build(sd.AdvQuantTemp(name="adv", sigma=1.0, temp=float(T)), 40, seed0=90)
        gap = abs(mean_feat(tr, "mean_entropy") - h_ent)
        if gap < best_gap:
            best_gap, best_T = gap, float(T)
    print(f"  tuned T={best_T:.3f} (residual entropy gap {best_gap:.4f})")
    plain = build(sd.QuantTarget(name="q", sigma=1.0), N_SAMPLES, seed0=5)
    adv = build(sd.AdvQuantTemp(name="adv", sigma=1.0, temp=best_T), N_SAMPLES, seed0=6)
    print(f"  {'variant':>16} | {'accept_fp':>10} {'entropy_fp':>11} {'recompute':>10}")
    for nm, tr in [("quant (plain)", plain), ("adv_quant_temp", adv)]:
        print(f"  {nm:>16} | "
              f"{feature_auc(fp,'accept_rate',honest_null,tr):>10.3f} "
              f"{feature_auc(fp,'mean_entropy',honest_null,tr):>11.3f} "
              f"{recompute_auc(honest_null,tr):>10.3f}")

    print("\nReading it:")
    print("  * accept_rate (draft-anchored, = 1−TV(p̂,q)) separates strong quant and, unlike")
    print("    the entropy fingerprint, SURVIVES the temperature retune -- matching entropy")
    print("    does not restore TV(p̂,q). Both fingerprints decay to the floor for small quant.")
    print("  * recompute (the full-M baseline the proxy approximates) stays ~1.0 throughout.")
    print("  * so the cheap proxy shrinks how often the exact recompute must fire; it does")
    print("    not replace it -- the recompute-dominant boundary the DiFR detectors hit.")


# ---------------------------------------------------------------------------
# (2) GPU: real proxy vs full recompute
# ---------------------------------------------------------------------------
def gpu_real_proxy():
    M = os.environ.get("IVGYM_M")
    PROXY = os.environ.get("IVGYM_PROXY")
    if not (M and PROXY):
        print("\n(2) real-proxy GPU section skipped -- set IVGYM_M and IVGYM_PROXY "
              "on a CUDA host to run it.")
        return

    import time
    from ivgym import attacks, harness, verifiers
    from ivgym.backends.hf_gpu import HFGPUBackend

    N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 12))
    N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 64))
    BATCH = int(os.environ.get("IVGYM_BATCH", 200))
    ATTACKS = ("quant_4bit", "kv_fp8")

    print("\n" + "=" * 78)
    print(f"(2) REAL proxy vs full recompute   M={M}  proxy={PROXY}")
    print("=" * 78)
    t0 = time.time()
    backend = HFGPUBackend(model_name=M, proxy_model_name=PROXY)
    ratio = backend.proxy_n_params / backend.n_params
    print(f"loaded ({time.time()-t0:.1f}s)  M={backend.n_params/1e9:.2f}B  "
          f"proxy={backend.proxy_n_params/1e9:.2f}B  ({1/ratio:.1f}x fewer params)\n")

    spec = SamplingSpec()
    td = verifiers.get("token_difr")
    surf = verifiers.get("surface_stat")             # proxy NLL: the cheap real-proxy signal

    honest = harness.generate_dataset(backend, attacks.get("honest"), spec,
                                      N_PROMPTS, N_TOKENS, record_activations=True)
    honest_td = harness.verify(backend, honest, spec, [td])
    honest_io = harness.verify(backend, honest, spec, [surf])

    # honest acceptance-rate anchor: accept_rate = 1 − TV(M, proxy) on served tokens,
    # exactly the speculative-decoding acceptance rate (see exp_family_correlation).
    accs = []
    for seq in honest:
        rows_p = np.stack([backend.reference_logits(seq.prompt_id, s.position) for s in seq.steps])
        rows_q = np.stack([backend.proxy_logits(seq.prompt_id, s.position) for s in seq.steps])
        pos = sd.positions_from_rows(rows_p, rows_q)
        accs.append(np.mean([sd.accept_rate(sd.softmax(pp.target_logprobs),
                                            sd.softmax(pp.proxy_logprobs)) for pp in pos]))
    print(f"honest accept_rate = 1 − TV(M, proxy) = {np.mean(accs):.3f}  "
          f"(the real-proxy speculative-decoding acceptance rate)\n")

    print(f"detection AUC (honest vs attack), batch={BATCH}:")
    print(f"  {'attack':>12} | {'surface_stat (cheap proxy)':>26} | {'token_difr (recompute)':>24}")
    print("  " + "-" * 68)
    for name in ATTACKS:
        seqs = harness.generate_dataset(backend, attacks.get(name), spec,
                                        N_PROMPTS, N_TOKENS, record_activations=True)
        a_td = harness.verify(backend, seqs, spec, [td])
        a_io = harness.verify(backend, seqs, spec, [surf])
        auc_io = harness.evaluate(honest_io, a_io, [surf], [BATCH], seed=7)[0].auc
        auc_td = harness.evaluate(honest_td, a_td, [td], [BATCH], seed=7)[0].auc
        print(f"  {name:>12} | {auc_io:>26.3f} | {auc_td:>24.3f}")

    print("\n  -> the cheap real proxy tracks the forward-pass cheats it can see, but the full")
    print("     recompute (token_difr) dominates -- the same recompute-dominant boundary,")
    print("     now with a genuine second model rather than a synthetic proxy.")
    print(f"\ntotal {time.time()-t0:.1f}s")


def main():
    cpu_sweep()
    gpu_real_proxy()


if __name__ == "__main__":
    main()
