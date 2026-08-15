"""The accuracy you cannot buy: the benign forward-pass shift, in d' units.

`exp_specdec_shape_gpu` established that holding the weights, the prefix and the
arithmetic fixed and changing only how the forward pass is *scheduled* moves the
logits -- 13.6% of values bitwise identical between a chunked and a sequential
pass, 0.88% of generated positions changing their argmax. `NEXT_EXPERIMENTS.md`
item 1 says the number that decides whether that matters is the one it did not
produce: **the same shift on `harness.evaluate`'s scale**, so it can be put beside
`d'(quant_4bit) = 0.0775`.

This produces it. One honest pool, generated once by a provider that decodes
sequentially. Then the *verifier* replays it -- the same weights, the same claimed
tokens, the same benign noise draw -- under four schedules a real verifier might
legitimately use:

    chunked_b1   one prefill over [prompt + claimed], batch of 1     (the BASELINE:
                 what every AUC in this repo calibrates its honest null on)
    chunked_b4   the same prefill sharing a batch with 3 unrelated rows -- what a
                 verifier does the moment it batches audits for throughput. Rows
                 cannot attend across, so any difference is the kernel's reduction
                 schedule, not the model.
    chunked_b1_sdpa  the same batch-1 prefill through a different attention kernel
                 (sdpa vs eager). Same math, same weights: a library upgrade.
    sequential   replay token-by-token through a KV cache -- the provider's own
                 schedule, and the most expensive way to verify.

Every pair of these is an honest-vs-honest comparison, so a verifier that could
not tell them apart would report d' = 0. What it reports instead is the FLOOR
under the false-positive side of every detection number in the repo, and the
poster's central asymmetry:

    a detector separates an attack from honest at d'_attack, but it separates
    honest-in-shape-A from honest-in-shape-B at d'_benign. Growing the token pool
    sharpens BOTH. The batch a verdict really needs is

        b = (delta* / (d'_attack - d'_benign))^2,

    which is not (delta'/d'_attack)^2, and is infinite when d'_benign >= d'_attack.
    Past that point more compute buys false accusations, not accuracy.

The comparison is PAIRED -- the same tokens, the same prompts, the same
`verifier_sigma` draw (memoized per prompt, so it is bit-identical across arms) --
so the reported interval is a paired sequence bootstrap and is far tighter than
the two-sample intervals in `exp_cost_of_a_verdict_gpu`.

Two noise arms, because the repo's honest null is partly synthetic. `noisy` keeps
the backend's injected `verifier_sigma = 0.02` logit noise, which is the model of
nondeterminism every committed d' was measured under -- that is the arm that is
comparable to 0.0775. `clean` sets it to zero, so the only nondeterminism left is
the real one: the GPU. Read the first for comparability and the second for the
physics.

    IVGYM_M=Qwen/Qwen3-1.7B python -m experiments.exp_benign_shape_dprime_gpu

Env: IVGYM_M(Qwen/Qwen3-1.7B), IVGYM_PROMPTS(80), IVGYM_TOKENS(256),
     IVGYM_BOOT(400), IVGYM_FILLER_BATCH(4), IVGYM_ARMS(noisy,clean).

Writes `docs/results/benign_shape_dprime.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, signal, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
N = int(os.environ.get("IVGYM_PROMPTS", 80))
T = int(os.environ.get("IVGYM_TOKENS", 256))
N_BOOT = int(os.environ.get("IVGYM_BOOT", 400))
FILLER_B = int(os.environ.get("IVGYM_FILLER_BATCH", 4))
ARMS = os.environ.get("IVGYM_ARMS", "noisy,clean").split(",")

DETS = ["token_difr", "cross_entropy", "token_toploc", "activation_difr"]
BASELINE = "chunked_b1"
SHAPES = [BASELINE, f"chunked_b{FILLER_B}", "chunked_b1_sdpa", "sequential"]
# the deviations this floor has to be read against (committed artifacts)
REFERENCE_DPRIMES = {"quant_4bit (token_difr, Qwen3-1.7B, pool_scaling.json)": 0.0775,
                     "kv_fp8 (token_difr, README grid)": 0.015}


class ShapeBackend(HFGPUBackend):
    """`HFGPUBackend` whose reference replay runs in a selectable forward-pass
    shape, with the injected verifier noise switchable off.

    Only `_populate_ref_cache` changes. Generation, sampling, the claimed tokens
    and the benign-noise draw are untouched, so the arms differ in exactly one
    thing: how the verifier's own forward pass was scheduled.
    """

    def __init__(self, *a, ref_shape: str = BASELINE, ref_noise: bool = True, **kw):
        super().__init__(*a, **kw)
        self.ref_shape = ref_shape
        self.ref_noise = ref_noise
        self._sdpa_model = None

    def _model_for(self, shape: str):
        """The eager-attention model, or a second copy with sdpa attention.

        Same weights and same dtype -- a different attention *kernel* is the point.
        Loaded lazily so an arm that never asks for it never pays the memory.
        """
        if not shape.endswith("_sdpa"):
            return self.model
        if self._sdpa_model is None:
            from transformers import AutoModelForCausalLM
            self._sdpa_model = (AutoModelForCausalLM.from_pretrained(
                self.model.config._name_or_path, dtype=self.model.dtype,
                attn_implementation="sdpa").to(self.device).eval())
        return self._sdpa_model

    def _populate_ref_cache(self, prompt_id, prompt_ids, claimed, n_tokens):
        torch = self._torch
        shape = self.ref_shape
        model = self._model_for(shape)
        claimed_t = torch.tensor([claimed], device=self.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, claimed_t], dim=1)
        L = int(prompt_ids.shape[1])

        if shape == "sequential":
            # The provider's own schedule: prefill the prompt, then one token at a
            # time through the KV cache. Position j's reference logits come off the
            # step that has consumed exactly [prompt + claimed[:j]].
            from transformers import DynamicCache
            lg, ac = [], []
            with torch.no_grad(), self._timed("reference"):
                cache = DynamicCache()
                out = model(prompt_ids, past_key_values=cache, use_cache=True,
                            output_hidden_states=True)
                lg.append(out.logits[0, -1])
                ac.append(out.hidden_states[-1][0, -1])
                for j in range(n_tokens - 1):
                    out = model(full[:, L + j: L + j + 1], past_key_values=cache,
                                use_cache=True, output_hidden_states=True)
                    lg.append(out.logits[0, -1])
                    ac.append(out.hidden_states[-1][0, -1])
            logits = torch.stack(lg).float().cpu().numpy()
            acts = torch.stack(ac).float().cpu().numpy()
        else:
            x = full
            nb = int(shape.split("_b")[1].split("_")[0])
            if nb > 1:
                # unrelated filler rows; they cannot reach row 0 through attention,
                # so whatever they change is the kernel's reduction schedule
                g = torch.Generator(device="cpu").manual_seed(7 + prompt_id)
                hi = min(int(model.config.vocab_size), 30000)
                filler = torch.randint(1000, hi, (nb - 1, full.shape[1]),
                                       generator=g).to(full.device)
                x = torch.cat([full, filler], 0)
            with torch.no_grad(), self._timed("reference"):
                out = model(x, output_hidden_states=True)
            idx = slice(L - 1, L - 1 + n_tokens)
            logits = out.logits[0, idx].float().cpu().numpy()
            acts = out.hidden_states[-1][0, idx].float().cpu().numpy()

        if self.ref_noise:
            # The SAME draw in every arm (pure function of model_seed/prompt_id),
            # so it cancels in the paired difference and only enters the honest sd.
            full_noise, full_act_noise = self._benign_noise(
                prompt_id, logits.shape[1], acts.shape[1])
            logits = logits + full_noise[:n_tokens]
            acts = acts + full_act_noise[:n_tokens]
        self._ref_cache[prompt_id] = {"logits": logits.astype(np.float32),
                                      "act": acts.astype(np.float32)}
        self._ref_depth[prompt_id] = n_tokens


def paired_d_prime(base: np.ndarray, alt: np.ndarray, n_seq: int, n_tok: int,
                   n_boot: int, seed: int = 0) -> dict:
    """`d'` of `alt` against `base` on `evaluate`'s scale, with a PAIRED bootstrap.

    Winsorized at the baseline's 99.9th percentile exactly as
    `signal.per_token_stats` (and therefore `harness.evaluate`) does, then
    `d' = (mean_alt - mean_base) / sd_base`. Resampling SEQUENCES (both arms
    together, since the arms are the same tokens scored twice) keeps the pairing,
    which is what makes this interval tight enough to compare against 0.0775.
    """
    st = signal.per_token_stats(base, alt)
    b2 = base.reshape(n_seq, n_tok)
    a2 = alt.reshape(n_seq, n_tok)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_seq, n_seq)
        draws[i] = signal.per_token_stats(b2[idx].ravel(), a2[idx].ravel())["d_prime"]
    return {"d_prime": st["d_prime"], "ci": [float(np.quantile(draws, 0.05)),
                                             float(np.quantile(draws, 0.95))],
            "base_mean": st["honest_mean"], "alt_mean": st["attack_mean"],
            "base_sd": st["honest_sd"],
            "frac_tokens_changed": float(np.mean(base != alt))}


def run_arm(arm: str, t0: float) -> dict:
    noise = arm == "noisy"
    print(f"\n{'='*84}\narm '{arm}': verifier_sigma = "
          f"{0.02 if noise else 0.0}, act sigma = {0.05 if noise else 0.0}\n{'='*84}",
          flush=True)
    backend = ShapeBackend(model_name=M, ref_noise=noise,
                           verifier_sigma=0.02 if noise else 0.0,
                           act_benign_sigma=0.05 if noise else 0.0)
    spec = SamplingSpec()
    dets = [verifiers.get(n) for n in DETS]

    # one honest pool; the provider decodes sequentially, as a provider does
    backend.ref_shape = BASELINE
    seqs = harness.generate_dataset(backend, attacks.get("honest"), spec, N, T,
                                    record_activations=True)
    print(f"  generated {N}x{T} honest tokens [{time.time()-t0:.0f}s]", flush=True)

    scores: dict[str, dict[str, np.ndarray]] = {}
    for shape in SHAPES:
        backend.ref_shape = shape
        backend.drop_reference_cache()
        for seq in seqs:                       # replay every sequence in this shape
            backend._populate_ref_cache(seq.prompt_id, backend._prompt_ids(seq.prompt_id),
                                        backend._claimed[seq.prompt_id], T)
        ts = harness.verify(backend, seqs, spec, dets)
        scores[shape] = {d: np.asarray(ts.scores[d], float) for d in DETS}
        print(f"  replayed + scored shape {shape:>16} [{time.time()-t0:.0f}s]", flush=True)

    out: dict[str, dict] = {}
    for shape in SHAPES:
        if shape == BASELINE:
            continue
        out[shape] = {}
        print(f"\n  {BASELINE}  ->  {shape}   (honest vs honest: d' must be ~0)")
        print(f"    {'verifier':>16} {'d prime (benign)':>28} {'tokens changed':>15}")
        for d in DETS:
            r = paired_d_prime(scores[BASELINE][d], scores[shape][d], N, T, N_BOOT)
            out[shape][d] = r
            print(f"    {d:>16} {r['d_prime']:>+12.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
                  f"{r['frac_tokens_changed']:>14.2%}", flush=True)
    # the control that says the machinery can report zero: replay the baseline
    # shape twice. A rerun at a fixed shape is bitwise identical, so this must be
    # exactly 0.0 -- if it is not, the arm is measuring its own bookkeeping.
    backend.ref_shape = BASELINE
    backend.drop_reference_cache()
    for seq in seqs:
        backend._populate_ref_cache(seq.prompt_id, backend._prompt_ids(seq.prompt_id),
                                    backend._claimed[seq.prompt_id], T)
    rerun = harness.verify(backend, seqs, spec, dets)
    ctrl = {d: paired_d_prime(scores[BASELINE][d], np.asarray(rerun.scores[d], float),
                              N, T, 50) for d in DETS}
    print(f"\n  control: {BASELINE} replayed twice (must be exactly 0)")
    for d in DETS:
        print(f"    {d:>16} d' = {ctrl[d]['d_prime']:+.6f}   "
              f"tokens changed {ctrl[d]['frac_tokens_changed']:.2%}")
    return {"shapes": out, "rerun_control": ctrl}


def main() -> None:
    t0 = time.time()
    print("=" * 84)
    print(f"The benign forward-pass shift on evaluate's d' scale   M={M}")
    print(f"  {N} prompts x {T} tokens, baseline shape = {BASELINE}, "
          f"shapes = {', '.join(SHAPES[1:])}")
    print("=" * 84, flush=True)

    arms = {arm: run_arm(arm, t0) for arm in ARMS}

    # ---- what it costs: the price multiplier on every verdict ---------------
    print(f"\n{'='*84}\nWhat the floor does to the price of a verdict\n{'='*84}")
    print("A verdict separating an attack from honest needs (delta*/d'_attack)^2 tokens.")
    print("If the verifier's audit shape differs from its calibration shape, it must")
    print("separate the attack from a SHIFTED honest null instead: d'_attack - d'_benign.")
    delta = signal.delta_for_pauc(0.90)
    price_table = {}
    for label, d_a in REFERENCE_DPRIMES.items():
        price_table[label] = {}
        print(f"\n  against {label}:  d'_attack = {d_a:.4f}, "
              f"{signal.batch_for_pauc(d_a):,} tokens with no floor")
        for arm in ARMS:
            for shape, per_det in arms[arm]["shapes"].items():
                d_b = per_det["token_difr"]["d_prime"]
                eff = d_a - d_b
                b = signal.batch_for_pauc(eff) if eff > 0 else -1
                mult = (b / signal.batch_for_pauc(d_a)) if b > 0 else float("inf")
                price_table[label][f"{arm}/{shape}"] = {
                    "d_benign_token_difr": d_b, "d_effective": eff,
                    "tokens_per_verdict": int(b),
                    "price_multiplier": mult}
                tok = f"{b:,}" if b > 0 else "UNREACHABLE at any budget"
                mul = f"x{mult:.2f}" if b > 0 else "infinite"
                print(f"    {arm:>6} / {shape:<16} d'_benign = {d_b:+.4f}  ->  "
                      f"effective {eff:+.4f}  ->  {tok:>26}  ({mul})")

    payload = {"M": M, "n_prompts": N, "tokens": T, "n_boot": N_BOOT,
               "baseline_shape": BASELINE, "shapes": SHAPES,
               "filler_batch": FILLER_B, "detectors": DETS,
               "reference_dprimes": REFERENCE_DPRIMES,
               "delta_star_090": float(delta),
               "arms": arms, "price_of_the_floor": price_table,
               "elapsed_s": time.time() - t0}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "benign_shape_dprime.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {RES/'benign_shape_dprime.json'}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
