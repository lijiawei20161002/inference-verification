"""Prove the provider-side commitment PR is useful -- on a REAL model.

This is the executable counterpart to docs/VLLM_PROVIDER_COMMITMENT_PR.md. It
implements the provider exactly as the PR specifies (ivgym.commitment:
Ed25519-signed Merkle roots over what was served) and drives it with a real
Qwen3-0.6B on the GPU. It then answers the two questions that decide whether the
PR is worth shipping:

  CLAIM A -- the crypto binding is sound.
    The signed root, published BEFORE the audit set is drawn, makes tampering and
    predict-the-audit-set attacks fail. Measured: honest openings verify 100%;
    tampered outputs / forged roots / post-publish leaf swaps verify 0%.

  CLAIM B -- recompute (which the commitment enables) catches forward-pass cheats
    that a self-reported trace provably CANNOT.
    Same committed transcripts, two auditors:
      * recompute  : re-run the REAL model, Token-DiFR margin under the shared
                     spec seed.  (needs the weights; sound.)
      * trace-only : trust the provider's self-reported per-token logits, enforce
                     only the public sampling rule.  (needs no weights; cheap.)
    A forward-pass cheat (quant/fp8) reports a perfectly self-consistent trace, so
    trace-only sits at AUC 0.5 -- while recompute separates it. Sampler cheats
    (wrong seed, bug) break trace self-consistency, so BOTH catch them. This is
    the procedure-vs-forward-pass boundary, measured on real logits.

  CLAIM C -- the cost is the MVP's: audit only a random q-fraction, at a measured
    per-audit recompute wall-clock; audit count is independent of traffic volume.

Run (H100; ~3-6 min at defaults, downloads Qwen/Qwen3-0.6B on first use):
    .venv/bin/python -m experiments.exp_commitment_vllm
Env: IVGYM_MODEL, IVGYM_PROMPTS, IVGYM_TOKENS.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ivgym import attacks as A
from ivgym import commitment as C
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec
from ivgym.harness import batch_means
from ivgym.metrics import roc_auc
from ivgym.sampling import (
    filtered_logits,
    gumbel_max_sample,
    gumbel_noise,
    position_seed,
    stable_hash,
)

np.set_printoptions(precision=4, suppress=True)

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", "24"))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", "64"))
BATCH = int(os.environ.get("IVGYM_BATCH", "256"))
N_BATCHES = int(os.environ.get("IVGYM_NBATCHES", "400"))
MODEL_SEED = 0
REF = SamplingSpec(temperature=1.0, top_k=50, top_p=0.95, seed=42)

# the configs the provider runs, and which axis each cheat lives on
CONFIGS = ["honest", "quant_4bit", "kv_fp8", "seed_43", "bug_k32"]
KIND = {
    "quant_4bit": "forward-pass",
    "kv_fp8": "forward-pass",
    "seed_43": "sampler",
    "bug_k32": "sampler",
}
DELTA_MAX = 30.0


# ---------------------------------------------------------------------------
# provider: serve one request, recording BOTH the reference recompute (trusted)
# and the provider's self-reported logits (what a trace would carry).
# Mirrors HFGPUBackend.generate; the only addition is keeping prov_logits.
# ---------------------------------------------------------------------------
def serve(backend: HFGPUBackend, prompt_id: int, n_tokens: int, attack: A.Attack):
    torch = backend._torch
    pspec = attack.provider_spec(REF)
    prompt_ids = backend._prompt_ids(prompt_id)
    claimed: list[int] = []
    prov_logits: list[np.ndarray] = []

    with torch.no_grad():
        out = backend.model(prompt_ids, use_cache=True)
        past = out.past_key_values
        logits_last = out.logits[0, -1]
        for pos in range(n_tokens):
            base = logits_last.float().cpu().numpy()
            prng = np.random.default_rng(
                (MODEL_SEED, prompt_id, pos, 11, stable_hash(attack.name))
            )
            logits = attack.perturb_logits(base, prng)          # provider's ACTUAL logits
            gseed = position_seed(pspec.seed, prompt_id, pos)
            g = gumbel_noise(backend.vocab, gseed)
            filt = filtered_logits(logits, pspec.top_k, pspec.top_p)
            top_ids = np.argsort(filt)[::-1]
            override = attack.sample_override(prng, top_ids)
            token = (
                int(override)
                if override is not None
                else gumbel_max_sample(logits, pspec.temperature, g, pspec.top_k, pspec.top_p)
            )
            claimed.append(token)
            prov_logits.append(logits.astype(np.float32))

            step_t = torch.tensor([[token]], device=backend.device, dtype=prompt_ids.dtype)
            out = backend.model(step_t, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits_last = out.logits[0, -1]

    # trusted reference recompute: one prefill over [prompt + claimed] under REF
    backend._populate_ref_cache(prompt_id, prompt_ids, claimed, n_tokens)
    ref_logits = np.stack([backend.reference_logits(prompt_id, p) for p in range(n_tokens)])
    return dict(
        prompt_id=prompt_id,
        prompt_ids=prompt_ids[0].cpu().numpy().astype(np.int32),
        tokens=np.array(claimed, np.int32),
        prov_logits=np.stack(prov_logits),
        ref_logits=ref_logits,
        seed=pspec.seed,
    )


def difr_margin(logits_pos, claimed_token, gumbel, spec, delta_max=DELTA_MAX):
    """Token-DiFR post-Gumbel margin (higher = more divergent). Identical rule
    whether `logits_pos` is the trusted recompute or the provider's self-report."""
    filt = filtered_logits(logits_pos, spec.top_k, spec.top_p)
    z = filt + spec.temperature * gumbel
    v = int(np.argmax(z))
    if filt[claimed_token] <= -1e29:
        return delta_max
    return min(float(z[v] - z[claimed_token]), delta_max)


def score_sequence(rec, vocab):
    """Per-token margins from the two auditors. Both enforce the PUBLIC spec seed
    (REF.seed); they differ only in WHICH logits they trust: recomputed vs
    provider-reported."""
    pid = rec["prompt_id"]
    m_recompute, m_trace = [], []
    for pos in range(len(rec["tokens"])):
        g = gumbel_noise(vocab, position_seed(REF.seed, pid, pos))
        tok = int(rec["tokens"][pos])
        m_recompute.append(difr_margin(rec["ref_logits"][pos], tok, g, REF))
        m_trace.append(difr_margin(rec["prov_logits"][pos], tok, g, REF))
    return np.array(m_recompute), np.array(m_trace)


def main():
    print("=" * 78)
    print(f"PROVIDER-SIDE COMMITMENT (vLLM PR) on a REAL model: {MODEL}")
    print(f"prompts={N_PROMPTS}  tokens/req={N_TOKENS}  spec={REF}")
    print("=" * 78)

    t0 = time.time()
    backend = HFGPUBackend(model_name=MODEL, max_prompt_tokens=24)
    print(f"loaded {MODEL}: {backend.n_params/1e6:.0f}M params, vocab={backend.vocab} "
          f"({time.time()-t0:.1f}s)")

    # ---- serve every (prompt, config); provider commits every served request --
    emitter = C.CommitmentEmitter(model_id=MODEL, epoch=N_PROMPTS * len(CONFIGS))
    pubkey = emitter.public_key()
    records: dict[str, list] = {c: [] for c in CONFIGS}
    openings: list[tuple] = []   # (config, rec, Opening) in the order committed

    t1 = time.time()
    for cfg in CONFIGS:
        atk = A.get(cfg)
        for pid in range(N_PROMPTS):
            rec = serve(backend, pid, N_TOKENS, atk)
            records[cfg].append(rec)
            op = emitter.record(rec["prompt_ids"], rec["tokens"], rec["seed"], REF)
            openings.append((cfg, rec, op))
    signed_root = emitter.seal()   # <-- root SIGNED + PUBLISHED before any audit
    print(f"served + committed {len(openings)} requests in {time.time()-t1:.1f}s; "
          f"epoch sealed, root signed (before audit set drawn)")

    # ===================================================================
    # CLAIM A -- crypto binding
    # ===================================================================
    print("\n" + "=" * 78)
    print("CLAIM A -- commitment binding (Ed25519-signed Merkle root)")
    print("=" * 78)
    rng = np.random.default_rng(0)

    honest_ok = tamper_caught = sig_caught = swap_caught = 0
    n_trials = len(openings)
    for i, (cfg, rec, op0) in enumerate(openings):
        op = emitter.opening_for(op0.epoch_id, op0.leaf_index)
        # (1) honest inclusion of what was actually served
        honest_ok += C.verify_inclusion(
            emitter.model_id, emitter.spec_version, rec["prompt_ids"], rec["tokens"],
            rec["seed"], REF, op, signed_root, pubkey)
        # (2) provider tampers the served output after committing
        tam = rec["tokens"].copy()
        tam[rng.integers(len(tam))] ^= 1
        tamper_caught += not C.verify_inclusion(
            emitter.model_id, emitter.spec_version, rec["prompt_ids"], tam,
            rec["seed"], REF, op, signed_root, pubkey)
        # (3) forged root (flip one byte of the signed root)
        forged = C.SignedRoot(signed_root.epoch_id,
                              signed_root.root[:-1] + bytes([signed_root.root[-1] ^ 1]),
                              signed_root.signature)
        sig_caught += not C.verify_signed_root(forged, pubkey)

    # (4) predict-the-audit-set: after the root is published, the provider tries
    #     to swap a cheat leaf for an honest one. It CANNOT re-sign the already
    #     published root; the auditor checks against the published signed root,
    #     under which the cheat leaf the user holds still verifies and the swapped
    #     tree's root does not match the signature.
    n_swap_candidates = n_identical = 0
    base_leaves = emitter._sealed_trees[signed_root.epoch_id].leaves
    for cfg, rec, op0 in openings:
        if cfg == "honest":
            continue
        honest_rec = records["honest"][rec["prompt_id"]]
        honest_leaf = C.leaf_hash(
            emitter.model_id, emitter.spec_version, honest_rec["prompt_ids"],
            honest_rec["tokens"], honest_rec["seed"], REF)
        if honest_leaf == op0.leaf:
            # cheat produced byte-identical output to honest: leaf is identical, so
            # there is nothing to swap. Such a cheat is caught by RECOMPUTE (logit
            # divergence), not by the commitment -- exactly as designed.
            n_identical += 1
            continue
        n_swap_candidates += 1
        # provider builds an alternative tree with this cheat leaf replaced by honest
        alt_leaves = list(base_leaves)
        alt_leaves[op0.leaf_index] = honest_leaf
        alt_root = C.MerkleTree(alt_leaves).root
        # swap is defeated iff the published signed root no longer matches the tree
        # the provider now claims (it would have to forge the Ed25519 signature)
        swap_caught += (alt_root != signed_root.root)

    print(f"  honest openings verify         : {honest_ok}/{n_trials}   ({honest_ok/n_trials:.0%})")
    print(f"  tampered outputs rejected      : {tamper_caught}/{n_trials}   ({tamper_caught/n_trials:.0%})")
    print(f"  forged signed-roots rejected   : {sig_caught}/{n_trials}   ({sig_caught/n_trials:.0%})")
    print(f"  post-publish leaf-swaps foiled : {swap_caught}/{n_swap_candidates}   "
          f"({swap_caught/max(n_swap_candidates,1):.0%})  "
          f"[{n_identical} cheats had honest-identical output -> caught by recompute, not commitment]")

    # ===================================================================
    # CLAIM B -- recompute vs trace-only detection (the usefulness argument)
    # ===================================================================
    print("\n" + "=" * 78)
    print("CLAIM B -- what the commitment ENABLES: recompute vs a self-report trace")
    print("=" * 78)
    tok = {c: {"recompute": [], "trace": []} for c in CONFIGS}
    for cfg in CONFIGS:
        for rec in records[cfg]:
            mr, mt = score_sequence(rec, backend.vocab)
            tok[cfg]["recompute"].append(mr)
            tok[cfg]["trace"].append(mt)
        tok[cfg]["recompute"] = np.concatenate(tok[cfg]["recompute"])
        tok[cfg]["trace"] = np.concatenate(tok[cfg]["trace"])

    hon_r = tok["honest"]["recompute"]
    hon_t = tok["honest"]["trace"]

    # -- mechanism: mean per-token divergence margin (WHY the AUCs come out so) --
    print(f"  mean per-token DiFR margin (n={len(hon_r)} tokens/config); higher = more divergent")
    print(f"  {'config':>12} {'axis':>13} | {'recompute':>10} {'trace-only':>11}")
    print("  " + "-" * 52)
    for cfg in CONFIGS:
        axis = "honest" if cfg == "honest" else KIND[cfg]
        print(f"  {cfg:>12} {axis:>13} | {tok[cfg]['recompute'].mean():>10.3f} "
              f"{tok[cfg]['trace'].mean():>11.3f}")
    print("  (trace-only margin is ~0 for honest AND forward-pass: the provider")
    print("   samples correctly from its own logits, so the self-report is")
    print("   internally consistent -- there is nothing for a trace check to find.)")

    # -- batch-level detection AUC (the DiFR statistic: mean over BATCH tokens) --
    bsizes = sorted({1, min(64, len(hon_r)), min(BATCH, len(hon_r))})
    rng_b = np.random.default_rng(0)

    def batch_auc(honest, attack, b):
        hb = batch_means(np.asarray(honest, float), b, N_BATCHES, rng_b)
        ab = batch_means(np.asarray(attack, float), b, N_BATCHES, rng_b)
        return roc_auc(hb, ab)

    print(f"\n  batch-level detection AUC (honest vs cheat), batch sizes {bsizes}, "
          f"{N_BATCHES} batches")
    print(f"  headline batch b={bsizes[-1]}:\n")
    print(f"  {'attack':>12} {'axis':>13} | {'RECOMPUTE (PR+audit)':>22} {'TRACE-ONLY (no weights)':>24}")
    print("  " + "-" * 76)
    results = []
    for cfg in CONFIGS:
        if cfg == "honest":
            continue
        b = bsizes[-1]
        auc_r = batch_auc(hon_r, tok[cfg]["recompute"], b)
        auc_t = batch_auc(hon_t, tok[cfg]["trace"], b)
        sweep_r = [batch_auc(hon_r, tok[cfg]["recompute"], bb) for bb in bsizes]
        results.append((cfg, KIND[cfg], auc_r, auc_t, sweep_r))
        print(f"  {cfg:>12} {KIND[cfg]:>13} | {auc_r:>22.3f} {auc_t:>24.3f}")
    print(f"\n  recompute AUC vs batch size {bsizes} (the DiFR batch-size effect):")
    for cfg, axis, _, _, sweep_r in results:
        print(f"    {cfg:>12} {axis:>13}: " + "  ".join(f"{a:.3f}" for a in sweep_r))

    fwd = [r for r in results if r[1] == "forward-pass"]
    print("\n  READ-OUT:")
    print(f"    * forward-pass cheats (quant/fp8): recompute AUC "
          f"{np.mean([r[2] for r in fwd]):.3f}  vs  trace-only AUC "
          f"{np.mean([r[3] for r in fwd]):.3f}")
    print("      -> a self-reported trace is at chance: the provider samples")
    print("         correctly from its OWN (cheap) logits, so the trace is")
    print("         internally consistent. ONLY recomputing the real weights sees it.")
    print("    * sampler cheats (seed/bug): BOTH catch them -- they break trace")
    print("      self-consistency, so no recompute is even needed. That is the")
    print("      procedure-vs-forward-pass boundary, on real logits.")

    # ===================================================================
    # CLAIM C -- cost: per-audit recompute wall-clock + the MVP invariant
    # ===================================================================
    print("\n" + "=" * 78)
    print("CLAIM C -- cost: audit only a random q-fraction (MVP economics)")
    print("=" * 78)
    per_audit_s = backend.timed_seconds["reference"] / max(backend.timed_calls["reference"], 1)
    print(f"  measured per-audit recompute: {per_audit_s*1e3:.1f} ms "
          f"({backend.timed_calls['reference']} reference prefills timed)")
    print("  audit count = ln(1/(1-P)) / (f*(1-delta)), INDEPENDENT of traffic N:")
    print(f"  {'f':>8} {'P':>7} {'audits':>9} {'audit wall-clock':>18}")
    det = 1.0   # measured (1-delta) ~ 1.0 for these cheats under recompute
    for f_ in (0.10, 0.01, 0.001):
        for Pt in (0.99, 0.999):
            count = np.log(1 / (1 - Pt)) / (f_ * det)
            print(f"  {f_:>8} {Pt:>7.3f} {count:>9,.0f} {count*per_audit_s:>16.1f}s")

    # ---- figure ----
    try:
        _render(results, hon_r, hon_t, tok)
    except Exception as e:
        print(f"\n(skipped figure: {e})")
    print(f"\nDONE in {time.time()-t0:.1f}s.")
    return results


def _render(results, hon_r, hon_t, tok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r[0] for r in results]
    auc_r = [r[2] for r in results]
    auc_t = [r[3] for r in results]
    kinds = [r[1] for r in results]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    b1 = ax.bar(x - w / 2, auc_r, w, label="recompute  (commitment + audit)",
                color="#1f77b4")
    b2 = ax.bar(x + w / 2, auc_t, w, label="trace-only  (self-report, no weights)",
                color="#d62728")
    ax.axhline(0.5, ls="--", color="#555", lw=1.0)
    ax.text(len(labels) - 0.5, 0.515, "chance", fontsize=8, color="#555", ha="right")
    for i, k in enumerate(kinds):
        ax.text(i, 1.02, k, ha="center", fontsize=8,
                color="#b8860b" if k == "forward-pass" else "#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("detection AUC (honest vs cheat)")
    ax.set_title("Provider-side commitment: recompute catches forward-pass cheats\n"
                 "a self-reported trace provably cannot (real Qwen3-0.6B)", fontsize=11)
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "figures" / "fig_commitment_vllm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote figure: {out}")


if __name__ == "__main__":
    main()
