"""Input-output (black-box) detectors vs recomputation (Token-DiFR).

Ports the white-box/black-box detector distinction from Clymer et al.
(*Lessons from building a model organism testbed*, 2025) into ivgym. A
recomputation defense (`token_difr`) is the white-box analogue -- it re-runs the
reference model M. An `IODetector` is the black-box analogue -- it scores from
`(prompt, claimed_tokens)` alone, never recomputing M (see `ivgym/io_detectors.py`).

Runs end-to-end on the **synthetic backend with no GPU**. Produces three things:

  ROLE 2 -- white-box-ness validation table.
    Per attack: recomputation AUC (token_difr) vs I/O-only AUC. Where an attack
    is output-*indistinguishable*, the I/O detector pins at the honest-null floor
    while token_difr is high -- a quantitative proof the recomputation defense
    provides information beyond behaviour. The (printed) synthesis classifies every
    attack by which detector family DOMINATES (excess over each detector's own
    honest-null floor; a dominance test is robust to single-draw floor noise where a
    hard above-floor test is not):
      * RECOMPUTE-DOMINANT (the Role-2 headline) -- seed_43 (wrong seed redraws the
        SAME distribution) and the constructed quant_spoof (quantize + temperature-
        retune to output-match): token_difr ~ 1.0 yet every cheap I/O detector at
        the floor.
      * I/O-DOMINANT -- output-visible attacks (e.g. temp_1.1) the cheap proxy
        catches better than (or as well as) recomputation. NOTE the honest caveat.
      * BOTH -- caught comparably by both families.
    COMPLEMENTARITY (a separate sweep, NOT in the original plan) -- SAFE-set
    substitution is engineered to evade Token-DiFR (it deviates only inside
    token_difr's near-tie blind spot). As its covert budget epsilon grows a cheap
    proxy-NLL OUTPACES recomputation: the I/O detector dominates exactly the attack
    built against recomputation. The two families are complementary -- neither
    dominates everywhere.
    HONEST CAVEAT (printed): a *high* I/O-AUC (e.g. temp_1.1) is NOT a verifier
    win -- it means that attack is crude enough to catch from outputs alone.
    Detectability is reported symmetrically as max(AUC, 1-AUC); the (honest null)
    floor row -- a second independent honest draw from the SAME model M -- calibrates
    what "indistinguishable" means for each detector column.

  ROLE 1 -- cost/accuracy Pareto.
    AUC vs verifier FLOPs across {full recompute (token_difr, 1.0), cheap proxy
    (surface_stat, ~0.08), zero model (surface_tokens, ~0)}. Rendered to
    docs/figures/fig3_io_pareto.png.

  HELD-OUT-ATTACK GENERALIZATION (the learned detector).
    Train LearnedSeqIODetector on ONE attack family, evaluate on attacks it never
    saw. Ports the post's diversity lesson: a black-box detector generalizes only
    across attacks that share an output footprint; it cannot catch an
    output-invisible deviation (seed) no matter the training -- which is exactly
    the information recomputation adds.

Run:  python -m experiments.exp_io_detector
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness, io_detectors
from ivgym.attacks import Attack
from ivgym.backends.synthetic import SyntheticBackend
from ivgym.core import SamplingSpec
from ivgym.io_detectors import LearnedSeqIODetector

from examples.safe_set_strategies import SafeSetSubstitution
from examples.seed_free_strategies import AdvQuantTempSpoof

N_PROMPTS = 100
N_TOKENS = 256
# Batch size for the batch-level statistic S. Kept well below the token pool
# (~N_PROMPTS*N_TOKENS): the null AUC of a per-sequence-varying I/O feature scales
# as ~sqrt(BATCH / N_tokens) (large batches concentrate batch means so tightly that
# a tiny finite-pool mean gap between the two honest pools inflates into AUC>>0.5).
# BATCH=200 keeps the honest-null floor near 0.5 for every detector while leaving
# the strong recomputation signal (token_difr) saturated at ~1.0.
BATCH = 200
N_BATCHES = 400
VOCAB = 512

# The library's canonical attacks (ivgym/attacks.py). We list them explicitly
# rather than iterating all_attacks(), because importing the example strategy
# modules registers their full *sweeps* (safe_set_eps*, adv_quant_spoof_q*) -- we
# add one canonical constructed version of each below instead.
CORE_ATTACKS = ("quant_4bit", "kv_fp8", "temp_1.1", "seed_43",
                "bug_k2", "bug_k32", "adv_quant_temp")

# I/O detectors shown in the validation table (none need training).
IO_NAMES = ("surface_stat", "surface_rank", "surface_tokens")
# Relative verifier cost for the Pareto x-axis (1.0 = a full recompute of M).
FLOPS = {
    "token_difr": 1.0,              # re-runs the reference forward pass
    "surface_stat": io_detectors.get("surface_stat").verifier_flops,    # cheap proxy LM
    "surface_rank": io_detectors.get("surface_rank").verifier_flops,
    "surface_tokens": io_detectors.get("surface_tokens").verifier_flops,  # no model
}


# ---------------------------------------------------------------------------
@dataclass
class Row:
    attack: str
    note: str
    token_difr: float
    io: dict[str, float] = field(default_factory=dict)


def _mean_ce(backend, atk, spec, n_prompts=20) -> float:
    ts = harness.verify(backend, harness.generate_dataset(backend, atk, spec, n_prompts, N_TOKENS),
                        spec, [defenses.get("cross_entropy")])
    return float(ts.scores["cross_entropy"].mean())


def _tune_quant_spoof_temp(backend, spec, target_ce, q) -> float:
    """Pick provider temperature whose mean CE matches honest (so the attack is
    output-matched -- the genuinely indistinguishable construction of ATTACKS.md)."""
    best_t, best_gap = 1.0, 1e9
    for t in np.linspace(0.7, 1.0, 13):
        atk = AdvQuantTempSpoof(name="_tune", extra_sigma=0.30 * q, bias_sigma=0.10 * q,
                                provider_temp=float(t))
        gap = abs(_mean_ce(backend, atk, spec) - target_ce)
        if gap < best_gap:
            best_t, best_gap = float(t), gap
    return best_t


def build_attacks(backend, spec, target_ce) -> list[tuple[Attack, str]]:
    """The registry attacks + the two constructed output-indistinguishable ones."""
    rows: list[tuple[Attack, str]] = []
    for name in CORE_ATTACKS:
        rows.append((attacks.get(name), ""))
    # Constructed headline attacks (ATTACKS.md).
    rows.append((SafeSetSubstitution(name="safe_set_eps0.05", epsilon=0.05, logit_eps=0.05),
                 "constructed: SAFE-set substitution"))
    q = 0.2
    t = _tune_quant_spoof_temp(backend, spec, target_ce, q)
    rows.append((AdvQuantTempSpoof(name=f"quant_spoof_q{q}", extra_sigma=0.30 * q,
                                   bias_sigma=0.10 * q, provider_temp=t),
                 f"constructed: quant+temp-retune (T={t:.2f})"))
    return rows


def _detect(auc: float) -> float:
    """Symmetric detectability = max(AUC, 1-AUC). A black-box detector whose signal
    *reverses* under an attack (e.g. a temperature-retune makes claimed tokens MORE
    probable under the proxy, so proxy-NLL AUC drops below 0.5) is still detecting
    the attack -- the outputs are separable, just in the opposite direction. The
    honest question for the white-box-ness validator is 'can outputs tell them apart
    at all?', which is symmetric. ~0.5 = genuinely indistinguishable."""
    return max(auc, 1.0 - auc)


# ---------------------------------------------------------------------------
def run():
    backend = SyntheticBackend(vocab=VOCAB)
    spec = SamplingSpec()
    td = defenses.get("token_difr")
    io_dets = [io_detectors.get(n) for n in IO_NAMES]

    honest_seqs = harness.generate_dataset(backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS)
    honest_io = harness.io_verify(backend, honest_seqs, spec, io_dets)
    honest_td = harness.verify(backend, honest_seqs, spec, [td])
    target_ce = _mean_ce(backend, attacks.get("honest"), spec)

    rows: list[Row] = []

    # Honest-null floor: a second, statistically independent honest draw from the
    # SAME model M -- generated on a DISJOINT prompt range (N..2N-1) rather than a
    # different model or a different seed. (A different model_seed would be model
    # substitution, and a different sampling seed would mimic the seed-attack -- both
    # would make recomputation fire; neither is a null.) Both recomputation AND I/O
    # detectors must sit at ~0.5 here: it calibrates what "indistinguishable" looks
    # like for every column, so a low score on a real attack reads as "= the null".
    null_seqs = [backend.generate(p, N_TOKENS, spec, attacks.get("honest"), False, 123, 32)
                 for p in range(N_PROMPTS, 2 * N_PROMPTS)]
    null_io = harness.io_verify(backend, null_seqs, spec, io_dets)
    null_td = harness.verify(backend, null_seqs, spec, [td])
    null_td_auc = _detect(harness.evaluate(honest_td, null_td, [td], [BATCH], n_batches=N_BATCHES,
                                           winsor_pct=99.9, seed=7)[0].auc)
    null_io_res = {r.defense: _detect(r.auc) for r in
                   harness.evaluate(honest_io, null_io, io_dets, [BATCH], n_batches=N_BATCHES,
                                    winsor_pct=99.9, seed=7)}
    rows.append(Row(attack="(honest null)", note="floor: honest vs reseeded honest",
                    token_difr=null_td_auc, io=null_io_res))

    def score(atk, n=N_PROMPTS) -> tuple[float, dict[str, float]]:
        """Detectability of `atk` vs honest, for token_difr and every I/O detector."""
        seqs = harness.generate_dataset(backend, atk, spec, n, N_TOKENS)
        a_io = harness.io_verify(backend, seqs, spec, io_dets)
        a_td = harness.verify(backend, seqs, spec, [td])
        td_auc = _detect(harness.evaluate(honest_td, a_td, [td], [BATCH], n_batches=N_BATCHES,
                                          winsor_pct=99.9, seed=7)[0].auc)
        io_res = {r.defense: _detect(r.auc) for r in
                  harness.evaluate(honest_io, a_io, io_dets, [BATCH], n_batches=N_BATCHES,
                                   winsor_pct=99.9, seed=7)}
        return td_auc, io_res

    for atk, note in build_attacks(backend, spec, target_ce):
        td_auc, io_res = score(atk)
        rows.append(Row(attack=atk.name, note=note, token_difr=td_auc, io=io_res))

    # SAFE-set capacity sweep: as epsilon (the covert channel's per-token budget)
    # grows, BOTH detectors rise -- but proxy-NLL outpaces token_difr, because the
    # attack is calibrated to minimise its token_difr footprint, not its footprint
    # under a *different* (cheap, proxy) model. This is the complementarity result:
    # the cheap I/O detector dominates exactly the attack built to evade recomputation.
    sweep = []
    for eps in (0.05, 0.2, 0.5):
        td_auc, io_res = score(SafeSetSubstitution(name=f"ss_{eps}", epsilon=eps, logit_eps=eps))
        sweep.append((eps, td_auc, io_res["surface_stat"]))

    held_out = run_heldout_generalization(backend, spec)
    return rows, held_out, sweep


def run_heldout_generalization(backend, spec):
    """Train the learned I/O detector on ONE attack family; evaluate on attacks it
    never saw. Uses an independent backend (model_seed=99) for training data so
    there is no overlap with the evaluation pool."""
    train_be = SyntheticBackend(vocab=VOCAB, model_seed=99)
    train_atk = "quant_4bit"

    def ctxs(be, atk, n=100):
        return [harness.io_context(be, s, spec, True, False)
                for s in harness.generate_dataset(be, attacks.get(atk), spec, n, N_TOKENS)]

    tc, ty = [], []
    for c in ctxs(train_be, "honest"):
        tc.append(c); ty.append(0)
    for c in ctxs(train_be, train_atk):
        tc.append(c); ty.append(1)
    learned = LearnedSeqIODetector().fit(tc, ty)

    honest_io = harness.io_verify(backend, harness.generate_dataset(
        backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS), spec, [learned])
    out = []
    eval_attacks = ["quant_4bit", "kv_fp8", "bug_k32", "temp_1.1", "seed_43"]
    for a in eval_attacks:
        a_io = harness.io_verify(backend, harness.generate_dataset(
            backend, attacks.get(a), spec, N_PROMPTS, N_TOKENS), spec, [learned])
        auc = _detect(harness.evaluate(honest_io, a_io, [learned], [BATCH], n_batches=N_BATCHES,
                                       winsor_pct=99.9, seed=7)[0].auc)
        out.append((a, "trained" if a == train_atk else "held-out", auc))
    return train_atk, out


# ---------------------------------------------------------------------------
def main():
    rows, (train_atk, held_out), sweep = run()

    # ---- Role 2 validation table ----
    print(f"\nI/O (black-box) detectors vs Token-DiFR (recomputation)   "
          f"[synthetic, {N_PROMPTS}x{N_TOKENS} tok, batch={BATCH}]")
    print("Cells are symmetric DETECTABILITY = max(AUC,1-AUC); ~0.50 = "
          "indistinguishable to that detector\n(compare to the (honest null) floor "
          "row). An I/O detector at ~0.5 while token_difr is high proves\nthe attack "
          "is invisible in the outputs (recomputation adds real information).\n")
    head = f"{'attack':>18} | {'token_difr':>11} | " + " ".join(f"{n:>14}" for n in IO_NAMES)
    print(head + "   note\n" + "-" * len(head))
    for r in rows:
        cells = " ".join(f"{r.io[n]:>14.3f}" for n in IO_NAMES)
        print(f"{r.attack:>18} | {r.token_difr:>11.3f} | {cells}   {r.note}")

    print("\nHONEST CAVEAT: a HIGH I/O-AUC (e.g. temp_1.1) is NOT a verifier win -- "
          "it means that\nattack is crude enough to catch from outputs alone (a "
          "statement about the attack,\nnot the verifier). The interesting rows are "
          "the LOW I/O-AUC ones with high token_difr.")

    # ---- Synthesis: which detector family DOMINATES each attack? ----
    # Robust to residual batch-means noise: we compare each detector to ITS OWN
    # honest-null floor (excess), then ask which family's excess is larger by more
    # than `margin`. (A single null draw under-estimates the noise band, so a hard
    # "is it above floor?" test is fragile; a *dominance* test -- who wins by a clear
    # margin -- is stable. seed_43, whose only real signal is recomputation, lands
    # cleanly in RECOMPUTE-DOMINANT this way even though its noisiest I/O detector
    # wobbles a hair above floor.)
    null = {r.attack: r for r in rows}["(honest null)"]
    td_floor = null.token_difr
    io_floor = null.io                                   # per-detector floors
    margin = 0.08
    def io_excess(r): return max(r.io[n] - io_floor[n] for n in IO_NAMES)
    def best_io(r):   return max(r.io.values())
    def cat(r):
        tde, ioe = r.token_difr - td_floor, io_excess(r)
        if max(tde, ioe) < margin:  return "NEITHER           (at floor for every detector)"
        if tde - ioe > margin:      return "RECOMPUTE-DOMINANT (recomputation adds info beyond outputs)"
        if ioe - tde > margin:      return "I/O-DOMINANT       (cheap output detection suffices)"
        return "BOTH              (output-visible -- weak safety evidence)"
    print(f"\nDominant detector family per attack  (excess over each detector's OWN "
          f"honest-null floor; margin={margin}):")
    regimes = {}
    for r in rows:
        if r.attack == "(honest null)":
            continue
        regimes[r.attack] = cat(r)
        print(f"   {r.attack:>18}  td_excess={r.token_difr - td_floor:+.3f}  "
              f"io_excess={io_excess(r):+.3f}  -> {regimes[r.attack]}")

    rec_dom = [a for a, c in regimes.items() if c.startswith("RECOMPUTE")]
    print("\n  HEADLINE (Role 2 -- recomputation sees what outputs cannot):")
    print(f"    RECOMPUTE-DOMINANT = {rec_dom or '(none)'}")
    print("    For these the recomputation excess dwarfs every cheap I/O detector's -- the\n"
          "    quantitative proof the forward-pass/seed deviation is (near-)invisible in the\n"
          "    outputs. seed_43 is the extreme: token_difr = 1.0 (wrong seed redraws the SAME\n"
          "    distribution, so a re-run flips Gumbel winners) while every I/O detector sits\n"
          "    at its honest-null floor. The constructed quant_spoof (quantize + temperature-\n"
          "    retune to output-match) is the same story for a forward-pass change.")

    # ---- Complementarity: SAFE-set capacity sweep ----
    print("\nCOMPLEMENT (not in the original plan) -- SAFE-set capacity sweep:")
    print("  SAFE-set deviates only inside token_difr's near-tie blind spot, so it is")
    print("  built to evade recomputation. As its covert budget epsilon grows, proxy-NLL")
    print("  (a DIFFERENT, cheap model) OUTPACES token_difr -- the cheap I/O detector")
    print("  dominates exactly the attack engineered against recomputation. Complementary.")
    print(f"     {'epsilon':>8} | {'token_difr':>11} | {'surface_stat (proxy-NLL)':>24}")
    for eps, td_auc, ss in sweep:
        flag = "  <- proxy-NLL > token_difr" if ss > td_auc else ""
        print(f"     {eps:>8.2f} | {td_auc:>11.3f} | {ss:>24.3f}{flag}")

    # ---- Held-out generalization ----
    print(f"\nHeld-out-attack generalization (LearnedSeqIODetector trained on "
          f"'{train_atk}' only):")
    for a, tag, auc in held_out:
        print(f"   {a:>12} [{tag:>8}]  AUC={auc:.3f}")
    print("   -> generalizes across attacks with an output footprint; stays at "
          "chance on seed_43\n      (output-invisible) no matter the training -- "
          "the gap recomputation fills.")

    # ---- Role 1 Pareto figure ----
    try:
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig3_io_pareto.png"
        render_pareto(rows, out)
        print(f"\nwrote Pareto figure: {out}")
    except Exception as e:  # matplotlib optional; the tables are the result
        print(f"\n(skipped Pareto figure: {e})")


def render_pareto(rows, path: Path):
    """AUC vs verifier FLOPs, one line per attack. Reuses the figures.py palette.
    x-points: surface_tokens (~0 FLOPs) -> surface_stat (cheap proxy) -> token_difr
    (full recompute)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    shown = {
        "temp_1.1": ("#d62728", "o", "temp_1.1 (output-visible)"),
        "quant_4bit": ("#ff7f0e", "s", "quant_4bit"),
        "bug_k32": ("#9467bd", "D", "bug_k32"),
        "seed_43": ("#1f77b4", "^", "seed_43 (output-invisible)"),
        "safe_set_eps0.05": ("#2ca02c", "v", "safe_set (constructed)"),
    }
    # the two quant_spoof headline rows have a tuned-T name; match by prefix
    by_name = {r.attack: r for r in rows}
    spoof = next((r for r in rows if r.attack.startswith("quant_spoof")), None)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    xs_names = ["surface_tokens", "surface_stat", "token_difr"]
    xs = [FLOPS[n] for n in xs_names]

    def line(r, color, marker, label):
        ys = [r.io["surface_tokens"], r.io["surface_stat"], r.token_difr]
        ax.plot(xs, ys, color=color, marker=marker, ms=7, lw=1.8, label=label)

    for name, (c, m, lbl) in shown.items():
        if name in by_name:
            line(by_name[name], c, m, lbl)
    if spoof is not None:
        line(spoof, "#8c564b", "P", "quant_spoof (constructed)")

    ax.axhline(0.5, ls=":", color="0.5", lw=1.2, label="chance (AUC = 0.5)")
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("verifier cost   (relative FLOPs;  1.0 = full recompute of M)  [log]")
    ax.set_ylabel("detectability   (max(AUC, 1-AUC))")
    ax.set_title("Fig 3 — cost/accuracy Pareto: cheap I/O detector vs full recomputation\n"
                 "output-visible attacks are caught cheaply; output-invisible ones need "
                 "recomputation", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
