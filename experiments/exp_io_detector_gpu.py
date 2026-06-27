"""I/O (black-box) detectors vs recomputation on a REAL model on a GPU.

The white-box-ness validator: a recomputation defense (`token_difr`, which re-runs
the reference model M) against output-only `IODetector`s that score from the
prompt + claimed tokens alone. Logits, activations, prompt text and continuations
come from a real LLM (default Qwen/Qwen3-0.6B) on CUDA via `ivgym.backends.hf_gpu`.

The black-box detectors:

  * **proxy-NLL / proxy-rank** read a cheap proxy over the model's *real* logit
    geometry (a noisier estimate of the reference distribution).
  * **surface_tokens** uses pure token-id statistics, no model at all.
  * the **`llm_judge`** -- a zero-shot Claude judge over (prompt, decoded
    continuation) -- which only runs on a real-text backend like this one.

The headline (Role 2): where an attack is output-*indistinguishable* (a wrong
sampling seed; quantization tuned to output-match), every black-box detector --
including a strong LLM judge -- sits at its honest-null floor while `token_difr`
is high. That gap is the quantitative proof recomputation provides information
beyond behaviour.

Run:  .venv/bin/python -m experiments.exp_io_detector_gpu      # ~5 min default
Headline (clean AUCs; ~20 min on an H100):
  IVGYM_PROMPTS=20 IVGYM_TOKENS=128 IVGYM_BATCH=200 \
      .venv/bin/python -m experiments.exp_io_detector_gpu

Env overrides:
  IVGYM_MODEL    HF model id (default Qwen/Qwen3-0.6B)
  IVGYM_PROMPTS  prompts per config (default 12; honest+null use DISJOINT ranges
                 [0,N) and [N,2N), so keep 2*PROMPTS <= the prompt bank to avoid
                 text overlap -- the bank holds 44 prompts, so N<=22)
  IVGYM_TOKENS   tokens per sequence (default 64). token_difr and the I/O null
                 floor both sharpen as the token pool (PROMPTS*TOKENS) grows.
  IVGYM_BATCH    batch size for the batch-level statistic S (default 100)
  IVGYM_JUDGE    1 (default) to run the Claude llm_judge; 0 to skip it
  IVGYM_JUDGE_MODEL  judge model id (default claude-sonnet-4-6)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness, io_detectors
from ivgym.backends.hf_gpu import HFGPUBackend, DEFAULT_PROMPTS
from ivgym.core import SamplingSpec
from ivgym.io_detectors import LLMJudgeIODetector

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 12))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 64))
BATCH = int(os.environ.get("IVGYM_BATCH", 100))
N_BATCHES = 400
RUN_JUDGE = os.environ.get("IVGYM_JUDGE", "1") != "0"
JUDGE_MODEL = os.environ.get("IVGYM_JUDGE_MODEL", "claude-sonnet-4-6")

# The paper's canonical attacks (same set exp_gpu sweeps).
CORE_ATTACKS = ("quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32")
# Cheap, no-fit I/O detectors that run on any backend.
IO_NAMES = ["surface_stat", "surface_rank", "surface_tokens"]
# Relative verifier cost for the Pareto x-axis (1.0 = a full recompute of M).
FLOPS = {
    "token_difr": 1.0,              # re-runs the reference forward pass
    "surface_stat": io_detectors.get("surface_stat").verifier_flops,    # cheap proxy LM
    "surface_rank": io_detectors.get("surface_rank").verifier_flops,
    "surface_tokens": io_detectors.get("surface_tokens").verifier_flops,  # no model
}


@dataclass
class Row:
    attack: str
    note: str
    token_difr: float
    io: dict[str, float] = field(default_factory=dict)


def _detect(auc: float) -> float:
    """Symmetric detectability = max(AUC, 1-AUC). A black-box detector whose signal
    *reverses* under an attack (e.g. a temperature-retune makes claimed tokens MORE
    probable under the proxy, so proxy-NLL AUC drops below 0.5) is still detecting
    the attack -- the outputs are separable, just in the opposite direction. The
    honest question for the white-box-ness validator is 'can outputs tell them apart
    at all?', which is symmetric. ~0.5 = genuinely indistinguishable."""
    return max(auc, 1.0 - auc)


def _maybe_judge():
    """Build the llm_judge if requested and an API key resolves; else None."""
    if not RUN_JUDGE:
        return None
    judge = LLMJudgeIODetector(model=JUDGE_MODEL)
    if judge._api_key() is None:
        print("  (llm_judge: no ANTHROPIC_API_KEY / key helper found -- skipping)",
              flush=True)
        return None
    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("  (llm_judge: `pip install anthropic` to enable -- skipping)", flush=True)
        return None
    return judge


def run():
    if 2 * N_PROMPTS > len(DEFAULT_PROMPTS):
        print(f"  NOTE: 2*PROMPTS ({2*N_PROMPTS}) > prompt bank ({len(DEFAULT_PROMPTS)}); "
              "honest and null pools will share some prompt text.", flush=True)

    t0 = time.time()
    print(f"loading {MODEL} ...", flush=True)
    backend = HFGPUBackend(model_name=MODEL)
    print(f"loaded in {time.time()-t0:.1f}s | vocab={backend.vocab} "
          f"hidden={backend.hidden_dim} | {N_PROMPTS} prompts x {N_TOKENS} tokens",
          flush=True)

    spec = SamplingSpec()
    td = defenses.get("token_difr")
    judge = _maybe_judge()
    io_dets = [io_detectors.get(n) for n in IO_NAMES] + ([judge] if judge else [])
    io_cols = IO_NAMES + (["llm_judge"] if judge else [])

    def score_pool(seqs):
        """Detectability arrays for one already-generated sequence pool: token_difr
        (recompute) scores and I/O-detector scores, materialised immediately while
        this config's reference cache is still fresh (the HF backend keys its cache
        by prompt_id and overwrites it on the next generate)."""
        return (harness.verify(backend, seqs, spec, [td]),
                harness.io_verify(backend, seqs, spec, io_dets))

    def eval_auc(honest_scores, attack_scores, dets):
        return {r.defense: _detect(r.auc) for r in harness.evaluate(
            honest_scores, attack_scores, dets, [BATCH], n_batches=N_BATCHES,
            winsor_pct=99.9, seed=7)}

    # honest reference pool (record activations so token_difr has fingerprints if needed)
    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, N_PROMPTS, N_TOKENS, record_activations=True)
    honest_td, honest_io = score_pool(honest_seqs)
    print(f"  honest pool scored ({time.time()-t0:.1f}s)", flush=True)

    # honest-null floor: a second independent honest draw from the SAME model M on a
    # DISJOINT prompt range (N..2N-1). Both recompute and I/O detectors must sit near
    # 0.5 here -- it calibrates what "indistinguishable" looks like per column.
    null_seqs = [backend.generate(p, N_TOKENS, spec, attacks.get("honest"), True, 123, 32)
                 for p in range(N_PROMPTS, 2 * N_PROMPTS)]
    null_td, null_io = score_pool(null_seqs)
    rows = [Row(attack="(honest null)", note="floor: honest vs reseeded honest",
                token_difr=_detect(eval_auc(honest_td, null_td, [td])["token_difr"]),
                io=eval_auc(honest_io, null_io, io_dets))]
    print(f"  null floor scored ({time.time()-t0:.1f}s)", flush=True)

    for name in CORE_ATTACKS:
        atk = attacks.get(name)
        seqs = harness.generate_dataset(
            backend, atk, spec, N_PROMPTS, N_TOKENS, record_activations=True)
        a_td, a_io = score_pool(seqs)
        rows.append(Row(attack=name, note="",
                        token_difr=_detect(eval_auc(honest_td, a_td, [td])["token_difr"]),
                        io=eval_auc(honest_io, a_io, io_dets)))
        print(f"  {name:>12} scored ({time.time()-t0:.1f}s)", flush=True)

    return rows, io_cols, time.time() - t0


def main():
    rows, io_cols, elapsed = run()

    print(f"\nI/O (black-box) detectors vs Token-DiFR (recomputation)   "
          f"[{MODEL}, {N_PROMPTS}x{N_TOKENS} tok, batch={BATCH}]")
    print("Cells are symmetric DETECTABILITY = max(AUC,1-AUC); ~0.50 = "
          "indistinguishable to that detector\n(compare to the (honest null) floor "
          "row). An I/O detector -- including the LLM judge -- at ~0.5 while\n"
          "token_difr is high proves the attack is invisible in the outputs "
          "(recomputation adds real information).\n")
    head = f"{'attack':>16} | {'token_difr':>11} | " + " ".join(f"{n:>14}" for n in io_cols)
    print(head + "   note\n" + "-" * len(head))
    for r in rows:
        cells = " ".join(f"{r.io[n]:>14.3f}" for n in io_cols)
        print(f"{r.attack:>16} | {r.token_difr:>11.3f} | {cells}   {r.note}")

    print("\nHONEST CAVEAT: a HIGH I/O-AUC (e.g. temp_1.1) is NOT a verifier win -- "
          "it means that\nattack is crude enough to catch from outputs alone (a "
          "statement about the attack,\nnot the verifier). The interesting rows are "
          "the LOW I/O-AUC ones with high token_difr.")
    if "llm_judge" in io_cols or "surface_tokens" in io_cols:
        print("\nNOTE: per-SEQUENCE-constant detectors (llm_judge, surface_tokens) emit one "
              "value per\nsequence broadcast to its tokens, so with few sequences their "
              "token-batch null FLOOR\ninflates well above 0.5 (a known finite-pool artifact). "
              "Read them via the dominance\nblock below -- excess over each detector's OWN "
              "floor -- not via the absolute AUC cell.")

    # ---- dominance synthesis: which detector family wins each attack? ----
    null = {r.attack: r for r in rows}["(honest null)"]
    td_floor, io_floor, margin = null.token_difr, null.io, 0.08

    def io_excess(r):
        return max(r.io[n] - io_floor[n] for n in io_cols)

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
        print(f"   {r.attack:>16}  td_excess={r.token_difr - td_floor:+.3f}  "
              f"io_excess={io_excess(r):+.3f}  -> {regimes[r.attack]}")

    rec_dom = [a for a, c in regimes.items() if c.startswith("RECOMPUTE")]
    print("\n  HEADLINE (Role 2 -- recomputation sees what outputs cannot, on a real model):")
    print(f"    RECOMPUTE-DOMINANT = {rec_dom or '(none)'}")
    print("    For these the recomputation excess dwarfs every cheap I/O detector's AND the\n"
          "    LLM judge's -- a forward-pass/seed deviation that is (near-)invisible in the\n"
          "    real model's outputs. seed_43 is the extreme: wrong seed redraws the SAME\n"
          "    distribution, so a re-run flips Gumbel winners while the text looks honest.")

    # ---- Role 1 Pareto figure ----
    try:
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig3_io_pareto_gpu.png"
        render_pareto(rows, out)
        print(f"\nwrote Pareto figure: {out}")
    except Exception as e:  # matplotlib optional; the tables are the result
        print(f"\n(skipped Pareto figure: {e})")

    print(f"\ntotal {elapsed:.1f}s on {MODEL}")


def render_pareto(rows, path: Path):
    """AUC vs verifier FLOPs, one line per attack.
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
    }
    by_name = {r.attack: r for r in rows}

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    xs_names = ["surface_tokens", "surface_stat", "token_difr"]
    xs = [FLOPS[n] for n in xs_names]

    def line(r, color, marker, label):
        ys = [r.io["surface_tokens"], r.io["surface_stat"], r.token_difr]
        ax.plot(xs, ys, color=color, marker=marker, ms=7, lw=1.8, label=label)

    for name, (c, m, lbl) in shown.items():
        if name in by_name:
            line(by_name[name], c, m, lbl)

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
