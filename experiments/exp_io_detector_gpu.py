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
  * the **`logit_judge`** -- recompute on the cheap proxy, then a Claude judge
    over the per-token surprisal/rank divergence (logit space, not text).

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

from ivgym import attacks, harness, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend, DEFAULT_PROMPTS
from ivgym.core import SamplingSpec
from ivgym.verifiers import LLMJudge, LogitJudge

MODEL = os.environ.get("IVGYM_MODEL", "Qwen/Qwen3-0.6B")
# Option B: a REAL cheap proxy model for the black-box detectors. When set, the
# backend loads it as a second network and `proxy_logits` is its genuine forward
# pass (not a noised read of M). Use a same-family proxy so the tokenizer/vocab
# match M, e.g. IVGYM_MODEL=Qwen/Qwen3-8B IVGYM_PROXY_MODEL=Qwen/Qwen3-0.6B.
PROXY_MODEL = os.environ.get("IVGYM_PROXY_MODEL") or None
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 12))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 64))
BATCH = int(os.environ.get("IVGYM_BATCH", 100))
N_BATCHES = 2000
RUN_JUDGE = os.environ.get("IVGYM_JUDGE", "1") != "0"
JUDGE_MODEL = os.environ.get("IVGYM_JUDGE_MODEL", "claude-sonnet-4-6")

# The paper's canonical attacks (same set exp_gpu sweeps).
CORE_ATTACKS = ("quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32")
# Cheap, no-fit I/O detectors that run on any backend.
IO_NAMES = ["surface_stat", "surface_rank", "surface_tokens"]
# The Pareto x-axis is the verifier's MEASURED wall-clock cost (see `measured_costs`),
# not the detectors' declared `verifier_flops` -- those static ratios are no longer
# read here.


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


def _judge_available(det, label):
    """True if an Anthropic-backed judge `det` can actually run (API key + sdk)."""
    if det._api_key() is None:
        print(f"  ({label}: no ANTHROPIC_API_KEY / key helper found -- skipping)",
              flush=True)
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        print(f"  ({label}: `pip install anthropic` to enable -- skipping)", flush=True)
        return False
    return True


def _maybe_judge():
    """Build the llm_judge if requested and an API key resolves; else None."""
    if not RUN_JUDGE:
        return None
    judge = LLMJudge(model=JUDGE_MODEL)
    return judge if _judge_available(judge, "llm_judge") else None


def _maybe_logit_judge():
    """Build the logit_judge (recompute-on-proxy + LLM judges the logit-level
    divergence) if requested and an API key resolves; else None. Unlike llm_judge
    it needs no text backend -- its evidence is per-token surprisal/rank off the
    cheap proxy -- but it does need proxy logits, which every HF-GPU backend
    provides (real proxy when IVGYM_PROXY_MODEL is set, else the noised fallback)."""
    if not RUN_JUDGE:
        return None
    judge = LogitJudge(model=JUDGE_MODEL)
    return judge if _judge_available(judge, "logit_judge") else None


def run():
    if 2 * N_PROMPTS > len(DEFAULT_PROMPTS):
        print(f"  NOTE: 2*PROMPTS ({2*N_PROMPTS}) > prompt bank ({len(DEFAULT_PROMPTS)}); "
              "honest and null pools will share some prompt text.", flush=True)

    t0 = time.time()
    print(f"loading {MODEL}"
          + (f"  + proxy {PROXY_MODEL}" if PROXY_MODEL else "") + " ...", flush=True)
    backend = HFGPUBackend(model_name=MODEL, proxy_model_name=PROXY_MODEL)
    print(f"loaded in {time.time()-t0:.1f}s | vocab={backend.vocab} "
          f"hidden={backend.hidden_dim} | {N_PROMPTS} prompts x {N_TOKENS} tokens",
          flush=True)
    if PROXY_MODEL:
        # Param ratio, reported as a cross-check on the MEASURED proxy/recompute time
        # ratio printed at the end (the figure x-axis itself is measured seconds now).
        ratio = backend.proxy_n_params / backend.n_params
        print(f"  REAL proxy: {backend.proxy_n_params/1e9:.2f}B params vs M "
              f"{backend.n_params/1e9:.2f}B  ->  proxy/M param ratio = {ratio:.3f} "
              f"({1/ratio:.1f}x fewer params)", flush=True)

    spec = SamplingSpec()
    td = verifiers.get("token_difr")
    judge = _maybe_judge()
    logit_judge = _maybe_logit_judge()
    io_dets = ([verifiers.get(n) for n in IO_NAMES]
               + ([judge] if judge else [])
               + ([logit_judge] if logit_judge else []))
    io_cols = (IO_NAMES + (["llm_judge"] if judge else [])
               + (["logit_judge"] if logit_judge else []))

    def score_pool(seqs):
        """Detectability arrays for one already-generated sequence pool: token_difr
        (recompute) scores and I/O-detector scores, materialised immediately while
        this config's reference cache is still fresh (the HF backend keys its cache
        by prompt_id and overwrites it on the next generate)."""
        return (harness.verify(backend, seqs, spec, [td]),
                harness.verify(backend, seqs, spec, io_dets))

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

    return rows, io_cols, time.time() - t0, measured_costs(backend), compute_flops(backend, backend.max_prompt_tokens, N_TOKENS)


def measured_costs(backend) -> dict[str, float]:
    """Real, run-specific verifier cost per detector: mean GPU-synchronised
    wall-clock SECONDS to produce that detector's evidence over one [prompt+claimed]
    sequence. token_difr = the full-M reference prefill (the recompute); surface_stat
    /surface_rank = the cheap-proxy prefill; surface_tokens = the decode (no forward).
    These replace the old param-count ratio on the Pareto x-axis with measured time."""
    sec, n = backend.timed_seconds, backend.timed_calls
    mean = {k: sec[k] / max(n[k], 1) for k in sec}
    # No real proxy loaded -> proxy_logits is a noised cache read (no forward pass),
    # so the cheap-proxy detector's measured cost collapses to its decode/text work.
    proxy = mean["proxy"] if n["proxy"] else mean["decode"]
    return {
        "token_difr": mean["reference"],
        "surface_stat": proxy,
        "surface_rank": proxy,
        "surface_tokens": mean["decode"],
    }


def compute_flops(backend, n_prompt_tokens: int, n_cont_tokens: int) -> dict[str, float]:
    """FLOPs per sequence for each verifier, using the standard 2*N_non_embed*T approx.

    N_non_embed excludes embedding lookup tables (pure index reads, not matmuls).
    T = n_prompt_tokens + n_cont_tokens (the full prefill length).
    surface_tokens does no neural forward pass; assigned a nominal 1e6 FLOPs so it
    renders as the leftmost point on the log-scale x-axis.
    """
    T = n_prompt_tokens + n_cont_tokens
    vocab = backend.vocab
    hidden = backend.hidden_dim
    main_non_embed = max(backend.n_params - vocab * hidden, 0)

    if backend.proxy_model is not None:
        proxy_hidden = int(backend.proxy_model.config.hidden_size)
        proxy_non_embed = max(backend.proxy_n_params - vocab * proxy_hidden, 0)
    else:
        # No real proxy: proxy detectors reuse M's cached logits (no second forward
        # pass), so proxy FLOPs collapse to zero.
        proxy_non_embed = 0

    return {
        "token_difr":    2 * main_non_embed  * T,
        "surface_stat":  2 * proxy_non_embed * T,
        "surface_rank":  2 * proxy_non_embed * T,
        "surface_tokens": 1e6,    # no model forward pass; nominal floor for log scale
    }


def main():
    rows, io_cols, elapsed, cost, flops = run()

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
    if any(c in io_cols for c in ("llm_judge", "logit_judge", "surface_tokens")):
        print("\nNOTE: per-SEQUENCE-constant detectors (llm_judge, logit_judge, surface_tokens) "
              "emit one value per\nsequence broadcast to its tokens, so with few sequences their "
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

    # ---- measured verifier cost (real wall-clock) ----
    print("\nMEASURED verifier cost  (mean GPU-synchronised seconds per [prompt+claimed] "
          "sequence):")
    for n in ("surface_tokens", "surface_stat", "token_difr"):
        print(f"   {n:>14}  {cost[n]*1e3:9.3f} ms/seq")
    if cost["token_difr"] > 0:
        print(f"   -> surface_stat is {cost['token_difr']/max(cost['surface_stat'],1e-12):.1f}x "
              f"cheaper than a full recompute (token_difr); surface_tokens "
              f"{cost['token_difr']/max(cost['surface_tokens'],1e-12):.0f}x cheaper.")

    # ---- FLOPs per sequence (2·N_non_embed·T approximation) ----
    print(f"\nVERIFIER FLOPs / sequence  (2·N_non_embed·T, T = prompt + {N_TOKENS} tokens):")
    for n in ("surface_tokens", "surface_stat", "token_difr"):
        f = flops[n]
        if f < 1e9:
            label = f"{f/1e6:.1f} MFLOPs  (nominal; no model forward pass)"
        elif f < 1e12:
            label = f"{f/1e9:.1f} GFLOPs"
        else:
            label = f"{f/1e12:.3f} TFLOPs"
        print(f"   {n:>14}  {label}")
    if flops["token_difr"] > 0 and flops["surface_stat"] > 0:
        print(f"   -> surface_stat is {flops['token_difr']/flops['surface_stat']:.1f}x "
              f"fewer FLOPs than a full recompute (token_difr).")

    # ---- Role 1 Pareto figure ----
    try:
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig3_io_pareto_gpu.png"
        render_pareto(rows, out, cost, flops)
        print(f"\nwrote Pareto figure: {out}")
    except Exception as e:  # matplotlib optional; the tables are the result
        print(f"\n(skipped Pareto figure: {e})")

    print(f"\ntotal {elapsed:.1f}s on {MODEL}")


def render_pareto(rows, path: Path, cost: dict[str, float],
                  flops: dict[str, float] | None = None):
    """Detectability vs verifier cost, one line per attack.

    x-axis: FLOPs/sequence (if `flops` provided) or measured GPU seconds.
    x-points: surface_tokens -> surface_stat (cheap-proxy prefill) ->
    token_difr (full-M recompute).
    """
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

    xs_names = ["surface_tokens", "surface_stat", "token_difr"]
    xs_data = flops if flops is not None else cost
    xs = [xs_data[n] for n in xs_names]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    def line(r, color, marker, label):
        ys = [r.io["surface_tokens"], r.io["surface_stat"], r.token_difr]
        ax.plot(xs, ys, color=color, marker=marker, ms=7, lw=1.8, label=label)

    for name, (c, m, lbl) in shown.items():
        if name in by_name:
            line(by_name[name], c, m, lbl)

    ax.axhline(0.5, ls=":", color="0.5", lw=1.2, label="chance (AUC = 0.5)")
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.03)

    if flops is not None:
        # annotate each x-tick with its GFLOPs value
        ax.set_xticks(xs)
        gf = [f"{x/1e9:.3g} GFLOPs" for x in xs]
        gf[0] = f"{xs[0]/1e6:.0f} MFLOPs\n(nominal)"   # surface_tokens ≈ 0
        ax.set_xticklabels(gf, fontsize=8)
        ax.set_xlabel("verifier cost   (FLOPs / sequence, 2·N·T approx,\n"
                      "N = non-embedding params;  token_difr = full recompute of M)  [log]")
    else:
        ax.set_xlabel("verifier cost   (measured GPU seconds per sequence;  "
                      "token_difr = full recompute of M)  [log]")

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
