"""Robustness sweep for the UNIFIED verification algorithm on real GPUs.

Runs the one unified statistic (`ivgym.verifiers` + `harness.verify`, the
"unify detection algorithms" design) against every registered attack, on a
*matrix of real models* -- multiple model FAMILIES, multiple model SIZES, with
and without a real same-family cheap proxy -- to test how well the detector
behaviour generalises beyond the single Qwen3-0.6B the repo was validated on.

For each (family, size) it reports, on the real model on an H100:

  * full-recompute detection AUC (attack x verifier) for the whole unified
    registry -- Tier-1 recompute verifiers (token_difr / cross_entropy /
    activation_difr / token_toploc) AND Tier-0 no-recompute verifiers
    (surface_stat / surface_rank / surface_tokens / accept_rate);
  * information-directed SELECTIVE recompute (harness.verify at budget<1) at a
    couple of budgets, so the cost-aware tier is checked on every model too.

It then writes a cross-model synthesis: does "token_difr catches every attack"
survive a change of family / size / tokenizer? which attacks are hardest where?

Run (all models, moderate pool ~clean AUCs):
    .venv/bin/python -m experiments.exp_robustness_gpu

Env overrides:
    IVGYM_PROMPTS, IVGYM_TOKENS, IVGYM_BATCH, IVGYM_NBATCHES
    IVGYM_SELECTIVE   comma list of budgets, e.g. "0.25,0.125"  ("" = skip)
    IVGYM_MODELS      comma list of tags to include (default: all below)
    IVGYM_MAX_PARAMS  skip any reference model above this many params (e.g. 5e9)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ivgym import attacks, harness, verifiers  # noqa: E402
from ivgym.backends.hf_gpu import HFGPUBackend  # noqa: E402
from ivgym.core import SamplingSpec  # noqa: E402
from ivgym.model_registry import REGISTRY  # noqa: E402

# ---------------------------------------------------------------------------
# The model matrix: families x sizes. All ungated on the HF hub. Within a family
# every model shares one tokenizer/vocab, so the smaller one is a valid cheap
# proxy for the black-box (Tier-0) verifiers and the selective-recompute triage.
# tag = "<family>-<size>". `proxy` must share the reference's vocab (same family).
# ---------------------------------------------------------------------------
MODEL_MATRIX = [
    # --- Qwen3 (the repo's validated family) : size ladder + proxy pairs ------
    dict(tag="qwen3-0.6b", ref="Qwen/Qwen3-0.6B", proxy=None,               params=0.6e9),
    dict(tag="qwen3-1.7b", ref="Qwen/Qwen3-1.7B", proxy="Qwen/Qwen3-0.6B",  params=1.7e9),
    dict(tag="qwen3-4b",   ref="Qwen/Qwen3-4B",   proxy="Qwen/Qwen3-0.6B",  params=4.0e9),
    # --- Llama 3.2 (ungated unsloth mirror): different family/tokenizer ------
    dict(tag="llama3.2-1b", ref="unsloth/Llama-3.2-1B-Instruct", proxy=None, params=1.2e9),
    dict(tag="llama3.2-3b", ref="unsloth/Llama-3.2-3B-Instruct",
         proxy="unsloth/Llama-3.2-1B-Instruct", params=3.2e9),
    # --- SmolLM2 (Llama-arch, GPT tokenizer): small size ladder -------------
    dict(tag="smollm2-360m", ref="HuggingFaceTB/SmolLM2-360M-Instruct",
         proxy="HuggingFaceTB/SmolLM2-135M-Instruct", params=0.36e9),
    dict(tag="smollm2-1.7b", ref="HuggingFaceTB/SmolLM2-1.7B-Instruct",
         proxy="HuggingFaceTB/SmolLM2-135M-Instruct", params=1.7e9),
    # --- Pythia (GPT-NeoX, a genuinely different architecture) --------------
    dict(tag="pythia-410m", ref="EleutherAI/pythia-410m",
         proxy="EleutherAI/pythia-160m", params=0.41e9),
    dict(tag="pythia-1.4b", ref="EleutherAI/pythia-1.4b",
         proxy="EleutherAI/pythia-160m", params=1.4e9),
]

ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32",
           "adv_quant_temp"]

N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 16))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 400))
N_BATCHES = int(os.environ.get("IVGYM_NBATCHES", 2000))
_sel_env = os.environ.get("IVGYM_SELECTIVE", "0.25,0.125")
SELECTIVE = [float(x) for x in _sel_env.split(",") if x.strip()]
VALUE_FN = os.environ.get("IVGYM_VALUE_FN", "entropy")
MAX_PARAMS = float(os.environ.get("IVGYM_MAX_PARAMS", 0)) or None

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"
OUT_JSON = OUT_DIR / "robustness_sweep.json"
OUT_MD = OUT_DIR / "robustness_sweep.md"


def _pick_models():
    want = os.environ.get("IVGYM_MODELS")
    matrix = MODEL_MATRIX
    if want:
        keep = {t.strip() for t in want.split(",") if t.strip()}
        matrix = [m for m in matrix if m["tag"] in keep]
    if MAX_PARAMS:
        matrix = [m for m in matrix if m["params"] <= MAX_PARAMS]
    return matrix


def _free(backend):
    try:
        del backend.model
        if getattr(backend, "proxy_model", None) is not None:
            del backend.proxy_model
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_one_model(cfg) -> dict:
    """Run the full unified registry + selective budgets against every attack on
    one real model. Returns a nested result dict (or {'error': ...})."""
    tag, ref, proxy = cfg["tag"], cfg["ref"], cfg["proxy"]
    t0 = time.time()
    print(f"\n{'='*78}\n[{tag}] loading ref={ref}  proxy={proxy or '(noised-M fallback)'}",
          flush=True)
    bkw = {"model_name": ref}
    if proxy:
        bkw["proxy_model_name"] = proxy
    backend = HFGPUBackend(**bkw)
    print(f"[{tag}] loaded in {time.time()-t0:.1f}s | vocab={backend.vocab} "
          f"hidden={backend.hidden_dim} params={backend.n_params/1e9:.2f}B"
          + (f" proxy_params={backend.proxy_n_params/1e9:.2f}B" if backend.proxy_n_params else ""),
          flush=True)

    defs = list(verifiers.all_verifiers().values())
    def_names = [d.name for d in defs]
    needs_act = any(d.needs_activation for d in defs)
    spec = SamplingSpec()

    def gen(atk):
        return harness.generate_dataset(backend, atk, spec, N_PROMPTS, N_TOKENS,
                                        record_activations=needs_act)

    # honest reference: full + one selective ranking reused across budgets
    honest_seqs = gen(attacks.get("honest"))
    honest_full = harness.verify(backend, honest_seqs, spec, defs)
    honest_sel = {}
    realized = {}
    if SELECTIVE:
        h_val = harness.token_values(backend, honest_seqs, spec, VALUE_FN)
        for b in SELECTIVE:
            ts = harness.verify(backend, honest_seqs, spec, defs, budget=b,
                                value_fn=VALUE_FN, values=h_val)
            honest_sel[b] = ts
            realized[b] = ts.recompute_ratio

    result = {
        "tag": tag, "ref": ref, "proxy": proxy,
        "params": backend.n_params,
        "proxy_params": backend.proxy_n_params,
        "vocab": backend.vocab, "hidden": backend.hidden_dim,
        "verifiers": def_names,
        "realized_recompute_ratio": realized,
        "full": {},           # attack -> verifier -> auc
        "selective": {b: {} for b in SELECTIVE},
    }

    for aname in ATTACKS:
        atk = attacks.get(aname)
        seqs = gen(atk)
        a_full = harness.verify(backend, seqs, spec, defs)
        res = harness.evaluate(honest_full, a_full, defs, [BATCH],
                               n_batches=N_BATCHES, winsor_pct=99.9)
        result["full"][aname] = {r.defense: r.auc for r in res}
        if SELECTIVE:
            a_val = harness.token_values(backend, seqs, spec, VALUE_FN)
            for b in SELECTIVE:
                a_ts = harness.verify(backend, seqs, spec, defs, budget=b,
                                      value_fn=VALUE_FN, values=a_val)
                res_b = harness.evaluate(honest_sel[b], a_ts, defs, [BATCH],
                                         n_batches=N_BATCHES, winsor_pct=99.9)
                result["selective"][b][aname] = {r.defense: r.auc for r in res_b}
        print(f"[{tag}] {aname:>14}  "
              + "  ".join(f"{d}={result['full'][aname][d]:.3f}"
                          for d in ("token_difr", "token_toploc", "accept_rate")
                          if d in result["full"][aname]),
              flush=True)

    result["seconds"] = time.time() - t0
    _print_model_table(result)
    _free(backend)
    return result


def _print_model_table(result):
    defs = result["verifiers"]
    header = f"{'attack':>15} | " + " ".join(f"{d:>15}" for d in defs)

    def tbl(title, rows):
        print(f"\n{title}\n{header}\n{'-'*len(header)}")
        for a in ATTACKS:
            if a not in rows:
                continue
            print(f"{a:>15} | " + " ".join(f"{rows[a][d]:>15.4f}" for d in defs))

    tag = result["tag"]
    tbl(f"[{tag}] FULL-recompute AUC @ batch={BATCH} ({N_PROMPTS}x{N_TOKENS} tok)",
        result["full"])
    for b in SELECTIVE:
        rr = result["realized_recompute_ratio"].get(b, b)
        tbl(f"[{tag}] SELECTIVE budget={b:g} (realized ratio {rr:.3f}) value={VALUE_FN}",
            result["selective"][b])


def synthesize(results: list[dict]) -> str:
    """Cross-model robustness synthesis in markdown."""
    ok = [r for r in results if "error" not in r]
    lines = []
    lines.append("# Unified verification algorithm - robustness sweep\n")
    lines.append(f"- config: {N_PROMPTS} prompts x {N_TOKENS} tokens, batch={BATCH}, "
                 f"n_batches={N_BATCHES}, value_fn={VALUE_FN}, selective={SELECTIVE}")
    lines.append(f"- models attempted: {len(results)}, succeeded: {len(ok)}\n")

    # model roster. `family`/`org` are read from ivgym/model_registry.py when the
    # reference model has a taxonomy entry there (soft lookup: this matrix is
    # allowed to include models nobody has added facts for yet).
    def _fam_org(ref):
        m = REGISTRY.get(ref)
        return (m.family, m.org) if m else ("?", "?")

    lines.append("## Models\n")
    lines.append("| tag | reference | family | org | params | proxy | proxy params | vocab |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['tag']} | {r['ref']} | - | - | - | - | - | FAILED: {r['error'][:60]} |")
            continue
        fam, org = _fam_org(r["ref"])
        pp = f"{r['proxy_params']/1e9:.2f}B" if r["proxy_params"] else "-"
        lines.append(f"| {r['tag']} | {r['ref']} | {fam} | {org} | {r['params']/1e9:.2f}B | "
                     f"{r['proxy'] or '(noised-M)'} | {pp} | {r['vocab']} |")
    lines.append("")

    if not ok:
        return "\n".join(lines)

    verifier_names = ok[0]["verifiers"]
    tier1 = {"token_difr", "cross_entropy", "activation_difr", "token_toploc"}

    # Per-verifier: mean / min AUC across all (model, attack) full cells.
    lines.append("## Per-verifier AUC across ALL (model x attack) full-recompute cells\n")
    lines.append("| verifier | mean AUC | min AUC | worst (model/attack) | cells |")
    lines.append("|---|---|---|---|---|")
    for d in verifier_names:
        cells = []
        for r in ok:
            for a in ATTACKS:
                if d in r["full"].get(a, {}):
                    cells.append((r["full"][a][d], r["tag"], a))
        if not cells:
            continue
        aucs = np.array([c[0] for c in cells])
        worst = min(cells, key=lambda c: c[0])
        lines.append(f"| {d} | {aucs.mean():.3f} | {worst[0]:.3f} | "
                     f"{worst[1]}/{worst[2]} | {len(cells)} |")
    lines.append("")

    # token_difr headline: is it >=0.95 everywhere?
    lines.append("## Headline check: does `token_difr` catch every attack on every model?\n")
    lines.append("| model | " + " | ".join(ATTACKS) + " | min |")
    lines.append("|---|" + "|".join(["---"] * (len(ATTACKS) + 1)) + "|")
    for r in ok:
        row = [r["full"].get(a, {}).get("token_difr", float("nan")) for a in ATTACKS]
        mn = np.nanmin(row)
        lines.append(f"| {r['tag']} | " + " | ".join(f"{v:.3f}" for v in row)
                     + f" | **{mn:.3f}** |")
    lines.append("")

    # Per-attack difficulty: mean token_difr AUC across models (lower = harder).
    lines.append("## Attack difficulty (mean `token_difr` AUC across models; lower = harder)\n")
    lines.append("| attack | mean token_difr AUC | min | max |")
    lines.append("|---|---|---|---|")
    diffs = []
    for a in ATTACKS:
        vals = [r["full"][a]["token_difr"] for r in ok
                if "token_difr" in r["full"].get(a, {})]
        if vals:
            diffs.append((a, float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))))
    for a, mean, mn, mx in sorted(diffs, key=lambda x: x[1]):
        lines.append(f"| {a} | {mean:.3f} | {mn:.3f} | {mx:.3f} |")
    lines.append("")

    # Selective tier: token_difr full vs selective, averaged over models/attacks.
    if SELECTIVE:
        lines.append("## Information-directed selective recompute (token_difr, mean over models x attacks)\n")
        lines.append("| budget | realized ratio | mean AUC | vs full |")
        lines.append("|---|---|---|---|")
        full_vals = [r["full"][a]["token_difr"] for r in ok for a in ATTACKS
                     if "token_difr" in r["full"].get(a, {})]
        full_mean = float(np.mean(full_vals)) if full_vals else float("nan")
        lines.append(f"| 1.0 (full) | 1.000 | {full_mean:.3f} | - |")
        for b in SELECTIVE:
            vals, ratios = [], []
            for r in ok:
                ratios.append(r["realized_recompute_ratio"].get(b, b))
                for a in ATTACKS:
                    cell = r["selective"].get(b, {}).get(a, {})
                    if "token_difr" in cell:
                        vals.append(cell["token_difr"])
            if vals:
                m = float(np.mean(vals))
                lines.append(f"| {b:g} | {np.mean(ratios):.3f} | {m:.3f} | "
                             f"{m-full_mean:+.3f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    models = _pick_models()
    print(f"robustness sweep over {len(models)} models: "
          f"{[m['tag'] for m in models]}", flush=True)
    print(f"config: {N_PROMPTS}x{N_TOKENS} tok, batch={BATCH}, n_batches={N_BATCHES}, "
          f"selective={SELECTIVE}, value_fn={VALUE_FN}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for cfg in models:
        try:
            results.append(run_one_model(cfg))
        except Exception as e:
            print(f"[{cfg['tag']}] FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"tag": cfg["tag"], "ref": cfg["ref"], "error": str(e)})
        # persist incrementally so a crash mid-sweep keeps finished models
        OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
        OUT_MD.write_text(synthesize(results))
        print(f"\n[progress] {len([r for r in results if 'error' not in r])}/"
              f"{len(models)} ok, elapsed {time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 78)
    print(synthesize(results))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
