"""Post-process the robustness sweep JSON into a richer, orientation-correct
report.

`exp_robustness_gpu.py` stores RAW AUCs. For Tier-1 recompute verifiers a higher
raw AUC is the right orientation (more divergence = attack). For Tier-0 black-box
detectors (surface_*, accept_rate) the signal legitimately REVERSES under some
attacks -- the DiFR/Clymer convention is detectability = max(AUC, 1-AUC). This
script applies that orientation, and adds the cross-cutting views the flat
synthesis does not: size trend within a family, family comparison, and the
full-vs-selective gain on the hard attacks.

    .venv/bin/python -m experiments.analyze_robustness            # reads default json
    .venv/bin/python -m experiments.analyze_robustness path.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ATTACKS = ["quant_4bit", "kv_fp8", "temp_1.1", "seed_43", "bug_k2", "bug_k32",
           "adv_quant_temp"]
TIER1 = {"token_difr", "cross_entropy", "activation_difr", "token_toploc"}
DEFAULT = Path(__file__).resolve().parents[1] / "docs" / "results" / "robustness_sweep.json"


def detect(auc: float, verifier: str) -> float:
    """Orientation-correct detectability: raw AUC for Tier-1, max(auc,1-auc) for
    Tier-0 (a reversed black-box signal still separates)."""
    if verifier in TIER1:
        return auc
    return max(auc, 1.0 - auc)


def family_of(tag: str) -> str:
    return tag.rsplit("-", 1)[0]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    data = json.loads(Path(path).read_text())
    ok = [r for r in data if "error" not in r]
    if not ok:
        print("no successful models in", path)
        return
    verifiers = ok[0]["verifiers"]

    out = []
    P = out.append
    P("# Unified verification algorithm — robustness across families, sizes, attacks\n")
    P(f"Source: `{path}`  •  {len(ok)} models succeeded"
      + (f", {len(data)-len(ok)} failed" if len(data) > len(ok) else "") + ".\n")

    # -- roster -------------------------------------------------------------
    P("## Model matrix\n")
    P("| model | ref params | proxy | proxy params | vocab |")
    P("|---|---|---|---|---|")
    for r in ok:
        pp = f"{r['proxy_params']/1e9:.2f}B" if r.get("proxy_params") else "—"
        P(f"| `{r['tag']}` | {r['params']/1e9:.2f}B | "
          f"{r['proxy'] or '(noised-M)'} | {pp} | {r['vocab']} |")
    for r in data:
        if "error" in r:
            P(f"| `{r['tag']}` | — | — | — | FAILED: {r['error'][:50]} |")
    P("")

    # -- token_difr headline heatmap (raw AUC) ------------------------------
    P("## Headline: `token_difr` full-recompute AUC (every model × attack)\n")
    P("| model | " + " | ".join(ATTACKS) + " | **min** |")
    P("|---|" + "|".join(["---"] * (len(ATTACKS) + 1)) + "|")
    for r in ok:
        row = [r["full"].get(a, {}).get("token_difr", float("nan")) for a in ATTACKS]
        P(f"| `{r['tag']}` | " + " | ".join(f"{v:.3f}" for v in row)
          + f" | **{np.nanmin(row):.3f}** |")
    P("")

    # -- per-verifier detectability across all cells ------------------------
    P("## Per-verifier detectability across ALL (model × attack) cells\n")
    P("Tier-1 = raw AUC; Tier-0 = max(AUC, 1−AUC).\n")
    P("| verifier | tier | mean | median | min | worst cell |")
    P("|---|---|---|---|---|---|")
    for d in verifiers:
        cells = [(detect(r["full"][a][d], d), r["tag"], a)
                 for r in ok for a in ATTACKS if d in r["full"].get(a, {})]
        if not cells:
            continue
        vals = np.array([c[0] for c in cells])
        worst = min(cells, key=lambda c: c[0])
        tier = "1" if d in TIER1 else "0"
        P(f"| `{d}` | {tier} | {vals.mean():.3f} | {np.median(vals):.3f} | "
          f"{worst[0]:.3f} | {worst[1]}/{worst[2]} |")
    P("")

    # -- best detector per attack (max detectability over verifiers, mean over models)
    P("## Best detector per attack (mean detectability over models)\n")
    P("| attack | best verifier | mean detect. | 2nd best |")
    P("|---|---|---|---|")
    for a in ATTACKS:
        scores = []
        for d in verifiers:
            vals = [detect(r["full"][a][d], d) for r in ok if d in r["full"].get(a, {})]
            if vals:
                scores.append((float(np.mean(vals)), d))
        scores.sort(reverse=True)
        if scores:
            b = scores[0]
            s2 = scores[1] if len(scores) > 1 else (float("nan"), "—")
            P(f"| `{a}` | `{b[1]}` | {b[0]:.3f} | `{s2[1]}` ({s2[0]:.3f}) |")
    P("")

    # -- attack difficulty (token_difr) -------------------------------------
    P("## Attack difficulty — mean `token_difr` AUC across models (lower = harder)\n")
    P("| attack | mean | min | max |")
    P("|---|---|---|---|")
    rows = []
    for a in ATTACKS:
        vals = [r["full"][a]["token_difr"] for r in ok if "token_difr" in r["full"].get(a, {})]
        if vals:
            rows.append((a, float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))))
    for a, m, mn, mx in sorted(rows, key=lambda x: x[1]):
        P(f"| `{a}` | {m:.3f} | {mn:.3f} | {mx:.3f} |")
    P("")

    # -- size trend within family (mean token_difr over attacks vs params) ---
    P("## Size trend — mean `token_difr` AUC over attacks, by model (does size help?)\n")
    P("| model | params | mean token_difr | mean best-detector |")
    P("|---|---|---|---|")
    for r in sorted(ok, key=lambda r: (family_of(r["tag"]), r["params"])):
        td = [r["full"][a]["token_difr"] for a in ATTACKS if "token_difr" in r["full"].get(a, {})]
        best = [max(detect(r["full"][a][d], d) for d in verifiers if d in r["full"].get(a, {}))
                for a in ATTACKS if a in r["full"]]
        P(f"| `{r['tag']}` | {r['params']/1e9:.2f}B | "
          f"{np.mean(td):.3f} | {np.mean(best):.3f} |")
    P("")

    # -- selective vs full on the hard forward-pass attacks -----------------
    budgets = sorted({float(b) for r in ok for b in r.get("selective", {})})
    if budgets:
        P("## Information-directed selective recompute vs full (`token_difr`)\n")
        hard = ["quant_4bit", "kv_fp8", "bug_k2", "adv_quant_temp"]
        P(f"Mean `token_difr` AUC over models, on the hard attacks {hard}:\n")
        P("| tier | recompute | mean AUC |")
        P("|---|---|---|")
        fv = [r["full"][a]["token_difr"] for r in ok for a in hard
              if "token_difr" in r["full"].get(a, {})]
        P(f"| full | 100% | {np.mean(fv):.3f} |")
        for b in budgets:
            bkey = b if b in ok[0].get("selective", {}) else str(b)
            sv, rr = [], []
            for r in ok:
                sel = r.get("selective", {})
                cell = sel.get(b, sel.get(str(b), {}))
                rr.append(r.get("realized_recompute_ratio", {}).get(
                    str(b), r.get("realized_recompute_ratio", {}).get(b, b)))
                for a in hard:
                    if a in cell and "token_difr" in cell[a]:
                        sv.append(cell[a]["token_difr"])
            if sv:
                P(f"| selective | {np.mean(rr)*100:.0f}% | {np.mean(sv):.3f} |")
        P("")

    report = "\n".join(out)
    dest = Path(path).with_name("robustness_report.md")
    dest.write_text(report)
    print(report)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
