"""Slice the selective-vs-full robustness sweep by family, size, attack type, and
value function.

Reads `docs/results/selective_robustness.json` (from exp_selective_robustness_gpu)
and answers, per slice rather than as one aggregate:

  * triage vs EQUAL-COST random -- the saving factor (random / triaged recompute
    ratio to reach the target AUC), the control the full sweep lacks;
  * selective vs full -- triaged AUC at a fixed budget vs full-recompute AUC;
  * attack-type -- forward-pass (near-tie flips, where a proxy can point) vs
    sampling-only (no such structure), the two regimes for the triage rationale;
  * value-fn robustness -- which cheap signal (entropy/tie_margin/surprisal) wins,
    and whether that holds across families/sizes/attacks.

    .venv/bin/python -m experiments.analyze_selective_robustness [path.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).resolve().parents[1] / "docs" / "results" / "selective_robustness.json"
HEADLINE = "token_difr"
FIXED_BUDGET = 0.25          # for the "selective vs full at a fixed budget" view


def family_of(tag: str) -> str:
    return tag.rsplit("-", 1)[0]


def interp(rhos, curve, x):
    return float(np.interp(x, rhos, curve))


def fmt(x, suf="", nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}{suf}"


def gmean(xs):
    xs = [x for x in xs if x]
    return float(np.exp(np.mean(np.log(xs)))) if xs else None


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    payload = json.loads(Path(path).read_text())
    cfg = payload["config"]
    ok = [m for m in payload["models"] if "error" not in m]
    if not ok:
        print("no successful models in", path)
        return
    rhos = cfg["rhos"]
    fwd, samp = cfg["forward_pass"], cfg["sampling_only"]
    attacks = cfg["attacks"]
    target = cfg["target"]
    rel_frac = cfg.get("rel_frac", 0.95)
    value_fns = cfg["value_fns"]

    out = []
    P = out.append
    P("# Selective vs full recompute — robustness across families, sizes, attack types\n")
    P(f"Source: `{path}`  •  {len(ok)} models.\n")
    P(f"Headline verifier: `{HEADLINE}`. Value signals: {', '.join(f'`{v}`' for v in value_fns)}. "
      f"Random baseline = equal-cost random subsample, {cfg['boot']} seeds.\n")
    P(f"Two cost metrics: **relative** = recompute ratio to recover {int(rel_frac*100)}% of "
      f"full recompute's *own* AUC (always defined — the primary selective-vs-full number); "
      f"**absolute** = ratio to reach AUC {target} (n/a when even full recompute misses it). "
      f"Fixed-budget view uses **{int(FIXED_BUDGET*100)}%**.\n")

    # ---- roster --------------------------------------------------------------
    P("## Model matrix\n")
    P("| model | family | params | proxy params |")
    P("|---|---|---|---|")
    for m in sorted(ok, key=lambda m: (family_of(m["tag"]), m["params"])):
        P(f"| `{m['tag']}` | {family_of(m['tag'])} | {m['params']/1e9:.2f}B | "
          f"{(m['proxy_params'] or 0)/1e9:.2f}B |")
    P("")

    def cell(m, atk):
        return m["cells"][atk][HEADLINE]

    # ---- headline: triage vs random saving, per model x attack ---------------
    P("## Triage vs equal-cost random — relative saving factor (per model × attack)\n")
    P(f"Saving = random ratio / triaged ratio to recover {int(rel_frac*100)}% of full-recompute "
      "AUC (>1 → triage is cheaper at equal cost; the value fn is the per-cell best).\n")
    P("| model | " + " | ".join(attacks) + " |")
    P("|---|" + "|".join(["---"] * len(attacks)) + "|")
    for m in sorted(ok, key=lambda m: (family_of(m["tag"]), m["params"])):
        row = [fmt(cell(m, a).get("saving_rel"), "×", 1) for a in attacks]
        P(f"| `{m['tag']}` | " + " | ".join(row) + " |")
    P("")

    # ---- attack-type slice ---------------------------------------------------
    P("## By attack type — does information-directed triage pay off?\n")
    P("Geometric-mean saving factor and how often each method reaches the target, "
      "over all (model × attack) cells in the group.\n")
    P("| attack group | mean full AUC | gmean rel-saving | triage reaches AUC "
      f"{target} | random reaches AUC {target} |")
    P("|---|---|---|---|---|")
    for label, group in [("forward-pass " + str(fwd), fwd), ("sampling-only " + str(samp), samp)]:
        savings, fulls, tri_hit, rnd_hit, n = [], [], 0, 0, 0
        for m in ok:
            for a in group:
                c = cell(m, a)
                n += 1
                fulls.append(c["full_auc"])
                savings.append(c.get("saving_rel"))
                tri_hit += c["cost"][f"triage:{c['best_value_fn']}"] is not None
                rnd_hit += c["cost"]["random"] is not None
        P(f"| {label} | {np.mean(fulls):.3f} | {fmt(gmean(savings),'×',2)} | "
          f"{tri_hit}/{n} | {rnd_hit}/{n} |")
    P("")

    # ---- selective vs full at a fixed budget ---------------------------------
    P(f"## Selective vs full at a fixed {int(FIXED_BUDGET*100)}% budget (`{HEADLINE}` AUC)\n")
    P("Triaged AUC at the fixed budget (best value fn) vs full-recompute AUC, mean over models.\n")
    P("| attack | mean full AUC | mean triaged AUC @ budget | Δ (triaged − full) |")
    P("|---|---|---|---|")
    for a in attacks:
        fulls, tris = [], []
        for m in ok:
            c = cell(m, a)
            fulls.append(c["full_auc"])
            best = c["best_value_fn"]
            tris.append(interp(rhos, c["triaged"][best], FIXED_BUDGET))
        grp = "FWD" if a in fwd else "samp"
        P(f"| `{a}` ({grp}) | {np.mean(fulls):.3f} | {np.mean(tris):.3f} | "
          f"{np.mean(tris)-np.mean(fulls):+.3f} |")
    P("")

    # ---- family slice (forward-pass only: where triage is motivated) ---------
    P("## By family — saving factor on forward-pass attacks (triage's home turf)\n")
    P("| family | models | gmean saving | mean full AUC |")
    P("|---|---|---|---|")
    fams = sorted({family_of(m["tag"]) for m in ok})
    for f in fams:
        ms = [m for m in ok if family_of(m["tag"]) == f]
        savings = [cell(m, a).get("saving_rel") for m in ms for a in fwd]
        fulls = [cell(m, a)["full_auc"] for m in ms for a in fwd]
        P(f"| {f} | {len(ms)} | {fmt(gmean(savings),'×',2)} | {np.mean(fulls):.3f} |")
    P("")

    # ---- size trend (within families that have a size ladder) ----------------
    P("## Size trend — forward-pass saving vs model size (within family)\n")
    P("| model | family | params | gmean saving (FWD) | mean full AUC (FWD) |")
    P("|---|---|---|---|---|")
    laddered = [f for f in fams if len([m for m in ok if family_of(m["tag"]) == f]) > 1]
    for m in sorted(ok, key=lambda m: (family_of(m["tag"]), m["params"])):
        if family_of(m["tag"]) not in laddered:
            continue
        savings = [cell(m, a).get("saving_rel") for a in fwd]
        fulls = [cell(m, a)["full_auc"] for a in fwd]
        P(f"| `{m['tag']}` | {family_of(m['tag'])} | {m['params']/1e9:.2f}B | "
          f"{fmt(gmean(savings),'×',2)} | {np.mean(fulls):.3f} |")
    P("")

    # ---- value-fn robustness -------------------------------------------------
    P("## Value-fn robustness — which cheap triage signal wins?\n")
    P(f"Win = lowest recompute ratio to recover {int(rel_frac*100)}% of full AUC on that cell "
      "(forward-pass attacks). Also the mean ratio each signal needs.\n")
    P("| value fn | wins (FWD cells) | mean ratio to rel target | defined |")
    P("|---|---|---|---|")
    wins = {vf: 0 for vf in value_fns}
    ratios = {vf: [] for vf in value_fns}
    hits = {vf: 0 for vf in value_fns}
    n_cells = 0
    for m in ok:
        for a in fwd:
            c = cell(m, a)
            n_cells += 1
            costs = {vf: c.get("cost_rel", {}).get(f"triage:{vf}") for vf in value_fns}
            for vf in value_fns:
                if costs[vf] is not None:
                    ratios[vf].append(costs[vf]); hits[vf] += 1
            avail = {vf: costs[vf] for vf in value_fns if costs[vf] is not None}
            if avail:
                wins[min(avail, key=avail.get)] += 1
    for vf in value_fns:
        mr = float(np.mean(ratios[vf])) if ratios[vf] else None
        P(f"| `{vf}` | {wins[vf]}/{n_cells} | {fmt(mr)} | {hits[vf]}/{n_cells} |")
    P("")

    report = "\n".join(out)
    dest = Path(path).with_name("selective_robustness_report.md")
    dest.write_text(report)
    print(report)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
