"""LaTeX table fragments for the report, generated from the experiment JSONs.

Every number the paper prints in a table comes from here, so a table can never
drift from the run that produced it. Writes into `tables/`, which `paper.tex`
\\input{}s.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

RES = REPO / "docs/results"
BASE = REPO / "runs/baseline_json"
OUT = HERE / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SHORT = {
    "token_difr": r"\td",
    "cross_entropy": r"\ce",
    "activation_difr": r"\ad",
    "token_toploc": r"\tl",
}
ATK = {
    "quant_4bit": r"\texttt{quant\_4bit}",
    "quant_2bit": r"\texttt{quant\_2bit}",
    "kv_fp8": r"\texttt{kv\_fp8}",
    "temp_1.1": r"\texttt{temp\_1.1}",
    "seed_43": r"\texttt{seed\_43}",
    "bug_k2": r"\texttt{bug\_k2}",
    "bug_k32": r"\texttt{bug\_k32}",
}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body.rstrip() + "\n")
    print("wrote", name)


# ------------------------------------------------------------------ headline
def tab_headline():
    d = load(RES / "headline_ratio.json")
    if d is None:
        return
    cells, dets = d["cells"], d["defenses"]
    r_pub = d["ratio"]["readme"] * 100
    r_val = d["ratio"]["ratioed"] * 100

    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{SHORT[x]}}}" for x in dets)
    rules = " ".join(rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(dets)))
    sub = " & ".join([rf"{r_pub:.0f}\% & {r_val:.1f}\%"] * len(dets))

    lines = []
    for a in d["attacks"]:
        vals = []
        for det in dets:
            pub = cells[a][det]["readme"]["auc"]
            val = cells[a][det]["ratioed"]["auc"]
            # bold the cell the portfolio would actually pick, per arm
            vals += [f"{pub:.3f}", f"{val:.3f}"]
        best_p = max(range(len(dets)), key=lambda i: cells[a][dets[i]]["readme"]["auc"])
        best_v = max(range(len(dets)), key=lambda i: cells[a][dets[i]]["ratioed"]["auc"])
        vals[2 * best_p] = r"\textbf{" + vals[2 * best_p] + "}"
        vals[2 * best_v + 1] = r"\textbf{" + vals[2 * best_v + 1] + "}"
        lines.append(f"{ATK[a]} & " + " & ".join(vals) + r" \\")

    body = rf"""\begin{{tabular}}{{l{'rr' * len(dets)}}}
\toprule
attack & {head} \\
{rules}
 & {sub} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}"""
    write("tab_headline.tex", body)


# --------------------------------------------------------------- pool scaling
def tab_poolscaling():
    d = load(RES / "pool_scaling.json")
    if d is None:
        return
    lines = []
    for r in d["rows"]:
        mark = r"$^{\dagger}$" if r["is_prediction_point"] else ""
        note = "" if r["in_ceiling"] else r" \textit{over ceiling}"
        lines.append(
            rf"{r['batch']:,}{mark} & {r['ratio']*100:.1f}\% & "
            rf"${r['auc']:.3f} \pm {r['sd']:.3f}$ & {r['predicted']:.3f} & "
            rf"${r['auc']-r['predicted']:+.3f}${note} \\".replace(",", "{,}")
        )
    body = rf"""\begin{{tabular}}{{rrlrl}}
\toprule
batch $b$ & ratio & measured AUC & predicted & residual \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}"""
    write("tab_poolscaling.tex", body)


# ------------------------------------------------------------------ reversal
def tab_reversal():
    d = load(RES / "reversal_check.json")
    if d is None:
        return
    lines = []
    for c in d["cells"]:
        model = c["model"].split("/")[-1].replace("-Instruct", "")
        lines.append(
            rf"\texttt{{{model}}} & {ATK[c['attack']]} & {c['published']:.3f} & "
            rf"${c['sweepcfg']['auc']:.3f} \pm {c['sweepcfg']['sd']:.3f}$ & "
            rf"${c['ratioed']['auc']:.3f} \pm {c['ratioed']['sd']:.3f}$ & "
            rf"${c['d_prime']:+.4f}$ & {'yes' if c['reversed_per_token'] else 'no'} \\"
        )
    body = rf"""\begin{{tabular}}{{llrllr l}}
\toprule
model & attack & published & re-run at {d['cells'][0]['sweepcfg']['ratio']*100:.0f}\% &
re-run at {d['cells'][0]['ratioed']['ratio']*100:.1f}\% & $d'$ & reversed? \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}"""
    write("tab_reversal.tex", body)


# ----------------------------------------------------- prefix cost, 3 attacks
PREFIX_RUNS = [
    ("quant_2bit", BASE / "prefix_cost_quant2bit.json"),
    ("kv_fp8", BASE / "prefix_cost_kvfp8.json"),
    ("bug_k32", BASE / "prefix_cost_bugk32.json"),
]


def tab_prefix_attacks():
    runs = [(a, load(p)) for a, p in PREFIX_RUNS]
    runs = [(a, d) for a, d in runs if d is not None]
    if len(runs) < 2:
        print("tab_prefix_attacks: need >=2 prefix runs; have", len(runs))
        return

    # Cost geometry at a 5% nominal budget, plus the full-audit reference, plus
    # the honest-Pareto pair the text compares.
    lines = []
    for a, d in runs:
        tk, pf = d["curves"]["topk"], d["curves"]["prefix"]
        i5 = tk["budget"].index(0.05)
        lines.append(
            rf"{ATK[a]} & {tk['prefill_ratio'][i5]*100:.1f}\% & {tk['seconds'][i5]:.2f} & "
            rf"{pf['prefill_ratio'][i5]*100:.1f}\% & {pf['seconds'][i5]:.2f} & "
            rf"{tk['seconds'][i5]/pf['seconds'][i5]:.1f}$\times$ & "
            rf"{tk['seconds'][-1]:.2f} & "
            rf"${d['curves']['topk']['auc'][-1]:.3f} \pm {d['curves']['topk']['auc_sd'][-1]:.3f}$ \\"
        )
    body = rf"""\begin{{tabular}}{{lrrrrrrl}}
\toprule
& \multicolumn{{2}}{{c}}{{global top-$k$ @ 5\%}} & \multicolumn{{2}}{{c}}{{prefix @ 5\%}}
& & \multicolumn{{2}}{{c}}{{full audit}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{7-8}}
attack & cost & sec & cost & sec & speed-up & sec & AUC \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}"""
    write("tab_prefix_attacks.tex", body)


def tab_prefix_pareto():
    """The honest Pareto: prefix points that dominate top-k points on BOTH axes."""
    runs = [(a, load(p)) for a, p in PREFIX_RUNS]
    runs = [(a, d) for a, d in runs if d is not None]
    # An attack whose FULL audit is at chance has no detection to allocate, so a
    # Pareto plane over it compares noise to noise. Drop it and say so.
    dropped = [a for a, d in runs if d["curves"]["topk"]["auc"][-1] < 0.6]
    runs = [(a, d) for a, d in runs if d["curves"]["topk"]["auc"][-1] >= 0.6]
    if dropped:
        print("  tab_prefix_pareto: excluded (full audit at chance):", dropped)
    lines = []
    for a, d in runs:
        tk, pf = d["curves"]["topk"], d["curves"]["prefix"]
        # best prefix point by AUC, then count top-k points it strictly dominates
        j = max(range(len(pf["budget"])), key=lambda i: pf["auc"][i] if pf["prefill_ratio"][i] < 1.0 else -1)
        dom = [i for i in range(len(tk["budget"]))
               if tk["prefill_ratio"][i] > pf["prefill_ratio"][j]
               and tk["auc"][i] < pf["auc"][j]]
        gaps = [pf["auc"][j] - tk["auc"][i] for i in dom]
        # unpaired separation in combined standard deviations -- conservative,
        # since the two arms share protocol seeds and we cannot pair them here.
        zs = [(pf["auc"][j] - tk["auc"][i])
              / ((pf["auc_sd"][j] ** 2 + tk["auc_sd"][i] ** 2) ** 0.5) for i in dom]
        gtxt = (rf"{min(gaps):.2f}--{max(gaps):.2f}" if dom else "---")
        ztxt = (rf"{min(zs):.1f}--{max(zs):.1f} ({sum(z > 2 for z in zs)})"
                if dom else "---")
        lines.append(
            rf"{ATK[a]} & {pf['prefill_ratio'][j]*100:.0f}\% & "
            rf"${pf['auc'][j]:.3f} \pm {pf['auc_sd'][j]:.3f}$ & "
            rf"${tk['auc'][-1]:.3f} \pm {tk['auc_sd'][-1]:.3f}$ & "
            rf"{len(dom)} of {len(tk['budget'])} & {gtxt} & {ztxt} \\"
        )
    body = rf"""\begin{{tabular}}{{lrllrll}}
\toprule
attack & cost & prefix AUC & full audit & top-$k$ dom. & AUC gap & sep.\ (sd) \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}"""
    write("tab_prefix_pareto.tex", body)


if __name__ == "__main__":
    tab_headline()
    tab_poolscaling()
    tab_reversal()
    tab_prefix_attacks()
    tab_prefix_pareto()
