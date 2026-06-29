"""Regenerate fig3_io_pareto_gpu.png with FLOPs on the x-axis using data
already captured in run_8b_N48.log (Qwen3-8B + proxy Qwen3-0.6B, N=48, T=128).

Run:
    .venv/bin/python -m experiments.replot_flops
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardcoded results from run_8b_N48.log
# (Qwen/Qwen3-8B + proxy Qwen/Qwen3-0.6B, 48 prompts x 128 tokens, batch=100)
# ---------------------------------------------------------------------------
@dataclass
class Row:
    attack: str
    note: str
    token_difr: float
    io: dict[str, float] = field(default_factory=dict)


ROWS = [
    Row("(honest null)", "floor: honest vs reseeded honest", 0.583,
        {"surface_stat": 0.687, "surface_rank": 0.582, "surface_tokens": 0.500,
         "llm_judge": 0.619, "logit_judge": 0.918}),
    Row("quant_4bit", "", 0.692,
        {"surface_stat": 0.640, "surface_rank": 0.605, "surface_tokens": 0.500,
         "llm_judge": 0.913, "logit_judge": 0.672}),
    Row("kv_fp8", "", 0.511,
        {"surface_stat": 0.621, "surface_rank": 0.550, "surface_tokens": 0.500,
         "llm_judge": 0.503, "logit_judge": 0.842}),
    Row("temp_1.1", "", 0.564,
        {"surface_stat": 0.645, "surface_rank": 0.570, "surface_tokens": 0.500,
         "llm_judge": 0.823, "logit_judge": 0.916}),
    Row("seed_43", "", 0.892,
        {"surface_stat": 0.664, "surface_rank": 0.589, "surface_tokens": 0.500,
         "llm_judge": 0.919, "logit_judge": 0.954}),
    Row("bug_k2", "", 0.521,
        {"surface_stat": 0.665, "surface_rank": 0.536, "surface_tokens": 0.500,
         "llm_judge": 0.955, "logit_judge": 0.955}),
    Row("bug_k32", "", 0.661,
        {"surface_stat": 0.572, "surface_rank": 0.748, "surface_tokens": 0.500,
         "llm_judge": 1.000, "logit_judge": 1.000}),
]

# ---------------------------------------------------------------------------
# Model parameters (from run log: Qwen3-8B + proxy Qwen3-0.6B)
# ---------------------------------------------------------------------------
# Qwen3-8B
N_PARAMS_MAIN   = 8_190_000_000   # 8.19B total parameters
VOCAB            = 151936
HIDDEN_MAIN      = 4096
EMBED_MAIN       = VOCAB * HIDDEN_MAIN          # 622M

# Qwen3-0.6B proxy
N_PARAMS_PROXY  = 600_000_000     # 0.60B total parameters
HIDDEN_PROXY     = 1024
EMBED_PROXY      = VOCAB * HIDDEN_PROXY         # 156M

# Sequence length for cost calculation
N_PROMPT_TOKENS = 32    # backend.max_prompt_tokens default
N_CONT_TOKENS   = 128   # N_TOKENS from experiment
T               = N_PROMPT_TOKENS + N_CONT_TOKENS   # = 160

MAIN_NON_EMBED  = max(N_PARAMS_MAIN  - EMBED_MAIN,  0)   # ≈ 7.57B
PROXY_NON_EMBED = max(N_PARAMS_PROXY - EMBED_PROXY, 0)   # ≈ 0.44B

FLOPS = {
    "token_difr":    2 * MAIN_NON_EMBED  * T,   # ≈ 2.42 TFLOPs
    "surface_stat":  2 * PROXY_NON_EMBED * T,   # ≈ 141 GFLOPs
    "surface_rank":  2 * PROXY_NON_EMBED * T,
    "surface_tokens": 1e6,                       # nominal; no model forward pass
}


def render_pareto(rows, path: Path, flops: dict[str, float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    shown = {
        "temp_1.1":  ("#d62728", "o", "temp_1.1 (output-visible)"),
        "quant_4bit": ("#ff7f0e", "s", "quant_4bit"),
        "bug_k32":   ("#9467bd", "D", "bug_k32"),
        "seed_43":   ("#1f77b4", "^", "seed_43 (output-invisible)"),
    }
    by_name = {r.attack: r for r in rows}

    xs_names = ["surface_tokens", "surface_stat", "token_difr"]
    xs = [flops[n] for n in xs_names]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for name, (c, m, lbl) in shown.items():
        if name not in by_name:
            continue
        r = by_name[name]
        ys = [r.io["surface_tokens"], r.io["surface_stat"], r.token_difr]
        ax.plot(xs, ys, color=c, marker=m, ms=7, lw=1.8, label=lbl)

    ax.axhline(0.5, ls=":", color="0.5", lw=1.2, label="chance (AUC = 0.5)")
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.03)

    # annotate each x-tick with its human-readable FLOPs value
    ax.set_xticks(xs)
    tick_labels = [
        f"{flops['surface_tokens']/1e6:.0f} MFLOPs\n(nominal; no fwd pass)",
        f"{flops['surface_stat']/1e9:.0f} GFLOPs\n(proxy {N_PARAMS_PROXY/1e9:.1f}B)",
        f"{flops['token_difr']/1e12:.2f} TFLOPs\n(recompute {N_PARAMS_MAIN/1e9:.1f}B)",
    ]
    ax.set_xticklabels(tick_labels, fontsize=8)

    ax.set_xlabel(
        "verifier cost   (FLOPs / sequence, 2·N·T approx,\n"
        f"N = non-embedding params, T = {T} tokens;  token_difr = full recompute)  [log]",
        fontsize=9,
    )
    ax.set_ylabel("detectability   (max(AUC, 1-AUC))")
    ax.set_title(
        "Fig 3 — cost/accuracy Pareto: cheap I/O detector vs full recomputation\n"
        "output-visible attacks are caught cheaply; output-invisible ones need recomputation",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {path}")


def main():
    print(f"Model: Qwen3-8B ({N_PARAMS_MAIN/1e9:.2f}B) + proxy Qwen3-0.6B ({N_PARAMS_PROXY/1e9:.2f}B)")
    print(f"T = {N_PROMPT_TOKENS} prompt + {N_CONT_TOKENS} continuation = {T} tokens/seq")
    print()
    print("Verifier FLOPs / sequence  (2·N_non_embed·T):")
    for n, f in FLOPS.items():
        if f < 1e9:
            label = f"{f/1e6:.1f} MFLOPs  (nominal; no model forward pass)"
        elif f < 1e12:
            label = f"{f/1e9:.1f} GFLOPs"
        else:
            label = f"{f/1e12:.3f} TFLOPs"
        print(f"   {n:>14}  {label}")
    td, ss = FLOPS["token_difr"], FLOPS["surface_stat"]
    print(f"\n   surface_stat is {td/ss:.1f}x fewer FLOPs than token_difr (full recompute).")

    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig3_io_pareto_gpu.png"
    render_pareto(ROWS, out, FLOPS)


if __name__ == "__main__":
    main()
