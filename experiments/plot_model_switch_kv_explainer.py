"""Illustrate exact token agreement despite an invalid model-switch cache handoff.

This is a declared one-slot cache toy, not a measured transformer result or a
counterexample to a fixed-model proof. The token transcript has a legal rebuild
explanation; only trusted evidence of the actual handoff exposes the violation.

Run: python -m experiments.plot_model_switch_kv_explainer
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.plot_trajectory_consistency_explainer import (
    BLUE, BLUE_BG, GREEN, GREEN_BG, INK, LINE, MUTED,
    ORANGE, ORANGE_BG, RED, RED_BG, arrow, box, text,
)


OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"


def b_logits(cache_value):
    """Exact toy output head after reading the single cached value."""
    return {"runs": cache_value + 1, "sleeps": 2}


def greedy(logits):
    return max(logits, key=logits.get)


def main():
    # A and B have different value projections of the same scalar prefix state.
    # A one-slot attention read has weight 1, so its output is the cached value.
    prefix_state = 1
    cache_a = {"K": prefix_state, "V": 2 * prefix_state}
    cache_b = {"K": prefix_state, "V": 4 * prefix_state}
    stale_logits = b_logits(cache_a["V"])
    fresh_logits = b_logits(cache_b["V"])
    stale_token, fresh_token = greedy(stale_logits), greedy(fresh_logits)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "svg.fonttype": "none",
        "figure.facecolor": "#F7F9FC",
    })
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    text(ax, .035, .962, "Exact tokens can hide an invalid KV-cache handoff",
         25, weight="bold")
    text(ax, .035, .922,
         "Model A → model B  •  Different weights  •  Exact arithmetic  •  Toy response: “cat runs”",
         13.5, MUTED)

    box(ax, .035, .818, .93, .065, fill="#EDF1F6", edge="#EDF1F6")
    text(ax, .055, .858,
         "ASSUMED CONTRACT: switching models is allowed; B must receive a cache valid for B.",
         12.8, weight="bold")
    text(ax, .055, .833,
         "Rebuilding the observed prefix or using a validated conversion is permitted. Unvalidated reuse is not.",
         11.8, MUTED)

    text(ax, .055, .769, "SHARED START", 11, MUTED, "bold")
    box(ax, .23, .730, .27, .065, 'Model A emits “cat”', fill=BLUE_BG,
        edge=BLUE, color=BLUE, size=15, weight="bold")
    arrow(ax, (.515, .762), (.59, .762), BLUE)
    box(ax, .605, .730, .33, .065, 'After processing “cat”: cache A',
        fill=BLUE_BG, edge=BLUE, color=BLUE, size=13, weight="bold")
    text(ax, .365, .703, "Exact token check 1: PASS", 11.5, GREEN, ha="center")
    text(ax, .77, .703, "One-slot toy: K = 1, V = 2", 11.5, BLUE, ha="center")

    # Actual execution: switching weights does not rebuild cached activations.
    box(ax, .035, .485, .93, .165, fill="white", edge=LINE)
    text(ax, .055, .617, "ACTUAL SERVER", 11.5, RED, "bold")
    text(ax, .055, .579, "Switches A → B", 12.5, weight="bold")
    text(ax, .055, .548, "Skips cache rebuild", 11.7, RED)
    box(ax, .29, .515, .22, .08, 'B reads cache A\nK = 1, V = 2',
        fill=RED_BG, edge=RED, color=RED, size=14, weight="bold")
    arrow(ax, (.525, .555), (.58, .555), RED)
    box(ax, .595, .515, .18, .08,
        f'runs: {stale_logits["runs"]}   sleeps: {stale_logits["sleeps"]}',
        fill="#F4F6F8", size=12.8)
    arrow(ax, (.788, .555), (.83, .555), MUTED)
    box(ax, .845, .515, .09, .08, f'“{stale_token}”',
        fill=GREEN_BG, edge=GREEN, color=GREEN, size=16, weight="bold")
    text(ax, .4, .617, "CARRIED CACHE", 10.5, MUTED, "bold", ha="center")
    text(ax, .685, .617, "B’S EXACT LOGITS", 10.5, MUTED, "bold", ha="center")
    text(ax, .89, .617, "SERVED TOKEN", 10.5, MUTED, "bold", ha="center")

    text(ax, .4, .452, "Cache values differ: 2 ≠ 4", 13, RED, "bold", ha="center")
    text(ax, .89, .452, "Tokens match ✓", 13, GREEN, "bold", ha="center")

    # Independent token verification supplies its own prefix-derived cache.
    box(ax, .035, .255, .93, .165, fill="white", edge=LINE)
    text(ax, .055, .387, "TOKEN VERIFIER", 11.5, GREEN, "bold")
    text(ax, .055, .349, 'Replays prefix “cat”', 12.5, weight="bold")
    text(ax, .055, .318, "Rebuilds under B", 11.7, ORANGE)
    box(ax, .29, .285, .22, .08, 'B reads fresh cache B\nK = 1, V = 4',
        fill=ORANGE_BG, edge=ORANGE, color=ORANGE, size=14, weight="bold")
    arrow(ax, (.525, .325), (.58, .325), ORANGE)
    box(ax, .595, .285, .18, .08,
        f'runs: {fresh_logits["runs"]}   sleeps: {fresh_logits["sleeps"]}',
        fill="#F4F6F8", size=12.8)
    arrow(ax, (.788, .325), (.83, .325), MUTED)
    box(ax, .845, .285, .09, .08, f'“{fresh_token}”',
        fill=GREEN_BG, edge=GREEN, color=GREEN, size=16, weight="bold")
    text(ax, .4, .387, "RECOMPUTED CACHE", 10.5, MUTED, "bold", ha="center")
    text(ax, .685, .387, "B’S EXACT LOGITS", 10.5, MUTED, "bold", ha="center")
    text(ax, .89, .387, "EXPECTED TOKEN", 10.5, MUTED, "bold", ha="center")

    box(ax, .035, .187, .93, .037,
        "Both exact token checks pass. They do not establish which cache the server used.",
        fill=GREEN_BG, edge=GREEN_BG, color=GREEN, size=14, weight="bold")

    text(ax, .045, .145, "WHAT A STATE-AWARE CHECK ADDS", 11, INK, "bold")
    text(ax, .045, .116,
         "With trusted handoff evidence, it rejects B reading A’s unvalidated cache. A legal rebuild must pass.",
         12.7, INK)
    text(ax, .045, .079,
         "Tokens alone still have a legal explanation: A → rebuild the prefix under B → “runs”.",
         12.4, MUTED)
    text(ax, .045, .055,
         "This is a limit on verifying execution history, not a counterexample to FlashProve’s fixed-model output proof.",
         11.3, MUTED)

    text(ax, .045, .022,
         "Declared toy: one cached value v; B’s logits (runs, sleeps) = (v + 1, 2). Greedy decoding picks the larger logit. Not measured model outputs.",
         10.2, MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig_model_switch_kv_exact_tokens"
    for extension in ("png", "svg"):
        target = stem.with_suffix(f".{extension}")
        fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
        print(target)
    plt.close(fig)
    print(f"Stale cache: {stale_logits} -> {stale_token}")
    print(f"Fresh cache: {fresh_logits} -> {fresh_token}")


if __name__ == "__main__":
    main()
