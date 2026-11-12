"""Draw a conceptual comparison of token and trajectory verification.

This is an illustrative toy example, not experiment output. Generate with:
    python -m experiments.plot_trajectory_consistency_explainer

Outputs editable SVG and PNG under docs/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
INK = "#172C40"
MUTED = "#596A7A"
BLUE = "#2563A6"
BLUE_BG = "#EAF2FC"
ORANGE = "#A96312"
ORANGE_BG = "#FFF2DF"
GREEN = "#167358"
GREEN_BG = "#E6F5EE"
RED = "#B23A47"
RED_BG = "#FCECEF"
LINE = "#D8E2EA"


def text(ax, x, y, label, size=12, color=INK, weight="normal", ha="left", **kw):
    return ax.text(
        x, y, label, fontsize=size, color=color, weight=weight,
        ha=ha, va="center", linespacing=1.4, **kw,
    )


def box(ax, x, y, w, h, label="", fill="white", edge=LINE,
        color=INK, size=12, weight="normal", linewidth=1.3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.023",
        facecolor=fill, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(patch)
    if label:
        text(ax, x+w/2, y+h/2, label, size, color, weight, ha="center")
    return patch


def arrow(ax, start, end, color=MUTED, dashed=False, width=2, curve=0):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=width, color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(patch)
    return patch


def panel(fig, rect, number, title, subtitle):
    ax = fig.add_axes(rect)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(ax, .012, .012, .976, .976, edge=LINE)
    box(ax, .045, .85, .065, .095, str(number), fill=INK,
        edge=INK, color="white", size=15, weight="bold")
    text(ax, .135, .899, title, 16, weight="bold")
    text(ax, .045, .788, subtitle, 11.4, MUTED)
    return ax


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "figure.facecolor": "#F7F9FC",
    })
    fig = plt.figure(figsize=(16, 12.3))
    fig.text(.035, .965, "Can the token explanations form one legal execution?",
             fontsize=24, weight="bold", color=INK, va="top")
    fig.text(.035, .917,
             'Toy response: “cat runs”     •     KV cache = the model’s stored memory of the prefix',
             fontsize=13.3, color=MUTED, va="top")

    ax = panel(
        fig, [.025, .535, .465, .345], 1,
        "Fixed deterministic semantics",
        "FlashProve’s setting: a specified integer model, greedy decoding",
    )
    text(ax, .5, .67, "ONE SPECIFIED NEXT-TOKEN FUNCTION", 10.5,
         MUTED, "bold", ha="center")
    box(ax, .07, .43, .19, .16, "Prompt", fill=BLUE_BG, edge=BLUE)
    box(ax, .40, .43, .19, .16, '“cat”', fill=BLUE_BG, edge=BLUE,
        color=BLUE, size=15, weight="bold")
    box(ax, .73, .43, .19, .16, '“runs”', fill=BLUE_BG, edge=BLUE,
        color=BLUE, size=15, weight="bold")
    arrow(ax, (.28, .51), (.38, .51), BLUE)
    arrow(ax, (.61, .51), (.71, .51), BLUE)
    text(ax, .495, .36, "check 1", 10.5, GREEN, ha="center")
    text(ax, .825, .36, "check 2", 10.5, GREEN, ha="center")
    text(ax, .5, .25, "Each proof checks the computation for the correct prefix.",
         11.1, MUTED, ha="center")
    box(ax, .055, .065, .89, .11,
        "Every token checked  →  the sequence is verified",
        fill=GREEN_BG, edge=GREEN_BG, color=GREEN, size=12.2, weight="bold")

    ax = panel(
        fig, [.51, .535, .465, .345], 2,
        "Separate BF16 token explanations",
        "Same weights; different arithmetic can change the cache and output",
    )
    text(ax, .42, .66, "First token", 11.7, weight="bold", ha="center")
    text(ax, .755, .66, 'Next token after “cat”', 11.7,
         weight="bold", ha="center")
    text(ax, .065, .505, "Mode A", 12, BLUE, "bold")
    text(ax, .065, .325, "Mode B", 12, ORANGE, "bold")
    box(ax, .285, .435, .255, .14, '“cat”  ✓', fill=BLUE_BG,
        edge=BLUE, color=BLUE, size=15, weight="bold", linewidth=2)
    box(ax, .605, .435, .30, .14, '“sleeps”', fill="#F4F6F8",
        color=MUTED, size=14)
    box(ax, .285, .255, .255, .14, '“dog”', fill="#F4F6F8",
        color=MUTED, size=14)
    box(ax, .605, .255, .30, .14, '“runs”  ✓', fill=ORANGE_BG,
        edge=ORANGE, color=ORANGE, size=15, weight="bold", linewidth=2)
    text(ax, .5, .185,
         'For column 2, both modes are given the observed prefix “cat”.',
         10.7, MUTED, ha="center")
    box(ax, .055, .045, .89, .09,
        'Each token has an explanation. Can A’s state connect to B’s?',
        fill="#EDF1F6", edge="#EDF1F6", size=11.6, weight="bold")

    ax = panel(
        fig, [.025, .15, .465, .345], 3,
        "IF there is no legal connection",
        "Whole-trajectory check: follow the state across token decisions",
    )
    box(ax, .075, .42, .31, .23, 'After “cat”\nKV cache A',
        fill=BLUE_BG, edge=BLUE, color=BLUE, size=14, weight="bold")
    box(ax, .61, .42, .31, .23, 'To explain “runs”\nneed KV cache B\nfor prefix “cat”',
        fill=ORANGE_BG, edge=ORANGE, color=ORANGE, size=12.7, weight="bold")
    arrow(ax, (.405, .535), (.59, .535), RED, dashed=True)
    text(ax, .5, .55, "×", 32, RED, "bold", ha="center",
         bbox={"facecolor": "white", "edgecolor": "none", "pad": .2})
    text(ax, .5, .345, "All allowed switches and rebuilds must be ruled out.",
         11.5, RED, ha="center")
    box(ax, .055, .16, .89, .11, "Local checks pass; the joint check rejects.",
        fill=RED_BG, edge=RED_BG, color=RED, size=12.5, weight="bold")
    text(ax, .5, .075, "This is the possible gap we still need to demonstrate.",
         11.3, MUTED, ha="center")

    ax = panel(
        fig, [.51, .15, .465, .345], 4,
        "IF rebuilding the state is allowed",
        "The same checker must include that legal transition",
    )
    box(ax, .055, .435, .23, .18, 'After “cat”\nKV cache A',
        fill=BLUE_BG, edge=BLUE, color=BLUE, size=12.5, weight="bold")
    box(ax, .39, .40, .25, .25, 'Rebuild prefix\n“cat” in mode B',
        fill=GREEN_BG, edge=GREEN, color=GREEN, size=12.4, weight="bold")
    box(ax, .745, .435, .20, .18, 'KV cache B\nfor “cat”',
        fill=ORANGE_BG, edge=ORANGE, color=ORANGE, size=12.3, weight="bold")
    arrow(ax, (.30, .525), (.375, .525), GREEN)
    arrow(ax, (.655, .525), (.73, .525), GREEN)
    arrow(ax, (.845, .417), (.845, .31), GREEN)
    text(ax, .845, .265, 'emit “runs”', 12.5, GREEN, "bold", ha="center")
    text(ax, .055, .30, '“cat” is already observed;\nrebuilding does not emit it again.',
         11.0, MUTED)
    box(ax, .055, .095, .89, .11,
        "Accept: a legal full path exists. The gap disappears.",
        fill=GREEN_BG, edge=GREEN_BG, color=GREEN, size=12.0, weight="bold")

    fig.text(.5, .109,
             "The experiment asks whether case 3 survives every transition a real service permits.",
             fontsize=14, color=INK, weight="bold", ha="center")
    fig.text(.5, .078,
             "We cannot create an advantage by banning legitimate switching or recomputation.",
             fontsize=13, color=MUTED, ha="center")
    fig.text(.035, .033,
             "Conceptual example, not a measured result. Panels 2–4 describe the proposed BF16 question, not a flaw in FlashProve.",
             fontsize=10.5, color=MUTED)
    fig.text(.035, .014,
             "Reference: anthropic.com/document/flashprove-technical-report, §5.3.  •  Fixed-model composition is our deduction.",
             fontsize=9.6, color=MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig_trajectory_consistency_explainer"
    for extension in ("png", "svg"):
        target = stem.with_suffix(f".{extension}")
        fig.savefig(target, dpi=180, facecolor=fig.get_facecolor())
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
        print(target)
    plt.close(fig)


if __name__ == "__main__":
    main()
