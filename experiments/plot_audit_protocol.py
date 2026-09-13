"""Draw the implemented receipted audit protocol as SVG, PNG and PDF.

    python -m experiments.plot_audit_protocol

This figure illustrates the v1 message flow, not a measured security guarantee.
It requires matplotlib only; no model, CUDA device or private artifact is read.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK = "#162C3F"
MUTED = "#526576"
LINE = "#A7B4BF"
BLUE = "#245CA5"
TEAL = "#137A70"
RUST = "#A35737"
GOLD = "#976515"


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "svg.fonttype": "none", "svg.hashsalt": "ivgym-audit-protocol-v1",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(16, 14.2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 16), ylim=(14.2, 0))
    ax.axis("off")
    texts = []

    def text(x, y, value, size=13, color=INK, weight="normal", ha="center", **kwargs):
        artist = ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                         ha=ha, va="center", linespacing=1.45, **kwargs)
        texts.append(artist)
        return artist

    def box(x, y, w, h, fill, edge="none", radius=.13, lw=1, zorder=2):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder)
        ax.add_patch(patch)
        return patch

    def arrow(x1, y1, x2, y2, color=INK, both=False, lw=1.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="<->" if both else "-|>", mutation_scale=17,
                     linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=3))

    def step(n, y):
        ax.add_patch(Circle((.51, y), .175, color=INK, zorder=4))
        text(.51, y, str(n), 10.5, "white", "bold", zorder=5)

    def message(n, y, x1, x2, title, detail=None, color=INK, both=False):
        step(n, y)
        arrow(x1, y, x2, y, color, both)
        text((x1+x2)/2, y-.26, title, 13.5, color, "bold")
        if detail:
            text((x1+x2)/2, y+.22, detail, 10.7, MUTED)

    text(.55, .47, "The inference audit protocol", 28, weight="bold", ha="left")
    text(.55, .97, "Commit before collection. Choose checks after closure. Recompute independently.",
         13.5, MUTED, ha="left")
    box(12.18, .26, 3.27, .42, "#EEF2F5")
    text(13.815, .47, "EXPERIMENTAL v1  /  RECEIPTED MODE", 9.4, MUTED, "bold")

    auditor, provider, reference = 3.05, 8.35, 13.55
    lanes = [
        (1.05, 4.0, auditor, "AUDITOR", "Local policy + trusted calibration", BLUE, "#F2F6FC"),
        (6.35, 4.0, provider, "PROVIDER", "Signs claims; execution is untrusted", RUST, "#FBF5F1"),
        (11.55, 4.0, reference, "REFERENCE", "Adopter controlled or explicitly trusted", TEAL, "#F0F8F5"),
    ]
    for x, w, center, title, subtitle, color, fill in lanes:
        box(x, 1.53, w, .82, fill)
        text(center, 1.81, title, 15, color, "bold")
        text(center, 2.12, subtitle, 10.2, MUTED)
        ax.plot([center, center], [2.52, 9.97], color=LINE, linewidth=1.0,
                linestyle=(0, (3, 5)), alpha=.8, zorder=0)

    message(1, 2.98, provider, auditor, "Pin the signed deployment manifest",
            "Model, spec, epoch and expiry", RUST)
    message(2, 4.08, auditor, provider, "Send signed plan + seed commitment",
            "Freeze schedule, budget and calibration; keep the seed secret", BLUE)
    message(3, 5.14, auditor, provider, "Send exact-token requests", color=BLUE)
    arrow(provider, 5.83, auditor, 5.83, RUST)
    text((auditor+provider)/2, 5.57, "Return outputs + chained final receipts", 13.5, RUST, "bold")
    text((auditor+provider)/2, 6.05, "Capture exact wire bodies and input/output token IDs", 10.7, MUTED)
    message(4, 6.89, auditor, provider, "Close all scheduled response slots",
            "Provider signs the set root + count; errors remain visible", INK, both=True)

    # The challenge must follow closure: this is the key causal boundary.
    box(1.05, 7.43, 14.5, .48, "#FFF4DE", edge="#E9CD94", radius=.08)
    text(8.3, 7.67, "RESPONSE SET CLOSED   /   The selection seed may now be revealed", 11.6, GOLD, "bold")

    step(5, 8.64)
    box(1.22, 8.22, 3.66, .86, "#F2F6FC", edge="#D1DEEF")
    text(auditor, 8.49, "Reveal seed & select", 13.3, BLUE, "bold")
    text(auditor, 8.83, "Uniform sample, without replacement", 10.0, MUTED)
    arrow(4.98, 8.65, 11.71, 8.65, TEAL)
    text(8.345, 8.37, "Selected prompts + returned prefixes", 12.5, TEAL, "bold")
    box(11.81, 8.16, 3.48, 1.03, "#F0F8F5", edge="#C6E3D8")
    text(reference, 8.43, "Teacher-force pinned weights", 11.5, TEAL, "bold")
    text(reference, 8.81, "NLL + capped log rank\nUse the provider-returned prefix", 9.8, MUTED)

    message(6, 9.72, reference, auditor, "Return signed reference scores + measured prefill cost", color=TEAL)

    step(7, 10.73)
    box(1.22, 10.25, 3.66, .96, "#F2F6FC", edge="#D1DEEF")
    text(auditor, 10.54, "Test & seal", 14, BLUE, "bold")
    text(auditor, 10.91, "Rank p-values + Bonferroni", 10.5, MUTED)
    arrow(4.98, 10.73, 6.02, 10.73, BLUE)
    box(6.13, 10.25, 9.16, .96, INK)
    text(10.71, 10.54, "Signed audit artifact", 15, "white", "bold")
    text(10.71, 10.91, "Transcript  /  selection opening  /  reference scores  /  verdict", 11.3, "#DBE5ED")

    # These are two different review operations on the same sealed artifact.
    ax.plot([10.71, 10.71], [11.21, 11.55], color=LINE, lw=1.5)
    ax.plot([4.48, 11.88], [11.55, 11.55], color=LINE, lw=1.5)
    arrow(4.48, 11.55, 4.48, 11.86, BLUE, lw=1.5)
    arrow(11.88, 11.55, 11.88, 11.86, TEAL, lw=1.5)
    box(1.22, 11.95, 6.52, 1.10, "#F2F6FC", edge="#D1DEEF")
    box(8.47, 11.95, 6.82, 1.10, "#F0F8F5", edge="#C6E3D8")
    text(4.48, 12.25, "VERIFY  /  no GPU", 14, BLUE, "bold")
    text(4.48, 12.69, "Check local trust, signatures, commitments and the decision.\nIntegrity: valid / invalid / unverifiable", 10.7, MUTED)
    text(11.88, 12.25, "REPLAY  /  GPU", 14, TEAL, "bold")
    text(11.88, 12.69, "Recompute scores; report numerical or verdict disagreement.\nKeep the recorded verdict unchanged.", 10.7, MUTED)

    text(8.26, 13.43, "Audit outcomes:  inconsistent  /  no_deviation_detected  /  inconclusive  /  unsupported", 11.1, INK)
    text(8.26, 13.82, "Signatures bind claims and bytes. Statistical consistency does not prove precision, GPU identity or compute spent.",
         10.3, MUTED)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for artist in texts:
        b = artist.get_window_extent(renderer)
        if not (b.x0 >= bounds.x0 and b.x1 <= bounds.x1 and b.y0 >= bounds.y0 and b.y1 <= bounds.y1):
            raise RuntimeError(f"Text outside figure: {artist.get_text()}")

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png", "pdf"):
        path = out / f"fig_audit_protocol.{ext}"
        metadata = {"Title": "IVGym authenticated audit protocol v1"}
        if ext == "svg":
            metadata["Date"] = None
        if ext == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(path, dpi=160, facecolor="white", metadata=metadata)
        if ext == "svg":
            # Matplotlib emits trailing spaces in multiline path attributes.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
