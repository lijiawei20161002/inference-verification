"""Render descriptive serving figures from a validated campaign summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((root / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)

    colors = {"graph_off": "#167d9a", "eager_off": "#be603a"}
    labels = {"graph_off": "Optimized", "eager_off": "Eager"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, concurrency in zip(axes, (1, 4)):
        for config in colors:
            for arm in ("full", "truncate"):
                for phase in ("pilot", "delayed_repeat"):
                    rows = [r for r in summary["phases"][phase]["cells"]
                            if (r["config"], r["concurrency"], r["arm"]) == (config, concurrency, arm)]
                    xs = [r["length"] for r in rows]
                    values = [r["post_first_ms_per_token"] for r in rows]
                    if phase == "pilot":
                        ax.plot(xs, [v["median"] for v in values], color=colors[config],
                                linestyle="-" if arm == "full" else "--", marker="o",
                                label=f"{labels[config]} / {arm}")
                        ax.fill_between(xs, [v["min"] for v in values], [v["max"] for v in values],
                                        color=colors[config], alpha=0.1)
                    else:
                        ax.plot(xs, [v["median"] for v in values], linestyle="none", marker="x",
                                color=colors[config], markersize=7)
        ax.set(title=f"Client concurrency {concurrency}", xlabel="Wire prompt tokens",
               xticks=[512, 2048, 8192, 16384])
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Post-first-content ms / returned token")
    axes[0].legend(fontsize=8)
    fig.suptitle("vLLM / Qwen3-1.7B / L40S: context response\nLines: pilot; crosses: delayed repeat; shading: six-block observed range", fontsize=11)
    save(fig, "context_response")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    windows = ("early_ms_per_token", "post_first_ms_per_token", "late_ms_per_token")
    for ax, concurrency in zip(axes, (1, 4)):
        for phase, color, label in (("pilot", "#167d9a", "Pilot"),
                                    ("delayed_repeat", "#be603a", "Delayed repeat")):
            row = next(r for r in summary["phases"][phase]["contrasts"]
                       if (r["config"], r["length"], r["concurrency"]) == ("graph_on", 8192, concurrency))
            values = [row[w] for w in windows]
            y = [v["median"] for v in values]
            ax.errorbar(range(3), y, yerr=[[v["median"]-v["min"] for v in values],
                                         [v["max"]-v["median"] for v in values]],
                        color=color, marker="o", capsize=4, label=label)
        ax.axhline(1, color="0.4", linestyle=":")
        ax.set(title=f"8K context, concurrency {concurrency}", xticks=range(3),
               xticklabels=["Early 0–32", "Post-first", "Late 32–127"])
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Paired cold / hit arrival-time ratio")
    axes[0].legend()
    fig.suptitle("Prefix-cache timing depends on the stream window\nExploratory window split; bars show six-block observed ranges", fontsize=11)
    save(fig, "cache_windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.output)


if __name__ == "__main__":
    main()
