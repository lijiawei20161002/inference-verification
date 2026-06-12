"""
E5 driver — map a model's per-token generation slack and where it lives.

  python -m e5_slack.run_e5 --n-prompts 4 --max-tokens 48 --n-seed-avg 4

Outputs:
  e5_slack/e5_slack.json     aggregates, per-category slice, correlation, sample transcript
  e5_slack/figs/*.png        slack histogram, slack-vs-entropy, slack-by-category, slack-vs-pos
  stdout                     markdown summary + an inline "slack heatmap" of one continuation
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from e4_stego.gls import GLSParams
from .slackmap import generate_and_map

PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main causes of inflation in modern economies?",
    "Describe the process of training a neural network from scratch.",
    "Give me a recipe for a simple vegetable soup.",
    "Summarize the plot of a hypothetical detective novel set in Venice.",
]

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"


def build_prompt_ids(tokenizer, text: str) -> list[int]:
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer(text)["input_ids"]


def shade(bits: float, lo: float = 0.0, hi: float = 4.0) -> str:
    """Map slack (bits) to a block glyph for the inline heatmap."""
    blocks = " ▁▂▃▄▅▆▇█"
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (bits - lo) / (hi - lo)))
    return blocks[int(round(t * (len(blocks) - 1)))]


def render_heatmap(tokens, width_chars: int = 96) -> str:
    """Inline annotated transcript: each token followed by its slack in bits as a block + value."""
    lines, cur = [], ""
    for t in tokens:
        piece = f"{t.token_str}{shade(t.slack_bits)}"
        if len(cur) + len(piece) > width_chars:
            lines.append(cur)
            cur = ""
        cur += piece
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def make_plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slack = np.array([r["slack_bits"] for r in rows])
    ent = np.array([r["entropy_bits"] for r in rows])
    pos = np.array([r["pos"] for r in rows])
    cats = [r["category"] for r in rows]
    FIGS.mkdir(parents=True, exist_ok=True)

    # 1. slack histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(slack, bins=24, color="#3b6", edgecolor="k", alpha=0.8)
    ax.axvline(slack.mean(), color="k", ls="--", label=f"mean {slack.mean():.2f} bits")
    ax.set_xlabel("per-token slack  log2(mean |SAFE|)  [bits]")
    ax.set_ylabel("# tokens")
    ax.set_title("Where the model leaves room: per-token slack")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "slack_hist.png", dpi=120); plt.close(fig)

    # 2. slack vs entropy
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ent, slack, s=10, alpha=0.4, color="#36b")
    if len(ent) > 2 and ent.std() > 0:
        m, b = np.polyfit(ent, slack, 1)
        xs = np.linspace(ent.min(), ent.max(), 50)
        r = float(np.corrcoef(ent, slack)[0, 1])
        ax.plot(xs, m * xs + b, "r-", label=f"fit  r={r:.2f}")
        ax.legend()
    ax.set_xlabel("next-token entropy [bits]")
    ax.set_ylabel("slack [bits]")
    ax.set_title("Slack tracks intrinsic uncertainty")
    fig.tight_layout(); fig.savefig(FIGS / "slack_vs_entropy.png", dpi=120); plt.close(fig)

    # 3. slack by category
    by = defaultdict(list)
    for c, s in zip(cats, slack):
        by[c].append(s)
    order = sorted(by, key=lambda k: -np.mean(by[k]))
    means = [np.mean(by[k]) for k in order]
    counts = [len(by[k]) for k in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(order, means, color="#a63", alpha=0.85)
    for i, (m, n) in enumerate(zip(means, counts)):
        ax.text(i, m + 0.03, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean slack [bits]")
    ax.set_title("Slack by syntactic category")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIGS / "slack_by_category.png", dpi=120); plt.close(fig)

    # 4. slack vs position (rolling mean)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pos, slack, s=8, alpha=0.25, color="#777")
    o = np.argsort(pos)
    ps, ss = pos[o], slack[o]
    if len(ps) >= 5:
        w = max(3, len(ps) // 20)
        roll = np.convolve(ss, np.ones(w) / w, mode="valid")
        ax.plot(ps[w - 1:], roll, "b-", lw=2, label=f"rolling mean (w={w})")
        ax.legend()
    ax.set_xlabel("token position in continuation")
    ax.set_ylabel("slack [bits]")
    ax.set_title("Slack across the generation")
    fig.tight_layout(); fig.savefig(FIGS / "slack_vs_position.png", dpi=120); plt.close(fig)

    return ["slack_hist.png", "slack_vs_entropy.png", "slack_by_category.png", "slack_vs_position.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n-seed-avg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gls-threshold", type=float, default=-5.0)
    ap.add_argument("--logit-rank-threshold", type=int, default=32)
    ap.add_argument("--out", default=str(HERE / "e5_slack.json"))
    args = ap.parse_args()

    params = GLSParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        gls_threshold=args.gls_threshold, logit_rank_threshold=args.logit_rank_threshold,
    )

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    eos_id = tok.eos_token_id if not isinstance(tok.eos_token_id, list) else tok.eos_token_id[0]

    maps = []
    for i, text in enumerate(PROMPTS[: args.n_prompts]):
        pids = build_prompt_ids(tok, text)
        m = generate_and_map(model, tok, pids, params, seed=args.seed,
                             max_tokens=args.max_tokens, n_seed_avg=args.n_seed_avg,
                             eos_id=eos_id, prompt_idx=i)
        maps.append(m)
        print(f"[map] prompt {i}: {len(m.tokens)} tokens, "
              f"mean slack {np.mean([t.slack_bits for t in m.tokens]):.2f} bits")

    rows = [vars(t) | {"prompt": m.prompt_idx} for m in maps for t in m.tokens]
    slack = np.array([r["slack_bits"] for r in rows])
    ent = np.array([r["entropy_bits"] for r in rows])
    corr = float(np.corrcoef(ent, slack)[0, 1]) if slack.std() > 0 and ent.std() > 0 else float("nan")

    by_cat = {}
    cat_rows = defaultdict(list)
    for r in rows:
        cat_rows[r["category"]].append(r["slack_bits"])
    for c, vals in cat_rows.items():
        by_cat[c] = {"n": len(vals), "mean_slack_bits": float(np.mean(vals)),
                     "median_slack_bits": float(np.median(vals))}

    figs = make_plots(rows)

    sample = maps[0]
    heatmap = render_heatmap(sample.tokens)

    report = {
        "model": args.model, "n_prompts": args.n_prompts, "max_tokens": args.max_tokens,
        "n_seed_avg": args.n_seed_avg, "seed": args.seed,
        "n_tokens": len(rows),
        "slack_bits": {
            "mean": float(slack.mean()), "median": float(np.median(slack)),
            "p90": float(np.percentile(slack, 90)), "max": float(slack.max()),
            "frac_zero_slack": float((slack < 0.01).mean()),
            "frac_ge_1bit": float((slack >= 1.0).mean()),
            "frac_ge_2bit": float((slack >= 2.0).mean()),
        },
        "entropy_bits_mean": float(ent.mean()),
        "corr_slack_entropy": corr,
        "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1]["mean_slack_bits"])),
        "figures": figs,
        "sample_transcript": {
            "prompt": PROMPTS[0],
            "text": "".join(t.token_str for t in sample.tokens),
            "tokens": [{"t": t.token_str, "cat": t.category,
                        "slack_bits": round(t.slack_bits, 2),
                        "entropy_bits": round(t.entropy_bits, 2),
                        "safe_mean": round(t.safe_size_mean, 1)} for t in sample.tokens],
        },
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # ---- stdout summary ----
    print("\n" + "=" * 84)
    print("E5 — GENERATION SLACK MAP")
    print("=" * 84)
    print(f"  tokens analysed       : {len(rows)}  ({args.n_prompts} prompts × ~{args.max_tokens} tok)")
    print(f"  mean slack            : {slack.mean():.2f} bits   median {np.median(slack):.2f}   "
          f"p90 {np.percentile(slack,90):.2f}   max {slack.max():.2f}")
    print(f"  forced (0-slack) tokens: {100*(slack<0.01).mean():.1f}%   "
          f"≥1 bit: {100*(slack>=1).mean():.1f}%   ≥2 bits: {100*(slack>=2).mean():.1f}%")
    print(f"  mean next-token entropy: {ent.mean():.2f} bits")
    print(f"  corr(slack, entropy)   : r = {corr:.3f}")
    print("\n  slack by category (mean bits, n):")
    for c, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["mean_slack_bits"]):
        bar = "█" * int(round(d["mean_slack_bits"] * 6))
        print(f"    {c:16s} {d['mean_slack_bits']:5.2f}  {bar:<24s} n={d['n']}")
    print("\n  inline slack heatmap (block height ∝ slack; prompt 0 continuation):")
    print("  " + "-" * 82)
    for line in heatmap.splitlines():
        print("  " + line)
    print("  " + "-" * 82)
    print(f"\n[saved] {args.out}")
    print(f"[saved] {FIGS}/  ({', '.join(figs)})")


if __name__ == "__main__":
    main()
