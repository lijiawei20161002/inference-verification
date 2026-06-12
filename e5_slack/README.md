# E5 — Generation Slack Map

A constructive experiment built **on top of** the GLS verifier from
*"Verifying LLM Inference to Detect Model Weight Exfiltration"* (arXiv:2511.02620), not a
test of its bound. The paper defines, per position, the set of tokens that score **SAFE**
(`GLS > −5`) — tokens statistically indistinguishable from an honest Gumbel-max sample —
and treats `|SAFE|` as covert *capacity* to be bounded away. **E5 flips the lens:** that
same `|SAFE|` is a per-token measure of how much *freedom* the model leaves at each step —
its local decision **slack**. We map where the slack lives in a real generation.

Reuses `e4_stego.gls` (the upstream-identical GLS scorer) unchanged. No attacker, no codec.

## Method
Honestly generate a continuation (emit the Gumbel-max winner `c*`). At each position:
- **slack** = `log2(mean |SAFE|)`, averaging `|SAFE|` over `n_seed_avg` independent Gumbel
  draws so the metric reflects the logit geometry, not one noise sample;
- **entropy** = Shannon entropy of the temperature/top-k/top-p sampling distribution;
- record the emitted token's **syntactic category** and raw-logit rank.

```bash
. ../.venv/bin/activate
python -m e5_slack.run_e5 --n-prompts 4 --max-tokens 48 --n-seed-avg 4
```

## Findings (Qwen2.5-3B-Instruct, 192 tokens)

**1. Per-token covert capacity ≈ next-token entropy.** slack vs entropy is essentially the
identity line, `r = 0.956`, slope ≈ 1 (`figs/slack_vs_entropy.png`). Mean slack 1.13 bits ≈
mean entropy 1.00 bits. So the GLS "capacity" the paper bounds is, per token, just the
model's intrinsic uncertainty `H` — a clean, intuitive restatement of what the SAFE set *is*.

**2. Slack is bimodal, not uniform.** The 1.13-bit mean hides the structure: **33% of tokens
are fully forced** (0 slack — the model has effectively one admissible choice), while ~half
carry ≥1 bit and ~23% carry ≥2 bits (max 3.75). Freedom comes in bursts, not a steady drip.

**3. Freedom lives on content-word *starts*; it collapses inside words, on numbers, and on
formatting.** Mean slack by category:

| category | mean slack (bits) | n |
|---|---|---|
| content-word (start) | **1.41** | 83 |
| function-word (start) | 1.11 | 67 |
| punctuation | 0.79 | 20 |
| word-continuation (subword) | 0.50 | 19 |
| number | 0.00 | 2 |
| whitespace | 0.00 | 1 |

The model's real choices are at content-bearing word boundaries; once it commits to a word,
the subword continuation is largely forced, and numbers/whitespace are deterministic.

**4. Slack fluctuates throughout the sequence** with no strong positional trend
(`figs/slack_vs_position.png`) — it's driven by local content, not by depth into the answer.

### Inline slack heatmap
The driver prints the continuation with a block glyph per token whose height ∝ slack, e.g.:

```
Photos ynthesis  is  a▂ fundamental▄ biological▃ process  that▄ occurs▅ in▂ plants▃, ...
```

Forced tokens (`Photos`, `ynthesis`, `process`) carry no block; the slack sits on
`fundamental`, `occurs`, `that` — exactly the semantically open choices.

## Outputs
- `e5_slack.json` — aggregates, per-category slice, slack↔entropy correlation, and a fully
  annotated sample transcript (per-token slack/entropy/SAFE-set size).
- `figs/` — `slack_hist`, `slack_vs_entropy`, `slack_by_category`, `slack_vs_position`.

## Takeaway
Read through the verifier rather than against it, `|SAFE|` is a free, per-token map of where
a language model is *deciding* versus merely *transcribing*. The same quantity the paper
bounds for security doubles as an interpretability signal: covert capacity = local entropy,
and that entropy is overwhelmingly spent at content-word boundaries.
