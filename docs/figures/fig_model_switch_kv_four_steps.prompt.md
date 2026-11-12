# Four step model switch and KV cache illustration

Created with the built-in `image_gen` tool, followed by one targeted edit.

- Final output: [four step illustration](fig_model_switch_kv_four_steps.png)
- Initial reference: [two path illustration](fig_model_switch_kv_flow.png)
- Numerical labels: stale-cache logits `(3, 2)` and fresh-cache logits `(5, 2)`; both choose `runs`.
- Scope: a toy cache-history violation; detecting the actual reuse requires trusted state evidence.

## Initial prompt

```text
Edit the attached algorithm illustration to show FOUR EXPLICIT NUMBERED STEPS, exactly matching the story below. Preserve the clean blue-A / orange-B / cream-token visual language and simple flow arrows. Keep it intuitive and mostly visual. Add the missing cache-construction arrow and two tiny horizontal logit bar charts. No hardware illustration, no paragraphs, no clutter.

Canvas: spacious landscape approximately 4:3, white background, crisp vector-style scientific explainer. A modest title occupying no more than 6% of height: “Both tokens pass. The cache handoff is wrong.”
One small subtitle: “Toy rule: switching is allowed; B needs a rebuilt or validated cache.”
Use four clearly ordered horizontal regions from top to bottom, each with a large numbered circle at the left. No heavy enclosing boxes. Leave generous vertical spacing. Make component labels and numbers easily readable.

STEP 1 — “A generates ‘cat’ and builds its cache”
Blue model block “A” has TWO clearly drawn outgoing branches: one to a cream token “cat”, another to a BLUE matrix-stack “KV A”. This illustrates A generating the token and constructing the corresponding prefix cache. The cache must visibly originate from A rather than floating unconnected. Position KV A so it can flow down into step 2. Do not use a cache from B in this step.
A small green “cat ✓” mark indicates the first exact token check passes.

STEP 2 — “Server switches to B, keeps KV A”
Show the BLUE KV A carried from step 1 into an ORANGE model block “B”. The blue-to-orange cache-input arrow is RED, labeled only “reuse”, with a small red warning mark. Use a tiny “A → B” switch tag above B if helpful. Do not recolor this cache orange. After B show a tiny horizontal bar chart titled “logits”, with exactly TWO bars:
runs  3
sleeps  2
Draw both bars from the same zero baseline. The runs bar is longer than the sleeps bar, 3 versus 2. Highlight runs. Then an arrow points to a cream token tile “runs”. This lane is the ACTUAL SERVER, using the wrong cache.

STEP 3 — “Verifier rebuilds the prefix under B”
Start with the SAME observed cream “cat” token; a faint dotted connector from step 1 may indicate prefix reuse, but it must not carry KV A into the verifier.
The cat token enters a compact orange “B” block labeled “rebuild”. This produces an ORANGE matrix-stack “KV B”. That orange cache feeds another orange “B” block representing next-token evaluation. Then show a tiny horizontal bar chart titled “logits”, with exactly TWO bars:
runs  5
sleeps  2
Use THE SAME SCALE as the step-2 chart: the value-5 runs bar is 5/3 as long as the earlier runs=3 bar, and both sleeps=2 bars have identical length. Then an arrow points to a cream “runs” tile vertically aligned with step 2's output. The model B parameters are the same in steps 2 and 3; the input cache differs.

STEP 4 — “Both exact token checks pass”
Visually compare two short token strips:
Server:   [cat] [runs]
Expected: [cat] [runs]
A green check beneath each matching token column, or two small equality marks, makes BOTH successful checks explicit. A green label says “2 / 2 PASS”.
Beside this output comparison, put the small BLUE KV A feeding ORANGE B motif with a red cross or warning and the short caption “Cache rule violated”.
The visual conclusion must be: two matching output tokens and green checks coexist with an invalid state handoff. Do not mark a cache audit green.

One restrained footer only: “Detecting the bad handoff requires trusted state evidence.”

Scientific meaning:
The exact toy logits are runs=3,sleeps=2 with A's stale cache, versus runs=5,sleeps=2 with B's fresh cache. Greedy decoding outputs runs in both. The model switch itself is allowed; unvalidated reuse of A's cache by B violates the declared rule. The verifier legally rebuilds B's cache. The output sequence therefore still has a legal explanation. This is not a counterexample to a fixed-model cryptographic proof. Do not claim tokens alone reveal the bad cache or that the sequence is impossible.

The viewer should understand the four actions by following colored blocks, cache stacks, bar lengths, and matching token strips. Keep the four short step headings as navigation, but do not add explanatory body text beyond the labels specified above. Avoid decorative machinery or oversized typography.
```

## Refinement prompt

```text
Make a minimal precision edit to this four-step figure. Preserve all four numbered steps, wording, colors, model blocks, cache stacks, output tokens, title, footer, and layout.

Correct ONLY these visual details:
1. The two logit charts must have an identical numerical scale and identical zero baseline positions. Use a fixed common maximum of 5. For step 2: runs=3 bar should be exactly 60% of the available full-scale width; sleeps=2 should be 40%. For step 3: runs=5 should be 100%; sleeps=2 should be 40%. The two sleeps=2 bars MUST have exactly the same pixel length. The step-3 runs=5 bar must be five-thirds the length of step-2 runs=3. Keep all four numeric labels 3,2,5,2 clearly outside the ends of their bars, so labels do not overlap chart borders. Increase the chart boxes slightly if needed, without changing the rest of the figure.
2. Remove the gray vertical dotted line that currently descends from model A to the verifier's cat token. It connects to the wrong origin and is unnecessary because the same prefix is already labeled explicitly. Do not replace it with another connector.
Do not change any other content, or add any new prose. This is a toy illustration; retain the footer about trusted state evidence.
```
