# Model switch and KV cache illustration prompt

Generated with the built-in `image_gen` tool. The reference figure supplied the
scientific example; the generated image replaces its visual style and layout.

- Output: [physical illustration](fig_model_switch_kv_physical.png)
- Reference: [exact arithmetic diagram](fig_model_switch_kv_exact_tokens.png)
- Scope: illustrative cache-history violation; tokens retain a legal rebuild explanation.

## Generation prompt

```text
Redesign the attached technical figure into a radically simpler, vivid physical illustration for a researcher who wants to understand the idea immediately. The attachment is a reference for the scientific meaning only: replace its layout, all equations, nearly all text, and flat diagram boxes.

Create a polished wide landscape scientific explainer with beautiful tactile 3D/isometric physical computer components on a warm off-white studio background. Large simple forms, restrained shadows, cobalt blue and amber orange, dark navy type. Editorial quality, friendly and concrete, not a sci-fi dashboard. Two clearly separated physical setups: SERVER on the left and VERIFIER on the right. The central visual lesson is that both output exactly the same token despite different cache histories.

LEFT — SERVER AFTER MODEL SWITCH:
A silver motherboard currently holds a large ORANGE processor visibly labeled “MODEL B”. Just behind it, a BLUE processor labeled “A” has been lifted from the same socket, with one clear short swap arrow toward the installed B. A separate BLUE memory cartridge with prominent embossed label “KV A” remains plugged into the motherboard. It physically fits and the machine runs; do not show a broken plug or disconnection. Highlight the retained blue cartridge with a small red warning halo and warning triangle. One tiny callout “kept” points to that BLUE memory. The orange B processor and blue A cache must be the most immediately visible mismatch. A small input slip “cat” enters the apparatus. A short output chute produces a large cream token tile with the word “runs”.

RIGHT — INDEPENDENT VERIFIER REPLAY:
An otherwise similar silver motherboard has the same ORANGE processor labeled “MODEL B”, and a matching ORANGE memory cartridge labeled “KV B”. One tiny callout “rebuilt” points to the orange memory. A small input slip “cat” feeds this apparatus; a subtle short arrow from that slip to the orange memory conveys rebuilding the prefix under B. Its output chute produces an IDENTICAL cream token tile labeled “runs”. This is a legitimate reference recomputation.

COMPARISON:
Place the two identical “runs” tiles prominently toward the lower center, one clearly connected to the server and one to the verifier. Put a large green equals sign between them. Beneath, one modest green checkmark badge says exactly “Exact token check: PASS”. The green badge checks only equality of the tokens, not the cache or server integrity. Preserve the RED warning around the server's BLUE cache. This apparent conflict—wrong blue cache, yet equal outputs and a green token check—is the main story.

TEXT:
Top title, large and spare: “Same token. Wrong cache.”
Setup headings: “Server” and “Verifier”.
Use only the short component/input/output labels specified above. No paragraphs, no logits, no numbers, no legends, no tables, no enclosing infographic cards. Optional tiny bottom note, exactly: “Toy example · cache history needs trusted evidence”.
Total text should stay extremely sparse, preferably under 45 words. Keep every label large and legible; clean sans-serif. Do not add extra explanatory prose.

SCIENTIFIC CONSTRAINTS:
A and B are different model weights. The server switches weights but wrongly keeps A's cache; the verifier rebuilds B's valid cache from the same observed prefix. Both can select the same next token “runs”. This is a cache-history violation invisible to token equality, not a false cryptographic proof or proof that the output is impossible. Show legal rebuilding positively on the verifier side. Do not depict a trajectory detector catching a bad cache from tokens alone. The blue cache must remain physically connected to the server running orange B. The verifier must not use blue A state. Preserve semantic accuracy while making the analogy visual and immediately intuitive.
```
