# Model switch and KV cache flow illustration

Generated with the built-in `image_gen` tool.

- Output: [algorithm sketch](fig_model_switch_kv_flow.png)
- Edit reference: [physical illustration](fig_model_switch_kv_physical.png)
- Scope: token agreement can hide a cache-history violation; trusted state evidence is required to detect the handoff.

## Generation prompt

```text
Redesign the attached image into an intuitive ALGORITHMIC sketch. Preserve the scientific meaning, but completely replace the physical hardware rendering. No motherboards, no chips, no metal, no 3D machinery. This should look like a beautifully clear research-paper algorithm figure with simple colored model blocks, token tiles, little cache/tensor grids, and arrows. Extremely little text; let the paths show the algorithm.

Wide landscape canvas, clean white background, no enclosing cards or heavy borders. Cobalt blue for model A and its cached state; amber orange for model B and its cached state; restrained red for invalid state reuse; green only for matching output tokens. Flat vector-like shapes, crisp strokes, abundant whitespace. Large readable component labels. Avoid ornamental shadows. Modest title: “Same output, different state”.

Show TWO HORIZONTAL LANES, left to right:

TOP LANE — “Server”
1. A blue rounded rectangle labeled “A” emits a cream token tile “cat”.
2. That “cat” tile feeds an orange rounded rectangle labeled “B” farther right. A small “switch” label above the path indicates switching from A to B.
3. Below the cat tile is a blue mini-stack of square matrices labeled “KV A”, representing the prefix state previously built by A. A clearly visible RED curved arrow labeled “reuse” carries this BLUE cache into a lower input port of the ORANGE B block. Put a small red warning mark at that port. This is the single highlighted mistake.
4. B outputs a cream token tile “runs” at the far right.
The emitted cat and the carried KV A are distinct inputs. Show blue KV A visibly entering B; do not show it changing to orange.

BOTTOM LANE — “Verifier”
1. A cream tile “cat”, aligned below the top cat tile, represents the SAME observed prefix.
2. The cat tile enters a small orange B block with the concise label “rebuild”.
3. This block produces an ORANGE mini-stack of matrices labeled “KV B”.
4. Orange KV B flows into a second orange B block (this is the next-token evaluation), aligned vertically with the server's B.
5. This B also outputs the identical cream tile “runs”, aligned vertically below the server output.
Use a faint vertical dotted link between the two “cat” tiles to indicate the same observed prefix, without suggesting server cache is shared with the verifier.

ON THE FAR RIGHT:
Connect the two “runs” tiles with one green comparison bracket and an equals sign. A concise label “Token check ✓” makes clear that token verification passes.
Keep the red cache-input warning on the top path visually distinct from this green output check.

Optional tiny annotation between the two cache matrices: “KV A ≠ KV B”. Do not cross the main flow arrows.
One small footer only: “Trusted state evidence is needed to detect the bad handoff.”

Visual lesson: EXACT output tokens match, but the server's B consumed stale A state, while the verifier's B consumed correctly rebuilt B state. The image must read immediately without reading a paragraph. It is a toy history-verification failure, not a false proof of a fixed model's output, and the output has a legal rebuild explanation. Do not draw an all-seeing verifier that learns the server cache from output alone.

Only the above short labels. No equations, no logits, no prose blocks, no decorative physical analogy. Compact, clear algorithmic visual flow, with the two different colored caches and the matching terminal tokens as the main visual comparison.
```
