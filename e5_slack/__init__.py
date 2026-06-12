"""
E5 — Generation slack map.

A constructive *interpretability* experiment built on the GLS verifier from
  "Verifying LLM Inference to Detect Model Weight Exfiltration" (arXiv:2511.02620).

The verifier defines, at each generated position, the set of tokens that score SAFE
(GLS > threshold) — tokens statistically indistinguishable from an honest Gumbel-max
sample. The paper treats |SAFE set| as covert *capacity* to be bounded away. Here we
flip the lens: |SAFE set| is a per-token measure of how much *freedom* the model leaves
at that step — its local decision slack. We map where that slack lives across a real
generation (token position, syntactic category) and relate it to next-token entropy.

Reuses e4_stego.gls (the validated, upstream-identical GLS scorer) unchanged.
"""
