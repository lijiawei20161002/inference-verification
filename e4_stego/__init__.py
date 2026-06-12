"""
E4 — Concrete end-to-end steganographic codec against the GLS verifier.

Red-team experiment E4 from the program against
  "Verifying LLM Inference to Detect Model Weight Exfiltration" (arXiv:2511.02620)
  repo: github.com/RoyRin/inference_verification_for_model_weight_exfiltration

GOAL: fill the gap the paper explicitly leaves open — build the encoder/decoder the
authors abstain from — and *measure* realized covert capacity against their REAL GLS
scorer (not a reimplementation). The defender side is upstream code, unmodified.

Honesty contract (so evasion claims are comparable to the paper):
  * Attacker (encoder/decoder) = code in this package.
  * Defender (verifier) = upstream `inference_verification.verify.verify_outputs` +
    `classify_tokens`, run verbatim on the attack transcript.
  * We inherit the paper's *generous* adversary affordances: exact logit copy
    (same model/dtype/device on both sides) and knowledge of the seed
    (the "adversary-has-seed" / conservative regime, which is also experiment E1's
    realistic insider). These assumptions are stated, not hidden.
"""
