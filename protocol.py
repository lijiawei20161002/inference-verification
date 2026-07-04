"""The inference-verification protocol, end to end.

Roles
-----
Provider: serves requests by generating with the model. Honest => full n_layer
  pass. Cheating => early-exit after n_layer//2 layers (a cheaper approximation
  that saves ~half the compute). Commits to all transcripts via a Merkle root.
Client: audits a random fraction q of requests. For each audited request it
  (1) verifies the Merkle authentication path (binding) and (2) re-scores the
  committed transcript with a full teacher-forced pass and compares to the
  committed logits within a tolerance band eps. A cheat is CAUGHT iff audited
  AND flagged.

Nondeterminism: the client's re-score adds Gaussian noise sigma to emulate
non-bit-exact hardware. eps must exceed the noise to avoid false positives.
"""
import numpy as np
import model as M
from commit import leaf_hash, MerkleTree


class Transcript:
    __slots__ = ('prompt', 'tokens', 'logits', 'cheated', 'gen_flops')
    def __init__(self, prompt, tokens, logits, cheated, gen_flops):
        self.prompt = prompt
        self.tokens = tokens
        self.logits = logits
        self.cheated = cheated
        self.gen_flops = gen_flops


def serve(mdl: M.Model, prompt, T, cheat: bool):
    """Provider serves one request. Returns a Transcript with real FLOP count."""
    n_layers = mdl.cfg.n_layer // 2 if cheat else mdl.cfg.n_layer
    M.flops_reset()
    toks, lg = mdl.generate(prompt, T, n_layers=n_layers)
    return Transcript(prompt, toks, lg, cheat, M.flops_get())


def commit(transcripts, quant=1e-3):
    """Provider builds the Merkle commitment over all transcripts."""
    leaves = [leaf_hash(t.prompt, t.tokens, t.logits, quant) for t in transcripts]
    return MerkleTree(leaves)


def audit_one(mdl: M.Model, t: Transcript, tree: MerkleTree, index, root,
              eps, sigma, rng, quant=1e-3, cache=None):
    """Client audits request `index`.

    Returns (binding_ok, flagged, linf, token_mismatch, verify_flops).
    `flagged` means the re-score disagrees with the committed transcript beyond
    tolerance.

    `cache` (optional dict) memoizes the *deterministic* teacher-forced re-score
    keyed by transcript identity. The forward pass is a pure function of the
    fixed weights and committed tokens, so recomputing it for the same
    transcript yields bit-identical logits; caching it avoids repeating that
    work across the thousands of audits in a Monte-Carlo sweep. Every audit
    still performs a *real* Merkle-path verification and draws *fresh*,
    independent hardware noise on the re-score, and `verify_flops` reports the
    true per-audit re-score cost the client pays whether or not it was cached.
    """
    # (1) binding: committed leaf really is under the published root
    leaf = leaf_hash(t.prompt, t.tokens, t.logits, quant)
    binding_ok = MerkleTree.verify(leaf, index, tree.proof(index), root)

    # (2) re-score: one full teacher-forced pass over prompt+committed tokens
    L = len(t.prompt)
    key = id(t)
    if cache is not None and key in cache:
        base, verify_flops = cache[key]                    # deterministic; reused
    else:
        full = np.concatenate([t.prompt, t.tokens])
        M.flops_reset()
        vlogits = mdl.forward_full(full)                   # full n_layer honest pass
        verify_flops = M.flops_get()
        base = vlogits[L - 1 : L - 1 + len(t.tokens)]
        if cache is not None:
            cache[key] = (base, verify_flops)

    # hardware nondeterminism: independent noise on the re-score (fresh per audit)
    recomputed = base + rng.normal(0, sigma, base.shape).astype(np.float32)

    # the client compares against the *committed* (quantized) logits
    committed = np.round(t.logits / quant) * quant
    linf = float(np.abs(recomputed - committed).max())
    token_mismatch = bool((recomputed.argmax(-1) != t.tokens).any())
    flagged = linf > eps                          # primary rule: logit L-inf band
    return binding_ok, flagged, linf, token_mismatch, verify_flops
