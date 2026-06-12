"""
Attacker-side view of the GLS verifier: given the (seed-derived) Gumbel noise and the
raw logits at a position, compute every candidate token's GLS score, logit rank, and
SAFE/SUSPICIOUS/DANGEROUS class — and from that the *admissible* (non-dangerous) set
that a covert channel may emit at that position.

The math here mirrors upstream exactly:
  GLS(c) = log Phi( -(g[c*] - g[c]) / (sigma*sqrt(2)) )
where g = logits + temperature * gumbel_noise (gumbel_noise = -log(Exp(1) draw)),
masked to the top-k/top-p support, and c* = argmax_c g[c] is the verifier's competitor.
A token c is:
  SAFE       if GLS(c) > gls_threshold
  DANGEROUS  if GLS(c) <= gls_threshold AND raw-logit rank(c) > logit_rank_threshold
  SUSPICIOUS otherwise.
(see upstream inference_verification/verify.py: compute_gumbel_likelihood_score + classify_tokens)

`check_fidelity()` asserts this vectorized computation is bit-for-bit consistent with
upstream's per-token `compute_gumbel_likelihood_score`, so any capacity/evasion number
produced here is a number against the *real* detector.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from dataclasses import dataclass

import torch

# --- make the upstream package importable without pulling in real vLLM -------------
# verify.py only needs vllm.RequestOutput as a type symbol on the GLS path; the scoring
# itself is torch-only. We satisfy that one import with a shim so we run the upstream
# scorer UNMODIFIED. (Real vLLM is only needed for generation / experiment E2.)
_UPSTREAM = Path(__file__).resolve().parent.parent / "upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
if "vllm" not in sys.modules:
    _shim = types.ModuleType("vllm")

    class RequestOutput:  # minimal stand-in; never instantiated by the GLS path
        pass

    _shim.RequestOutput = RequestOutput
    sys.modules["vllm"] = _shim

from inference_verification.verify import (  # noqa: E402  (after shim)
    apply_top_k_top_p,
    classify_tokens,
    TokenClassification,
)
from inference_verification.scoring_functions import (  # noqa: E402
    compute_gumbel_likelihood_score,
    exponential_to_gumbel,
)

EPSILON = 1e-12
SAFE = TokenClassification.SAFE
SUSPICIOUS = TokenClassification.SUSPICIOUS
DANGEROUS = TokenClassification.DANGEROUS


@dataclass
class GLSParams:
    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float = 0.95
    gumbel_sigma: float = 1.0
    gls_threshold: float = -5.0
    logit_rank_threshold: int = 32

    def _tk(self, device):
        return torch.tensor([self.top_k], device=device) if self.top_k is not None else None

    def _tp(self, device):
        return torch.tensor([self.top_p], device=device)


@dataclass
class PositionScores:
    """Per-position scoring of every vocab token (attacker's view of the verifier)."""
    gls: torch.Tensor          # [V] GLS log-prob per candidate (-inf outside support)
    ranks: torch.Tensor        # [V] raw-logit rank per candidate (0 = top)
    classes: torch.Tensor      # [V] int8: 0=SAFE 1=SUSPICIOUS 2=DANGEROUS
    competitor: int            # verifier's Gumbel-max token c*

    @property
    def admissible_mask(self) -> torch.Tensor:
        """Non-dangerous tokens (SAFE or SUSPICIOUS)."""
        return self.classes != 2

    @property
    def safe_mask(self) -> torch.Tensor:
        return self.classes == 0


def score_position(logits_V: torch.Tensor, exp_noise_V: torch.Tensor, p: GLSParams) -> PositionScores:
    """
    Vectorized GLS scoring of ALL vocab tokens at one position.

    logits_V:     [V] raw (pre-temperature) logits, float.
    exp_noise_V:  [V] the Exp(1) draws the verifier used for Gumbel-max at this position.
    """
    device = logits_V.device
    V = logits_V.shape[0]

    # support mask (top-k/top-p) — identical to upstream
    temp_logits = logits_V.clone()
    if p.temperature > 0.0:
        temp_logits = temp_logits / p.temperature
    masked = apply_top_k_top_p(temp_logits[None, :], p._tk(device), p._tp(device)).squeeze(0)
    support = torch.isfinite(masked)

    # perturbed logits g = logits + temperature * gumbel_noise, masked to support
    gumbel = exponential_to_gumbel(exp_noise_V.float(), EPSILON)
    g = logits_V + gumbel * p.temperature
    g = g.masked_fill(~support, float("-inf"))

    competitor = int(g.argmax().item())
    gap = g[competitor] - g          # >= 0 on support, +inf off support
    std = p.gumbel_sigma * (2.0 ** 0.5)
    z = (-gap / std).to(torch.float64)
    gls = torch.special.log_ndtr(z).to(torch.float32)   # log Phi(z); -inf off support

    # raw-logit rank (0 = highest). argsort desc -> position of each token id.
    order = torch.argsort(logits_V, descending=True)
    ranks = torch.empty(V, dtype=torch.long, device=device)
    ranks[order] = torch.arange(V, device=device)

    # classify (matches classify_tokens)
    classes = torch.empty(V, dtype=torch.int8, device=device)
    safe = gls > p.gls_threshold
    dangerous = (~safe) & (ranks > p.logit_rank_threshold)
    classes[safe] = 0
    classes[dangerous] = 2
    classes[(~safe) & (~dangerous)] = 1
    return PositionScores(gls=gls, ranks=ranks, classes=classes, competitor=competitor)


def admissible_tokens(scores: PositionScores, allow: str) -> torch.Tensor:
    """
    Ordered (ascending token-id) tensor of tokens the channel may emit at this position.
      allow='safe'        -> SAFE only (GLS > threshold): the regime that yields the
                             paper's ~0.5% headline; never raises the suspicious-token rate.
      allow='nondangerous'-> SAFE + SUSPICIOUS: higher capacity, but every SUSPICIOUS
                             token is exactly the signal the verifier samples for.
    Canonical ordering (sorted token id) is shared by encoder and decoder.
    """
    if allow == "safe":
        mask = scores.safe_mask
    elif allow == "nondangerous":
        mask = scores.admissible_mask
    else:
        raise ValueError(f"allow must be 'safe' or 'nondangerous', got {allow!r}")
    toks = torch.nonzero(mask, as_tuple=False).flatten()
    return torch.sort(toks).values


def check_fidelity(logits_V: torch.Tensor, exp_noise_V: torch.Tensor, p: GLSParams,
                   n_probe: int = 64, tol: float = 1e-3) -> float:
    """
    Assert the vectorized GLS equals upstream's per-token compute_gumbel_likelihood_score
    for n_probe random in-support tokens. Returns the max abs deviation.
    Raises AssertionError if any deviation exceeds tol.
    """
    device = logits_V.device
    sc = score_position(logits_V, exp_noise_V, p)
    support = torch.isfinite(sc.gls)
    cand = torch.nonzero(support, as_tuple=False).flatten()
    if cand.numel() == 0:
        return 0.0
    sel = cand[torch.randperm(cand.numel(), device=device)[:n_probe]]
    max_dev = 0.0
    for c in sel.tolist():
        ref = compute_gumbel_likelihood_score(
            logits_V=logits_V,
            exponential_noise_V=exp_noise_V,
            temperature=p.temperature,
            top_k=p._tk(device),
            top_p=p._tp(device),
            gold_idx=torch.tensor(c, device=device),
            noise_sigma=p.gumbel_sigma,
            apply_top_k_top_p_fn=apply_top_k_top_p,
            epsilon=EPSILON,
        )
        dev = abs(ref - float(sc.gls[c].item()))
        max_dev = max(max_dev, dev)
    assert max_dev <= tol, f"vectorized GLS deviates from upstream by {max_dev} > {tol}"
    return max_dev
