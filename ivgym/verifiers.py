"""Information-directed verification: ONE abstraction for every detector.

Motivation
----------
Inference verification used to be three separate families in this repo:

  * white-box **recomputation defenses** (`token_difr`, `cross_entropy`,
    `token_toploc`, `activation_difr`) that re-run the reference model M;
  * black-box **I/O detectors** (`surface_stat`/`surprise`, `surface_rank`,
    `surface_tokens`) that read only outputs / a cheap proxy;
  * the **acceptance-rate fingerprint** (`accept_rate`) comparing the served
    distribution `p` to the client's proxy `q`.

They looked like different species, but each one is the *same* object: a per-token
aggregation of an evidence signal. Every score above is a cheap surrogate for the
per-token log-likelihood-ratio contribution in the hypothesis test "did the
provider run the intended model M under spec phi?". The only axis that actually
separates the families is **which distribution the verifier paid to obtain** --
the cheap proxy `q` (Tier-0) or a real recompute of M (Tier-1).

The unified statistic
---------------------
A `Verifier` is a triple ``(value, evidence, aggregation)`` evaluated under a
recompute ``budget`` B::

    S  =  A_t (  w(value_t) * evidence_t  )     with Tier-1 evidence computed
                                                only on the top-B fraction of
                                                positions by value_t

where

  * ``evidence_t`` estimates the per-token LLR contribution (what to measure);
  * ``value_t`` is a CHEAP, proxy-only estimate of how much verification value a
    position carries -- i.e. where to spend the expensive recompute. The
    principled default is the proxy's entropy ``H(q_t)``: positions that are
    near-deterministic under many plausible serving configurations carry almost
    no evidence about whether M was run, so recompute spent there is wasted. This
    is the token-level analogue of allocating expensive computation only to
    high-entropy positions (cf. "The Flexibility Trap", ICML 2025, which
    evaluates costly probability ratios only at the top-entropy tokens).

Prior verifiers are recovered as corners of this statistic:

    token_difr / cross_entropy / toploc  ->  value = uniform, evidence Tier-1, B=1
    surface_stat / surface_rank          ->  value = uniform, evidence Tier-0, B=0
    accept_rate                          ->  value = uniform, evidence Tier-0, B=0
    selective recompute                  ->  value = entropy, evidence Tier-1, B<1

Only the last row uses a non-uniform ``value`` -- that single degree of freedom
is what "information-directed" adds.

Extending
---------
Subclass `Verifier` (Tier-0) or `Tier1Verifier` (needs a recompute of M),
`@register` it, and `harness.verify` scores every config with it. This is the one
registry -- it replaces the old `defenses` and `io_detectors` registries.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import spec_decode as sd
from .core import VContext
from .sampling import filtered_logits, log_softmax

_EPS = 1e-12
RANK_CAP = 64.0

# ---------------------------------------------------------------------------
# Registry (the ONE registry -- replaces defenses._REGISTRY + io_detectors._REGISTRY)
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, "Verifier"] = {}


def register(v):
    """Register a verifier. Accepts a `Verifier` *instance* or *subclass*
    (instantiated with its defaults), so it works as a class decorator and
    restores a subclass-declared `name`. Returns its argument unchanged."""
    if isinstance(v, type):
        inst = v()
        inst.name = v.name
        v_inst = inst
    else:
        v_inst = v
    _REGISTRY[v_inst.name] = v_inst
    return v


def get(name: str) -> "Verifier":
    return _REGISTRY[name]


def all_verifiers() -> dict[str, "Verifier"]:
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Value functions: cheap, PROXY-ONLY estimates of per-token verification value.
# Every one reads only ctx.proxy_logits (a small, cheap, DIFFERENT model), never
# ctx.ref_logits -- so computing where to spend recompute never itself recomputes
# M. Higher value => audit this position first.
# ---------------------------------------------------------------------------
def _proxy_probs(ctx: VContext) -> np.ndarray:
    """Row-normalized proxy probabilities `q`, shape [T, V], at spec temperature."""
    if ctx.proxy_logits is None:
        raise RuntimeError("value/evidence needs proxy logits (set needs_proxy=True)")
    temp = max(ctx.sampling.temperature, 1e-6)
    z = ctx.proxy_logits / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def value_uniform(ctx: VContext) -> np.ndarray:
    """Every position equally valuable -- reproduces the classic uniform token
    budget (all prior verifiers). Needs no proxy."""
    return np.ones(len(ctx.claimed_tokens))


def value_entropy(ctx: VContext) -> np.ndarray:
    """Proxy entropy H(q_t) -- the principled default. Near-deterministic
    positions (low entropy) are served identically under most configurations and
    carry little evidence; the evidence lives where q is uncertain."""
    q = _proxy_probs(ctx)
    return -np.sum(q * np.log(q + _EPS), axis=1)


def value_tie_margin(ctx: VContext) -> np.ndarray:
    """Near-tie-ness -(q1 - q2) of the two largest proxy probabilities (higher =
    more tie-like). The cheapest value signal and the one tuned for quant/fp8
    corruption, which flips the sampled token only at near-tie positions."""
    q = _proxy_probs(ctx)
    top2 = np.partition(q, -2, axis=1)[:, -2:]
    return -(top2[:, 1] - top2[:, 0])


def value_surprisal(ctx: VContext) -> np.ndarray:
    """Proxy surprisal -log q(x_t) of the claimed token: a claimed token that is
    unlikely under the proxy is a high-value position to audit."""
    q = _proxy_probs(ctx)
    idx = np.asarray(ctx.claimed_tokens, int)
    return -np.log(q[np.arange(len(idx)), idx] + _EPS)


_VALUE_FNS = {
    "uniform": value_uniform,
    "entropy": value_entropy,
    "tie_margin": value_tie_margin,
    "surprisal": value_surprisal,
}


def value_of(name: str, ctx: VContext) -> np.ndarray:
    try:
        return _VALUE_FNS[name](ctx)
    except KeyError:
        raise ValueError(f"unknown value fn {name!r}; choose from {sorted(_VALUE_FNS)}")


def register_value_fn(name: str, fn) -> None:
    """Add a per-token value signal to the registry, so `harness.verify(value_fn=)`
    reaches it like a built-in. The extension point a *learned* triage head plugs
    into: `ivgym.triage.head_value_fn(head)` returns exactly such a callable
    (`VContext -> [T] float`), and it must read only Tier-0 context fields."""
    _VALUE_FNS[name] = fn


def value_fn_names() -> list[str]:
    return sorted(_VALUE_FNS)


# ---------------------------------------------------------------------------
# Verifier base
# ---------------------------------------------------------------------------
class Verifier:
    """A per-token evidence scorer. `evidence(ctx)` returns a length-T array
    (higher => more likely a deviation from spec phi, so it can be thresholded /
    fed to `evaluate` directly). `value(ctx)` is the cheap allocation signal used
    when the driver runs at a partial recompute budget or with value-weighting.

    Tier-0 verifiers read only Tier-0 context fields (proxy/served logits, text);
    Tier-1 verifiers (see `Tier1Verifier`) recompute M and read `ref_logits`.
    """

    name: str = "base"
    tier: int = 0                  # 0 = no recompute of M; 1 = needs ref_logits/activation
    needs_seed: bool = False       # requires shared Gumbel noise? (Tier-1)
    needs_activation: bool = False
    needs_proxy: bool = False      # driver fills ctx.proxy_logits?
    needs_served: bool = False     # driver fills ctx.served_logits?
    needs_text: bool = False       # needs a real-text backend (prompt_text/decode)?
    needs_fit: bool = False        # must be .fit(...) on labeled data before scoring?
    verifier_flops: float = 0.0    # relative verifier cost (1.0 = full recompute of M)
    value_fn: str = "entropy"      # which cheap value signal directs the budget
    neutral: float = 0.0           # score for a position that was not audited (Tier-1)

    def value(self, ctx: VContext) -> np.ndarray:
        return value_of(self.value_fn, ctx)

    def evidence(self, ctx: VContext) -> np.ndarray:
        raise NotImplementedError


class Tier1Verifier(Verifier):
    """Base for verifiers that recompute M. Implement `score_token`; the shared
    `evidence` loop scores only the audited rows (`ctx.audit_mask`, filled by the
    driver at the requested budget) and leaves the rest at `neutral` -- unaudited
    positions cost no recompute. At budget 1.0 the whole mask is True, recovering
    a full recompute."""

    tier: int = 1

    def score_token(self, ref_logits: np.ndarray, gumbel: np.ndarray | None,
                    claimed: int, sampling, fingerprint=None, ref_fingerprint=None) -> float:
        raise NotImplementedError

    def evidence(self, ctx: VContext) -> np.ndarray:
        n = len(ctx.claimed_tokens)
        out = np.full(n, float(self.neutral))
        mask = ctx.audit_mask if ctx.audit_mask is not None else np.ones(n, bool)
        for i in np.nonzero(mask)[0]:
            fp = ctx.fingerprints[i] if ctx.fingerprints is not None else None
            ref_fp = ctx.ref_fingerprints[i] if ctx.ref_fingerprints is not None else None
            g = ctx.gumbel[i] if ctx.gumbel is not None else None
            out[i] = self.score_token(ctx.ref_logits[i], g, ctx.claimed_tokens[i],
                                      ctx.sampling, fp, ref_fp)
        return out


# ===========================================================================
# Tier-1 verifiers: recompute the reference model M.
# ===========================================================================
@dataclass
class TokenDiFR(Tier1Verifier):
    """Clipped post-Gumbel logit margin (DiFR's default metric). delta =
    z[v_token] - z[claimed]; v_token = verifier's argmax under shared noise. 0 if
    tokens agree, grows with divergence, +delta_max if the claimed token was
    filtered out by top-k/top-p."""

    name: str = "token_difr"
    needs_seed: bool = True
    delta_max: float = 30.0

    def score_token(self, ref_logits, gumbel, claimed, sampling, fingerprint=None,
                    ref_fingerprint=None):
        s = sampling
        filt = filtered_logits(ref_logits, s.top_k, s.top_p)
        z = filt + s.temperature * gumbel
        v_token = int(np.argmax(z))
        if filt[claimed] <= -1e29:
            return self.delta_max
        return min(float(z[v_token] - z[claimed]), self.delta_max)


@dataclass
class CrossEntropy(Tier1Verifier):
    """Negative log-likelihood of the claimed token under the verifier softmax.
    Seed-free -- and defeatable by temperature tuning."""

    name: str = "cross_entropy"
    needs_seed: bool = False

    def score_token(self, ref_logits, gumbel, claimed, sampling, fingerprint=None,
                    ref_fingerprint=None):
        logp = log_softmax(ref_logits / max(sampling.temperature, 1e-6))
        return float(-logp[claimed])


@dataclass
class ActivationDiFR(Tier1Verifier):
    """L2 distance between provider and verifier projected activations."""

    name: str = "activation_difr"
    needs_seed: bool = False
    needs_activation: bool = True

    def score_token(self, ref_logits, gumbel, claimed, sampling, fingerprint=None,
                    ref_fingerprint=None):
        if fingerprint is None or ref_fingerprint is None:
            return 0.0
        return float(np.linalg.norm(fingerprint - ref_fingerprint))


@dataclass
class TokenTOPLOC(Tier1Verifier):
    """TOPLOC-flavoured score: rank of the claimed token in the verifier's
    filtered (top-k/top-p) distribution. 0 = verifier's argmax; larger = the
    provider emitted something the verifier ranks lower. Seed-free."""

    name: str = "token_toploc"
    needs_seed: bool = False
    rank_cap: float = 50.0

    def score_token(self, ref_logits, gumbel, claimed, sampling, fingerprint=None,
                    ref_fingerprint=None):
        s = sampling
        filt = filtered_logits(ref_logits, s.top_k, s.top_p)
        if filt[claimed] <= -1e29:
            return self.rank_cap
        rank = float(np.sum(filt > filt[claimed]))
        return min(rank, self.rank_cap)


# ===========================================================================
# Tier-0 verifiers: no recompute of M.
# ===========================================================================
# Shared output-only feature extraction (proxy perplexity/rank + pure token-id
# statistics). Every feature is a function ONLY of the claimed tokens and, when
# allowed, the CHEAP proxy logits -- never M's recomputed logits.
# ---------------------------------------------------------------------------
def proxy_nll(ctx: VContext) -> np.ndarray:
    """Per-token NLL of the claimed token under the cheap proxy's (temperature-
    scaled) softmax. 'Cheap model polices expensive model.'"""
    q = _proxy_probs(ctx)
    idx = np.asarray(ctx.claimed_tokens, int)
    return -np.log(q[np.arange(len(idx)), idx] + _EPS)


def proxy_rank(ctx: VContext) -> np.ndarray:
    """Per-token rank of the claimed token under the proxy logits (0 = proxy's
    argmax), capped at RANK_CAP. TOPLOC-flavoured but off a cheap proxy."""
    if ctx.proxy_logits is None:
        raise RuntimeError("proxy_rank needs proxy logits (set needs_proxy=True)")
    out = np.empty(len(ctx.claimed_tokens))
    for i, t in enumerate(ctx.claimed_tokens):
        row = ctx.proxy_logits[i]
        out[i] = min(float(np.sum(row > row[t])), RANK_CAP)
    return out


def token_surface_features(tokens: list[int]) -> dict[str, np.ndarray]:
    """Output-only features needing NO model at all -- pure token-id statistics
    (the zero-FLOP extreme). Only genuinely per-token features: `is_repeat` (does
    this token equal the previous one?) aggregates to the repetition rate under
    batch averaging. Sequence-level scalars are deliberately excluded (broadcast
    to tokens they collapse the batch-mean statistic to a spurious offset)."""
    n = len(tokens)
    arr = np.asarray(tokens)
    is_repeat = np.zeros(n)
    if n > 1:
        is_repeat[1:] = (arr[1:] == arr[:-1]).astype(float)
    return {"is_repeat": is_repeat}



@dataclass
class SurfaceStat(Verifier):
    """Scores each claimed token by its surprisal (NLL) or rank under a small,
    cheap proxy LM -- the Tier-0 analogue of `cross_entropy`/`token_toploc`,
    reading a cheap proxy instead of recomputing M. The cheap end of the
    verifier-FLOPs axis."""

    name: str = "surface_stat"
    needs_proxy: bool = True
    feature: str = "proxy_nll"          # "proxy_nll" or "proxy_rank"
    verifier_flops: float = 0.08

    def evidence(self, ctx: VContext) -> np.ndarray:
        return proxy_nll(ctx) if self.feature == "proxy_nll" else proxy_rank(ctx)


@dataclass
class TokenSurface(Verifier):
    """The zero-cost extreme: pure token-id statistics, NO model call at all.
    Scores a token by whether it repeats the previous one (a crude degeneracy /
    sampling-bug signal)."""

    name: str = "surface_tokens"
    needs_proxy: bool = False
    value_fn: str = "uniform"           # no proxy => can't compute an entropy value
    verifier_flops: float = 0.0005

    def evidence(self, ctx: VContext) -> np.ndarray:
        return token_surface_features(ctx.claimed_tokens)["is_repeat"]


@dataclass
class AcceptRate(Verifier):
    """The acceptance-rate fingerprint, as a first-class Tier-0 verifier. Per
    position it scores the total-variation distance TV(p, q) = 1 - accept_rate
    between the provider's served distribution `p` and the client's cheap proxy
    `q`. This is the draft-anchored signal that survives the temperature-retune
    evasion an entropy/cross-entropy fingerprint does not (matching entropy does
    not restore TV(p, q)). The CPU-testable acceptance-rate numerics live in
    `ivgym.spec_decode`; this class plugs them into the token harness so the
    fingerprint sits alongside every other verifier on the same Pareto."""

    name: str = "accept_rate"
    needs_proxy: bool = True
    needs_served: bool = True
    verifier_flops: float = 0.08

    def evidence(self, ctx: VContext) -> np.ndarray:
        if ctx.served_logits is None:
            raise RuntimeError("accept_rate needs served logits (set needs_served=True)")
        temp = max(ctx.sampling.temperature, 1e-6)
        out = np.empty(len(ctx.claimed_tokens))
        for i in range(len(ctx.claimed_tokens)):
            p = sd.softmax(ctx.served_logits[i] / temp)
            q = sd.softmax(ctx.proxy_logits[i] / temp)
            out[i] = sd.tv(p, q)             # = 1 - accept_rate(p, q)
        return out


# ---------------------------------------------------------------------------
# LLM-judge verifiers (Tier-0, external judge model). Kept out of the default
# registry: they need an API key and (for llm_judge) a text backend, so the
# experiments construct them explicitly.
# ---------------------------------------------------------------------------
@dataclass
class LLMJudge(Verifier):
    """Zero-shot LLM judge: feed (prompt, claimed continuation) to a judge model
    and ask whether it is a faithful sample from M at temperature T or a
    deviation; score = P(deviation), broadcast across tokens. Expected to be weak
    -- a judge cannot tell 4-bit-quant text from fp16 text -- which is exactly
    what the white-box-ness validator wants to demonstrate."""

    name: str = "llm_judge"
    needs_text: bool = True
    value_fn: str = "uniform"
    model: str = "claude-sonnet-4-6"
    max_chars: int = 600
    verifier_flops: float = 0.5
    _client: object | None = field(default=None, repr=False)

    def _api_key(self) -> str | None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
        for helper in (Path(__file__).resolve().parents[1] / ".claude" / "anthropic_key.sh",
                       Path.home() / ".claude" / "anthropic_key.sh"):
            if helper.exists():
                try:
                    out = subprocess.run(["sh", str(helper)], capture_output=True,
                                         text=True, timeout=10).stdout.strip()
                    if out:
                        return out
                except Exception:
                    continue
        return None

    def _judge(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key())
        return self._client

    def _judge_create(self, **kwargs):
        """One judge call, with exponential backoff on TRANSIENT API errors so a
        single 529/overload does not abort a long GPU run."""
        import anthropic
        transient = (anthropic.APIStatusError, anthropic.APIConnectionError,
                     anthropic.RateLimitError, anthropic.APITimeoutError)
        delays = (2, 4, 8, 16, 32, 60, 60, 60)
        for attempt, delay in enumerate(delays):
            try:
                return self._judge().messages.create(**kwargs)
            except transient as e:
                status = getattr(e, "status_code", None)
                if isinstance(e, anthropic.APIStatusError) and status is not None \
                        and 400 <= status < 500 and status != 429:
                    raise
                print(f"    [judge] transient API error ({type(e).__name__}"
                      f"{f' {status}' if status else ''}); retry "
                      f"{attempt + 1}/{len(delays)} in {delay}s", flush=True)
                time.sleep(delay)
        return self._judge().messages.create(**kwargs)

    def _p_deviation(self, prompt_text: str, continuation: str, temperature: float) -> float:
        sys_prompt = (
            "You are an inference-verification judge. You are given a PROMPT and a "
            "CONTINUATION that a model provider claims it sampled from a fixed "
            f"reference model at temperature {temperature:.2f}. Decide whether the "
            "continuation is a faithful sample from that model, or a DEVIATION "
            "(wrong model, wrong temperature/seed, quantized/corrupted decoding, or "
            "a sampling bug). Reply with ONLY a probability in [0,1] that it is a "
            "deviation -- e.g. '0.12'. No other text.")
        user = f"PROMPT:\n{prompt_text}\n\nCONTINUATION:\n{continuation[: self.max_chars]}"
        msg = self._judge_create(model=self.model, max_tokens=8, system=sys_prompt,
                                 messages=[{"role": "user", "content": user}])
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        try:
            return float(min(max(float(txt.split()[0]), 0.0), 1.0))
        except (ValueError, IndexError):
            return 0.5

    def evidence(self, ctx: VContext) -> np.ndarray:
        if ctx.prompt_text is None:
            raise RuntimeError("llm_judge needs a text backend (hf_gpu); this backend "
                               "exposed no prompt text.")
        prompt, _, cont = ctx.prompt_text.partition("\x00")
        p = self._p_deviation(prompt, cont, ctx.sampling.temperature)
        return np.full(len(ctx.claimed_tokens), p)


# ---------------------------------------------------------------------------
# Register the built-ins that need neither an API key nor a text backend.
# Each name below carries a claim the report stands behind; `llm_judge` needs API
# access, so `exp_io_detector_gpu` constructs it explicitly.
# ---------------------------------------------------------------------------
for _v in [
    TokenDiFR(), CrossEntropy(), ActivationDiFR(), TokenTOPLOC(),
    SurfaceStat(),
    SurfaceStat(name="surface_rank", feature="proxy_rank"),
    TokenSurface(),
    AcceptRate(),
]:
    register(_v)
