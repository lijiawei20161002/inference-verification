"""Speculative-decoding **trace verification** (client-side, no-recompute first line).

Motivation
----------
vLLM runs on the *provider* side -- the party we may not fully trust. So it is
the wrong place to put a verifier: a cheating provider would simply run a hollow
one. The useful thing vLLM *can* do is expose an **auditable trace** of its
speculative-decoding step, so an *independent client* can check the trace after
the fact. This module implements that client-side check, plus a faithful
simulator of the trace vLLM would emit so the whole thing is testable without a
GPU or vLLM.

The trace (what the vLLM PR exposes)
------------------------------------
For each drafted position vLLM's rejection sampler already computes everything we
need (`vllm/v1/sample/rejection_sampler.py`): the target distribution `p`, the
draft distribution `q`, the drafted token `x ~ q`, a uniform "coin" `u ~ U(0,1)`,
the accept decision, and -- on rejection -- the recovered token drawn from the
residual `max(p-q, 0)`; on a fully-accepted draft chain, a bonus token from `p`.
`DraftStep` / `BonusStep` / `SpecDecodeTrace` are exactly that record.

The accept rule mirrors vLLM's kernel verbatim::

    accept  <=>  target_prob(x) >= u * draft_prob(x)     # i.e. u <= p(x)/q(x)

which realizes standard speculative sampling `min(1, p/q)` and, with the residual
recovery, makes the marginal output **identical to the target distribution**
(Leviathan/Chen). See docs/SPEC_DECODE_TRACE_VERIFICATION.md.

What a client can verify from the trace alone (the trust model)
--------------------------------------------------------------
The provider reports `(p, q, x, u, accept, output)`; we trust none of it. What we
*can* check cheaply (no forward pass) is **internal consistency**:

* the reported accept/reject decisions follow the reported `p,q,u` (deterministic);
* rejected tokens land in the residual's support (deterministic);
* the coins really are Uniform(0,1) (statistical);
* the *observed* accept rate matches the rate the reported `p,q` imply -- a
  self-consistency test that needs no coins at all (statistical);
* recovered tokens are distributed as the residual, not as the draft (statistical);
* reported logprobs are normalized (deterministic).

These catch every *sampler-level* cheat -- over-accepting to skip the target
check, skipping the residual correction, fudging coins to justify decisions.
They do **not** catch a provider that runs only the cheap draft model and
**relabels its distribution as the target** (`draft_as_target`): that trace is
perfectly self-consistent. Catching *that* requires recomputing a subset of
target logits -- the `target_spotcheck` check, which is the (expensive)
recompute line and ties back to DiFR's "recomputation is necessary" result.

Everything here is pure numpy and backend-agnostic: `synthetic_positions`
fabricates correlated (p, q) pairs so the gym runs on CPU; a real backend would
instead supply target/draft logprobs from actual models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import SamplingSpec

# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
_EPS = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    return z - np.log(np.exp(z).sum())


def residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Normalized recovery distribution max(p-q, 0) / Z (vLLM's residual)."""
    r = np.clip(p - q, 0.0, None)
    z = r.sum()
    if z <= _EPS:                       # p == q everywhere -> nothing to recover
        return p.copy()
    return r / z


# ---------------------------------------------------------------------------
# Trace schema -- the record the vLLM PR would emit, per drafted position.
# ---------------------------------------------------------------------------
@dataclass
class DraftStep:
    """One drafted position: the accept/reject decision on token `draft_token`.

    Logprobs (not probs) are stored because that is what vLLM exposes and what an
    auditor wants to see; the verifier exponentiates as needed.
    """

    request_id: int
    chain_pos: int
    target_logprobs: np.ndarray   # reported log p, shape [V]
    draft_logprobs: np.ndarray    # reported log q, shape [V]
    draft_token: int              # x ~ q, the token the draft proposed
    coin: float                   # u ~ U(0,1) used for the accept test
    accepted: bool
    output_token: int             # emitted token (= draft_token if accepted else recovered)


@dataclass
class BonusStep:
    """A bonus token sampled directly from the target `p` after a fully-accepted
    draft chain (no draft token involved)."""

    request_id: int
    chain_pos: int
    target_logprobs: np.ndarray
    output_token: int


@dataclass
class SpecDecodeTrace:
    provider_name: str
    spec: SamplingSpec
    steps: list[DraftStep] = field(default_factory=list)
    bonus: list[BonusStep] = field(default_factory=list)
    # Hidden ground truth used only by the recompute spot-check oracle (a real
    # verifier recomputes these; here they stand in for the trusted forward pass).
    # Kept parallel to `steps` and `bonus` respectively so the oracle aligns with
    # the reported target logprobs exactly -- NOT part of the wire trace.
    _truth_steps: list[np.ndarray] = field(default_factory=list)
    _truth_bonus: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic position generator (CPU stand-in for real target/draft models)
# ---------------------------------------------------------------------------
@dataclass
class Position:
    """The true target/draft distributions at one position (what the models say)."""

    target_logprobs: np.ndarray
    draft_logprobs: np.ndarray


def synthetic_positions(rng: np.random.Generator, n: int, vocab: int = 64,
                        agreement: float = 0.85, sharpness: float = 2.0) -> list[Position]:
    """Fabricate `n` correlated (target, draft) distribution pairs.

    `agreement` in [0,1] blends the draft toward the target (higher -> higher
    acceptance rate, as in a within-family draft); `sharpness` scales the target
    logits (peakier distributions). Pure numpy -- the real backend supplies these
    from actual model forward passes instead.
    """
    out = []
    for _ in range(n):
        t = rng.standard_normal(vocab) * sharpness
        t_lp = _log_softmax(t)
        # Draft = target logits nudged by independent noise, mixed by agreement.
        noise = rng.standard_normal(vocab) * sharpness
        d = agreement * t + (1.0 - agreement) * noise
        d_lp = _log_softmax(d)
        out.append(Position(t_lp, d_lp))
    return out


# ---------------------------------------------------------------------------
# Cheat registry -- provider deviations, analogous to ivgym.attacks
# ---------------------------------------------------------------------------
_CHEATS: dict[str, "CheatStrategy"] = {}


def register_cheat(c):
    """Register a cheat (instance or subclass, like ivgym.attacks.register)."""
    if isinstance(c, type):
        inst = c()
        inst.name = c.name
    else:
        inst = c
    _CHEATS[inst.name] = inst
    return c


def get_cheat(name: str) -> "CheatStrategy":
    return _CHEATS[name]


def all_cheats() -> dict[str, "CheatStrategy"]:
    return dict(_CHEATS)


@dataclass
class CheatStrategy:
    """A provider's speculative-decoding step, honest by default.

    `step` fully determines the *reported* record for one drafted position given
    the TRUE distributions `p_true`/`q_true` and the drafted token `x`. Cheats
    override it to save compute while trying to keep the trace plausible. The
    honest implementation is exactly vLLM's rejection sampler.
    """

    name: str = "honest"

    # Reported distributions default to the truth (an honest provider reports
    # what it computed). A "relabel" cheat overrides these.
    def report_target_logprobs(self, p_true_lp, q_true_lp):
        return p_true_lp

    def report_draft_logprobs(self, p_true_lp, q_true_lp):
        return q_true_lp

    def step(self, rng, p_true_lp, q_true_lp, x, request_id, chain_pos):
        p_lp = self.report_target_logprobs(p_true_lp, q_true_lp)
        q_lp = self.report_draft_logprobs(p_true_lp, q_true_lp)
        p, q = np.exp(p_lp), np.exp(q_lp)
        u = float(rng.random())
        accepted = p[x] >= u * q[x]                     # vLLM's rule
        if accepted:
            out = x
        else:
            out = int(rng.choice(len(p), p=residual(p, q)))
        return DraftStep(request_id, chain_pos, p_lp, q_lp, x, u, bool(accepted), out)

    # Bonus token after a fully-accepted chain: honest = sample from target p.
    def bonus(self, rng, p_true_lp, q_true_lp, request_id, chain_pos):
        p_lp = self.report_target_logprobs(p_true_lp, q_true_lp)
        p = np.exp(p_lp)
        out = int(rng.choice(len(p), p=p))
        return BonusStep(request_id, chain_pos, p_lp, out)


@dataclass
class OverAcceptNaive(CheatStrategy):
    """Accept a fraction of would-be rejects to skip residual work -- but leaves
    the honest coin in the trace, so the decision no longer follows p,q,u.
    Caught by `decision_consistency`."""

    name: str = "over_accept_naive"
    extra_accept: float = 0.5     # of rejects, this fraction is flipped to accept

    def step(self, rng, p_true_lp, q_true_lp, x, request_id, chain_pos):
        s = super().step(rng, p_true_lp, q_true_lp, x, request_id, chain_pos)
        if not s.accepted and rng.random() < self.extra_accept:
            s.accepted = True
            s.output_token = x            # keep the cheap draft token
        return s


@dataclass
class OverAcceptCoinFudge(CheatStrategy):
    """Same over-acceptance, but *recomputes the coin* so each decision is
    self-consistent with p,q,u. Defeats `decision_consistency` -- but the coins
    are no longer Uniform(0,1) and the observed accept rate exceeds what p,q
    imply, so `coin_uniformity` and `accept_rate_consistency` catch it."""

    name: str = "over_accept_coinfudge"
    extra_accept: float = 0.5

    def step(self, rng, p_true_lp, q_true_lp, x, request_id, chain_pos):
        p_lp, q_lp = p_true_lp, q_true_lp
        p, q = np.exp(p_lp), np.exp(q_lp)
        ratio = min(1.0, p[x] / max(q[x], _EPS))         # honest accept prob
        if rng.random() < ratio:
            u = float(rng.uniform(0.0, ratio))           # any coin below ratio accepts
            return DraftStep(request_id, chain_pos, p_lp, q_lp, x, u, True, x)
        # would-be reject: flip some to accept with a justifying (small) coin
        if rng.random() < self.extra_accept:
            u = float(rng.uniform(0.0, ratio + _EPS))    # coin that "accepts" x
            return DraftStep(request_id, chain_pos, p_lp, q_lp, x, u, True, x)
        u = float(rng.uniform(ratio, 1.0))               # honest-looking reject coin
        out = int(rng.choice(len(p), p=residual(p, q)))
        return DraftStep(request_id, chain_pos, p_lp, q_lp, x, u, False, out)


@dataclass
class SkipResidual(CheatStrategy):
    """On rejection, resample from the *draft* q (or its argmax) instead of the
    residual -- cheaper, but the output distribution is no longer the target's.
    Caught by `residual_support` (draft-favored tokens have zero residual mass)
    and `recovered_distribution`."""

    name: str = "skip_residual"

    def step(self, rng, p_true_lp, q_true_lp, x, request_id, chain_pos):
        p_lp, q_lp = p_true_lp, q_true_lp
        p, q = np.exp(p_lp), np.exp(q_lp)
        u = float(rng.random())
        accepted = p[x] >= u * q[x]
        if accepted:
            out = x
        else:
            out = int(rng.choice(len(q), p=q))           # WRONG: from draft, not residual
        return DraftStep(request_id, chain_pos, p_lp, q_lp, x, u, bool(accepted), out)


@dataclass
class DraftAsTarget(CheatStrategy):
    """Run ONLY the cheap draft model and report its distribution as BOTH the
    draft and the target. Then p==q, so everything accepts (min(1,p/q)=1) and the
    trace is perfectly self-consistent -- every no-recompute check passes. Only
    `target_spotcheck` (which recomputes true target logprobs) catches it. This
    is the fundamental limit of trace self-consistency, and the reason
    recomputation exists."""

    name: str = "draft_as_target"

    def report_target_logprobs(self, p_true_lp, q_true_lp):
        return q_true_lp                                 # lie: report draft as target


@dataclass
class QuantTarget(CheatStrategy):
    """A *forward-pass* deviation, the SD analogue of `quant_4bit` / `kv_fp8`: the
    provider runs a **quantized target model**, so the true target logits are
    perturbed. It then does textbook rejection sampling on those perturbed logits
    and reports them. The trace is therefore perfectly self-consistent -- the
    corruption is in `p` itself, not in the procedure. Like `draft_as_target`, no
    self-consistency check can see it; only `target_spotcheck` (recompute the
    *unquantized* target) catches it. This is the forward-pass/recompute boundary,
    identical to the DiFR result."""

    name: str = "quant_target"
    sigma: float = 0.5        # quantization logit-error scale
    temp: float = 1.0         # temperature retune on the (quantized) target (see AdvQuantTemp)

    def _corrupt(self, rng, p_true_lp):
        """The quantized (and optionally temperature-retuned) reported target."""
        return _log_softmax((p_true_lp + rng.normal(0.0, self.sigma, p_true_lp.shape))
                            / max(self.temp, 1e-6))

    def step(self, rng, p_true_lp, q_true_lp, x, request_id, chain_pos):
        # Quantized target: corrupt the target logits, then sample honestly on them.
        p_lp = self._corrupt(rng, p_true_lp)
        q_lp = q_true_lp
        p, q = np.exp(p_lp), np.exp(q_lp)
        u = float(rng.random())
        accepted = p[x] >= u * q[x]
        out = x if accepted else int(rng.choice(len(p), p=residual(p, q)))
        return DraftStep(request_id, chain_pos, p_lp, q_lp, x, u, bool(accepted), out)

    def bonus(self, rng, p_true_lp, q_true_lp, request_id, chain_pos):
        p_lp = self._corrupt(rng, p_true_lp)
        out = int(rng.choice(len(p_lp), p=np.exp(p_lp)))
        return BonusStep(request_id, chain_pos, p_lp, out)


@dataclass
class AdvQuantTemp(QuantTarget):
    """Quantize the target, then **retune the temperature** so the reported
    distribution's entropy/surprisal matches the honest reference again -- the SD
    analogue of the repo's `adv_quant_temp`. Its purpose is to defeat *offline
    reference-fingerprint* detectors (entropy / cross-entropy), which are the main
    no-recompute alternative to self-consistency. See
    `experiments/exp_spec_decode_fingerprint.py`: the entropy fingerprint is
    neutralized, testing whether the *acceptance-rate* fingerprint (unique to SD,
    anchored on the trusted draft) survives. Still caught by `target_spotcheck`."""

    name: str = "adv_quant_temp"
    sigma: float = 0.5
    temp: float = 0.85        # tuned at runtime to match honest entropy


@dataclass
class SamplingBugCheat(CheatStrategy):
    """A *sampler* deviation, the SD analogue of `bug_k2` / `bug_k32`: with prob
    `rate` the sampler ignores the accept machinery and emits a uniform token from
    the top-`k` of the draft, while still reporting the honest `p,q,u` and accept
    bit. That corrupts the emitted token relative to the reported decision, so
    `decision_consistency` (accepted-but-output!=drafted) and the residual checks
    catch it WITHOUT recomputation -- because the deviation is in the procedure,
    not in the logits."""

    name: str = "sampling_bug"
    rate: float = 0.05
    k: int = 8

    def step(self, rng, p_true_lp, q_true_lp, x, request_id, chain_pos):
        s = super().step(rng, p_true_lp, q_true_lp, x, request_id, chain_pos)
        if rng.random() < self.rate:
            q = np.exp(q_true_lp)
            topk = np.argsort(q)[::-1][:self.k]
            s.output_token = int(rng.choice(topk))       # hijack the emitted token
        return s


# Register the built-ins.
for _c in [CheatStrategy(), OverAcceptNaive(), OverAcceptCoinFudge(),
           SkipResidual(), DraftAsTarget(), QuantTarget(), SamplingBugCheat(),
           AdvQuantTemp()]:
    register_cheat(_c)


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------
def generate_trace(rng: np.random.Generator, cheat: CheatStrategy,
                   positions: list[Position], spec: SamplingSpec | None = None,
                   draft_len: int = 4) -> SpecDecodeTrace:
    """Run one provider (honest or cheating) over `positions`, grouped into
    draft chains of length `draft_len`, producing the trace it would report.

    Faithful to vLLM's chain semantics: a chain stops at the first reject (with a
    recovered token); a fully-accepted chain appends a bonus token from target.
    """
    spec = spec or SamplingSpec()
    tr = SpecDecodeTrace(provider_name=cheat.name, spec=spec)
    i = 0
    req = 0
    n = len(positions)
    while i < n:
        chain = positions[i:i + draft_len]
        for cp, pos in enumerate(chain):
            q = np.exp(pos.draft_logprobs)
            x = int(rng.choice(len(q), p=q))             # draft proposes x ~ q
            s = cheat.step(rng, pos.target_logprobs, pos.draft_logprobs, x, req, cp)
            tr.steps.append(s)
            tr._truth_steps.append(pos.target_logprobs)  # oracle ground truth
            if not s.accepted:
                break
        else:
            # whole chain accepted -> bonus token from the position after it, which
            # is then consumed (the next chain drafts *after* the bonus position).
            if i + draft_len < n:
                bpos = positions[i + draft_len]
                b = cheat.bonus(rng, bpos.target_logprobs, bpos.draft_logprobs,
                                req, len(chain))
                tr.bonus.append(b)
                tr._truth_bonus.append(bpos.target_logprobs)
            i += 1                                        # skip the consumed bonus position
        i += draft_len
        req += 1
    return tr


# ---------------------------------------------------------------------------
# Checks -- each maps a trace to a scalar anomaly score (higher = more suspect).
# ---------------------------------------------------------------------------
_CHECKS: dict[str, "Check"] = {}


def register_check(c):
    if isinstance(c, type):
        inst = c()
        inst.name = c.name
        c_inst = inst
    else:
        c_inst = c
    _CHECKS[c_inst.name] = c_inst
    return c


def get_check(name: str) -> "Check":
    return _CHECKS[name]


def all_checks() -> dict[str, "Check"]:
    return dict(_CHECKS)


class Check:
    name: str = "base"
    needs_oracle: bool = False        # requires recomputing true target logprobs?

    def score(self, trace: SpecDecodeTrace, oracle=None) -> float:
        """Higher = more anomalous. `oracle(i) -> true target logprobs` for the
        i-th (steps + bonus) position, provided only to `needs_oracle` checks."""
        raise NotImplementedError


@register_check
class DecisionConsistency(Check):
    """Fraction of steps whose reported mechanics contradict the accept rule:
    the accept bit must equal `p[x] >= u*q[x]`, AND an accepted step must emit the
    drafted token itself (`output == draft_token`). Any nonzero value is hard
    proof of a contradictory trace -- this is what a sampler that hijacks the
    emitted token trips."""

    name = "decision_consistency"

    def score(self, trace, oracle=None):
        if not trace.steps:
            return 0.0
        bad = 0
        for s in trace.steps:
            p = np.exp(s.target_logprobs)
            q = np.exp(s.draft_logprobs)
            expect = bool(p[s.draft_token] >= s.coin * q[s.draft_token])
            wrong_bit = expect != bool(s.accepted)
            wrong_out = s.accepted and (s.output_token != s.draft_token)
            bad += (wrong_bit or wrong_out)
        return bad / len(trace.steps)


@register_check
class ResidualSupport(Check):
    """Fraction of *rejected* steps whose recovered token has zero residual mass
    max(p-q,0)==0 -- impossible under honest recovery. Hard evidence."""

    name = "residual_support"

    def score(self, trace, oracle=None):
        rej = [s for s in trace.steps if not s.accepted]
        if not rej:
            return 0.0
        bad = 0
        for s in rej:
            p = np.exp(s.target_logprobs)
            q = np.exp(s.draft_logprobs)
            r = np.clip(p - q, 0.0, None)
            bad += (r[s.output_token] <= _EPS)
        return bad / len(rej)


@register_check
class CoinUniformity(Check):
    """Kolmogorov-Smirnov distance of the reported coins to Uniform(0,1). Honest
    coins are U(0,1); a provider that fudges coins to justify over-acceptance
    skews them, raising the KS statistic."""

    name = "coin_uniformity"

    def score(self, trace, oracle=None):
        u = np.sort(np.array([s.coin for s in trace.steps], float))
        n = len(u)
        if n == 0:
            return 0.0
        # KS vs the U(0,1) CDF F(x)=x.
        i = np.arange(1, n + 1)
        d_plus = np.max(i / n - u)
        d_minus = np.max(u - (i - 1) / n)
        return float(max(d_plus, d_minus))


@register_check
class AcceptRateConsistency(Check):
    """Self-consistency z-test needing NO coins: the observed number of accepts
    must match sum_x min(1, p[x]/q[x]) implied by the reported p,q. A provider
    that over-accepts to save compute inflates the observed rate above what its
    own reported distributions justify."""

    name = "accept_rate_consistency"

    def score(self, trace, oracle=None):
        if not trace.steps:
            return 0.0
        acc_prob = np.empty(len(trace.steps))
        obs = np.empty(len(trace.steps))
        for j, s in enumerate(trace.steps):
            p = np.exp(s.target_logprobs)
            q = np.exp(s.draft_logprobs)
            acc_prob[j] = min(1.0, p[s.draft_token] / max(q[s.draft_token], _EPS))
            obs[j] = float(s.accepted)
        expected = acc_prob.sum()
        var = np.sum(acc_prob * (1.0 - acc_prob))
        if var <= _EPS:
            return 0.0
        return float(abs(obs.sum() - expected) / np.sqrt(var))


@register_check
class RecoveredDistribution(Check):
    """Log-likelihood-ratio that recovered tokens came from the draft q rather
    than the residual: mean over rejects of log q[y] - log residual[y]. Honest
    recovery (y ~ residual) sits low; `skip_residual` (y ~ q) sits high."""

    name = "recovered_distribution"

    def score(self, trace, oracle=None):
        rej = [s for s in trace.steps if not s.accepted]
        if not rej:
            return -np.inf              # no rejections -> no evidence, never flags
        vals = []
        for s in rej:
            p = np.exp(s.target_logprobs)
            q = np.exp(s.draft_logprobs)
            r = residual(p, q)
            y = s.output_token
            vals.append(np.log(q[y] + _EPS) - np.log(r[y] + _EPS))
        return float(np.mean(vals))


@register_check
class LogprobNormalization(Check):
    """Max |logsumexp(reported logprobs)| over the trace -- catches malformed /
    unnormalized reported distributions."""

    name = "logprob_normalization"

    def score(self, trace, oracle=None):
        worst = 0.0
        for s in trace.steps:
            for lp in (s.target_logprobs, s.draft_logprobs):
                worst = max(worst, abs(float(np.log(np.exp(lp).sum()))))
        for b in trace.bonus:
            worst = max(worst, abs(float(np.log(np.exp(b.target_logprobs).sum()))))
        return worst


@register_check
class TargetSpotcheck(Check):
    """The recompute line: for a sampled fraction of positions, recompute the TRUE
    target logprobs and compare to the reported ones (total-variation distance).
    This is the ONLY check that catches `draft_as_target`, whose trace is
    otherwise perfectly self-consistent. Expensive -- it runs the target model on
    the sampled positions."""

    name = "target_spotcheck"
    needs_oracle = True

    def __init__(self, fraction: float = 0.1, seed: int = 0):
        self.fraction = fraction
        self.seed = seed

    def score(self, trace, oracle=None):
        if oracle is None:
            return 0.0
        reported = [s.target_logprobs for s in trace.steps] + \
                   [b.target_logprobs for b in trace.bonus]
        n = len(reported)
        if n == 0:
            return 0.0
        rng = np.random.default_rng(self.seed)
        k = max(1, int(round(self.fraction * n)))
        idx = rng.choice(n, size=k, replace=False)
        tvs = []
        for i in idx:
            p_true = np.exp(oracle(int(i)))
            p_rep = np.exp(reported[i])
            tvs.append(0.5 * np.abs(p_true - p_rep).sum())      # total variation
        return float(np.max(tvs))


def make_oracle(trace: SpecDecodeTrace):
    """A trusted recompute oracle for `TargetSpotcheck`: returns the TRUE target
    logprobs for the i-th (steps ++ bonus) position. A real client recomputes
    these with the reference target model; here they are the stored ground truth.

    Ordering matches `TargetSpotcheck`'s `reported` list: all step positions (in
    step order) then all bonus positions (in bonus order)."""
    truth = trace._truth_steps + trace._truth_bonus
    return lambda i: truth[i]


# ---------------------------------------------------------------------------
# The client-side verifier: calibrate thresholds on honest traces, then flag.
# ---------------------------------------------------------------------------
@dataclass
class CheckVerdict:
    check: str
    score: float
    threshold: float
    flagged: bool


@dataclass
class TraceVerdict:
    provider_name: str
    flagged: bool
    checks: list[CheckVerdict]

    def summary(self) -> str:
        head = f"provider={self.provider_name!r}  ->  " + \
               ("FLAGGED (trace inconsistent with speculative decoding)"
                if self.flagged else "consistent")
        rows = "\n".join(
            f"    {'FLAG' if c.flagged else ' ok '}  {c.check:<24} "
            f"score={c.score:.4g}  thr={c.threshold:.4g}"
            for c in self.checks)
        return head + "\n" + rows


class TraceVerifier:
    """Independent client-side verifier for speculative-decoding traces.

    Deterministic checks (decision/residual/logprob-normalization) flag on ANY
    violation (threshold 0). Statistical checks (coin/accept-rate/recovered) get a
    threshold from an honest calibration set at a target false-positive rate. Give
    it `use_oracle=True` to also run the recompute spot-check.
    """

    _DETERMINISTIC = {"decision_consistency", "residual_support",
                      "logprob_normalization"}

    def __init__(self, checks: list[Check] | None = None, use_oracle: bool = False):
        chosen = checks if checks is not None else list(all_checks().values())
        if not use_oracle:
            chosen = [c for c in chosen if not c.needs_oracle]
        self.checks = chosen
        self.use_oracle = use_oracle
        self.thresholds: dict[str, float] = {}

    def calibrate(self, honest_traces: list[SpecDecodeTrace], fpr: float = 0.01,
                  det_tol: float = 1e-9, bonferroni: bool = True) -> "TraceVerifier":
        """Set each check's flag threshold from an honest calibration set.

        Deterministic checks flag on any violation (threshold `det_tol`, and honest
        traces score 0). Statistical checks use the honest `(1 - per_fpr)` quantile.
        Because a trace is flagged if *any* check trips, the per-check budget is
        Bonferroni-split across the statistical checks so the *overall* honest
        false-positive rate stays near `fpr` (deterministic checks cost ~0 FPR)."""
        stat = [c for c in self.checks if c.name not in self._DETERMINISTIC]
        per_fpr = fpr / max(len(stat), 1) if bonferroni else fpr
        for c in self.checks:
            if c.name in self._DETERMINISTIC:
                self.thresholds[c.name] = det_tol
                continue
            scores = np.array([c.score(t, make_oracle(t) if c.needs_oracle else None)
                               for t in honest_traces], float)
            scores = scores[np.isfinite(scores)]         # drop "no evidence" sentinels
            # (1 - per_fpr) honest quantile; if no finite honest scores, never flag.
            self.thresholds[c.name] = float(np.quantile(scores, 1.0 - per_fpr)) \
                if len(scores) else np.inf
        return self

    def verify(self, trace: SpecDecodeTrace) -> TraceVerdict:
        verdicts = []
        for c in self.checks:
            oracle = make_oracle(trace) if c.needs_oracle else None
            s = c.score(trace, oracle)
            thr = self.thresholds.get(c.name, 0.0)
            verdicts.append(CheckVerdict(c.name, s, thr, s > thr))
        return TraceVerdict(trace.provider_name, any(v.flagged for v in verdicts),
                            verdicts)


# ---------------------------------------------------------------------------
# Offline reference fingerprint -- the one no-recompute algorithm BEYOND self-
# consistency that can (partly) see forward-pass cheats like quantization.
# ---------------------------------------------------------------------------
# Self-consistency is blind to quantization because the trace is a faithful sample
# from the corrupted p-hat and carries zero bits about the true p*. The ONLY way
# to see a wrong p-hat without recomputing p* per request is to bring in an
# EXTERNAL anchor about the honest target -- here, a *one-time, offline* honest
# characterization ("fingerprint") computed once on a trusted run and then reused
# for every verification (no per-request forward pass). The verifier compares
# aggregate statistics of the reported trace to that fingerprint.
#
# Two feature families:
#   * entropy       -- mean entropy of the reported target dist. Generic (this is
#                      the DiFR cross-entropy signal's cousin) and DEFEATABLE by
#                      temperature retuning (`adv_quant_temp`).
#   * accept_rate   -- realized acceptance rate = 1 - TV(p-hat, q). SD-specific: it
#                      is anchored on the (trusted, cheap) draft q, so a quantizer
#                      that shifts TV(p*,q) -> TV(p-hat,q) moves it. The experiment
#                      measures whether it survives temperature retuning that kills
#                      the entropy feature.
#
# LIMITATIONS (why this is not a free lunch): the fingerprint must be built on a
# TRUSTED honest run (a one-time recompute, amortized), it only catches quant
# large enough to move a feature past honest run-to-run variance, and every
# feature is in principle gameable by an attacker who also matches that feature
# (accept-rate can be matched by tuning the draft or the corruption). It shrinks
# how often the exact `target_spotcheck` must fire; it does not replace it.


def trace_features(trace: SpecDecodeTrace) -> dict[str, float]:
    """Aggregate, no-recompute statistics of one trace (features for the
    fingerprint). All are computed from the REPORTED trace only."""
    ent = []
    for s in trace.steps:
        p = np.exp(s.target_logprobs)
        ent.append(float(-np.sum(p * np.log(p + _EPS))))
    for b in trace.bonus:
        p = np.exp(b.target_logprobs)
        ent.append(float(-np.sum(p * np.log(p + _EPS))))
    acc = np.mean([s.accepted for s in trace.steps]) if trace.steps else 0.0
    # self-surprisal of emitted tokens under the reported target
    sur = [float(-s.target_logprobs[s.output_token]) for s in trace.steps]
    return {
        "accept_rate": float(acc),
        "mean_entropy": float(np.mean(ent)) if ent else 0.0,
        "mean_surprisal": float(np.mean(sur)) if sur else 0.0,
    }


class ReferenceFingerprint:
    """Detect forward-pass corruption (e.g. quantization) WITHOUT per-request
    recompute, by comparing a trace's aggregate features to a one-time honest
    fingerprint. `fit` characterizes honest traces (mean/std per feature); `score`
    returns a per-feature standardized distance |z| (higher = more anomalous) and
    an aggregate. Not a `Check` (it needs an external reference, not just the
    trace), which is precisely the point: it requires an anchor to the true target.
    """

    def __init__(self, features: tuple[str, ...] = ("accept_rate", "mean_entropy",
                                                    "mean_surprisal")):
        self.features = features
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, honest_traces: list[SpecDecodeTrace]) -> "ReferenceFingerprint":
        rows = [trace_features(t) for t in honest_traces]
        for f in self.features:
            vals = np.array([r[f] for r in rows], float)
            self.mean[f] = float(vals.mean())
            self.std[f] = float(vals.std() + _EPS)
        return self

    def score(self, trace: SpecDecodeTrace) -> dict[str, float]:
        feats = trace_features(trace)
        out = {f: abs(feats[f] - self.mean[f]) / self.std[f] for f in self.features}
        out["aggregate"] = float(max(out.values())) if out else 0.0
        return out
