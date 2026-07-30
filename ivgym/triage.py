"""A learned, calibrated triage head: where to spend the expensive recompute.

Why
---
`harness.verify(budget<1)` audits only the top-`budget` fraction of tokens, ranked
by a cheap per-token `value` signal. Until now those signals were four
hand-crafted functions (`verifiers._VALUE_FNS`: uniform / entropy / tie_margin /
surprisal) -- each a *guess* at which positions carry evidence about whether `M`
was really run.

This module replaces the guess with a **learned head**, following the confidence
head of *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive
Generation* (arXiv:2607.05147). DSpark predicts, from cheap draft-side features, a
per-position **survival probability** -- how likely the target is to accept that
draft token -- trains it with binary cross-entropy against analytically computed
acceptance rates, and then **post-hoc calibrates** it (Sequential Temperature
Scaling) so the number is a probability rather than a ranking, which is what lets
their scheduler make admission decisions against a cost model.

What we change about it
-----------------------
DSpark ranks by ``P(token is accepted)``. That is the wrong target here. We are
not trying to accept tokens; we are trying to *detect a lying provider*. The
batch statistic is a mean of per-token evidence scores over the audited tokens,
so the quantity that decides detection power at a position is the **standardized
per-token effect size**

    Delta(t) / sigma_h(t) ,   Delta(t) = E[score | deviating] - E[score | honest]

(the per-token d-prime; its square is the Fisher information the position
contributes). `value_entropy` is a crude surrogate for this -- entropy is high
where the distribution is uncertain, and uncertain positions *tend* to be
sensitive -- but it is a surrogate, not the thing itself. So the head here is
trained to predict *sensitivity*, not acceptance.

Two label sources, and only one of them is deployable
-----------------------------------------------------
`surrogate_sensitivity` (**deployable**) needs honest data only. For each honest
token it perturbs the *reference* logit row with a generic zero-mean Gaussian
probe -- a stand-in deviation, committed to no particular attack -- and measures
how much the Tier-1 score moves. That is a direct Monte-Carlo estimate of
`Delta(t)` for the whole logit-perturbation threat family (which is exactly how
`attacks.Quantization` / `KVCacheFP8` are modeled, and how real quant/fp8 error
enters). It reads `ref_logits`, but only in a **one-time offline calibration
run** -- the same trusted-`M`-run amortization `spec_decode.ProxyReference.fit`
already assumes. At inference the head reads the cheap proxy only, so it stays
Tier-0.

`paired_effect_size` (**oracle, not deployable**) uses labeled honest/attack
pairs to measure `Delta(t)/sigma_h(t)` directly. It exists to upper-bound how
much the deployable head leaves on the table, and to measure cross-attack
transfer -- train the head on one attack, triage against another.

The head itself is a numpy logistic regression (BCE + L2, full-batch gradient
descent) over proxy-only features, so the core stays dependency-free.

Calibration buys something the top-k rule cannot have
-----------------------------------------------------
`harness.select_triaged` takes the top-`budget` fraction: an operator must pick
the fraction, and the same fraction is spent whether or not the tokens are worth
it. A *calibrated* head instead reports expected evidence per token, so admission
becomes a threshold on value -- "spend while marginal expected evidence exceeds
the marginal cost" -- and the realized ratio adapts to the data.
`SequentialTemperatureScaling` is the DSpark-style post-hoc fit: a global
temperature plus per-position-bucket offsets, fit by NLL on a held-out honest
split. It is monotone in each bucket, so it changes the *numbers* without
scrambling the within-bucket *ranking*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import VContext
from .sampling import filtered_logits

_EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic. Branching on the sign keeps `exp` under 1 on both
    sides, so no clipping is needed -- and that matters here beyond tidiness: the
    head's output is a RANKING key for `harness.select_triaged`, and clipping the
    logit at +-30 collapses everything in the tail to one value, making those
    positions unorderable.
    """
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

# ---------------------------------------------------------------------------
# Proxy-only features. EVERY feature here must be computable from the cheap
# proxy `q`, the claimed token ids, and the position index -- never from
# `ref_logits`. That invariant is what keeps the head Tier-0 at inference time,
# and `feature_matrix` is the only place it has to be enforced.
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    "entropy",        # H(q): the current library default value signal
    "renyi2",         # -log sum q^2, collision entropy (mass concentration)
    "tie_margin",     # -(q1 - q2): near-tie-ness, the quant-tuned signal
    "log_top1",       # log q1
    "top5_mass",      # sum of the 5 largest q
    "surprisal",      # -log q(claimed): claimed token unlikely under proxy
    "log_rank",       # log(1 + rank of claimed under q)
    "log_support",    # log(1 + #{v: q_v > 1/V}): effective support size
    "rel_position",   # t / T: DSpark's positional weighting analogue
)


def _proxy_probs(proxy_logits: np.ndarray, temperature: float) -> np.ndarray:
    z = proxy_logits / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def feature_matrix(ctx: VContext) -> np.ndarray:
    """`[T, len(FEATURE_NAMES)]` proxy-only features for one sequence.

    Vectorized over positions: one `[T, V]` softmax and a single `np.partition`
    replace the per-position Python loop the hand-crafted value functions use.
    """
    if ctx.proxy_logits is None:
        raise RuntimeError("triage features need proxy logits (Tier-0 field)")
    q = _proxy_probs(ctx.proxy_logits, ctx.sampling.temperature)
    t, v = q.shape
    idx = np.asarray(ctx.claimed_tokens, int)
    rows = np.arange(t)

    top5 = np.partition(q, -5, axis=1)[:, -5:]
    top5.sort(axis=1)                      # ascending: [..., q2, q1]
    q1, q2 = top5[:, -1], top5[:, -2]
    q_claimed = q[rows, idx]

    ent = -np.sum(q * np.log(q + _EPS), axis=1)
    renyi2 = -np.log(np.sum(q * q, axis=1) + _EPS)
    # rank of the claimed token = how many proxy probs strictly exceed it
    rank = np.sum(q > q_claimed[:, None], axis=1)
    support = np.sum(q > 1.0 / v, axis=1)

    return np.column_stack([
        ent,
        renyi2,
        -(q1 - q2),
        np.log(q1 + _EPS),
        top5.sum(axis=1),
        -np.log(q_claimed + _EPS),
        np.log1p(rank),
        np.log1p(support),
        rows / max(t - 1, 1),
    ])


# ---------------------------------------------------------------------------
# Label source A (DEPLOYABLE): surrogate sensitivity from honest data only.
# ---------------------------------------------------------------------------
def surrogate_sensitivity(ref_logits: np.ndarray, gumbel: np.ndarray,
                          claimed_tokens: list[int], sampling, verifier,
                          rng: np.random.Generator, probe_sigma: float = 0.15,
                          n_probe: int = 8) -> np.ndarray:
    """Per-token `Delta(t)` under a *generic* logit-perturbation probe.

    For each position: score the honest reference row, then re-score it
    `n_probe` times with `N(0, probe_sigma)` added to the logits, and return the
    mean increase in the Tier-1 score. High values mark positions where a
    forward-pass deviation of *any* kind in this family would actually show up in
    the evidence; low values mark positions the audit would waste itself on.

    The probe commits to no specific attack -- no attack name, no attack sigma --
    it only assumes the deviation perturbs logits, which is what quantization,
    fp8 KV cache, and a swapped checkpoint all do. `probe_sigma` sets the probe
    scale; the *ranking* it induces is stable over a wide range of it (that is
    what `experiments/exp_confidence_head_gpu.py` sweeps).
    """
    n = len(claimed_tokens)
    out = np.zeros(n)
    for i in range(n):
        base = verifier.score_token(ref_logits[i], gumbel[i], claimed_tokens[i], sampling)
        acc = 0.0
        for _ in range(n_probe):
            pert = ref_logits[i] + rng.normal(0.0, probe_sigma, ref_logits[i].shape)
            acc += verifier.score_token(pert, gumbel[i], claimed_tokens[i], sampling)
        out[i] = acc / n_probe - base
    return out


# ---------------------------------------------------------------------------
# Label source B (ORACLE): paired standardized effect size from labeled data.
# ---------------------------------------------------------------------------
def paired_effect_size(honest_scores: np.ndarray, attack_scores: np.ndarray
                       ) -> np.ndarray:
    """Per-token realized effect size `|s_attack(t) - s_honest(t)| / sigma_honest`.

    Both arrays are flat `[N]` in the same `(sequence, step)` order, generated
    from the same prompts and positions, so they are *paired*: the difference at a
    token is that token's realized contribution to the separation, and dividing by
    the global honest spread puts it on a d-prime scale.

    Used ONLY as a training label for the oracle head. It reads the attack's own
    Tier-1 scores, so it is not available at deployment -- and it is deliberately
    not used directly as a value signal either, because reading the attack scores
    you are about to be graded on is circular. Fitting the SAME proxy-feature head
    on this label answers the non-circular question: how much better would triage
    be if the head were trained on real deviation labels instead of the honest-only
    surrogate probe? That gap is the price of not having labeled attack data.
    """
    sigma = float(honest_scores.std()) + 1e-6
    return np.abs(attack_scores - honest_scores) / sigma


# ---------------------------------------------------------------------------
# The head: numpy logistic regression, BCE + L2.
# ---------------------------------------------------------------------------
@dataclass
class ConfidenceHead:
    """Logistic head mapping proxy-only features -> P(high-value position).

    `fit` standardizes features, binarizes the continuous sensitivity label at
    its `label_quantile`, and minimizes BCE with L2 by full-batch gradient
    descent. `score` returns the (optionally calibrated) probability, which
    `harness` consumes as a per-token `value`.
    """

    l2: float = 1e-3
    lr: float = 0.5
    n_steps: int = 4000
    label_quantile: float = 0.70
    w: np.ndarray | None = None
    b: float = 0.0
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    calibrator: "SequentialTemperatureScaling | None" = None
    history: list[float] = field(default_factory=list)

    # -- internals ---------------------------------------------------------
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    def _raw_logit(self, x: np.ndarray) -> np.ndarray:
        return self._standardize(x) @ self.w + self.b

    # -- api ---------------------------------------------------------------
    def fit(self, x: np.ndarray, sensitivity: np.ndarray) -> "ConfidenceHead":
        """Train on features `x` `[N, F]` and continuous `sensitivity` `[N]`."""
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0) + 1e-8
        xs = self._standardize(x)
        thr = np.quantile(sensitivity, self.label_quantile)
        y = (sensitivity > thr).astype(float)

        n, f = xs.shape
        self.w = np.zeros(f)
        self.b = 0.0
        self.history = []
        for step in range(self.n_steps):
            p = _sigmoid(xs @ self.w + self.b)
            g = p - y
            gw = xs.T @ g / n + self.l2 * self.w
            gb = g.mean()
            self.w -= self.lr * gw
            self.b -= self.lr * gb
            if step % 200 == 0:
                nll = -np.mean(y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS))
                self.history.append(float(nll))
        return self

    def score(self, x: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
        """Calibrated `P(high-value)` if a calibrator is fit, else the raw sigmoid."""
        z = self._raw_logit(x)
        if self.calibrator is not None:
            z = self.calibrator.transform(z, positions)
        return _sigmoid(z)

    def calibrate(self, x: np.ndarray, sensitivity: np.ndarray,
                  positions: np.ndarray, n_buckets: int = 4
                  ) -> "ConfidenceHead":
        """Fit Sequential Temperature Scaling on a HELD-OUT honest split."""
        thr = np.quantile(sensitivity, self.label_quantile)
        y = (sensitivity > thr).astype(float)
        self.calibrator = SequentialTemperatureScaling(n_buckets=n_buckets).fit(
            self._raw_logit(x), y, positions)
        return self

    def weights(self) -> dict[str, float]:
        """Learned coefficient per feature (on standardized features), for the
        readout in the experiment: which cheap signals the head actually uses."""
        return dict(zip(FEATURE_NAMES, self.w.tolist())) if self.w is not None else {}

    # -- persistence -------------------------------------------------------
    # Fitting the head needs `ref_logits` (the one-time offline surrogate-probe
    # run), so a fitted head is the expensive artifact -- not the scoring. Being
    # able to save one and load it into a DIFFERENT experiment is what lets the
    # prefix scheduler in `exp_prefix_cost_gpu.py` schedule against the calibrated
    # head without refitting it (and keeps there being exactly one fit on record).
    def to_dict(self) -> dict:
        """JSON-safe snapshot of everything `score` needs. Raises if unfitted."""
        if self.w is None:
            raise RuntimeError("cannot serialize an unfitted ConfidenceHead")
        out = {
            "feature_names": list(FEATURE_NAMES),
            "w": self.w.tolist(), "b": float(self.b),
            "mu": self.mu.tolist(), "sd": self.sd.tolist(),
            "label_quantile": self.label_quantile, "calibrator": None,
        }
        if self.calibrator is not None:
            out["calibrator"] = {"n_buckets": self.calibrator.n_buckets,
                                 "log_t": float(self.calibrator.log_t),
                                 "bias": self.calibrator.bias.tolist()}
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceHead":
        """Rebuild a fitted head. Refuses a snapshot whose feature layout differs
        from this module's -- the weights are positional, so a reordered or
        extended `FEATURE_NAMES` would silently score against the wrong columns."""
        names = tuple(d.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError(
                f"head was fit on features {names} but this build uses "
                f"{FEATURE_NAMES}; refusing to score positionally against a "
                f"different layout")
        head = cls(label_quantile=d.get("label_quantile", 0.70))
        head.w = np.asarray(d["w"], float)
        head.b = float(d["b"])
        head.mu = np.asarray(d["mu"], float)
        head.sd = np.asarray(d["sd"], float)
        cal = d.get("calibrator")
        if cal:
            sts = SequentialTemperatureScaling(n_buckets=int(cal["n_buckets"]))
            sts.log_t = float(cal["log_t"])
            sts.bias = np.asarray(cal["bias"], float)
            head.calibrator = sts
        return head


# ---------------------------------------------------------------------------
# DSpark's post-hoc calibration, ported.
# ---------------------------------------------------------------------------
@dataclass
class SequentialTemperatureScaling:
    """Global temperature + per-position-bucket bias, fit by NLL.

    DSpark calibrates confidence *sequentially* because acceptance probability
    decays with depth inside a draft block, so one global temperature
    mis-calibrates the tail. The analogue here is depth in the generated
    sequence: early positions sit close to the prompt and behave differently from
    late ones. Buckets are equal-width in relative position; within a bucket the
    map `z -> z/T + c_k` is strictly monotone, so calibration cannot reorder
    tokens that share a bucket -- it only makes the number a probability.
    """

    n_buckets: int = 4
    log_t: float = 0.0
    bias: np.ndarray | None = None
    lr: float = 0.05
    n_steps: int = 2000

    def _bucket(self, positions: np.ndarray) -> np.ndarray:
        if positions is None:
            return np.zeros(0, int)
        return np.clip((positions * self.n_buckets).astype(int), 0, self.n_buckets - 1)

    def fit(self, z: np.ndarray, y: np.ndarray, positions: np.ndarray
            ) -> "SequentialTemperatureScaling":
        k = self._bucket(positions)
        self.bias = np.zeros(self.n_buckets)
        self.log_t = 0.0
        onehot = np.eye(self.n_buckets)[k]
        for _ in range(self.n_steps):
            t = np.exp(self.log_t)
            p = _sigmoid(z / t + onehot @ self.bias)
            g = p - y
            # d zz / d log_t = -z / t
            self.log_t -= self.lr * float(np.mean(g * (-z / t)))
            self.bias -= self.lr * (onehot.T @ g) / max(len(z), 1)
        return self

    def transform(self, z: np.ndarray, positions: np.ndarray | None) -> np.ndarray:
        t = np.exp(self.log_t)
        if positions is None or self.bias is None:
            return z / t
        return z / t + self.bias[self._bucket(positions)]

    def temperature(self) -> float:
        return float(np.exp(self.log_t))


def head_value_fn(head: ConfidenceHead):
    """Wrap a fitted head as a `verifiers` value function (`VContext -> [T]`).

    Register it with `verifiers.register_value_fn("learned", head_value_fn(head))`
    and every existing call site -- `harness.verify(value_fn="learned")`,
    `harness.token_values` -- reaches the learned head with no other change. The
    closure touches `ctx.proxy_logits` only, so the Tier-0 invariant holds.
    """
    def _fn(ctx: VContext) -> np.ndarray:
        x = feature_matrix(ctx)
        pos = x[:, FEATURE_NAMES.index("rel_position")]
        return head.score(x, pos)
    return _fn


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Reliability curve + ECE for the calibration figure."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, frac = [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        conf.append(float(p[m].mean()))
        acc.append(float(y[m].mean()))
        frac.append(float(m.mean()))
        ece += frac[-1] * abs(conf[-1] - acc[-1])
    return np.array(conf), np.array(acc), np.array(frac), float(ece)
