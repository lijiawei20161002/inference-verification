"""Input-output (black-box) detectors.

A `Defense` (see `defenses.py`) is the **white-box** analogue of an inference
verifier: it is handed `ref_logits` -- the verifier re-ran the reference model
M's forward pass -- and scores per-token divergence-from-recomputation. An
`IODetector` is the **black-box** analogue: it decides "is this a faithful
sample from M under spec phi?" from `(prompt, claimed_tokens)` alone, *without*
recomputing M (see `IOContext` for why that boundary is the whole point).

Two roles motivate this (Clymer et al., *Lessons from building a model organism
testbed*, 2025; the white-box vs black-box detector split maps onto DiFR's
recomputation vs I/O-only split):

  1. **Cost/accuracy Pareto.** A pure I/O detector calling only a *tiny* proxy
     model -- or no model at all -- is the cheap extreme of "verifier spends less
     than re-running inference."
  2. **White-box-ness validator.** If an I/O detector *cannot* tell an attack
     apart from honest (AUC ~ 0.5) while a recomputation defense nails it
     (AUC ~ 1.0), that is a *quantitative* proof the attack is invisible in the
     outputs -- so the recomputation defense is providing genuine information
     beyond behaviour.

**Honest caveat (state it in any writeup):** a *high* I/O-AUC on an attack is NOT
a win for the verifier. It means the attack was crude enough to catch from
outputs alone -- a statement about the *attack*, not the verifier.

Mirrors the `Defense` registry pattern: subclass `IODetector`, `@register` it,
and the harness (`harness.io_verify`) scores every config with it. Detectors
return a *per-token* score array so the scores flow through the SAME
`winsorize` / `batch_means` / `evaluate` pipeline as `Defense` scores.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core import IOContext
from .sampling import log_softmax

_REGISTRY: dict[str, "IODetector"] = {}


def register(d):
    """Register an I/O detector. Accepts a `IODetector` *instance* or *subclass*
    (instantiated with its defaults), mirroring `defenses.register` /
    `attacks.register` so it is usable as a class decorator. Returns its argument
    unchanged, so it is decorator-safe."""
    if isinstance(d, type):
        inst = d()
        inst.name = d.name              # restore a subclass-declared class attr
        d_inst = inst
    else:
        d_inst = d
    _REGISTRY[d_inst.name] = d_inst
    return d


def get(name: str) -> "IODetector":
    return _REGISTRY[name]


def all_io_detectors() -> dict[str, "IODetector"]:
    return dict(_REGISTRY)


class IODetector:
    """Sequence-level, output-only detector. Higher score => more likely a
    deviation from spec phi (so it can be thresholded directly, like a Defense)."""

    name: str = "io_base"
    needs_proxy: bool = False      # harness should fill IOContext.proxy_logits?
    needs_text: bool = False       # needs a real-text backend (prompt_text/decode)?
    needs_fit: bool = False        # must be .fit(...) on labeled data before scoring?
    verifier_flops: float = 0.0    # relative verifier cost (1.0 = full recompute of M)

    def score_sequence(self, ctx: IOContext) -> np.ndarray:
        """Return a per-token score array of length len(ctx.claimed_tokens)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared output-only feature extraction (used by the surface-stat + learned
# detectors). Every feature is a function ONLY of the claimed tokens (and, when
# allowed, the CHEAP proxy logits) -- never of M's recomputed logits.
# ---------------------------------------------------------------------------
RANK_CAP = 64.0


def proxy_nll(ctx: IOContext) -> np.ndarray:
    """Per-token negative log-likelihood of each claimed token under the cheap
    proxy LM's (temperature-scaled) softmax. 'Cheap model polices expensive
    model': honest tokens are typical under the proxy, crude deviations less so."""
    if ctx.proxy_logits is None:
        raise RuntimeError("proxy_nll needs proxy logits (set needs_proxy=True)")
    temp = max(ctx.sampling.temperature, 1e-6)
    out = np.empty(len(ctx.claimed_tokens))
    for i, t in enumerate(ctx.claimed_tokens):
        out[i] = -log_softmax(ctx.proxy_logits[i] / temp)[t]
    return out


def proxy_rank(ctx: IOContext) -> np.ndarray:
    """Per-token rank of the claimed token under the proxy logits (0 = proxy's
    argmax). One-sided, TOPLOC-flavoured -- but read off a *cheap* proxy rather
    than the reference model."""
    if ctx.proxy_logits is None:
        raise RuntimeError("proxy_rank needs proxy logits (set needs_proxy=True)")
    out = np.empty(len(ctx.claimed_tokens))
    for i, t in enumerate(ctx.claimed_tokens):
        row = ctx.proxy_logits[i]
        out[i] = min(float(np.sum(row > row[t])), RANK_CAP)
    return out


def token_surface_features(tokens: list[int]) -> dict[str, np.ndarray]:
    """Output-only features that need NO model at all -- pure token-id statistics
    (the zero-FLOP extreme of the Pareto).

    NOTE on what is *deliberately excluded*: sequence-level scalars (length,
    unique-token fraction, whole-sequence entropy) are NOT used here. Broadcasting
    a per-sequence constant to its tokens is pathological for the per-token
    `batch_means` statistic: a batch of ~1000 tokens spans most sequences, so the
    batch mean collapses to the near-constant global mean and amplifies a
    statistically meaningless offset into AUC ~ 1.0 (verified: wrong-seed sampling,
    which is output-*indistinguishable*, spuriously scored ~1.0 via a 0.005
    entropy gap). The plan calls for per-token scores precisely to reuse this
    pipeline; we therefore keep only genuinely per-token features. `is_repeat`
    (does this token equal the previous one?) is per-token and aggregates to the
    repetition *rate* under batch averaging -- a real degeneracy signal."""
    n = len(tokens)
    arr = np.asarray(tokens)
    is_repeat = np.zeros(n)
    if n > 1:
        is_repeat[1:] = (arr[1:] == arr[:-1]).astype(float)
    return {"is_repeat": is_repeat}


def feature_matrix(ctx: IOContext, use_proxy: bool) -> tuple[np.ndarray, list[str]]:
    """Stack the per-token feature columns into [T, F] plus their names. Only
    per-token features (see `token_surface_features`); proxy features add the
    cheap-model perplexity/rank signal that carries most of the I/O information."""
    cols = [token_surface_features(ctx.claimed_tokens)["is_repeat"]]
    names = ["is_repeat"]
    if use_proxy:
        cols += [proxy_nll(ctx), proxy_rank(ctx)]
        names += ["proxy_nll", "proxy_rank"]
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
@dataclass
class SurfaceStatIODetector(IODetector):
    """The interesting one for the Pareto. Scores each claimed token by its
    perplexity (NLL) under a small, cheap proxy LM -- the analogue of
    `cross_entropy`, but reading a *cheap proxy* instead of recomputing M. This
    is the cheap end of the verifier-FLOPs axis ("cheap model polices expensive
    model"). The HF-GPU backend models the proxy as a noisier, lower-fidelity
    read of the reference logits (`HFGPUBackend.proxy_logits`)."""

    name: str = "surface_stat"
    needs_proxy: bool = True
    feature: str = "proxy_nll"          # "proxy_nll" or "proxy_rank"
    verifier_flops: float = 0.08        # ~12x cheaper than full recompute (illustrative)

    def score_sequence(self, ctx: IOContext) -> np.ndarray:
        return proxy_nll(ctx) if self.feature == "proxy_nll" else proxy_rank(ctx)


@dataclass
class TokenSurfaceIODetector(IODetector):
    """The zero-cost extreme: pure token-id statistics, NO model call at all.
    Scores a token by whether it repeats the previous one (a crude degeneracy /
    sampling-bug signal). Expected to sit near chance on most attacks -- that is
    itself an honest data point on the Pareto: zero FLOPs buys little detection."""

    name: str = "surface_tokens"
    needs_proxy: bool = False
    verifier_flops: float = 0.0005      # essentially free

    def score_sequence(self, ctx: IOContext) -> np.ndarray:
        return token_surface_features(ctx.claimed_tokens)["is_repeat"]


@dataclass
class LearnedSeqIODetector(IODetector):
    """The strongest I/O detector: a small classifier on output-only features
    (`feature_matrix`), trained on labeled `(prompt+tokens -> honest/attack)`
    sequences. Training data is FREE -- `harness.generate_dataset` emits honest +
    attack sequences tagged with `config_name`.

    Pure-numpy logistic regression (no sklearn dependency, matching the repo's
    metrics). `score_sequence` returns the per-token P(attack). Must be `.fit`
    before use. The crucial evaluation is **held-out-attack generalization**:
    fit on some attacks, test on attacks never seen in training -- a detector you
    trust to generalize must catch attacks it was not trained on."""

    name: str = "learned_io"
    needs_proxy: bool = True
    needs_fit: bool = True
    use_proxy: bool = True
    epochs: int = 800
    lr: float = 1.5
    l2: float = 1e-3
    verifier_flops: float = 0.08        # dominated by the proxy call; classifier is free
    # learned state (set by .fit)
    _w: np.ndarray | None = field(default=None, repr=False)
    _b: float = field(default=0.0, repr=False)
    _mu: np.ndarray | None = field(default=None, repr=False)
    _sd: np.ndarray | None = field(default=None, repr=False)
    _names: list[str] = field(default_factory=list, repr=False)

    # -- training ----------------------------------------------------------
    def fit(self, contexts: list[IOContext], labels: list[int]) -> "LearnedSeqIODetector":
        """`contexts` are per-sequence IOContexts; `labels` the matching
        sequence label (0 = honest, 1 = attack). Features are computed per token
        and the sequence label is broadcast to its tokens."""
        rows, ys, names = [], [], None
        for ctx, y in zip(contexts, labels):
            fm, names = feature_matrix(ctx, self.use_proxy)
            rows.append(fm)
            ys.append(np.full(len(fm), float(y)))
        X = np.vstack(rows)
        Y = np.concatenate(ys)
        self._names = names

        # standardize
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-8
        Xs = (X - self._mu) / self._sd

        n, f = Xs.shape
        w = np.zeros(f)
        b = 0.0
        for _ in range(self.epochs):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - Y
            gw = Xs.T @ g / n + self.l2 * w
            gb = float(g.mean())
            w -= self.lr * gw
            b -= self.lr * gb
        self._w, self._b = w, b
        return self

    # -- scoring -----------------------------------------------------------
    def score_sequence(self, ctx: IOContext) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("LearnedSeqIODetector must be .fit(...) before scoring")
        fm, _ = feature_matrix(ctx, self.use_proxy)
        Xs = (fm - self._mu) / self._sd
        z = Xs @ self._w + self._b
        return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
@dataclass
class LLMJudgeIODetector(IODetector):
    """Zero-shot LLM judge (mirrors the model-organism post's GPT-4 use): feed
    `(prompt, claimed continuation)` to a judge model and ask whether it is a
    faithful sample from M at temperature T or a deviation; score = P(deviation).
    No training.

    Expected to be *weak* -- a judge cannot tell 4-bit-quant text from fp16 text --
    and that weakness is exactly what the white-box-ness validator (Role 2) wants
    to demonstrate. Needs a real-text backend (`hf_gpu`): if a backend exposes no
    prompt text, `score_sequence` raises so the experiment can skip it.

    Uses the project's Anthropic access: the API key is resolved from
    `ANTHROPIC_API_KEY`, the repo's `.claude/anthropic_key.sh` helper, or
    `~/.claude/anthropic_key.sh`. Defaults to the latest capable Claude model.
    """

    name: str = "llm_judge"
    needs_text: bool = True
    model: str = "claude-sonnet-4-6"    # latest fast judge; opus-4-8 = most capable
    max_chars: int = 600                # truncate the continuation fed to the judge
    verifier_flops: float = 0.5         # a separate judge LM call (illustrative)
    _client: object | None = field(default=None, repr=False)

    def _api_key(self) -> str | None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            return key
        # Resolve via an apiKeyHelper script. Check the repo-local `.claude/`
        # first (project override), then the standard Claude Code location
        # (`~/.claude/anthropic_key.sh`) so the judge works without copying the
        # key into the repo.
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
            import anthropic  # lazy: only needed when a judge actually runs
            self._client = anthropic.Anthropic(api_key=self._api_key())
        return self._client

    def _p_deviation(self, prompt_text: str, continuation: str, temperature: float) -> float:
        sys_prompt = (
            "You are an inference-verification judge. You are given a PROMPT and a "
            "CONTINUATION that a model provider claims it sampled from a fixed "
            f"reference model at temperature {temperature:.2f}. Decide whether the "
            "continuation is a faithful sample from that model, or a DEVIATION "
            "(wrong model, wrong temperature/seed, quantized/corrupted decoding, or "
            "a sampling bug). Reply with ONLY a probability in [0,1] that it is a "
            "deviation -- e.g. '0.12'. No other text."
        )
        user = f"PROMPT:\n{prompt_text}\n\nCONTINUATION:\n{continuation[: self.max_chars]}"
        msg = self._judge().messages.create(
            model=self.model, max_tokens=8, system=sys_prompt,
            messages=[{"role": "user", "content": user}],
        )
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        try:
            return float(min(max(float(txt.split()[0]), 0.0), 1.0))
        except (ValueError, IndexError):
            return 0.5

    def score_sequence(self, ctx: IOContext) -> np.ndarray:
        if ctx.prompt_text is None:
            raise RuntimeError(
                "LLMJudgeIODetector needs a text backend (hf_gpu); this backend "
                "exposed no prompt text.")
        # `prompt_text` carries the decoded continuation appended by io_verify when
        # the backend supports decode(); see harness.io_verify.
        prompt, _, cont = ctx.prompt_text.partition("\x00")
        p = self._p_deviation(prompt, cont, ctx.sampling.temperature)
        # one judgement per sequence, broadcast across its tokens so the score
        # flows through the same per-token batch pipeline.
        return np.full(len(ctx.claimed_tokens), p)


@dataclass
class LogitJudgeIODetector(LLMJudgeIODetector):
    """Hybrid detector: recompute on a CHEAP PROXY, then let an LLM judge the
    LOGIT-level divergence (not the decoded text). For every claimed token the
    verifier computes its surprisal (NLL, nats) and rank under the proxy's
    temperature-scaled distribution -- "a cheap model recomputes, the LLM reads
    the divergence pattern." This is the logit-space sibling of `llm_judge`
    (which reads the continuation text): SAME judge model and API plumbing
    (inherited from `LLMJudgeIODetector`), but the evidence handed to the judge
    is numbers off `proxy_logits`, so it needs a proxy backend rather than a
    text one.

    Cost = one cheap-proxy forward pass + one judge call. `verifier_flops`
    carries the illustrative default; the experiment overrides the proxy part
    with the MEASURED params(proxy)/params(M) ratio.

    Expected behaviour matches the other black-box detectors and is the point of
    Role 2: on output-INVISIBLE attacks (e.g. seed_43, which redraws the SAME
    distribution) the proxy-surprisal/rank pattern is statistically unchanged, so
    the judge sits at its honest-null floor -- only a recomputation of M itself
    separates honest from attack. On output-VISIBLE attacks (temp_1.1, a crude
    bug) the surprisal pattern shifts and the judge can flag it.
    """

    name: str = "logit_judge"
    needs_proxy: bool = True
    needs_text: bool = False
    max_rows: int = 64                  # cap per-token rows handed to the judge
    verifier_flops: float = 0.58        # proxy (~0.08) + judge call (~0.5), illustrative

    def _evidence(self, nll: np.ndarray, rank: np.ndarray, temp: float) -> str:
        """Compact, model-free summary of the per-token proxy divergence. Numbers
        only -- the whole point is to judge in logit space, not from text."""
        n = len(nll)
        rows = " ".join(
            f"{i}:{s:.2f}/{int(r)}"
            for i, (s, r) in enumerate(zip(nll, rank)) if i < self.max_rows)
        more = "" if n <= self.max_rows else f" (+{n - self.max_rows} more tokens)"
        return (
            f"temperature={temp:.2f}  tokens={n}\n"
            f"per-token  surprisal_nats/proxy_rank  (rank 0 = proxy's argmax, "
            f"capped at {int(RANK_CAP)}):\n{rows}{more}\n"
            f"summary: mean_surprisal={float(nll.mean()):.2f}  "
            f"median_surprisal={float(np.median(nll)):.2f}  "
            f"max_surprisal={float(nll.max()):.2f}  mean_rank={float(rank.mean()):.1f}  "
            f"frac_not_argmax={float(np.mean(rank > 0)):.2f}  "
            f"frac_in_top8={float(np.mean(rank < 8)):.2f}")

    def _p_deviation_logits(self, nll: np.ndarray, rank: np.ndarray, temp: float) -> float:
        sys_prompt = (
            "You are an inference-verification judge working in LOGIT space. A "
            "verifier re-scored a provider's claimed tokens under a CHEAP PROXY "
            "model -- a smaller model from the same family as the reference model "
            "M, sharing M's tokenizer. For each claimed token you are given its "
            "surprisal (negative log-likelihood, nats) and its rank under the "
            "proxy's temperature-scaled distribution. Because the proxy is smaller "
            "than M, even a FAITHFUL sample from M shows nonzero surprisal and is "
            "often not the proxy's argmax -- that alone is NOT evidence of a "
            "deviation. Decide whether the PATTERN of surprisals and ranks is "
            "consistent with faithful sampling from M at the stated temperature, "
            "or indicates a DEVIATION (wrong model, wrong temperature/seed, "
            "quantized/corrupted decoding, or a sampling bug). Reply with ONLY a "
            "probability in [0,1] that it is a deviation -- e.g. '0.12'. No other text."
        )
        msg = self._judge().messages.create(
            model=self.model, max_tokens=8, system=sys_prompt,
            messages=[{"role": "user", "content": self._evidence(nll, rank, temp)}],
        )
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        try:
            return float(min(max(float(txt.split()[0]), 0.0), 1.0))
        except (ValueError, IndexError):
            return 0.5

    def score_sequence(self, ctx: IOContext) -> np.ndarray:
        if ctx.proxy_logits is None:
            raise RuntimeError(
                "logit_judge needs proxy logits (set needs_proxy=True); run with a "
                "proxy-capable backend.")
        nll = proxy_nll(ctx)
        rank = proxy_rank(ctx)
        p = self._p_deviation_logits(nll, rank, ctx.sampling.temperature)
        # one judgement per sequence, broadcast across its tokens (same shape
        # contract as every IODetector; see LLMJudgeIODetector).
        return np.full(len(ctx.claimed_tokens), p)


# Register the no-text, no-fit detectors so they appear in `--list` / the runner.
# (LearnedSeqIODetector needs fitting and LLMJudgeIODetector needs a text backend,
# so the experiment constructs those explicitly rather than from the registry.)
for d in [
    SurfaceStatIODetector(),
    SurfaceStatIODetector(name="surface_rank", feature="proxy_rank"),
    TokenSurfaceIODetector(),
]:
    register(d)
