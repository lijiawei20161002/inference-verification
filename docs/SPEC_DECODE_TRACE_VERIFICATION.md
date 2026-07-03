# Verifiable speculative decoding: expose the trace in vLLM, verify it on the client

*Design note + vLLM PR proposal for `ivgym.spec_decode`.*

## 1. Why not verify inside vLLM

vLLM runs on the **provider** side — exactly the party a verification game does
not trust. A verifier compiled into the serving process is one the provider can
stub out, feed cooked inputs, or simply not run. So "add speculative-decoding
verification to vLLM" is the wrong shape.

The right shape splits the two concerns:

* **vLLM (provider):** emit an **auditable trace** of what the speculative
  decoder actually did — the raw material an auditor needs. This is pure
  *observability*, no trust required, and a clean, mergeable vLLM PR.
* **Client (verifier):** an **independent** checker that reads the trace and
  decides whether it is *consistent with the speculative-decoding procedure the
  provider claims to run*. This lives outside vLLM and trusts nothing in it.

`ivgym.spec_decode` implements the client side in full and a faithful simulator
of the provider trace, so the whole loop is testable on CPU today
(`experiments/exp_spec_decode_trace.py`, `tests/test_spec_decode.py`). The vLLM
change is the trace-emission PR sketched in §4.

## 2. What speculative decoding does (and the one invariant we verify)

For a drafted token `x ~ q` (draft model `q`) verified against the target `p`,
vLLM's rejection sampler
([`vllm/v1/sample/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py))
accepts iff, verbatim from `rejection_random_sample_kernel`:

```python
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

i.e. `p(x) >= u · q(x)` with `u ~ U(0,1)` — standard `min(1, p/q)` acceptance. On
rejection it draws a **recovered token** from the residual computed in
`sample_recovered_tokens_kernel`:

```python
prob = max(target_prob - draft_prob, 0.0)      # ∝ residual; argmax-sampled, no explicit /sum
```

and, when **all** drafted tokens in a chain are accepted, appends a **bonus
token** from the target (`if not rejected: bonus_token_id = ...`). Accept +
residual recovery is the Leviathan/Chen correction that makes the marginal output
**identical to the target distribution `p`** — the invariant the whole scheme
sells, and the invariant our checks police. (`test_honest_marginal_output_equals_target`
confirms the simulator reproduces `p` to TV < 0.02;
`test_skip_residual_breaks_marginal` confirms a wrong recovery does not.)

## 3. The trace (wire format)

Per drafted position, everything below is already computed inside the kernel; the
PR only *records and returns* it. `ivgym.spec_decode.DraftStep` / `BonusStep` are
exactly this record:

| field | meaning | vLLM source |
|---|---|---|
| `target_logprobs` | log `p` (top-k suffices in practice) | `logits` → log_softmax |
| `draft_logprobs`  | log `q` | `draft_probs` |
| `draft_token`     | `x`, proposed by the draft | `metadata.draft_token_ids` |
| `coin`            | `u`, the uniform used in the accept test | `uniform_probs` in the kernel |
| `accepted`        | the accept/reject decision | kernel `accepted` |
| `output_token`    | emitted token (`x` if accepted, else recovered) | sampler output |
| `BonusStep`       | bonus token + its `target_logprobs` after a fully-accepted chain | bonus path |

Full per-token `p`/`q` over a 150k-vocab is bandwidth-heavy; in practice a vLLM PR
exposes **top-k logprobs** (vLLM already has `SamplingParams.logprobs`) plus the
scalar `coin` and the decision. Every check in §5 works on top-k with the
convention that an out-of-top-k token contributes ~0 probability (a conservative
bound). The CPU gym uses full vocab for exactness.

## 4. The vLLM PR (opt-in trace emission)

Grounded in the current file. Add an opt-in flag (per request or engine-level,
e.g. `SamplingParams(spec_decode_trace=True)`), and in `RejectionSampler.forward`

```python
def forward(self, metadata: SpecDecodeMetadata, draft_probs,
            logits, sampling_metadata) -> SamplerOutput:
```

record, alongside the existing outputs:

1. the per-position `uniform_probs` the kernel draws for the accept test (today
   they are consumed and discarded — the key new export);
2. the accept/reject bitmap (already computed as `accepted`);
3. `metadata.draft_token_ids`, the recovered/bonus token ids (already produced);
4. top-k target logprobs from `logits` and top-k `draft_probs`.

Plumb these back through `SamplerOutput` into the API response as a
`spec_decode_trace` field (parallel to the existing `logprobs` plumbing). Notes:

* **Opt-in and off by default** — zero overhead on the hot path when disabled; the
  `SYNTHETIC_MODE` constexpr already in the kernels shows the code is comfortable
  carrying an extra recording mode.
* **Observability, not verification** — the PR adds no trust logic to vLLM. That
  framing (a) is what makes it mergeable and (b) keeps the trusted verifier off
  the provider. Same spirit as SGLang's `eagle_utils` exposing target/draft probs
  and the "coins".
* **Determinism hook (optional, stronger):** derive `u` from a
  client-auditable seed (request seed × position) rather than a private RNG. Then
  the client can recompute the coins and check the decisions bit-exactly instead
  of only statistically (turns check A below from "given the reported `u`" into
  "given the *seed*"). This mirrors the seed-synchronized Gumbel contract already
  in `ivgym/sampling.py`.

## 5. Client-side verification: what a trace can prove about itself

The client trusts none of the reported numbers. What it *can* establish cheaply
(no model forward pass) is **internal consistency** of the trace with the accept
rule — enough to catch every *sampler-level* cheat a provider would use to save
compute. `ivgym.spec_decode` ships these as a check registry; `TraceVerifier`
calibrates thresholds on an honest set (Bonferroni-split across the statistical
checks so the union false-positive rate stays near target) and flags a trace if
any check trips.

| check | kind | catches | trusts |
|---|---|---|---|
| `decision_consistency` | deterministic | accept/reject that doesn't follow reported `p,q,u` | reported `p,q,u` |
| `residual_support` | deterministic | recovered token with zero residual mass | reported `p,q` |
| `logprob_normalization` | deterministic | malformed/unnormalized reported logprobs | — |
| `coin_uniformity` | statistical (KS) | coins fudged to justify decisions | reported `u` |
| `accept_rate_consistency` | statistical (z-test) | observed accepts ≠ `Σ min(1,p/q)` — **needs no coins** | reported `p,q` |
| `recovered_distribution` | statistical (LLR) | recovered tokens drawn from `q`, not the residual | reported `p,q` |
| `target_spotcheck` | **recompute** | reported `p` ≠ true `p` (relabelling) | trusted recompute of `p` |

The escalation is deliberate: `decision_consistency` is a hard, free proof but is
defeated by recomputing coins; `coin_uniformity` + `accept_rate_consistency` then
catch the coin-fudger without any forward pass; and only `target_spotcheck` (which
recomputes a subset of true target logprobs) sees a provider that lied about `p`
itself.

## 6. Results (CPU, `experiments/exp_spec_decode_trace.py`)

Detection AUC, honest vs each cheating provider (80 traces × 200 positions, vocab
64, honest accept rate 0.807). Full grid in the experiment output; the winners:

| cheating provider (compute saved) | kind | caught by | no-recompute? |
|---|---|---|---|
| `over_accept_naive` (skip target check, keep coin) | sampler | `decision_consistency` (1.00), `accept_rate_consistency` (1.00) | ✅ |
| `over_accept_coinfudge` (skip check, refit coin) | sampler | `accept_rate_consistency` (1.00), `coin_uniformity` (0.83) | ✅ |
| `skip_residual` (resample from draft on reject) | sampler | `residual_support` (1.00), `recovered_distribution` (1.00) | ✅ |
| `sampling_bug` (hijack the emitted token — SD analogue of `bug_k2`/`bug_k32`) | sampler | `decision_consistency` (1.00), `residual_support` (0.82) | ✅ |
| `quant_target` (quantized target — SD analogue of `quant_4bit`/`kv_fp8`) | **forward-pass** | `target_spotcheck` (1.00) only | ❌ recompute needed |
| `draft_as_target` (**run only the draft**, relabel as target) | **forward-pass** | `target_spotcheck` (1.00) only | ❌ recompute needed |

Calibrated honest false-positive rate: **0.0–5%** at a 1% target (Bonferroni).

### The dividing line: procedure vs. forward pass

The grid splits exactly along **where the deviation lives** — the same axis DiFR
splits on:

* **Sampler / procedure deviations** (over-accept, coin-fudge, skip-residual,
  sampling-bug) alter *what the acceptance/recovery step does with the reported
  logits*. They break an invariant that is checkable **from the trace alone**, so
  the no-recompute checks catch them. This is the new value the trace adds: for
  ordinary decoding these needed recomputation (they are DiFR's `bug_*` /
  seed / temp attacks); once vLLM exposes the coins and decisions, they become
  free to detect.
* **Forward-pass deviations** (`quant_target`, `kv_fp8`, `draft_as_target`)
  corrupt the target **logits themselves**. The provider then runs *textbook*
  rejection sampling on the corrupted `p` and reports it, so the trace is
  perfectly self-consistent — every no-recompute check sits at chance. Detecting a
  wrong `p` is definitionally "is the reported `p` the true target's `p`?", which
  only `target_spotcheck` (recompute) can answer. **The trace gives no free lunch
  here** — this is the recompute-dominant boundary of
  `docs/SPEC_DECODING_AND_PROXY_DETECTION.md`, unchanged.

Two nuances the trace *does* buy on the forward-pass side, short of full target
recompute:

* **The draft `q` is cheap to recompute** (the client can hold the small draft
  model), so *draft*-side forward-pass corruption is catchable at draft cost — but
  corrupting the draft never biases the SD output (it only lowers throughput), so
  it is not where a rational provider cheats.
* **Wrong seed / wrong temperature** are only invisible if the coin RNG is
  private. With the optional public-coin-seed hook (§4), the client recomputes the
  expected coins and catches a wrong seed with **no forward pass** (the DiFR
  seed-sync idea, ported to the coin); a temperature that is *reported metadata*
  rather than baked into corrupted logits is a trivial metadata-equality check.
  A temperature baked into corrupted logits is a forward-pass deviation and needs
  recompute.

## 7. The limit — and why it is the point

`draft_as_target` reports the draft distribution as *both* draft and target, so
`p == q`: every token accepts, `Σ min(1,p/q)` predicts exactly that, coins stay
uniform, and there are no rejections to inspect. The trace is **perfectly
self-consistent** and every no-recompute check sits at chance
(`test_draft_as_target_evades_selfconsistency_but_caught_by_spotcheck`). It runs
only the cheap draft model — the maximal compute cheat — and no amount of
trace-internal logic can see it. Only recomputing a sample of true target
logprobs (`target_spotcheck`) catches it.

That is the same conclusion the DiFR side of this repo reaches for ordinary
decoding (see `docs/SPEC_DECODING_AND_PROXY_DETECTION.md`): **output-/trace-only
signals bound out at the recompute-dominant gap; recomputation is necessary for
the hardest cheats.** Trace verification is the cheap first line — it makes
sampler-level cheating provably visible for free and shrinks how often the
expensive spot-check must run — not a replacement for it.

*(A note on `draft_as_target` and two-sided tests: its accept-rate z-score is
anomalously* low *— all-accept is perfectly expected under `p==q` — so a two-sided
"too consistent" flag could in principle catch this specific instance. That signal
is fragile: an attacker who sets `p` fractionally above `q` restores natural
variance while still accepting ~everything. The deployed verifier is therefore
one-sided, and the robust catch is the spot-check.)*

## 7b. Beyond self-consistency: can the trace catch *quantization* without recompute?

Self-consistency cannot (§7): the trace is a faithful sample from the corrupted
`p̂` and carries **zero bits about the true `p*`**. So *any* detection of
quantization needs an **external anchor** of information about the honest target.
The design question is which anchors avoid a *per-request* target forward pass.
The full taxonomy:

| anchor | catches quant w/o per-request recompute? | cost / catch |
|---|---|---|
| **Offline honest fingerprint** (entropy / cross-entropy / surprisal stats collected once on a trusted run) | partially | one-time trusted recompute; **gameable by temp-retune**; only sees gross quant |
| **Acceptance-rate fingerprint** (realized `1−TV(p̂,q)` vs the honest `(target,draft)` profile) | partially — **the SD-specific lever** | uses the *trusted, cheap draft* `q` as anchor; more robust than entropy |
| **Cheap draft recompute** | only *draft*-side corruption | draft-cost; but corrupting the draft never biases SD output |
| **Proxy-model recompute** (a different cheap model) | no, for realistic quant | bounded by `KL(M‖proxy)` ≫ quant error — the repo's `exp_detectability_vs_kl` result |
| **Activation/TOPLOC commitment** in the trace | yes, if exposed | needs a reference fingerprint = someone recomputes once; shifts *who/when*, not *whether* |
| **Cross-provider redundancy** | yes | providers recompute *each other*; no client recompute, needs a provider pool |
| **TEE / ZK attestation** | yes | trust moves to hardware/crypto, not the algorithm |
| **Quantization-lattice self-signature** (discretization artifacts in the logits) | in principle, no reference | fragile — softmax/accumulation usually erase clean lattice structure; defeatable by dither |

`ivgym.spec_decode.ReferenceFingerprint` implements the first two; measured in
`experiments/exp_spec_decode_fingerprint.py`
(`docs/results/exp_spec_decode_fingerprint_cpu.txt`):

```
(1) quant-strength sweep          entropy_fp  accept_fp  spotcheck
    sigma 0.1  (small/realistic)      0.490      0.553      1.000   <- invisible w/o recompute
    sigma 0.5                         0.693      0.816      1.000
    sigma 1.0  (gross)               1.000      1.000      1.000
(2) temp-retune evasion, sigma 1.0
    quant (plain)                     1.000      1.000      1.000
    adv_quant_temp                    0.588      0.998      1.000   <- entropy killed, accept survives
```

Two findings: (a) the fingerprint only catches **gross** quantization — small,
realistic quant hides under honest run-to-run variance; and (b) the classic
temperature-retune evasion (the repo's `adv_quant_temp`) **collapses the generic
entropy fingerprint** (1.00 → 0.59, ~floor) but the **acceptance-rate fingerprint
survives** (0.998) — matching entropy does not restore `TV(p̂,q)`, so the trusted
draft still sees the corruption. That draft-anchored acceptance-rate test is the
one no-recompute algorithm the SD trace adds *over* generic black-box statistics.

![Acceptance-rate fingerprint](figures/fig_accept_rate_fingerprint.png)

*How it works, in three panels (`experiments/plot_accept_rate_fingerprint.py`).*
**A** — the mechanism: the realized acceptance rate is `1 − TV(p̂, q)`, so as the
target is quantized more the distribution of per-trace acceptance rates shifts
left, away from the tight honest cluster. **B** — detection AUC rises with
quantization strength; the acceptance-rate fingerprint (anchored on the trusted
draft) leads the generic entropy fingerprint at every σ, while the recompute
spot-check is flat at 1.0. **C** — why the acceptance-rate anchor survives
`adv_quant_temp`: in `(entropy, acceptance-rate)` space the adversary tunes its
temperature back **into** the honest entropy band (grey vertical band → an entropy
detector goes blind) but it cannot re-enter the honest acceptance-rate band (grey
horizontal band), because matching entropy does not restore `TV(p̂, q)`. The
draft-anchored axis still separates it. `fig_accept_rate_mechanism.png` is panel C
standalone.

None of these dominates `target_spotcheck` (1.00 throughout, and robust to
gaming). The honest bottom line: **no purely no-recompute algorithm detects
quantization robustly** — it always needs an external anchor, and the best
cheap ones (offline fingerprints) are partial and gameable. They *shrink how often*
the exact spot-check must fire; they do not replace it.

## 8. Run it

```bash
# client-side verifier + provider simulator, pure numpy, no GPU:
python -m experiments.exp_spec_decode_trace
python -m experiments.exp_spec_decode_fingerprint    # can quant be caught w/o recompute?
python tests/test_spec_decode.py

# knobs: IVGYM_TRACES, IVGYM_POS, IVGYM_VOCAB, IVGYM_AGREE, IVGYM_FPR
IVGYM_AGREE=0.6 IVGYM_TRACES=120 python -m experiments.exp_spec_decode_trace
```

A real backend replaces `synthetic_positions` with target/draft logprobs pulled
from an actual vLLM `spec_decode_trace` (the §4 PR); every check and the verifier
are unchanged.

## 9. Status

* ✅ Client verifier, check registry, provider simulator, `TraceVerifier`, tests,
  CPU experiment — in this repo.
* ⏳ The vLLM trace-emission PR (§4) — patch sketched against
  `rejection_sampler.py`; opening it requires a vLLM fork + a CUDA host to
  validate the top-k plumbing and the opt-in overhead. Ready to draft on request.
