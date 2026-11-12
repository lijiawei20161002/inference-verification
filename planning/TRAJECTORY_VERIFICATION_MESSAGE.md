# Message on the proposed verification improvement

The potential advance over Anthropic's released verification setting is support
for realistic floating-point execution variability. FlashProve already proves
the computation behind checked tokens for its specified integer model. With
fixed deterministic semantics, checking every token against its correct prefix
also establishes the sequence, so whole-sequence coverage alone is not our
contribution. See the [technical report](https://www.anthropic.com/document/flashprove-technical-report#page=12)
and [release limitations](https://www.anthropic.com/research/provable-inference#limitations).

We want to test whether tokens that each have a valid BF16 explanation can
nevertheless lack a single legal state history connecting them. The decisive
requirement is that the checker accept every switch, cache transfer, and prefix
recomputation permitted by the real serving contract. We cannot create an
advantage by forbidding legitimate behavior. If legal recomputation connects
all the local explanations, there is no additional trajectory guarantee under
that contract, and we should report that result.

If a gap survives, the improvement would be a demonstrated ability to reject
jointly incompatible behavior while accepting legitimate numerical variation,
with the required observations and trust assumptions stated explicitly. This
extends the execution semantics we can analyze; it is not a claim of stronger
cryptographic security than FlashProve. The
[experiment plan](../docs/TRAJECTORY_CONSISTENCY_EXPERIMENTS.md) tests that boundary
before a larger GPU study.
