"""Seed-free regime: an indistinguishable attack that actually saves compute.

When the verifier *cannot* replay the provider's sampling RNG (the provider owns
the sampler / non-deterministic decode), seed-synchronized **Token-DiFR is
unavailable**. The verifier is left with seed-free scores:

  * `cross_entropy`   -- NLL of the claimed token under the reference logits,
  * `activation_difr` -- L2 between the provider-*reported* activation fingerprint
                         and the verifier's recomputed one,
  * `topk_overlap`    -- rank of the claimed token (a TOPLOC-style check).

All three are functions only of what the provider *emits* (the claimed token, or
the reported fingerprint). That is the opening: the attacker quantizes for the
compute win, then shapes what it emits so those emitted quantities match honest.

This attack composes three moves:
  1. **Quantize the forward pass** (`extra_sigma`/`bias_sigma` logit
     perturbation) -- the genuine compute saving.
  2. **Retune temperature** (`provider_temp`, tuned at runtime) so mean
     cross-entropy matches honest -- collapses the CE detector (DiFR Fig. 2).
  3. **Spoof the activation fingerprint** -- report a clean (reference) activation
     decoupled from the cheap computation. Activation-DiFR is only as strong as
     the *binding* between the reported fingerprint and the served output; the
     seed-free setting provides no such binding.

The point of the experiment (`experiments/exp_seed_free.py`) is that against the
seed-free panel this is indistinguishable (AUC ~ 0.5, TPR@1% ~ chance) while
genuinely quantizing -- yet the SAME attack is caught instantly by Token-DiFR
once seed-sync is available. Seed synchronization is what buys robustness.
"""
from __future__ import annotations

from dataclasses import dataclass

from ivgym.attacks import Attack, register as register_attack
from ivgym.core import SamplingSpec


@dataclass
class AdvQuantTempSpoof(Attack):
    """Quantize + temperature-retune + (optionally) spoof the fingerprint."""

    name: str = "adv_quant_spoof"
    extra_sigma: float = 0.30        # random logit noise from the quantized pass
    bias_sigma: float = 0.10         # systematic quantization error
    act_sigma: float = 0.30          # the TRUE activation perturbation it incurs
    provider_temp: float = 1.0       # tuned at runtime to match honest mean CE
    spoof_activation: bool = True    # report a clean fingerprint vs. honest report

    def logit_bias_sigma(self):
        return self.extra_sigma, self.bias_sigma

    def activation_extra_sigma(self):
        # spoof=True  -> 0.0: report the reference-clean activation (the quant
        #                perturbation never reaches the verifier's fingerprint).
        # spoof=False -> act_sigma: honestly expose the quantized activation,
        #                which activation_difr then catches.
        return 0.0 if self.spoof_activation else self.act_sigma

    def provider_spec(self, ref: SamplingSpec) -> SamplingSpec:
        return ref.replace(temperature=self.provider_temp)


# A couple of registered instances for `--list` / the standard runner. NOTE:
# provider_temp here is a placeholder -- exp_seed_free.py tunes it per quant
# strength at runtime so mean CE matches honest. Untuned, the CE detector fires.
for q in (0.7, 1.0, 1.4):
    register_attack(AdvQuantTempSpoof(
        name=f"adv_quant_spoof_q{q}", extra_sigma=0.30 * q, bias_sigma=0.10 * q))
register_attack(AdvQuantTempSpoof(name="adv_quant_honest_act", spoof_activation=False))
