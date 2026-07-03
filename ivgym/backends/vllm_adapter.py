"""vLLM backend adapter (contract + skeleton).

NOT runnable on this machine (no CUDA / no vLLM here). This file documents the
exact contract a real backend must satisfy so that the *same* attacks,
defenses, and harness run unchanged against real models on a GPU box.

How the abstractions map to real inference
------------------------------------------
Attacks (provider side) become real vLLM config:
    honest        -> reference LLM(...) config
    quant_4bit    -> LLM(quantization="awq"/"gptq"/...)         # or bitsandbytes 4-bit
    kv_fp8        -> LLM(kv_cache_dtype="fp8")
    temp_1.1      -> SamplingParams(temperature=1.1)
    seed_43       -> SamplingParams(seed=43)
    bug_k*        -> a custom LogitsProcessor / patched sampler
    adv_quant_temp-> quantization + tuned SamplingParams(temperature=t)

Defenses are backend-independent and need no change: they consume
`reference_logits` / `reference_activation` exactly as the HF-GPU backend
provides them.

Implementation notes
--------------------
* reference_*: run ONE prefill forward pass over [prompt + claimed tokens]
  under the reference config (Section 3.3 -- prefill is 3-5x faster), then read
  per-position logits and final-hidden-state activations. Cache per sequence.
* generate: issue a normal vLLM request with a per-request seed
  (vLLM passes a torch.Generator to exponential() for Gumbel noise -- match it
  in ivgym.sampling.gumbel_noise so the verifier reconstructs identical noise).
* Activation fingerprints: register a forward hook on the final norm layer,
  project with the shared orthogonal matrix (ivgym.sampling.projection),
  and store on TokenStep.fingerprint.
"""
from __future__ import annotations

import numpy as np

from ..core import SamplingSpec, Sequence


class VLLMBackend:
    """Skeleton. Fill in with a real vLLM LLM handle on a GPU host."""

    def __init__(self, model: str, hidden_dim: int, vocab: int, **llm_kwargs):
        self.model = model
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self._llm = None  # = vllm.LLM(model=model, **llm_kwargs)
        raise NotImplementedError(
            "VLLMBackend is a documented contract; instantiate on a CUDA host "
            "with vLLM installed and implement the three methods below.")

    def reference_logits(self, prompt_id: int, position: int) -> np.ndarray:
        raise NotImplementedError

    def reference_activation(self, prompt_id: int, position: int) -> np.ndarray:
        raise NotImplementedError

    def generate(self, prompt_id, n_tokens, spec: SamplingSpec, attack,
                 record_activations=False, proj_seed=123, proj_dim=32) -> Sequence:
        raise NotImplementedError

    # --- speculative-decoding trace verification (ivgym.spec_decode) ---------
    # If the server exposes the opt-in speculative-decoding trace (per-position
    # target/draft logprobs,
    # the accept coin, the decision, and the recovered/bonus tokens), surface it
    # here as an `ivgym.spec_decode.SpecDecodeTrace`. The client-side
    # `TraceVerifier` then checks it WITHOUT trusting this provider; the
    # `target_spotcheck` check calls back into `reference_logits` to recompute a
    # sampled subset of true target logprobs.
    def spec_decode_trace(self, prompt_id: int):
        """Return the provider-reported ivgym.spec_decode.SpecDecodeTrace, or
        None if the server was not run with trace emission enabled."""
        raise NotImplementedError
