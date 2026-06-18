"""Real-model GPU backend (HuggingFace transformers).

Runs the *same* attacks / defenses / harness as the synthetic backend, but the
logits and activations come from a real LLM on a CUDA device (validated on an
NVIDIA A100-80GB with Qwen/Qwen3-0.6B).

Mapping to the Backend protocol
-------------------------------
Synthetic `_true_logits(prompt, pos)` depended only on (prompt, pos). A real
model's logits at a position depend on the *claimed token prefix*, so we cannot
recompute them from (prompt, pos) alone. Instead we follow the contract in
`vllm_adapter.py`:

  generate():  autoregressive rollout under the provider/attack config (real
               forward passes, KV-cached). Records the claimed tokens, then runs
               ONE reference prefill over [prompt + claimed_tokens] under the
               reference model and caches per-position reference logits and
               final-hidden-state activations, keyed by prompt_id.
  reference_*: read straight from that cache.

The harness always calls `verify(...)` on a dataset immediately after
`generate_dataset(...)` for that same config, so the per-prompt cache is fresh
when the verifier reads it (honest scores are materialised up front and kept as
numbers, so they survive later overwrites).

How attacks map onto a real model
----------------------------------
* honest / quant_4bit / kv_fp8 : the forward-pass perturbations the synthetic
  backend models as extra logit / activation noise (`Attack.perturb_logits`,
  `activation_extra_sigma`) are applied on top of the *real* Qwen3 logits and
  activations. The base distribution is now a real LLM's; the attack signal is
  the same one the paper studies.
* temp_1.1 / seed_43 : real `SamplingSpec` changes via `attack.provider_spec`.
* bug_k* : `attack.sample_override` hijacks the sampler, unchanged.

Both provider and verifier carry small independent benign noise, mirroring the
synthetic design (the verifier is itself a correct-but-noisy deployment) and the
paper's premise that honest recomputation is non-deterministic.
"""
from __future__ import annotations

import numpy as np

from ..attacks import Attack
from ..core import SamplingSpec, Sequence, TokenStep
from ..sampling import (
    filtered_logits,
    gumbel_max_sample,
    gumbel_noise,
    position_seed,
    stable_hash,
)
from .synthetic import _projection

# A small bank of diverse prompts; prompt_id indexes into it.
DEFAULT_PROMPTS = [
    "The capital of France is",
    "In a shocking turn of events, scientists discovered that",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a kingdom by the sea,",
    "The three laws of thermodynamics state that",
    "To make a good cup of espresso you should",
    "The history of the Roman Empire can be summarised as",
    "Q: What is the speed of light?\nA:",
    "Dear hiring manager, I am writing to apply for",
    "The most important idea in linear algebra is",
    "Breaking news: the stock market today",
    "A haiku about autumn leaves:\n",
    "The difference between TCP and UDP is",
    "She opened the ancient book and read aloud:",
    "Climate change is primarily driven by",
    "The recipe calls for two cups of flour and",
]


class HFGPUBackend:
    """transformers backend producing real logits / activations on a GPU."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        prompts: list[str] | None = None,
        model_seed: int = 0,
        verifier_sigma: float = 0.02,
        act_benign_sigma: float = 0.05,
        max_prompt_tokens: int = 32,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device
        self.model_seed = model_seed
        self.verifier_sigma = verifier_sigma
        self.act_benign_sigma = act_benign_sigma
        self.max_prompt_tokens = max_prompt_tokens
        self.prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)

        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch_dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
        self.vocab = int(self.model.config.vocab_size)
        self.hidden_dim = int(self.model.config.hidden_size)
        # prompt_id -> {"logits": [n_tokens, V] float32, "act": [n_tokens, H] float32}
        self._ref_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------ utils
    def _prompt_ids(self, prompt_id: int):
        text = self.prompts[prompt_id % len(self.prompts)]
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids[:, : self.max_prompt_tokens].to(self.device)

    # --------------------------------------------------- trusted reference side
    def reference_logits(self, prompt_id: int, position: int) -> np.ndarray:
        return self._ref_cache[prompt_id]["logits"][position]

    def reference_activation(self, prompt_id: int, position: int) -> np.ndarray:
        return self._ref_cache[prompt_id]["act"][position]

    def _populate_ref_cache(self, prompt_id, prompt_ids, claimed, n_tokens):
        """One prefill pass over [prompt + claimed], under the reference model."""
        torch = self._torch
        claimed_t = torch.tensor([claimed], device=self.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, claimed_t], dim=1)
        L = prompt_ids.shape[1]
        with torch.no_grad():
            out = self.model(full, output_hidden_states=True)
        # logits at input index L-1+pos predict generated token `pos`; the
        # final-layer hidden state at that same index produced those logits.
        idx = slice(L - 1, L - 1 + n_tokens)
        logits = out.logits[0, idx].float().cpu().numpy()
        acts = out.hidden_states[-1][0, idx].float().cpu().numpy()
        # verifier benign noise (independent, deterministic per position)
        nrng = np.random.default_rng((self.model_seed, prompt_id, 7))
        logits = logits + nrng.normal(0.0, self.verifier_sigma, logits.shape)
        acts = acts + nrng.normal(0.0, self.act_benign_sigma, acts.shape)
        self._ref_cache[prompt_id] = {
            "logits": logits.astype(np.float32),
            "act": acts.astype(np.float32),
        }

    # ------------------------------------------------------ provider generation
    def generate(
        self,
        prompt_id: int,
        n_tokens: int,
        spec: SamplingSpec,
        attack: Attack,
        record_activations: bool = False,
        proj_seed: int = 123,
        proj_dim: int = 32,
    ) -> Sequence:
        torch = self._torch
        pspec = attack.provider_spec(spec)
        proj = _projection(proj_seed, proj_dim, self.hidden_dim) if record_activations else None
        seq = Sequence(prompt_id=prompt_id, config_name=attack.name)

        prompt_ids = self._prompt_ids(prompt_id)
        claimed: list[int] = []

        with torch.no_grad():
            out = self.model(prompt_ids, use_cache=True, output_hidden_states=record_activations)
            past = out.past_key_values
            logits_last = out.logits[0, -1]
            hidden_last = out.hidden_states[-1][0, -1] if record_activations else None

            for pos in range(n_tokens):
                base = logits_last.float().cpu().numpy()
                prng = np.random.default_rng(
                    (self.model_seed, prompt_id, pos, 11, stable_hash(attack.name))
                )
                logits = attack.perturb_logits(base, prng)

                gseed = position_seed(pspec.seed, prompt_id, pos)
                g = gumbel_noise(self.vocab, gseed)

                filt = filtered_logits(logits, pspec.top_k, pspec.top_p)
                top_ids = np.argsort(filt)[::-1]
                override = attack.sample_override(prng, top_ids)
                if override is not None:
                    token = int(override)
                else:
                    token = gumbel_max_sample(
                        logits, pspec.temperature, g, pspec.top_k, pspec.top_p
                    )

                fp = None
                if record_activations:
                    act = hidden_last.float().cpu().numpy()
                    act = act + prng.normal(0.0, self.act_benign_sigma, self.hidden_dim)
                    extra = attack.activation_extra_sigma()
                    if extra:
                        act = act + prng.normal(0.0, extra, self.hidden_dim)
                    fp = proj @ act

                seq.steps.append(
                    TokenStep(position=pos, claimed_token=token, sampling=spec, fingerprint=fp)
                )
                claimed.append(token)

                step_t = torch.tensor([[token]], device=self.device, dtype=prompt_ids.dtype)
                out = self.model(
                    step_t, past_key_values=past, use_cache=True,
                    output_hidden_states=record_activations,
                )
                past = out.past_key_values
                logits_last = out.logits[0, -1]
                hidden_last = out.hidden_states[-1][0, -1] if record_activations else None

        self._populate_ref_cache(prompt_id, prompt_ids, claimed, n_tokens)
        return seq
