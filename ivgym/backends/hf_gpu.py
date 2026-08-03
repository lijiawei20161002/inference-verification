"""Real-model GPU backend (HuggingFace transformers) -- the arena `ivgym` runs in.

Produces the logits and activations the attacks/defenses/harness operate on from
a real LLM on a CUDA device (validated on an NVIDIA H100-80GB with
Qwen/Qwen3-0.6B).

Mapping to the Backend protocol
-------------------------------
A real model's logits at a position depend on the *claimed token prefix*, so we
cannot recompute them from (prompt, pos) alone. Instead we follow the contract in
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
* honest / quant_4bit / kv_fp8 : the forward-pass perturbations (`Attack.perturb_logits`,
  `activation_extra_sigma`) are applied on top of the *real* Qwen3 logits and
  activations -- a real LLM's base distribution with the attack signal the paper
  studies layered on.
* temp_1.1 / seed_43 : real `SamplingSpec` changes via `attack.provider_spec`.
* bug_k* : `attack.sample_override` hijacks the sampler.

Both provider and verifier carry small independent benign noise (the verifier is
itself a correct-but-noisy deployment), matching the paper's premise that honest
recomputation is non-deterministic.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from ..attacks import Attack
from ..core import SamplingSpec, Sequence, TokenStep
from ..sampling import (
    filtered_logits,
    gumbel_max_sample,
    gumbel_noise,
    position_seed,
    projection,
    stable_hash,
)

# A bank of diverse prompts; prompt_id indexes into it (wrapping). It is kept
# large enough that experiments needing two DISJOINT honest pools -- e.g. the
# I/O detector's honest vs reseeded-honest null floor (exp_io_detector_gpu uses
# prompts [0,N) and [N,2N)) -- can draw non-overlapping text for a reasonable N.
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
    "The mitochondria is best described as",
    "import numpy as np\n\ndef softmax(x):\n    ",
    "My favourite memory from childhood is",
    "The fall of the Berlin Wall in 1989 marked",
    "A polite email declining a meeting might begin:",
    "The chemical formula for table salt is",
    "In machine learning, overfitting refers to",
    "The plot of Hamlet can be summarised as",
    "To change a flat tyre, first you should",
    "The largest planet in the solar system is",
    "A limerick about a curious cat:\n",
    "The French Revolution began because",
    "SELECT name FROM users WHERE",
    "The benefits of regular exercise include",
    "Translate 'good morning' into Spanish:",
    "The theory of evolution by natural selection states",
    "A good opening line for a mystery novel is",
    "The Pythagorean theorem says that",
    "When baking bread, the role of yeast is to",
    "The economic concept of supply and demand explains",
    "Dear diary, today was the kind of day where",
    "The main cause of the 2008 financial crisis was",
    "How does a transistor work? In short,",
    "A motivational quote to start the week:",
    "The water cycle consists of the following stages:",
    "In chess, a common opening for white is",
    "The northern lights are caused by",
    # --- extended bank: more prompts => more INDEPENDENT sequences, which is the
    # only thing that tightens the per-SEQUENCE-constant verifiers' (llm_judge,
    # surface_tokens) honest-null floor toward 0.5. honest + null use
    # disjoint ranges [0,N) and [N,2N), so the bank must hold >= 2N prompts.
    "The capital of Japan is",
    "A brief history of the internet begins with",
    "def quicksort(arr):\n    ",
    "The smell of rain on dry earth is called",
    "Newton's first law of motion states that",
    "To brew a proper pot of green tea you should",
    "The causes of World War One include",
    "Q: How far away is the Moon?\nA:",
    "To whom it may concern, I am writing to request",
    "The key insight behind calculus is",
    "Markets tumbled this morning after news that",
    "A haiku about the first snow:\n",
    "The difference between a virus and a bacterium is",
    "He unfolded the yellowed map and whispered:",
    "Ocean acidification happens when",
    "The recipe needs three eggs, a cup of sugar, and",
    "In biology, a ribosome is responsible for",
    "import torch\n\ndef relu(x):\n    return ",
    "The proudest moment of my life was when",
    "The signing of the Magna Carta in 1215 established",
    "A friendly reminder email might open with:",
    "The chemical symbol for gold is",
    "In statistics, the central limit theorem says",
    "The story of Romeo and Juliet ends with",
    "To replace a light switch safely, first",
    "The smallest prime number is",
    "A limerick about a forgetful wizard:\n",
    "The Industrial Revolution transformed society by",
    "SELECT COUNT(*) FROM orders GROUP BY",
    "Three habits of highly productive people are",
    "Translate 'thank you very much' into French:",
    "Plate tectonics is the theory that",
    "A gripping first sentence for a thriller could be:",
    "Euler's identity connects five numbers:",
    "When fermenting vegetables, salt works by",
    "The concept of opportunity cost means",
    "Dear journal, the strangest thing happened today:",
    "The primary trigger of the Great Depression was",
    "How does a vaccine train the immune system? Briefly,",
    "A proverb about patience goes:",
    "Photosynthesis can be summarised by the equation",
    "In music theory, a major scale is built from",
    "Black holes form when",
    "The capital of Australia is",
    "An overview of the French language would note that",
    "def is_palindrome(s):\n    ",
    "The phenomenon of déjà vu refers to",
    "The law of conservation of energy states that",
    "To sharpen a kitchen knife properly you should",
    "The legacy of the Ottoman Empire includes",
    "Q: Why is the sky blue?\nA:",
    "Dear professor, I am emailing about my grade in",
    "The foundational idea of probability is",
    "Investors reacted nervously today as",
    "A haiku about a quiet river:\n",
    "The distinction between weather and climate is",
    "She pressed the hidden button and the wall",
    "Deforestation contributes to climate change by",
    "For this soup you will need onions, garlic, and",
    "In computing, a hash function is used to",
    "function debounce(fn, delay) {\n  ",
    "The happiest day I can remember involved",
    "The moon landing of 1969 demonstrated that",
    "A thank-you note after an interview might say:",
    "The chemical formula for water is",
    "In economics, inflation is best described as",
    "The ending of Moby-Dick sees",
    "To jump-start a car battery, you first connect",
    "The largest mammal on Earth is",
    "A limerick about a rainy Monday:\n",
    "The Renaissance reshaped art by",
    "UPDATE accounts SET balance = balance -",
    "Two simple ways to reduce stress are",
    "Translate 'where is the station' into German:",
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
        proxy_model_name: str | None = None,
        proxy_sigma: float = 0.6,
        lazy_reference: bool = False,
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
        self.proxy_sigma = proxy_sigma

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
        self.n_params = sum(p.numel() for p in self.model.parameters())
        # prompt_id -> {"logits": [n_tokens, V] float32, "act": [n_tokens, H] float32}
        self._ref_cache: dict[int, dict] = {}
        # prompt_id -> [n_tokens, V] float32: the distribution the provider actually
        # SERVED UNDER, i.e. M's logits with the attack's forward-pass perturbation
        # applied (identity for an honest provider). This is what a logprob-returning
        # inference API hands the client for free -- NOT a client-side recompute of M.
        # The client-side speculative verifier (ivgym.spec_decode.ProxySpecVerifier)
        # reads it as `p` and pairs it with its own cheap proxy `q`; see
        # `served_logits` and the `accept_rate` Tier-0 verifier.
        self._served_cache: dict[int, np.ndarray] = {}

        # --- lazy reference prefill: making the recompute cost REAL ------------
        # By default `generate` eagerly prefills M over the whole
        # [prompt + claimed] sequence for every prompt, so `reference_logits` is a
        # cache read and a selective (budget<1) audit saves nothing measurable --
        # `TokenScores.recompute_ratio` is then a *notional* token fraction, not a
        # cost. With `lazy_reference=True` nothing is prefilled during generation;
        # the verifier must call `prefill_reference(prompt_id, depth)` itself, and
        # pays the real prefill for the prefix it asks for. That turns the audit
        # budget into what it physically is -- reference-forward TOKENS -- which is
        # the cost model `harness.select_prefix_scheduled` schedules against.
        self.lazy_reference = bool(lazy_reference)
        self._claimed: dict[int, list[int]] = {}
        self._prompt_len: dict[int, int] = {}
        self._ref_depth: dict[int, int] = {}      # prompt_id -> rows currently valid
        # prompt_id -> (logit noise, activation noise) at FULL sequence length; a
        # pure function of (model_seed, prompt_id). See `_benign_noise`.
        self._ref_noise: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.prefill_tokens = 0                   # M-forward input tokens spent verifying
        self.prefill_calls = 0                    # number of reference prefills issued

        # --- MEASURED verifier cost (real wall-clock, not a param-count proxy) ---
        # Each detector's cost is the GPU-synchronised forward pass that produces its
        # evidence over one [prompt+claimed] sequence: the full-M reference prefill
        # (token_difr, the recompute) and the cheap-proxy prefill (surface_stat/rank).
        # `verifier cost` figures read mean seconds = seconds[k] / max(calls[k], 1).
        # Text-only detectors (surface_tokens) do no forward pass; their cost is the
        # decode timed in `decode()`. perf_counter is wall-clock; CUDA is async, so we
        # synchronise around the region or the number measures only kernel-launch time.
        self.timed_seconds: dict[str, float] = {"reference": 0.0, "proxy": 0.0, "decode": 0.0}
        self.timed_calls: dict[str, int] = {"reference": 0, "proxy": 0, "decode": 0}

        # --- optional REAL cheap proxy model (Option B) -------------------
        # When set, `proxy_logits` returns a genuine separate-model forward pass
        # instead of a noised read of M's cached logits. The proxy MUST share M's
        # tokenizer/vocab so claimed token ids index its logits directly -- true
        # for a same-family pair (e.g. M=Qwen3-8B, proxy=Qwen3-0.6B).
        self.proxy_model = None
        self.proxy_n_params = None
        self._proxy_cache: dict[int, np.ndarray] = {}
        if proxy_model_name:
            self.proxy_model = (
                AutoModelForCausalLM.from_pretrained(
                    proxy_model_name, dtype=torch_dtype, attn_implementation="eager"
                )
                .to(device)
                .eval()
            )
            proxy_vocab = int(self.proxy_model.config.vocab_size)
            if proxy_vocab != self.vocab:
                raise ValueError(
                    f"proxy vocab ({proxy_vocab}) != reference vocab ({self.vocab}); "
                    f"a black-box proxy detector reads M's claimed token ids against "
                    f"the proxy's logits, so they must share a tokenizer. Use a "
                    f"same-family proxy (e.g. Qwen3-0.6B for Qwen3-8B).")
            self.proxy_n_params = sum(p.numel() for p in self.proxy_model.parameters())

    # ------------------------------------------------------------------ utils
    @contextmanager
    def _timed(self, kind: str):
        """Accumulate GPU-synchronised wall-clock for one timed region into
        `self.timed_seconds[kind]`. Synchronising before AND after is what makes the
        number the real device time rather than the async kernel-launch latency."""
        torch = self._torch
        on_cuda = "cuda" in str(self.device)
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if on_cuda:
                torch.cuda.synchronize()
            self.timed_seconds[kind] += time.perf_counter() - t0
            self.timed_calls[kind] += 1

    def _prompt_ids(self, prompt_id: int):
        text = self.prompts[prompt_id % len(self.prompts)]
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids[:, : self.max_prompt_tokens].to(self.device)

    # --------------------------------------------------- trusted reference side
    def reference_logits(self, prompt_id: int, position: int) -> np.ndarray:
        if self.lazy_reference and position >= self._ref_depth.get(prompt_id, 0):
            raise RuntimeError(
                f"reference_logits(prompt {prompt_id}, pos {position}) not prefilled "
                f"(depth={self._ref_depth.get(prompt_id, 0)}). In lazy_reference mode the "
                f"verifier must call prefill_reference(prompt_id, depth) for the prefix "
                f"it intends to audit -- that call IS the cost being measured.")
        return self._ref_cache[prompt_id]["logits"][position]

    def reference_activation(self, prompt_id: int, position: int) -> np.ndarray:
        return self._ref_cache[prompt_id]["act"][position]

    def prompt_len(self, prompt_id: int) -> int:
        """Prompt token count -- the fixed part of a reference prefill's cost."""
        if prompt_id not in self._prompt_len:
            self._prompt_len[prompt_id] = int(self._prompt_ids(prompt_id).shape[1])
        return self._prompt_len[prompt_id]

    def _benign_noise(self, prompt_id: int, vocab: int, hidden: int):
        """The verifier's benign noise for EVERY position of `prompt_id`, memoized.

        The noise has to be drawn at the sequence's full length and sliced, so that
        a position's noise does not depend on how deep the audit happened to go --
        otherwise the audit budget would leak into the scores. But it is a pure
        function of `(model_seed, prompt_id)`, and `prefill_reference` is called
        once per prompt per BUDGET per scheduler (a cost sweep re-pays prefills on
        purpose, via `drop_reference_cache`), so redrawing it each time was
        re-generating the same ~20M float64 normals hundreds of times -- ~200 ms a
        call at V=151936, which dominated `exp_prefix_cost_gpu`'s runtime.

        Memoizing cannot change a single score (same seed, same shape, same values)
        and cannot change a single reported cost: `prefill_tokens` counts
        model-forward input tokens, and the timed region wraps only the forward
        pass, so this draw was never inside either measurement. It is kept out of
        `drop_reference_cache`, which exists to make the model recompute what an
        audit actually pays for -- benign noise is not that.
        """
        cached = self._ref_noise.get(prompt_id)
        n_full = len(self._claimed[prompt_id])
        if cached is not None and cached[0].shape == (n_full, vocab):
            return cached
        nrng = np.random.default_rng((self.model_seed, prompt_id, 7))
        noise = (nrng.normal(0.0, self.verifier_sigma, (n_full, vocab)),
                 nrng.normal(0.0, self.act_benign_sigma, (n_full, hidden)))
        self._ref_noise[prompt_id] = noise
        return noise

    def prefill_reference(self, prompt_id: int, depth: int) -> int:
        """Prefill M over `[prompt + claimed[:depth-1]]` so reference rows
        `0..depth-1` become readable, and return the input tokens it cost.

        Scoring generated position `j` needs M's prediction at input index
        `L-1+j`, which needs the input to contain `claimed[0..j-1]`. So `depth`
        readable rows require an input of `L + depth - 1` tokens -- the honest
        minimal prefill, not a padded one. Monotone and incremental: asking for a
        depth already covered costs nothing, and a deeper ask re-prefills (a real
        serving verifier would extend the KV cache instead; the cost accounting is
        the same either way because we count the tokens of the deepest prefill,
        see `_ref_depth`).
        """
        torch = self._torch
        have = self._ref_depth.get(prompt_id, 0)
        if depth <= have:
            return 0
        claimed = self._claimed[prompt_id][: max(depth - 1, 0)]
        prompt_ids = self._prompt_ids(prompt_id)
        L = int(prompt_ids.shape[1])
        claimed_t = torch.tensor([claimed], device=self.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, claimed_t], dim=1) if claimed else prompt_ids
        with torch.no_grad(), self._timed("reference"):
            out = self.model(full, output_hidden_states=True)
        idx = slice(L - 1, L - 1 + depth)
        logits = out.logits[0, idx].float().cpu().numpy()
        acts = out.hidden_states[-1][0, idx].float().cpu().numpy()
        full_noise, full_act_noise = self._benign_noise(prompt_id, logits.shape[1],
                                                        acts.shape[1])
        logits = logits + full_noise[:depth]
        acts = acts + full_act_noise[:depth]
        self._ref_cache[prompt_id] = {"logits": logits.astype(np.float32),
                                      "act": acts.astype(np.float32)}
        self._ref_depth[prompt_id] = depth
        # Charge the MARGINAL tokens: a first prefill costs the whole input; a
        # deeper ask costs only the extra generated positions, because a real
        # serving verifier extends the KV cache rather than re-prefilling. (We
        # re-prefill here for simplicity; charging for that would be billing an
        # implementation detail.)
        cost = int(full.shape[1]) if not have else (depth - have)
        self.prefill_tokens += max(cost, 0)
        self.prefill_calls += 1
        return max(cost, 0)

    def reset_cost(self) -> None:
        """Zero the measured verification cost counters (per-experiment scoping)."""
        self.prefill_tokens = 0
        self.prefill_calls = 0
        self.timed_seconds = {k: 0.0 for k in self.timed_seconds}
        self.timed_calls = {k: 0 for k in self.timed_calls}

    def drop_reference_cache(self) -> None:
        """Forget every prefilled reference row (so the next budget pays again)."""
        self._ref_cache.clear()
        self._ref_depth.clear()

    def served_logits(self, prompt_id: int, position: int) -> np.ndarray:
        """The logits the provider SERVED UNDER at this position (M's logits with
        the attack's forward-pass perturbation applied; identity when honest).

        This is the `p` the client-side speculative verifier scores against its own
        proxy `q`. It stands in for a logprob-returning API's output -- the provider
        computed it anyway to serve the request, so reading it costs the client
        NOTHING (unlike `reference_logits`, which is the verifier re-running M)."""
        return self._served_cache[prompt_id][position]

    # --- cheap proxy + raw I/O (black-box detectors; NOT a recompute of M) ---
    def proxy_logits(self, prompt_id: int, position: int) -> np.ndarray:
        """Cheap-proxy LM logits for the black-box detectors -- NEVER a recompute
        of M.

        If a real proxy model is configured (`proxy_model_name`, Option B), this
        returns that small separate model's genuine forward-pass logits over
        [prompt + claimed_tokens] -- a true "cheap model polices the expensive
        model" read, with its own distribution that differs from M's wherever the
        two models actually disagree.

        Otherwise it falls back to the legacy stand-in: a noised read of M's
        cached reference logits (a weaker, cheaper estimator). The fallback reuses
        the ref cache only as a *base distribution* -- it is not offered to
        recomputation defenses and adds proxy noise on top."""
        if self.proxy_model is not None:
            return self._proxy_cache[prompt_id][position]
        base = self._ref_cache[prompt_id]["logits"][position]
        prng = np.random.default_rng((self.model_seed, prompt_id, position, 555))
        return base + prng.normal(0.0, self.proxy_sigma, base.shape)

    def prompt_text(self, prompt_id: int) -> str | None:
        return self.prompts[prompt_id % len(self.prompts)]

    def decode(self, token_ids: list[int]) -> str | None:
        # The only "compute" a text-only detector (surface_tokens) spends per
        # sequence -- timed so it lands on the verifier-cost axis as a real number.
        with self._timed("decode"):
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _populate_ref_cache(self, prompt_id, prompt_ids, claimed, n_tokens):
        """One prefill pass over [prompt + claimed], under the reference model."""
        torch = self._torch
        claimed_t = torch.tensor([claimed], device=self.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, claimed_t], dim=1)
        L = prompt_ids.shape[1]
        with torch.no_grad(), self._timed("reference"):
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

    def _populate_proxy_cache(self, prompt_id, prompt_ids, claimed, n_tokens):
        """One prefill pass over [prompt + claimed] under the REAL cheap proxy
        model -- the black-box detector's view of the sequence. No activations,
        no benign verifier noise: the proxy is a genuinely different (smaller)
        model, so the distribution gap to M IS the signal."""
        torch = self._torch
        claimed_t = torch.tensor([claimed], device=self.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, claimed_t], dim=1)
        L = prompt_ids.shape[1]
        with torch.no_grad(), self._timed("proxy"):
            out = self.proxy_model(full)
        idx = slice(L - 1, L - 1 + n_tokens)
        logits = out.logits[0, idx].float().cpu().numpy()
        self._proxy_cache[prompt_id] = logits.astype(np.float32)

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
        proj = projection(proj_seed, proj_dim, self.hidden_dim) if record_activations else None
        seq = Sequence(prompt_id=prompt_id, config_name=attack.name)

        prompt_ids = self._prompt_ids(prompt_id)
        claimed: list[int] = []
        served: list[np.ndarray] = []   # per-position served logits p (what the API returns)

        # `sample_override` is the only consumer of the descending top-id ordering,
        # and building that ordering costs a full-vocabulary argsort (4.8 ms at
        # V=151936) plus a second `filtered_logits` -- ~13% of a generation step.
        # `Attack.sample_override` returns None unconditionally and, crucially,
        # WITHOUT drawing from `rng`, so for an attack that does not override the
        # hook the whole block is dead work and skipping it leaves the RNG stream
        # (and therefore every sampled token) bit-identical. Only the sampler-bug
        # family pays for it. Asserted in
        # `tests/test_smoke.py::test_top_id_ordering_is_skipped_only_when_unused`.
        overrides_sampler = type(attack).sample_override is not Attack.sample_override

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
                served.append(logits.astype(np.float32))

                gseed = position_seed(pspec.seed, prompt_id, pos)
                g = gumbel_noise(self.vocab, gseed)

                override = None
                if overrides_sampler:
                    filt = filtered_logits(logits, pspec.top_k, pspec.top_p)
                    override = attack.sample_override(prng, np.argsort(filt)[::-1])
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

        self._served_cache[prompt_id] = np.stack(served) if served else np.empty((0, self.vocab), np.float32)
        # Record what a lazy reference prefill would need to replay.
        self._claimed[prompt_id] = list(claimed)
        self._prompt_len[prompt_id] = int(prompt_ids.shape[1])
        if self.lazy_reference:
            # Deliberately DO NOT prefill M here. The verifier pays for the prefix
            # it actually audits, via `prefill_reference`.
            self._ref_depth[prompt_id] = 0
        else:
            self._populate_ref_cache(prompt_id, prompt_ids, claimed, n_tokens)
            self._ref_depth[prompt_id] = n_tokens
        if self.proxy_model is not None:
            self._populate_proxy_cache(prompt_id, prompt_ids, claimed, n_tokens)
        return seq
