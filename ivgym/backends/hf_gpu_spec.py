"""A real speculative-decoding *provider* on a GPU.

`HFGPUBackend` generates by ordinary sequential decoding: one target forward pass
per emitted token, and the token is `argmax(filt(logits) + T*g)` with `g` drawn
from the public per-position seed -- the map a seed-replaying verifier inverts.
This subclass adds the other way a real server produces the same distribution:

  1. a small **draft** model proposes `gamma` tokens autoregressively;
  2. the **target** scores all `gamma` of them in ONE batched forward pass;
  3. each draft is accepted with probability `min(1, p/q)`, and the first
     rejection is corrected by a draw from the normalized residual `(p - q)_+`
     (`ivgym.spec_server`).

Both paths are real: two real models, real KV caches, a real batched verify pass.
Nothing about the speculative path is simulated at the logit level, which matters
because the *reason* it defeats a seed-replaying verifier is procedural, and a
simulation of the procedure would beg the question.

Two properties are load-bearing and easy to lose in an implementation:

* **The target verify pass is genuinely batched.** `_verify_step` forwards
  `gamma` tokens at once against the KV cache, exactly as a serving stack does. It
  is not `gamma` single-token decodes in a loop. The distinction is the entire
  subject of `experiments/exp_spec_batch_numerics_gpu.py`: a batched forward and a
  sequential decode reduce the same arithmetic in different orders and return
  different floats, so the served logits differ from *any* verifier's recompute
  before sampling is even reached.
* **The two arms share everything else.** Same model instance, same prompt
  tokenization, same benign-noise scale, same reference-prefill code path
  (`_populate_ref_cache`, inherited untouched), same `SamplingSpec`. The only
  difference between the honest non-speculative arm and the honest speculative arm
  is how randomness maps to tokens -- otherwise the false positive could be
  attributed to the harness rather than to speculative decoding.

Which path runs is decided by `getattr(attack, "spec_mode", None)`, so every
attack already in `ivgym.attacks` generates exactly as it did before and this
backend is a drop-in replacement for `HFGPUBackend`.
"""
from __future__ import annotations

import numpy as np

from ..attacks import Attack
from ..core import SamplingSpec, Sequence, TokenStep
from ..sampling import position_seed, projection, stable_hash
from ..spec_server import (
    AcceptRule,
    SpecTrace,
    categorical_from_uniform,
    position_uniforms,
    spec_probs,
    speculative_round,
)
from .hf_gpu import HFGPUBackend


class SpecDecodeHFBackend(HFGPUBackend):
    """`HFGPUBackend` plus a real speculative-decoding generation path.

    `draft_model_name` must share the target's tokenizer and vocabulary -- a
    same-family pair (Qwen3-0.6B drafting for Qwen3-1.7B). That is not a
    convenience: speculative decoding requires it, so a provider's draft model is
    always tokenizer-compatible with what it serves, and the client's Tier-0 proxy
    can therefore be the same class of model. When `draft_model_name` equals
    `proxy_model_name` the weights are loaded once and shared, which is also the
    realistic worst case for a client whose proxy happens to be the provider's
    draft.
    """

    def __init__(self, *args, draft_model_name: str = "Qwen/Qwen3-0.6B",
                 gamma: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = int(gamma)
        self.draft_model_name = draft_model_name
        if draft_model_name == kwargs.get("proxy_model_name") and self.proxy_model is not None:
            self.draft_model = self.proxy_model
        else:
            from transformers import AutoModelForCausalLM
            self.draft_model = (
                AutoModelForCausalLM.from_pretrained(
                    draft_model_name, dtype=next(self.model.parameters()).dtype,
                    attn_implementation="eager")
                .to(self.device)
                .eval()
            )
        draft_vocab = int(self.draft_model.config.vocab_size)
        if draft_vocab != self.vocab:
            raise ValueError(
                f"draft vocab ({draft_vocab}) != target vocab ({self.vocab}); "
                f"speculative decoding scores the draft's token ids against the "
                f"target's logits, so the pair must share a tokenizer. Use a "
                f"same-family draft (e.g. Qwen3-0.6B for Qwen3-1.7B).")
        self.draft_n_params = sum(p.numel() for p in self.draft_model.parameters())
        # prompt_id -> list[SpecTrace] (one per emitted token) and run statistics.
        # This is the provider's *disclosure surface*: everything a spec-aware
        # verifier would have to be told. `exp_spec_aware_verifier_gpu` reads it.
        self.spec_traces: dict[int, list[SpecTrace]] = {}
        self.spec_stats: dict[int, dict] = {}

    # ------------------------------------------------------------------ helpers
    def _row(self, logits_row, attack: Attack, prng: np.random.Generator) -> np.ndarray:
        """One target logit row as the *server* has it: real logits plus the same
        benign forward-pass noise the sequential path applies. Identical function,
        identical scale -- so the arms differ only in the sampler."""
        return attack.perturb_logits(logits_row.float().cpu().numpy(), prng)

    def _verify_step(self, tokens: list[int], past, want_hidden: bool):
        """THE batched verify pass: score `len(tokens)` drafted tokens in one
        forward against the KV cache. Returns `(logits[n, V], hidden[n, H], past)`
        where row `j` is the target's prediction for the position *after*
        `tokens[:j+1]`."""
        torch = self._torch
        step = torch.tensor([tokens], device=self.device, dtype=torch.long)
        out = self.model(step, past_key_values=past, use_cache=True,
                         output_hidden_states=want_hidden)
        hidden = out.hidden_states[-1][0] if want_hidden else None
        return out.logits[0], hidden, out.past_key_values

    @staticmethod
    def _crop(past, length: int):
        """Roll a KV cache back to `length` tokens -- what a serving stack does when
        a speculative round's rejected tail is discarded."""
        if past is not None and hasattr(past, "crop"):
            past.crop(length)
        return past

    def _uniforms(self, attack, pspec: SamplingSpec, prompt_id: int, gen_pos: int,
                  srng: np.random.Generator) -> np.ndarray:
        """The three uniforms one speculative slot consumes: draft draw, acceptance
        test, correction draw.

        An ordinary server takes them from its own entropy (`srng`) -- unreplayable
        by anyone, which is the whole problem. A `seeded` server derives them from
        the same public per-position seed the Gumbel path uses, which makes it
        replayable by a verifier that also holds the draft model. The output law is
        the same either way: a seeded uniform is still uniform.
        """
        if getattr(attack, "seeded", False):
            return position_uniforms(position_seed(pspec.seed, prompt_id, gen_pos), 3)
        return srng.random(3)

    # --------------------------------------------------------------- generation
    def generate(self, prompt_id: int, n_tokens: int, spec: SamplingSpec, attack: Attack,
                 record_activations: bool = False, proj_seed: int = 123,
                 proj_dim: int = 32) -> Sequence:
        if getattr(attack, "spec_mode", None) is None:
            return super().generate(prompt_id, n_tokens, spec, attack,
                                    record_activations, proj_seed, proj_dim)
        return self._generate_spec(prompt_id, n_tokens, spec, attack,
                                   record_activations, proj_seed, proj_dim)

    def _generate_spec(self, prompt_id, n_tokens, spec, attack, record_activations,
                       proj_seed, proj_dim) -> Sequence:
        torch = self._torch
        pspec = attack.provider_spec(spec)
        rule: AcceptRule = attack.accept_rule()
        gamma = int(getattr(attack, "gamma", self.gamma))
        proj = projection(proj_seed, proj_dim, self.hidden_dim) if record_activations else None
        seq = Sequence(prompt_id=prompt_id, config_name=attack.name)

        prompt_ids = self._prompt_ids(prompt_id)
        L = int(prompt_ids.shape[1])
        # The server's own randomness -- deliberately NOT the public position seed.
        srng = np.random.default_rng(
            (self.model_seed, prompt_id, 991, stable_hash(attack.name)))

        claimed: list[int] = []
        served: list[np.ndarray] = []
        traces: list[SpecTrace] = []
        n_accepts: list[int] = []
        rounds = 0

        with torch.no_grad():
            t_out = self.model(prompt_ids, use_cache=True,
                               output_hidden_states=record_activations)
            past_t = t_out.past_key_values
            p0_logits = t_out.logits[0, -1]
            p0_hidden = t_out.hidden_states[-1][0, -1] if record_activations else None

            d_out = self.draft_model(prompt_ids, use_cache=True)
            past_d = d_out.past_key_values
            q_last = d_out.logits[0, -1]

            while len(claimed) < n_tokens:
                prefix_len = L + len(claimed)
                base_pos = len(claimed)          # generated index of this round's slot 0
                slot_u = np.stack([
                    self._uniforms(attack, pspec, prompt_id, base_pos + i, srng)
                    for i in range(gamma + 1)])

                # --- 1. draft `gamma` tokens autoregressively under q ---------
                drafted: list[int] = []
                q_rows: list[np.ndarray] = []
                for i in range(gamma):
                    qrow = spec_probs(q_last.float().cpu().numpy(), pspec)
                    x = categorical_from_uniform(qrow, float(slot_u[i, 0]))
                    drafted.append(x)
                    q_rows.append(qrow)
                    if i < gamma - 1:
                        step = torch.tensor([[x]], device=self.device, dtype=torch.long)
                        d_out = self.draft_model(step, past_key_values=past_d, use_cache=True)
                        past_d = d_out.past_key_values
                        q_last = d_out.logits[0, -1]

                # --- 2. ONE batched target pass over all `gamma` drafts -------
                v_logits, v_hidden, past_t = self._verify_step(
                    drafted, past_t, record_activations)

                # Slot 0's target row came from the previous step's single-token
                # decode; slots 1..gamma come from the batched pass above. That
                # split is real -- and it is why the served logits cannot match a
                # verifier's uniform prefill even before sampling is considered.
                p_logits, p_hidden = [], []
                for slot in range(gamma + 1):
                    prng = np.random.default_rng(
                        (self.model_seed, prompt_id, rounds, slot, 11,
                         stable_hash(attack.name)))
                    raw = p0_logits if slot == 0 else v_logits[slot - 1]
                    p_logits.append(self._row(raw, attack, prng))
                    if record_activations:
                        h = (p0_hidden if slot == 0 else v_hidden[slot - 1])
                        act = h.float().cpu().numpy()
                        act = act + prng.normal(0.0, self.act_benign_sigma, self.hidden_dim)
                        extra = attack.activation_extra_sigma()
                        if extra:
                            act = act + prng.normal(0.0, extra, self.hidden_dim)
                        p_hidden.append(act)

                # --- 3. accept / reject / resample ---------------------------
                p_rows = np.stack([spec_probs(r, pspec) for r in p_logits])
                emitted, n_acc, details = speculative_round(
                    p_rows, np.stack(q_rows), drafted, slot_u[:, 1:], rule)
                rounds += 1
                n_accepts.append(n_acc)

                for slot, tok in enumerate(emitted):
                    if len(claimed) >= n_tokens:
                        break
                    pos = len(claimed)
                    fp = (proj @ p_hidden[slot]) if record_activations else None
                    seq.steps.append(TokenStep(position=pos, claimed_token=int(tok),
                                               sampling=spec, fingerprint=fp))
                    claimed.append(int(tok))
                    served.append(p_logits[slot].astype(np.float32))
                    d = details[slot]
                    traces.append(SpecTrace(
                        position=pos, token=int(tok), draft_token=d["draft_token"],
                        accepted=d["accepted"], draft_prob=d["draft_prob"],
                        target_prob=d["target_prob"], u_accept=d["u_accept"],
                        u_sample=d["u_sample"], round_index=rounds - 1, slot=slot,
                        is_bonus=d["is_bonus"]))

                if len(claimed) >= n_tokens:
                    break

                # --- 4. roll both caches forward to the accepted prefix ------
                # The target cache holds prefix + every draft; the accepted ones
                # are exactly `emitted[:n_acc]`, so cropping there keeps real work
                # and discards the speculative tail. The final emitted token (the
                # correction, or the bonus) is not in either cache and is fed as a
                # single-token decode -- which is what produces the next round's
                # slot-0 row.
                past_t = self._crop(past_t, prefix_len + n_acc)
                last = torch.tensor([[emitted[n_acc]]], device=self.device, dtype=torch.long)
                t_out = self.model(last, past_key_values=past_t, use_cache=True,
                                   output_hidden_states=record_activations)
                past_t = t_out.past_key_values
                p0_logits = t_out.logits[0, -1]
                p0_hidden = t_out.hidden_states[-1][0, -1] if record_activations else None

                past_d = self._crop(past_d, prefix_len)
                step = torch.tensor([emitted[: n_acc + 1]], device=self.device,
                                    dtype=torch.long)
                d_out = self.draft_model(step, past_key_values=past_d, use_cache=True)
                past_d = d_out.past_key_values
                q_last = d_out.logits[0, -1]

        self._served_cache[prompt_id] = (np.stack(served) if served
                                         else np.empty((0, self.vocab), np.float32))
        self._claimed[prompt_id] = list(claimed)
        self._prompt_len[prompt_id] = L
        self.spec_traces[prompt_id] = traces
        self.spec_stats[prompt_id] = {
            "rounds": rounds,
            "gamma": gamma,
            "mean_accepted": float(np.mean(n_accepts)) if n_accepts else 0.0,
            "accept_rate": float(np.sum(n_accepts) / max(rounds * gamma, 1)),
            "tokens_per_round": len(claimed) / max(rounds, 1),
            "accepted_frac": float(np.mean([t.accepted for t in traces])) if traces else 0.0,
            "bonus_frac": float(np.mean([t.is_bonus for t in traces])) if traces else 0.0,
        }
        if self.lazy_reference:
            self._ref_depth[prompt_id] = 0
        else:
            self._populate_ref_cache(prompt_id, prompt_ids, claimed, len(claimed))
            self._ref_depth[prompt_id] = len(claimed)
        if self.proxy_model is not None:
            self._populate_proxy_cache(prompt_id, prompt_ids, claimed, len(claimed))
        return seq

    # ------------------------------------------------------- spec-aware verifier
    def draft_prefill(self, prompt_id: int, claimed: list[int]) -> np.ndarray:
        """Teacher-forced draft logits over `[prompt + claimed]`, `[n, V]`.

        One prefill of the *attested* draft model, the cheap half of a spec-aware
        audit. Row `j` is the draft's distribution conditioned on the claimed
        prefix -- which is the right conditioning for slot 0 of a round and the
        wrong one for later slots, where the real server conditioned on its own
        speculative prefix. That mismatch is why a cheap prefill-shaped spec-aware
        check cannot reconstruct the round structure, and why `spec_replay` below
        has to be sequential."""
        torch = self._torch
        prompt_ids = self._prompt_ids(prompt_id)
        L = int(prompt_ids.shape[1])
        full = torch.cat([prompt_ids, torch.tensor([claimed], device=self.device,
                                                   dtype=prompt_ids.dtype)], dim=1)
        with torch.no_grad(), self._timed("proxy"):
            out = self.draft_model(full)
        return out.logits[0, L - 1: L - 1 + len(claimed)].float().cpu().numpy()

    def spec_replay(self, prompt_id: int, claimed: list[int], spec: SamplingSpec,
                    gamma: int | None = None, *, trust_declared: bool = False,
                    ) -> np.ndarray:
        """Spec-aware replay audit: per-token disagreement, 1.0 = inconsistent.

        The verifier re-runs the speculative loop itself -- draft `gamma` tokens
        from the attested draft model, score them with one batched target pass,
        apply the accept/reject rule with the uniforms derived from the PUBLIC
        per-position seed -- and checks that the tokens it reproduces are the tokens
        the provider claimed. This is the sound version of a DiFR audit under
        speculative decoding, and it works only if the server was built to draw its
        randomness from that public seed (`spec_server.SpecDecodeSeeded`); against a
        server using its own entropy it disagrees just as `token_difr` does.

        Rounds are replayed on the CLAIMED prefix (teacher forcing, as every DiFR
        detector does), so a single divergence cannot cascade: after each round the
        caches are advanced over `claimed`, not over what was replayed.

        `trust_declared=True` is the unsound shortcut the accompanying experiment
        exists to kill: instead of rerunning the draft, the verifier believes the
        provider's assertion about what its draft proposed. The provider is then
        free to assert `draft_token = claimed_token` with whatever draft probability
        makes the acceptance test pass, so the audit returns 0 for every token no
        matter what was served. Implemented so the vacuity is measured rather than
        argued.

        NOTE this costs a full re-generation, not a prefill: `len(claimed)/gamma`
        sequential rounds, each a `gamma`-step draft rollout plus a batched target
        pass. `exp_spec_aware_verifier_gpu` measures it against the single prefill
        `token_difr` needs.
        """
        torch = self._torch
        n = len(claimed)
        out = np.zeros(n)
        if n == 0:
            return out
        if trust_declared:
            # The provider declares draft_token := claimed_token and a draft
            # probability small enough that min(1, p/q_hat) >= u. Always possible
            # while p(claimed) > 0, so every token is certified. No forward pass at
            # all -- the shortcut saves the draft model and loses the audit.
            return out
        gamma = int(gamma if gamma is not None else self.gamma)
        rule = AcceptRule("exact")

        prompt_ids = self._prompt_ids(prompt_id)
        L = int(prompt_ids.shape[1])
        with torch.no_grad(), self._timed("reference"):
            t_out = self.model(prompt_ids, use_cache=True)
            past_t = t_out.past_key_values
            p0 = t_out.logits[0, -1]
            d_out = self.draft_model(prompt_ids, use_cache=True)
            past_d = d_out.past_key_values
            q_last = d_out.logits[0, -1]

            pos = 0
            while pos < n:
                prefix_len = L + pos
                slot_u = np.stack([
                    position_uniforms(position_seed(spec.seed, prompt_id, pos + i), 3)
                    for i in range(gamma + 1)])
                drafted, q_rows = [], []
                for i in range(gamma):
                    qrow = spec_probs(q_last.float().cpu().numpy(), spec)
                    drafted.append(categorical_from_uniform(qrow, float(slot_u[i, 0])))
                    q_rows.append(qrow)
                    if i < gamma - 1:
                        step = torch.tensor([[drafted[-1]]], device=self.device,
                                            dtype=torch.long)
                        d_out = self.draft_model(step, past_key_values=past_d, use_cache=True)
                        past_d = d_out.past_key_values
                        q_last = d_out.logits[0, -1]

                v_logits, _, past_t = self._verify_step(drafted, past_t, False)
                p_rows = np.stack([spec_probs(
                    (p0 if s == 0 else v_logits[s - 1]).float().cpu().numpy(), spec)
                    for s in range(gamma + 1)])
                emitted, n_acc, _ = speculative_round(
                    p_rows, np.stack(q_rows), drafted, slot_u[:, 1:], rule)

                take = min(len(emitted), n - pos)
                for k in range(take):
                    out[pos + k] = 0.0 if emitted[k] == claimed[pos + k] else 1.0
                pos_new = pos + take
                if pos_new >= n:
                    break

                # Advance both caches over the CLAIMED tokens, not the replayed ones.
                past_t = self._crop(past_t, prefix_len)
                fwd = torch.tensor([claimed[pos:pos_new]], device=self.device,
                                   dtype=torch.long)
                t_out = self.model(fwd, past_key_values=past_t, use_cache=True)
                past_t = t_out.past_key_values
                p0 = t_out.logits[0, -1]
                past_d = self._crop(past_d, prefix_len)
                d_out = self.draft_model(fwd, past_key_values=past_d, use_cache=True)
                past_d = d_out.past_key_values
                q_last = d_out.logits[0, -1]
                pos = pos_new
        return out

    # ------------------------------------------------------------------ summary
    def spec_summary(self) -> dict:
        """Run statistics pooled over every speculative sequence generated so far:
        the realized acceptance rate, tokens emitted per target verify pass (the
        provider's actual speedup), and the share of emitted tokens that came from
        each path. Reported alongside every detection number so a reader can see
        the server was really speculating and how hard."""
        if not self.spec_stats:
            return {}
        keys = ("accept_rate", "mean_accepted", "tokens_per_round", "accepted_frac",
                "bonus_frac")
        out = {k: float(np.mean([s[k] for s in self.spec_stats.values()])) for k in keys}
        out["rounds"] = int(sum(s["rounds"] for s in self.spec_stats.values()))
        out["gamma"] = int(next(iter(self.spec_stats.values()))["gamma"])
        out["draft_params"] = int(self.draft_n_params)
        out["target_params"] = int(self.n_params)
        return out
