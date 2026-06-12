"""
Autoregressive steganographic codec against the GLS verifier.

Channel model (one machine, paper's generous affordances):
  * Encoder and decoder share the model, dtype, device, sampling params and SEED.
  * Per position the verifier draws one block of V Exp(1) variates from a generator
    seeded once with `seed` (see verify_outputs). `GumbelStream` reproduces that exact
    sequence so encoder/decoder/verifier all see the same Gumbel noise.
  * Logits are taken from a single full-prefix forward pass (causal => the logit at a
    position is independent of later tokens), so encoder and decoder agree bit-for-bit
    in the clean case. A `logit_noise_sigma` knob injects verifier-side numerical
    divergence to study BER / ECC overhead (preview of E8).

Coding:
  * variable rate (rate=None): at each position emit k=floor(log2|A|) payload bits by
    indexing the canonical (sorted-token-id) admissible set A. Decoder reconstructs A
    and reads the emitted token's index -> k bits. Max capacity; clean BER=0.
  * fixed rate (rate=r): only positions with |A|>=2**r carry r bits; others emit the
    forced competitor and carry 0. Avoids k-desync so BER under injected noise is
    measurable, and ECC overhead can be charged against a constant rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from .gls import GLSParams, score_position, admissible_tokens


class GumbelStream:
    """Reproduces verify_outputs' per-token Exp(1) draws: one generator seeded once,
    `vocab`-sized exponential draw per position, in generation order."""

    def __init__(self, seed: int, vocab: int, device):
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.vocab = vocab
        self.device = device

    def next(self) -> torch.Tensor:
        noise = torch.empty(self.vocab, device=self.device)
        noise.exponential_(generator=self.gen)
        return noise


def _bits_to_int(bits) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _int_to_bits(v: int, k: int) -> list[int]:
    return [(v >> (k - 1 - i)) & 1 for i in range(k)]


def _logits_full(model, ids: list[int]) -> torch.Tensor:
    """Single forward pass over `ids`; returns [L, V] float logits."""
    device = model.device
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=inp).logits
    return out.squeeze(0).float()


def per_position_logits(model, prompt_ids: list[int], gen_ids: list[int], mode: str):
    """
    Logits used to predict each generated token.
      mode='prefix' : for token j, forward over prompt+gen[:j], take last row. This MIRRORS
                      the encoder's online computation exactly => bit-identical => clean BER 0.
      mode='full'   : one forward over prompt+gen, slice row (n_prompt+j-1). This is what the
                      UPSTREAM verifier does; differs from 'prefix' only by bf16 reduction
                      order => the prefill-vs-decode numerical gap (experiment E8).
    Returns a list of [V] tensors, one per generated token.
    """
    n_prompt = len(prompt_ids)
    if mode == "full":
        full = _logits_full(model, prompt_ids + gen_ids)
        return [full[n_prompt + j - 1] for j in range(len(gen_ids))]
    elif mode == "prefix":
        return [_logits_full(model, prompt_ids + gen_ids[:j])[-1] for j in range(len(gen_ids))]
    raise ValueError(f"mode must be 'prefix' or 'full', got {mode!r}")


@dataclass
class EncodeResult:
    prompt_ids: list[int]
    gen_ids: list[int]
    per_pos_bits: list[int]          # payload bits carried at each position
    n_payload_bits: int              # total payload bits embedded
    competitors: list[int]           # verifier's c* per position (for diagnostics)
    admissible_sizes: list[int]


def encode(model, prompt_ids: list[int], payload_bits, params: GLSParams, *,
           seed: int, allow: str = "safe", max_tokens: int = 256,
           rate: int | None = None, eos_id: int | None = None) -> EncodeResult:
    """Embed `payload_bits` (iterable of 0/1) into a generated continuation of prompt_ids."""
    device = model.device
    vocab = model.config.vocab_size
    stream = GumbelStream(seed, vocab, device)

    payload = list(int(b) for b in payload_bits)
    cursor = 0
    gen_ids: list[int] = []
    per_pos_bits: list[int] = []
    competitors: list[int] = []
    adm_sizes: list[int] = []

    for _ in range(max_tokens):
        logits_LV = _logits_full(model, prompt_ids + gen_ids)
        logits_V = logits_LV[-1]                      # predicts next token
        exp_noise = stream.next()
        sc = score_position(logits_V, exp_noise, params)
        A = admissible_tokens(sc, allow)
        competitors.append(sc.competitor)
        adm_sizes.append(int(A.numel()))

        a = int(A.numel())
        if rate is None:
            k = max(0, int(torch.floor(torch.log2(torch.tensor(float(a)))).item())) if a > 0 else 0
        else:
            k = rate if a >= (1 << rate) else 0

        if k > 0 and cursor < len(payload):
            chunk = payload[cursor:cursor + k]
            if len(chunk) < k:                        # pad final chunk with zeros
                chunk = chunk + [0] * (k - len(chunk))
            idx = _bits_to_int(chunk)
            idx = min(idx, int(A.numel()) - 1)
            tok = int(A[idx].item())
            consumed = min(k, len(payload) - cursor)
            cursor += k
        else:
            # no capacity here (or payload done): emit the forced competitor (always SAFE)
            tok = sc.competitor
            k = 0
            consumed = 0

        per_pos_bits.append(consumed)
        gen_ids.append(tok)
        if eos_id is not None and tok == eos_id:
            break
        if cursor >= len(payload) and rate is None:
            # payload embedded; stop early (keeps transcript short for the smoke test)
            break

    return EncodeResult(
        prompt_ids=prompt_ids, gen_ids=gen_ids, per_pos_bits=per_pos_bits,
        n_payload_bits=min(cursor, len(payload)), competitors=competitors,
        admissible_sizes=adm_sizes,
    )


@dataclass
class DecodeResult:
    recovered_bits: list[int]
    per_pos_bits: list[int]
    n_desync: int                    # positions where decoder's |A| disagreed (fixed-rate)
    n_token_oob: int                 # emitted token not found in decoder's A


def decode(model, prompt_ids: list[int], gen_ids: list[int], params: GLSParams, *,
           seed: int, allow: str = "safe", rate: int | None = None,
           logit_noise_sigma: float = 0.0,
           noise_seed: int = 12345, logit_mode: str = "prefix") -> DecodeResult:
    """Recover embedded bits from an emitted transcript by reconstructing A per position.

    `logit_mode` selects how the decoder recomputes each position's logits:
      'prefix' (default) : forward over prompt+gen[:j], last row — the SAME online path the
                           encoder used, so in the clean case the logits are bit-identical and
                           BER is 0 by construction (the README's stated invariant).
      'full'             : one forward over the full transcript, sliced — what the UPSTREAM
                           verifier does. Differs from 'prefix' only by bf16 reduction order
                           (the prefill-vs-decode numerical gap); enabling it MEASURES that
                           gap's effect on the channel (experiment E8) with no synthetic noise.
    `logit_noise_sigma` adds Gaussian logit noise on top — a first-order stand-in for
    hardware/quantization divergence, independent of `logit_mode`.
    """
    device = model.device
    vocab = model.config.vocab_size
    stream = GumbelStream(seed, vocab, device)

    pos_logits = per_position_logits(model, prompt_ids, gen_ids, logit_mode)
    if logit_noise_sigma > 0.0:
        ng = torch.Generator(device=device)
        ng.manual_seed(noise_seed)
        pos_logits = [lv + torch.randn(lv.shape, generator=ng, device=device) * logit_noise_sigma
                      for lv in pos_logits]

    recovered: list[int] = []
    per_pos_bits: list[int] = []
    n_desync = 0
    n_oob = 0

    for j, tok in enumerate(gen_ids):
        logits_V = pos_logits[j]
        exp_noise = stream.next()
        sc = score_position(logits_V, exp_noise, params)
        A = admissible_tokens(sc, allow)
        a = int(A.numel())

        if rate is None:
            k = max(0, int(torch.floor(torch.log2(torch.tensor(float(a)))).item())) if a > 0 else 0
        else:
            k = rate if a >= (1 << rate) else 0

        if k == 0:
            per_pos_bits.append(0)
            continue

        hit = torch.nonzero(A == tok, as_tuple=False).flatten()
        if hit.numel() == 0:
            n_oob += 1
            idx = 0                                   # erasure -> guess 0 (counts as errors)
        else:
            idx = int(hit[0].item())
            idx = min(idx, (1 << k) - 1)
        recovered.extend(_int_to_bits(idx, k))
        per_pos_bits.append(k)

    return DecodeResult(recovered_bits=recovered, per_pos_bits=per_pos_bits,
                        n_desync=n_desync, n_token_oob=n_oob)


def make_request_output(prompt_ids: list[int], gen_ids: list[int]):
    """Build the minimal object upstream verify_outputs expects (.prompt_token_ids, .outputs[0].token_ids)."""
    return SimpleNamespace(
        prompt_token_ids=list(prompt_ids),
        outputs=[SimpleNamespace(token_ids=list(gen_ids))],
    )
