"""Shared model-quantization helpers for the subtle-quant experiments.

The subtle-quant experiments (exp_real_quant_triage, exp_tie_triage_margin, ...)
need a QUANTIZED provider model M_hat derived from the reference M. Two paths:

  * fake_quant_(model, bits)  -- deterministic per-output-channel signed n-bit
    weight-only quant, in place. Dependency-free (pure torch), runs on any CUDA
    box, and sweeps bit-width so the SUBTLE regime (8/6/5-bit) is reachable. This
    is the default and what the committed figures were produced with.
  * load_nf4(name)            -- a REAL bitsandbytes NF4 4-bit weight-only load,
    i.e. the exact format a client would actually ship. It is the ground truth the
    deterministic fake-quant is meant to approximate; use it to confirm the
    flip-rate / sparsity numbers hold on true 4-bit weights. Requires the optional
    `bitsandbytes` dependency and a CUDA host.

Select the path with the IVGYM_QUANT env var:

    IVGYM_QUANT=fake   (default)  deterministic fake-quant, swept over IVGYM_QBITS
    IVGYM_QUANT=nf4               real bitsandbytes NF4 (a single 4-bit config;
                                  the bit-width sweep collapses to one "nf4" run)

Both paths skip `lm_head` (transformers keeps it out of the 4-bit conversion by
default, and fake_quant_ skips it explicitly), so the two are apples-to-apples on
the transformer Linear weights.
"""
from __future__ import annotations

import os

# "fake" (deterministic, sweepable) | "nf4" (real bitsandbytes 4-bit).
QUANT_MODE = os.environ.get("IVGYM_QUANT", "fake").lower()


def load(name, torch):
    """Full-precision (bf16) reference load, on CUDA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    return tok, mdl


def fake_quant_(model, bits, torch):
    """Deterministic per-output-channel signed n-bit weight-only quant, in place.
    Skips lm_head. Error is a structured function of each row's magnitude.
    Returns the number of quantized Linear layers."""
    import torch.nn as nn
    lvl = 2 ** (bits - 1) - 1
    n_q = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            W = mod.weight.data
            scale = W.abs().amax(dim=1, keepdim=True) / lvl + 1e-8
            q = torch.clamp(torch.round(W / scale), -lvl - 1, lvl)
            mod.weight.data = (q * scale).to(W.dtype)
            n_q += 1
    return n_q


def load_nf4(name, torch):
    """Load `name` with REAL bitsandbytes NF4 4-bit weight-only quantization -- the
    ground truth the deterministic fake_quant_ approximates. Double-quantized NF4
    with a bf16 compute dtype (the shipped default, matching experiments/generate.py's
    quant_4bit config). transformers keeps lm_head/embeddings out of the 4-bit
    conversion, mirroring fake_quant_'s lm_head skip."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=qc, device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    return tok, mdl


def quant_settings(bits_list):
    """The list of quant settings to sweep, given the requested fake-quant bit
    widths. In `nf4` mode this collapses to a single real-4-bit run; otherwise it
    is `bits_list` unchanged. Each setting is either an int (fake-quant bits) or
    the string "nf4"."""
    if QUANT_MODE == "nf4":
        return ["nf4"]
    return list(bits_list)


def quant_label(setting) -> str:
    """Short human/plot label for a quant setting."""
    return "nf4" if setting == "nf4" else f"{setting}-bit"


def make_quant(name, setting, torch):
    """Return `(tokenizer, quantized_model)` for one quant `setting`:
    "nf4" -> real bitsandbytes NF4; an int -> fresh bf16 load + in-place fake-quant
    to that many bits. The model is always a fresh load (fake-quant is destructive)."""
    if setting == "nf4":
        return load_nf4(name, torch)
    tok, mdl = load(name, torch)
    fake_quant_(mdl, int(setting), torch)
    return tok, mdl
