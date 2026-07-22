"""Canonical `ModelIdentity` facts for every HF id used by the model-distance
ladder experiments (`exp_proxy_distance_grid`, `exp_cross_family_accept`,
`exp_family_correlation`, `exp_robustness_gpu`).

One entry per model, stated once. `ivgym.model_taxonomy.distance()` /
`describe()` derive every ladder's ordering and group labels from these facts
-- no experiment file hand-picks an integer or types out a group string.

Facts here (family/base/org/generation/domain/tokenizer) are the same
public-model-card-level claims the experiments' own docstrings already made in
prose (e.g. "Qwen3, Qwen2.5, Qwen2.5-Coder, and DeepSeek-R1-Distill-Qwen all
encode text to identical ids" in `exp_cross_family_accept.py`); this module is
where that prose becomes structured and reusable instead of re-typed per file.
Adding a new model to any experiment means adding one entry here.
"""
from __future__ import annotations

from dataclasses import replace

from ivgym.model_taxonomy import ModelIdentity, quantized

# ---------------------------------------------------------------------------
# Qwen family. Qwen3 / Qwen2.5 / Qwen2.5-Coder / DeepSeek-R1-Distill-Qwen all
# share Qwen's exact tokenizer/vocab (verified in exp_cross_family_accept.py),
# so `tokenizer="qwen"` throughout even where `generation` / `org` differ.
_QWEN25_7B = ModelIdentity(
    id="Qwen/Qwen2.5-7B-Instruct", label="Qwen2.5-7B", org="Qwen",
    family="qwen", base="qwen2.5-7b", generation="2.5", domain="general",
    tokenizer="qwen", params=7.6e9)
_QWEN25_3B = ModelIdentity(
    id="Qwen/Qwen2.5-3B-Instruct", label="Qwen2.5-3B", org="Qwen",
    family="qwen", base="qwen2.5-3b", generation="2.5", domain="general",
    tokenizer="qwen", params=3.1e9)
_QWEN25_CODER_7B = ModelIdentity(
    id="Qwen/Qwen2.5-Coder-7B-Instruct", label="Qwen2.5-Coder-7B", org="Qwen",
    family="qwen", base="qwen2.5-7b", generation="2.5", domain="code",
    tokenizer="qwen", params=7.6e9)
_QWEN25_CODER_1_5B = ModelIdentity(
    id="Qwen/Qwen2.5-Coder-1.5B", label="Qwen2.5-Coder-1.5B", org="Qwen",
    family="qwen", base="qwen2.5-1.5b", generation="2.5", domain="code",
    tokenizer="qwen", params=1.5e9)
_QWEN3_8B = ModelIdentity(
    id="Qwen/Qwen3-8B", label="Qwen3-8B", org="Qwen",
    family="qwen", base="qwen3-8b", generation="3", domain="general",
    tokenizer="qwen", params=8.2e9)
_QWEN3_4B = ModelIdentity(
    id="Qwen/Qwen3-4B", label="Qwen3-4B", org="Qwen",
    family="qwen", base="qwen3-4b", generation="3", domain="general",
    tokenizer="qwen", params=4.0e9)
_QWEN3_1_7B = ModelIdentity(
    id="Qwen/Qwen3-1.7B", label="Qwen3-1.7B", org="Qwen",
    family="qwen", base="qwen3-1.7b", generation="3", domain="general",
    tokenizer="qwen", params=1.7e9)
_QWEN3_0_6B = ModelIdentity(
    id="Qwen/Qwen3-0.6B", label="Qwen3-0.6B", org="Qwen",
    family="qwen", base="qwen3-0.6b", generation="3", domain="general",
    tokenizer="qwen", params=0.6e9)
_QWEN25_1_5B = ModelIdentity(
    id="Qwen/Qwen2.5-1.5B", label="Qwen2.5-1.5B", org="Qwen",
    family="qwen", base="qwen2.5-1.5b", generation="2.5", domain="base",
    tokenizer="qwen", params=1.5e9)
_QWEN25_0_5B = ModelIdentity(
    id="Qwen/Qwen2.5-0.5B", label="Qwen2.5-0.5B", org="Qwen",
    family="qwen", base="qwen2.5-0.5b", generation="2.5", domain="base",
    tokenizer="qwen", params=0.5e9)
_DS_DISTILL_QWEN_7B = ModelIdentity(
    id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", label="DS-R1-Distill-Qwen",
    org="DeepSeek", family="qwen", base="qwen2.5-7b", generation="2.5",
    domain="reasoning-distill", tokenizer="qwen", params=7.6e9)
_DS_DISTILL_QWEN_1_5B = ModelIdentity(
    id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", label="DS-R1-Qwen-1.5B",
    org="DeepSeek", family="qwen", base="qwen2.5-1.5b", generation="2.5",
    domain="reasoning-distill", tokenizer="qwen", params=1.5e9)

# ---------------------------------------------------------------------------
# Llama family (Llama-3.x + its DeepSeek reasoning-distill), Meta.
_LLAMA31_8B = ModelIdentity(
    id="unsloth/Meta-Llama-3.1-8B-Instruct", label="Llama-3.1-8B", org="Meta",
    family="llama", base="llama-3.1-8b", generation="3.1", domain="general",
    tokenizer="llama3", params=8.0e9)
_LLAMA31_8B_BASE = ModelIdentity(
    id="unsloth/Meta-Llama-3.1-8B", label="Llama-3.1-8B base", org="Meta",
    family="llama", base="llama-3.1-8b", generation="3.1", domain="base",
    tokenizer="llama3", params=8.0e9)
_LLAMA30_8B = ModelIdentity(
    id="unsloth/llama-3-8b-Instruct", label="Llama-3-8B", org="Meta",
    family="llama", base="llama-3.0-8b", generation="3.0", domain="general",
    tokenizer="llama3", params=8.0e9)
_LLAMA32_3B = ModelIdentity(
    id="unsloth/Llama-3.2-3B-Instruct", label="Llama-3.2-3B", org="Meta",
    family="llama", base="llama-3.2-3b", generation="3.2", domain="general",
    tokenizer="llama3", params=3.2e9)
_LLAMA32_1B = ModelIdentity(
    id="unsloth/Llama-3.2-1B-Instruct", label="Llama-3.2-1B", org="Meta",
    family="llama", base="llama-3.2-1b", generation="3.2", domain="general",
    tokenizer="llama3", params=1.2e9)
_DS_DISTILL_LLAMA_8B = ModelIdentity(
    id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", label="DS-R1-Distill-Llama",
    org="DeepSeek", family="llama", base="llama-3.1-8b", generation="3.1",
    domain="reasoning-distill", tokenizer="llama3", params=8.0e9)

# ---------------------------------------------------------------------------
# SmolLM2 (HuggingFaceTB) and Pythia (EleutherAI, GPT-NeoX arch): the two
# genuinely different architectures/tokenizers in the robustness matrix.
_SMOLLM2_1_7B = ModelIdentity(
    id="HuggingFaceTB/SmolLM2-1.7B-Instruct", label="SmolLM2-1.7B", org="HuggingFace",
    family="smollm2", base="smollm2-1.7b", generation="2", domain="general",
    tokenizer="smollm2", params=1.7e9)
_SMOLLM2_360M = ModelIdentity(
    id="HuggingFaceTB/SmolLM2-360M-Instruct", label="SmolLM2-360M", org="HuggingFace",
    family="smollm2", base="smollm2-360m", generation="2", domain="general",
    tokenizer="smollm2", params=0.36e9)
_SMOLLM2_135M = ModelIdentity(
    id="HuggingFaceTB/SmolLM2-135M-Instruct", label="SmolLM2-135M", org="HuggingFace",
    family="smollm2", base="smollm2-135m", generation="2", domain="general",
    tokenizer="smollm2", params=0.135e9)
_PYTHIA_410M = ModelIdentity(
    id="EleutherAI/pythia-410m", label="Pythia-410M", org="EleutherAI",
    family="pythia", base="pythia-410m", generation="1", domain="base",
    tokenizer="neox", params=0.41e9)
_PYTHIA_160M = ModelIdentity(
    id="EleutherAI/pythia-160m", label="Pythia-160M", org="EleutherAI",
    family="pythia", base="pythia-160m", generation="1", domain="base",
    tokenizer="neox", params=0.16e9)
_PYTHIA_1_4B = ModelIdentity(
    id="EleutherAI/pythia-1.4b", label="Pythia-1.4B", org="EleutherAI",
    family="pythia", base="pythia-1.4b", generation="1", domain="base",
    tokenizer="neox", params=1.4e9)

# ---------------------------------------------------------------------------
# GPT-2 (OpenAI): a fifth family and a clean, ungated size ladder (124M ->
# 355M -> 774M) that all share one tokenizer/vocab, so any smaller sibling is a
# valid same-family cheap proxy for a larger one -- exactly what the cost-curve
# sweep (exp_cost_curve_gpu) needs. A pre-instruct base-LM architecture, which
# is the point: it shows the verification method is architecture-agnostic, not a
# quirk of modern instruct models.
_GPT2_124M = ModelIdentity(
    id="gpt2", label="GPT2-124M", org="OpenAI",
    family="gpt2", base="gpt2-124m", generation="1", domain="base",
    tokenizer="gpt2", params=0.124e9)
_GPT2_355M = ModelIdentity(
    id="gpt2-medium", label="GPT2-355M", org="OpenAI",
    family="gpt2", base="gpt2-355m", generation="1", domain="base",
    tokenizer="gpt2", params=0.355e9)
_GPT2_774M = ModelIdentity(
    id="gpt2-large", label="GPT2-774M", org="OpenAI",
    family="gpt2", base="gpt2-774m", generation="1", domain="base",
    tokenizer="gpt2", params=0.774e9)

# ---------------------------------------------------------------------------
# Quantized copies. bnb variants use a "base_id::method" id (loaded via
# bitsandbytes off the base repo, see exp_proxy_distance_grid._parse_id) via
# `quantized()`; official GPTQ/AWQ checkpoints are separate HF repos with
# their own id, so those use a plain `dataclasses.replace`.
# `id_suffix` matches the "::int8"/"::nf4"/"::fp4" markers
# `exp_proxy_distance_grid._parse_id` strips off before a bitsandbytes load;
# `quant` is the taxonomy's own (more specific) lossiness label.
_QUANTIZED = [
    quantized(_QWEN25_7B, "bnb-int8", "Qwen2.5-7B int8", id_suffix="int8"),
    quantized(_QWEN25_7B, "bnb-nf4", "Qwen2.5-7B NF4", id_suffix="nf4"),
    quantized(_QWEN25_7B, "bnb-fp4", "Qwen2.5-7B FP4", id_suffix="fp4"),
    replace(_QWEN25_7B, id="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
            label="Qwen2.5-7B GPTQ-i8", quant="gptq-int8"),
    replace(_QWEN25_7B, id="Qwen/Qwen2.5-7B-Instruct-AWQ",
            label="Qwen2.5-7B AWQ", quant="awq-int4"),
    replace(_QWEN25_7B, id="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            label="Qwen2.5-7B GPTQ-i4", quant="gptq-int4"),
    quantized(_LLAMA31_8B, "bnb-int8", "Llama-3.1-8B int8", id_suffix="int8"),
    quantized(_LLAMA31_8B, "bnb-nf4", "Llama-3.1-8B NF4", id_suffix="nf4"),
    quantized(_LLAMA31_8B, "bnb-fp4", "Llama-3.1-8B FP4", id_suffix="fp4"),
]

_ENTRIES = [
    _QWEN25_7B, _QWEN25_3B, _QWEN25_CODER_7B, _QWEN25_CODER_1_5B, _QWEN3_8B,
    _QWEN3_4B, _QWEN3_1_7B, _QWEN3_0_6B, _QWEN25_1_5B, _QWEN25_0_5B,
    _DS_DISTILL_QWEN_7B, _DS_DISTILL_QWEN_1_5B,
    _LLAMA31_8B, _LLAMA31_8B_BASE, _LLAMA30_8B, _LLAMA32_3B, _LLAMA32_1B,
    _DS_DISTILL_LLAMA_8B,
    _SMOLLM2_1_7B, _SMOLLM2_360M, _SMOLLM2_135M,
    _PYTHIA_410M, _PYTHIA_160M, _PYTHIA_1_4B,
    _GPT2_124M, _GPT2_355M, _GPT2_774M,
    *_QUANTIZED,
]

REGISTRY: dict[str, ModelIdentity] = {m.id: m for m in _ENTRIES}


def identity(hf_id: str) -> ModelIdentity:
    """Look up a model's taxonomy facts by HF id (or `base_id::quant_method`
    for a bitsandbytes load). Raises with a pointer to this file if the model
    hasn't been added yet -- there is no silent guess."""
    try:
        return REGISTRY[hf_id]
    except KeyError:
        raise KeyError(
            f"{hf_id!r} has no ModelIdentity in ivgym/model_registry.py -- "
            "add one (family/base/org/generation/domain/tokenizer/params) "
            "before using it in a distance ladder.") from None
