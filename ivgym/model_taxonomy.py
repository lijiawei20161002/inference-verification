"""A principled model-relationship taxonomy, shared by every "distance ladder"
experiment (`exp_proxy_distance_grid`, `exp_cross_family_accept`,
`exp_family_correlation`, `exp_robustness_gpu`).

Those experiments all ask the same question in different clothes: *how
different is model B from reference model A?* Before this module, each file
answered it by hand -- a free-text `group` string ("same company, next gen",
"cross domain", ...) plus a manually-picked integer `dist` -- re-derived
independently per file, with no guarantee the same phrase meant the same thing
twice.

Here the only manual step is stating FACTS about a model (its family, its
size, which pretrained checkpoint it was fine-tuned from, who trained it, ...).
The relationship between any two models -- and an ordinal distance between
them -- is then DERIVED, not typed out per row:

    same node class      -- `family`      (architecture + tokenizer lineage)
    same size             -- `size_class`  (params within `SIZE_CLASS_RATIO` of each other)
    fine-tuned on each other -- `base`     (the pretrained checkpoint both derive from)
    same company           -- `org`       (the lab/company that trained it)
    same generation         -- `generation` (version within the family lineage)
    same post-training domain -- `domain`  (general-instruct / code / reasoning-distill / base)
    same tokenizer           -- `tokenizer` (shared vocab -> token-aligned metrics are valid)

`distance()` orders two models by comparing these axes in a fixed priority
--`AXIS_PRIORITY`, in the order a human asking "how different is this model?"
would ask them, per the ordering given when this module was requested: node
class, then size, then shared-lineage fine-tuning, then company, with the
remaining axes as tie-breakers. There is no single "correct" priority over
independent axes -- that is exactly the subjectivity this module makes
explicit and inspectable instead of burying it in 12 hand-picked integers.
Re-ordering `AXIS_PRIORITY` is the one place to change the ladder's intuition;
nothing else needs to change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Priority order the axes are compared in (highest priority first). Two models
# are "farther apart" starting at the first axis, in this order, where they
# disagree -- e.g. two models that share family, size and base but differ in
# org are closer than two that already differ in family.
AXIS_PRIORITY = ("family", "size_class", "base", "org", "generation", "domain", "tokenizer")

# How much size can differ and still count as "the same size class".
SIZE_CLASS_RATIO = 1.5

# Lossiness sub-order for the "same weights, quantized" case (0 = least lossy).
# All of these are the SAME checkpoint compressed differently, so they occupy
# their own sub-ladder strictly between "identical" (0) and any real model
# substitution (>= 100, see `distance`).
QUANT_RANK = {
    "bnb-int8": 0,
    "gptq-int8": 1,
    "awq-int4": 2,
    "gptq-int4": 3,
    "bnb-nf4": 4,
    "bnb-fp4": 5,
}


@dataclass(frozen=True)
class ModelIdentity:
    """Facts about ONE model. The relationship to another model is derived by
    `relationship()` / `distance()` below -- nothing here is pairwise."""

    id: str          # HF repo id (this model's own identity for loading/display)
    label: str       # short display label
    org: str         # training organization/lab, e.g. "Qwen", "Meta", "DeepSeek"
    family: str      # architecture + tokenizer lineage ("node class"), e.g. "qwen", "llama"
    base: str        # the pretrained checkpoint this model's weights derive from
                      # (a fine-tune/distill points at its parent pretrain's `base`,
                      # which is exactly "fine-tuned on each other" between two models)
    generation: str  # version within the family lineage, e.g. "2.5", "3", "3.2"
    domain: str      # post-training objective: "general" / "code" / "reasoning-distill" / "base"
    tokenizer: str   # vocab identity key; equal <=> token-aligned metrics are valid
    params: float    # parameter count
    quant: str | None = None  # quantization method, if this is a lossy copy of `base`


def quantized(m: ModelIdentity, quant: str, label: str, id_suffix: str | None = None) -> ModelIdentity:
    """A lossy-compressed copy of `m`: same base/family/org/etc, only `quant` differs."""
    return replace(m, id=f"{m.id}::{id_suffix or quant}", label=label, quant=quant)


@dataclass(frozen=True)
class Relationship:
    same_family: bool
    same_size_class: bool
    same_base: bool
    same_org: bool
    same_generation: bool
    same_domain: bool
    same_tokenizer: bool

    def mismatches(self) -> tuple[str, ...]:
        flags = {
            "family": not self.same_family, "size_class": not self.same_size_class,
            "base": not self.same_base, "org": not self.same_org,
            "generation": not self.same_generation, "domain": not self.same_domain,
            "tokenizer": not self.same_tokenizer,
        }
        return tuple(axis for axis in AXIS_PRIORITY if flags[axis])


def relationship(ref: ModelIdentity, other: ModelIdentity) -> Relationship:
    """The independent-axis comparison of `other` against reference `ref`."""
    return Relationship(
        same_family=other.family == ref.family,
        same_size_class=abs(math.log(other.params / ref.params)) <= math.log(SIZE_CLASS_RATIO),
        same_base=other.base == ref.base,
        same_org=other.org == ref.org,
        same_generation=other.generation == ref.generation,
        same_domain=other.domain == ref.domain,
        same_tokenizer=other.tokenizer == ref.tokenizer,
    )


def _is_quantized_self(ref: ModelIdentity, other: ModelIdentity) -> bool:
    """True iff `other` is `ref`'s own weights, just compressed -- every axis
    matches except `quant`."""
    if other.quant is None:
        return False
    rel = relationship(ref, other)
    return all((rel.same_family, rel.same_size_class, rel.same_base, rel.same_org,
                rel.same_generation, rel.same_domain, rel.same_tokenizer))


def distance(ref: ModelIdentity, other: ModelIdentity) -> int:
    """An ordinal distance from `ref`, DERIVED from the axes above:

        0            identical (same id)
        1..len(QUANT_RANK)   a lossy-compressed copy of `ref`'s own weights,
                             sub-ordered by lossiness (least lossy first)
        100 + code   a genuinely different model, `code` a lexicographic
                     encoding of which axes mismatch (in `AXIS_PRIORITY` order:
                     an earlier mismatch always outweighs any number of later
                     ones -- this is a plain bitmask over `AXIS_PRIORITY`, so
                     comparing `code` is exactly comparing the axis-mismatch
                     tuples lexicographically)

    Two models can tie (same code) if they mismatch on exactly the same axes;
    that is a feature, not a bug -- the taxonomy has no opinion about which of
    two such models is "closer" beyond what the stated axes say.
    """
    if other.id == ref.id:
        return 0
    if _is_quantized_self(ref, other):
        return 1 + QUANT_RANK[other.quant]
    rel = relationship(ref, other)
    n = len(AXIS_PRIORITY)
    code = sum(1 << (n - 1 - i) for i, axis in enumerate(AXIS_PRIORITY) if axis in rel.mismatches())
    return 100 + code


def describe(ref: ModelIdentity, other: ModelIdentity) -> str:
    """A short human-readable label for the relationship -- generated from
    whichever axes mismatch, in priority order, instead of typed by hand."""
    if other.id == ref.id:
        return "identical"
    if _is_quantized_self(ref, other):
        return f"quant: {other.quant}"
    rel = relationship(ref, other)
    mism = rel.mismatches()
    if not mism:
        return "same everything (distinct checkpoint)"
    first = mism[0]
    if first == "family":
        return "different family" if not rel.same_tokenizer else "different family, shared tokenizer"
    if first == "size_class":
        bigger = other.params > ref.params
        return f"same family, {'larger' if bigger else 'smaller'}"
    if first == "base":
        return "same family+size, different pretrain"
    if first == "org":
        return "same base, different org" if not rel.same_domain else "same base+domain, different org"
    if first == "generation":
        return "same company, next gen" if rel.same_org else "cross-org, different generation"
    if first == "domain":
        return "same fam+size, diff domain" if rel.same_size_class else "same family, diff domain+size"
    return "same lineage, different tokenizer"  # first == "tokenizer": everything else matched


def sorted_by_distance(ref: ModelIdentity, others: list[ModelIdentity]) -> list[tuple[int, str, ModelIdentity]]:
    """`(distance, description, model)` triples, ascending distance from `ref`."""
    rows = [(distance(ref, m), describe(ref, m), m) for m in others]
    return sorted(rows, key=lambda r: r[0])
