"""Explicit matched-input reduction recipes; torch is imported only on use.

These are diagnostic implementations, not claims about cuBLAS split-K or the
native HF attention kernel. A/B differ only in the order of the same FP32
partial sums; both cast once to the captured operand dtype at the end.
"""
from __future__ import annotations

import numpy as np

from experiments.numerical.frozen_logits import MODES, decision


def ordered_sum(parts, order):
    if not parts or order not in ("ascending", "descending"):
        raise ValueError("Need nonempty partials and a declared reduction order")
    values = parts if order == "ascending" else list(reversed(parts))
    result = values[0].clone()
    for value in values[1:]:
        result = result + value
    return result


def gemm(x, weight, *, chunk, order):
    """Bias-free linear map with fixed K partitions and FP32 partial GEMMs."""
    import torch

    if chunk < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("Invalid GEMM operands or partition")
    xf, wf = x.float(), weight.float()
    parts = [torch.matmul(xf[..., i:i + chunk], wf[:, i:i + chunk].T)
             for i in range(0, weight.shape[1], chunk)]
    return ordered_sum(parts, order).to(x.dtype)


def attention(query, key, value, mask, *, scale, chunk, order):
    """Fixed split-KV softmax partials, rescaled by a common maximum.

Q/K/V already include RoPE, GQA expansion, and the exact captured mask. Entirely
masked partitions contribute zero, never a fabricated uniform distribution.
Only the final numerator/denominator summation order differs between A and B.
"""
    import torch

    if (chunk < 1 or query.ndim != 4 or key.shape != value.shape
            or query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]
            or mask.shape[-2:] != (query.shape[-2], key.shape[-2])):
        raise ValueError("Invalid attention operands, mask or partition")
    maxima, numerators, denominators = [], [], []
    q, k, v = query.float(), key.float(), value.float()
    for i in range(0, key.shape[-2], chunk):
        scores = torch.matmul(q, k[..., i:i + chunk, :].transpose(-1, -2)) * scale
        block_mask = mask[..., i:i + chunk].float()
        scores = (scores + block_mask).masked_fill(block_mask < -1e20, -torch.inf)
        maximum = scores.amax(dim=-1, keepdim=True)
        safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        weights = torch.exp(scores - safe_maximum)
        maxima.append(maximum)
        denominators.append(weights.sum(dim=-1, keepdim=True))
        numerators.append(torch.matmul(weights, v[..., i:i + chunk, :]))
    maximum = torch.stack(maxima).amax(dim=0)
    if not torch.isfinite(maximum).all():
        raise ValueError("Attention query has no allowed keys")
    factors = [torch.exp(m - maximum) for m in maxima]
    numerator = ordered_sum([n * f for n, f in zip(numerators, factors)], order)
    denominator = ordered_sum([d * f for d, f in zip(denominators, factors)], order)
    return (numerator / denominator).to(query.dtype).transpose(1, 2).contiguous()


def calculate(family, operands, recipe, order):
    if family == "gemm":
        return gemm(operands["x"], operands["weight"], chunk=recipe["gemm_chunk"], order=order)
    if family == "attention":
        return attention(operands["query"], operands["key"], operands["value"], operands["mask"],
                         scale=recipe["attention_scale"], chunk=recipe["attention_chunk"], order=order)
    raise ValueError("Unsupported mechanism family")


def select_cases(events, count, layers=(0, 14, 27)):
    """Freeze balanced historical/control selection before any intervention."""
    if type(count) is not int or count < 2 or count > 32 or count % 2:
        raise ValueError("Use an even number of 2..32 cases per family")
    chosen = []
    for stratum in ("historical", "control"):
        pool = sorted((e for e in events if (e["historical_event_id"] is not None
                       if stratum == "historical" else e["control"])),
                      key=lambda e: (e["position"], e["prompt_id"]))
        n = count // 2
        if len(pool) < n:
            raise ValueError("Insufficient discovery cases")
        chosen += [(stratum, pool[(2 * i + 1) * len(pool) // (2 * n)]) for i in range(n)]
    cases = []
    for family in ("gemm", "attention"):
        for i, (stratum, event) in enumerate(chosen):
            layer = layers[i % len(layers)]
            pid, pos = event["prompt_id"], event["position"]
            cases.append(dict(case_id=f"{family}_p{pid}_t{pos}_l{layer}", family=family,
                              stratum=stratum, layer=layer, prompt_id=pid, position=pos,
                              evidence=event["evidence"], evidence_sha256=event["evidence_sha256"]))
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case selection")
    return cases


def validate_controls(states, layer):
    pairs = [("native", "capture"), ("a", "sham_a"), ("b", "sham_b"),
             ("b", "forward"), ("a", "reverse")]
    for left, right in pairs:
        if states[left] != states[right]:
            raise ValueError(f"Intervention control failed: {left} vs {right}")
    native = states["native"]
    for name, state in states.items():
        if state["prefix_logits"] != native["prefix_logits"]:
            raise ValueError(f"Intervention changed preceding token outputs: {name}")
        if state["kv"][:layer + 1] != native["kv"][:layer + 1]:
            raise ValueError(f"Intervention changed ancestor KV state: {name}")
    return dict(capture_exact=True, sham_a_exact=True, sham_b_exact=True,
                forward_exact=True, reverse_exact=True, ancestors_exact=True)


def sampling_matrix(a, b, noise):
    cells = []
    for dtype in ("float32", "float64"):
        for convention in ("legacy", "crossing_inclusive_v1"):
            for mode, (k, p, greedy) in MODES.items():
                ad, am = decision(a.astype(dtype), noise, k, p, convention, greedy)
                bd, bm = decision(b.astype(dtype), noise, k, p, convention, greedy)
                cells.append(dict(dtype=dtype, convention=convention, mode=mode,
                                  a=ad, b=bd, token_changed=ad["selected_token"] != bd["selected_token"],
                                  support_delta=int(np.count_nonzero(am["support_mask"] != bm["support_mask"]))))
    return cells
