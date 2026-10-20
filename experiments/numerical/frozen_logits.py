"""CPU diagnostics for exact BF16 logits and the actual captured Gumbel draws.

``original`` means the archived sampler arithmetic (provider float32, reference
float64), not filtering directly in BF16. The native BF16 bits remain preserved.
The inclusive nucleus variant is experimental; production sampling is untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ivgym.sampling import NEG_INF, apply_top_k, filtered_logits


def array_hash(x):
    x = np.asarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(json.dumps(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def bf16_bits(x):
    """Recover BF16 bits from an exact finite float32 widening, never round."""
    x = np.asarray(x)
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Expected finite float32 widened BF16 values")
    u = x.view(np.uint32)
    if np.any(u & 0xFFFF):
        raise ValueError("Values are not an exact BF16 widening")
    return (u >> 16).astype(np.uint16)


def widen_bf16(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.uint16:
        raise ValueError("BF16 storage must be uint16")
    result = (bits.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite captured logits")
    return result


def filter_detail(logits, k, p, convention="legacy"):
    x = np.asarray(logits)
    if (x.ndim != 1 or not len(x) or not np.isfinite(x).all()
            or x.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        raise ValueError("Expected a finite, nonempty float32/float64 logit vector")
    if convention not in ("legacy", "crossing_inclusive_v1"):
        raise ValueError("Unknown nucleus convention")
    if k is not None and (type(k) is not int or k < 1):
        raise ValueError("Invalid top-k")
    if p is not None and not 0 < p <= 1:
        raise ValueError("Invalid top-p")
    topk = apply_top_k(x, k)
    candidates = np.flatnonzero(topk > NEG_INF / 2)
    order = candidates[np.argsort(topk[candidates])[::-1]]
    e = np.exp(topk[order] - topk[order].max())
    cum = np.cumsum(e / e.sum())
    if p is None or p >= 1:
        keep = np.ones(len(order), dtype=bool)
    else:
        keep = cum <= p
        if convention == "crossing_inclusive_v1":
            # Keep through the first strictly crossing token, including exact ties.
            crossing = np.flatnonzero(cum > p)
            if len(crossing):
                keep[crossing[0]] = True
        keep[0] = True
    result = np.full_like(x, NEG_INF)
    result[order[keep]] = x[order[keep]]
    if convention == "legacy" and not np.array_equal(result, filtered_logits(x, k, p)):
        raise RuntimeError("Diagnostic filtering differs from the production sampler")
    return result, {
        "top_k_mask": topk > NEG_INF / 2,
        "support_mask": result > NEG_INF / 2,
        "sorted_ids": order,
        "sorted_logits": topk[order],
        "cumulative_probability": cum,
    }


def decision(logits, noise, k=50, p=0.95, convention="legacy", greedy=False,
             claimed=None, temperature=1.0):
    noise = np.asarray(noise)
    if noise.shape != np.shape(logits) or not np.isfinite(noise).all():
        raise ValueError("Noise must be finite and match logits")
    filt, detail = filter_detail(logits, k, p, convention)
    z = filt + (0.0 if greedy else temperature) * noise
    chosen = int(np.argmax(z))
    token = chosen if claimed is None else int(claimed)
    if not 0 <= token < len(z):
        raise ValueError("Claimed token is out of range")
    excluded = bool(filt[token] <= -1e29)
    margin = float(z[chosen] - z[token])
    return dict(selected_token=chosen, claimed_token=token,
                claimed_in_support=not excluded,
                # A finite sentinel subtraction is not a physical numerical margin.
                uncapped_margin=None if excluded else margin,
                margin_status="excluded_from_support" if excluded else "finite",
                sentinel_gap=margin if excluded else None,
                capped_score=30.0 if excluded else min(margin, 30.0),
                support_size=int(detail["support_mask"].sum())), detail


MODES = {
    "greedy": (None, None, True),
    "gumbel": (None, None, False),
    "top_k": (50, None, False),
    "top_p": (None, 0.95, False),
    "top_k_top_p": (50, 0.95, False),
}


def replay_event(path):
    with np.load(path, allow_pickle=False) as z:
        required = {"provider_bf16", "reference_bf16", "provider_gumbel",
                    "reference_gumbel", "claimed_token",
                    "provider_top_k_mask", "provider_support_mask",
                    "reference_top_k_mask", "reference_support_mask"}
        if not required.issubset(z.files):
            raise ValueError("Incomplete event evidence")
        pv, rf = widen_bf16(z["provider_bf16"]), widen_bf16(z["reference_bf16"])
        g, rg = z["provider_gumbel"], z["reference_gumbel"]
        if not np.array_equal(g, rg):
            raise ValueError("Provider/reference random tensors differ")
        claimed = int(z["claimed_token"])
        for name, logits in (("provider", pv), ("reference", rf.astype(np.float64))):
            _, d = filter_detail(logits, 50, 0.95)
            for mask in ("top_k_mask", "support_mask"):
                if not np.array_equal(d[mask], z[f"{name}_{mask}"]):
                    raise ValueError(f"Actual captured {name} {mask} disagrees with replay")
        original, _ = decision(pv, g)
        if original["selected_token"] != claimed:
            raise ValueError("Captured provider decision does not replay")
    cells = []
    for dtype in ("original", "float32", "float64"):
        pl = pv.astype(np.float32 if dtype == "original" else dtype)
        rl = rf.astype(np.float64 if dtype == "original" else dtype)
        for convention in ("legacy", "crossing_inclusive_v1"):
            for mode, (k, p, greedy) in MODES.items():
                a, ad = decision(pl, g, k, p, convention, greedy)
                b, bd = decision(rl, g, k, p, convention, greedy,
                                 claimed=a["selected_token"])
                cells.append(dict(arithmetic=dtype, convention=convention, mode=mode,
                                  provider=a, reference=b,
                                  token_agreement=a["selected_token"] == b["selected_token"],
                                  support_symmetric_difference=int(np.count_nonzero(
                                      ad["support_mask"] != bd["support_mask"]))))
    same_input = []
    for name, values in (("provider", pv), ("reference", rf)):
        a, _ = decision(values.astype(np.float32), g)
        b, _ = decision(values.astype(np.float64), g)
        same_input.append(dict(source=name, float32_token=a["selected_token"],
                               float64_token=b["selected_token"],
                               mixed_dtype_disagreement=a["selected_token"] != b["selected_token"]))
    original_cell = next(c for c in cells if c["arithmetic"] == "original"
                         and c["convention"] == "legacy" and c["mode"] == "top_k_top_p")
    matched = [c for c in cells if c["arithmetic"] in ("float32", "float64")
               and c["convention"] == "legacy" and c["mode"] == "top_k_top_p"]
    if any(not c["token_agreement"] for c in matched):
        label = "residual_under_matched_protocol"
    elif (not original_cell["token_agreement"]
          and any(s["mixed_dtype_disagreement"] for s in same_input)):
        label = "protocol_explained"
    elif original_cell["token_agreement"]:
        label = "no_default_disagreement"
    else:
        label = "unresolved"
    return dict(evidence_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                label=label, same_input_dtype_controls=same_input, cells=cells)
