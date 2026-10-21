"""Partition coverage and full-shape native bridges for the second M3 pilot.

The native attention bridge follows the pinned Qwen3 eager implementation.
It reconstructs its observable arithmetic, not undocumented kernel internals.
"""
from __future__ import annotations

import numpy as np


def partition_coverage(mask, chunk, minimum=3):
    mask = np.asarray(mask)
    if (mask.ndim != 4 or mask.shape[-2] != 1 or mask.shape[-1] < 1
            or type(chunk) is not int or chunk < 1 or minimum < 1
            or np.isnan(mask).any()):
        raise ValueError("Invalid attention partition coverage inputs")
    allowed = mask >= -1e20
    active = np.stack([allowed[..., i:i + chunk].any(axis=-1)
                       for i in range(0, mask.shape[-1], chunk)], axis=-1).sum(axis=-1)
    if int(active.min()) < minimum:
        raise ValueError("Too few active attention partitions")
    return dict(active_partitions_min=int(active.min()), active_partitions_max=int(active.max()),
                allowed_keys_min=int(allowed.sum(axis=-1).min()),
                allowed_keys_max=int(allowed.sum(axis=-1).max()),
                total_partitions=(mask.shape[-1] + chunk - 1) // chunk, chunk=chunk)


def native_reference(family, operands, recipe):
    """Replay full-shape BF16 operators, with captured strides and expanded KV."""
    import torch

    if family == "gemm":
        return torch.nn.functional.linear(operands["full_x"], operands["weight"])
    if family != "attention":
        raise ValueError("Unsupported native bridge family")
    query, key, value = operands["full_query"], operands["key"], operands["value"]
    weights = torch.matmul(query, key.transpose(2, 3)) * recipe["attention_scale"]
    weights = weights + operands["full_mask"][..., :key.shape[-2]]
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=0.0, training=False)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


def sampling_changes(cells):
    primary = next(c for c in cells if (c["dtype"], c["convention"], c["mode"])
                   == ("float32", "legacy", "top_k_top_p"))
    return dict(primary_token_changed=primary["token_changed"],
                primary_support_delta=primary["support_delta"],
                any_token_changed=any(c["token_changed"] for c in cells),
                any_support_changed=any(c["support_delta"] > 0 for c in cells))


def summarize_diagnostics(rows):
    if not rows or any("diagnostics" not in r for r in rows):
        raise ValueError("Missing partition/bridge diagnostics")
    if any(not r["diagnostics"]["native_full_exact"]
           or not r["controls"].get("native_sham_exact")
           or not r["controls"].get("native_restore_exact") for r in rows):
        raise ValueError("Native bridge control failed")
    result = dict(fp32_local_changed=sum(r["diagnostics"]["fp32_difference"]["changed"] > 0 for r in rows),
                  selected_logits_changed=sum(r["diagnostics"]["selected_logits_changed"] for r in rows),
                  any_sampling_token_changed=sum(sampling_changes(r["sampling"])["any_token_changed"] for r in rows),
                  any_sampling_support_changed=sum(sampling_changes(r["sampling"])["any_support_changed"] for r in rows),
                  native_full_exact_cases=len(rows), native_sham_exact_cases=len(rows), native_restore_exact_cases=len(rows))
    for side in ("a", "b"):
        changes = [sampling_changes(r["diagnostics"]["native_sampling"][side]) for r in rows]
        result[f"native_to_{side}"] = dict(
            local_changed=sum(r["native_bridge"][side]["changed"] > 0 for r in rows),
            primary_token_changed=sum(c["primary_token_changed"] for c in changes),
            primary_support_changed=sum(c["primary_support_delta"] > 0 for c in changes),
            any_token_changed=sum(c["any_token_changed"] for c in changes),
            any_support_changed=sum(c["any_support_changed"] for c in changes))
    coverage = [r["diagnostics"]["coverage"] for r in rows if r["case"]["family"] == "attention"]
    if coverage:
        result["active_partitions_min"] = min(c["active_partitions_min"] for c in coverage)
        result["active_partitions_max"] = max(c["active_partitions_max"] for c in coverage)
    return result
