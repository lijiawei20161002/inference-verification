"""Native execution-shape interventions on identical effective BF16 operands.

The attention contrast changes query extent/layout and removes a fully masked
KV suffix. A query-only control separates those local contrasts. This tests
observable native arithmetic; it does not identify a kernel's reduction tree.
"""
from __future__ import annotations

import numpy as np

from experiments.numerical.mechanism_diagnostics import native_reference
from experiments.numerical.mechanism_replay import (
    OperationHook, operand_hashes, tensor_layouts,
)


def shape_operands(family, operands, provider_layouts, case, *, query_only=False):
    """Project reference values into the observed provider's call geometry."""
    import torch

    end = case["token_index"] + 1
    start = 0 if case["position"] == 0 else end - 1
    if family == "gemm":
        values = dict(full_x=operands["full_x"][:, start:end], weight=operands["weight"])
    elif family == "attention":
        mask = operands["full_mask"][:, :, start:end]
        if end > operands["key"].shape[-2] or torch.any(mask[..., end:] >= -1e20):
            raise ValueError("Cannot remove an unmasked KV suffix")
        stop = operands["key"].shape[-2] if query_only else end
        values = dict(full_query=operands["full_query"][:, :, start:end],
                      key=operands["key"][:, :, :stop],
                      value=operands["value"][:, :, :stop], full_mask=mask[..., :stop])
    else:
        raise ValueError("Unknown native operator family")
    result = {}
    for name, value in values.items():
        # The observed provider layout is part of the declared intervention.
        if query_only and name in ("key", "value", "full_mask"):
            result[name] = value.contiguous()
            continue
        layout = provider_layouts[name]
        if list(value.shape) != layout["shape"] or str(value.dtype) != layout["dtype"]:
            raise ValueError(f"Provider geometry does not align with the reference: {name}")
        copy = torch.empty_strided(value.shape, layout["stride"], dtype=value.dtype, device=value.device)
        copy.copy_(value)
        result[name] = copy
    return result


def effective_names(family):
    return ("full_x", "weight") if family == "gemm" else ("full_query", "key", "value", "full_mask")


def shape_output(family, operands, recipe, provider_layouts, case, *, query_only=False):
    alternate = shape_operands(family, operands, provider_layouts, case, query_only=query_only)
    return native_reference(family, alternate, recipe)[:, -1:]


class NativeShapeHook(OperationHook):
    def __init__(self, *args, provider_layouts=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_layouts = provider_layouts

    def process(self, operands, output):
        import torch

        self.calls += 1
        if self.calls != 1:
            raise ValueError("Target operation called more than once")
        if self.expected is not None and operand_hashes(operands) != self.expected:
            raise ValueError("Native intervention operands drifted")
        if self.expected_layouts is not None and tensor_layouts(operands) != self.expected_layouts:
            raise ValueError("Native intervention layouts drifted")
        if self.operands is None:
            self.operands = operands
        self.native = output.detach().clone()
        if self.mode == "capture":
            return output
        initial = "a" if self.mode in ("a", "sham_a", "forward") else "b"
        if initial == "a":
            full = native_reference(self.case["family"], operands, self.recipe)
            idx = self.case["token_index"]
            result = full[:, idx:idx + 1]
        else:
            result = shape_output(self.case["family"], operands, self.recipe,
                                  self.provider_layouts, self.case)
        if not torch.equal(result, self.replacements[initial]):
            raise ValueError("Standalone native output differs inside model replay")
        target = "b" if self.mode == "forward" else "a" if self.mode == "reverse" else initial
        if self.mode in ("sham_a", "sham_b", "forward", "reverse"):
            result = self.replacements[target].clone()
        if result.dtype != output.dtype or result.shape != output.shape:
            raise ValueError("Replacement shape or dtype differs")
        return result


def summarize(rows):
    if not rows or len({r["case"]["case_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate native-shape records")
    result = {}
    for family in ("gemm", "attention"):
        selected = [r for r in rows if r["case"]["family"] == family]
        if not selected:
            raise ValueError("Missing native operator family")
        for row in selected:
            if not all(row["controls"].values()):
                raise ValueError("Unvalidated native-shape control")
        result[family] = dict(
            cases=len(selected),
            historical_cases=sum(r["case"]["stratum"] == "historical" for r in selected),
            local_changed=sum(r["local_difference"]["changed"] > 0 for r in selected),
            selected_logits_changed=sum(r["selected_logits_changed"] for r in selected),
            primary_token_changed=sum(r["primary_token_changed"] for r in selected),
            primary_support_changed=sum(r["primary_support_delta"] > 0 for r in selected),
            any_token_changed=sum(any(c["token_changed"] for c in r["sampling"]) for r in selected),
            any_support_changed=sum(any(c["support_delta"] > 0 for c in r["sampling"]) for r in selected),
            effective_provider_operands_exact=sum(r["historical_bridge"]["effective_operands_exact"] for r in selected),
            local_provider_output_exact=sum(r["historical_bridge"]["alternate_provider_output_exact"] for r in selected),
            selected_provider_logits_exact=sum(r["historical_bridge"]["alternate_provider_logits_exact"] for r in selected),
            primary_disagreement_removed=sum(r["historical_bridge"]["primary_disagreement_removed"] for r in selected),
            primary_disagreement_introduced=sum(r["historical_bridge"]["primary_disagreement_introduced"] for r in selected),
        )
    return result
