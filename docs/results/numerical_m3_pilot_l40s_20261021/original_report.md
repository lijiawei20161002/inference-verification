# M3 controlled reduction-order pilot

Finished: 2026-10-21T02:24:58.525669+00:00

- Cases per family: 32; independent repeats: 3.
- Native outputs and KV states matched the parent M1 reference path.
- Capture, both shams, forward/reverse swaps and ancestor checks passed.
- Fresh saved-operand replay reproduced all recorded outputs and controls.

- gemm: {"cases": 32, "strata": {"historical": 16, "control": 16}, "local_changed": 7, "logits_changed": 2, "primary_support_changed": 0, "primary_token_changed": 0, "any_sampling_cell_changed": 0}
- attention: {"cases": 32, "strata": {"historical": 16, "control": 16}, "local_changed": 0, "logits_changed": 0, "primary_support_changed": 0, "primary_token_changed": 0, "any_sampling_cell_changed": 0}

These are explicit diagnostic FP32 reduction recipes on historical inputs.
They are not evidence of native cuBLAS split-K or a native attention merge policy.
No historical mechanism attribution, deployment rate, fault or attack is established.
Every model replay started from the full token checkpoint with empty KV and recomputed all descendants.
The full M3 mechanism gate remains pending; zero propagated effects are valid negative results.
