# M3 controlled reduction-order pilot

Finished: 2026-10-21T08:58:18.546213+00:00

- Cases per family: 32; independent repeats: 3.
- Native outputs and KV states matched the parent M1 reference path.
- Capture, both shams, forward/reverse swaps and ancestor checks passed.
- Fresh saved-operand replay reproduced all recorded outputs and controls.

- gemm: {"cases": 32, "strata": {"historical": 16, "control": 16}, "local_changed": 7, "logits_changed": 2, "primary_support_changed": 0, "primary_token_changed": 0, "any_sampling_cell_changed": 0, "diagnostics": {"fp32_local_changed": 32, "selected_logits_changed": 2, "any_sampling_token_changed": 0, "any_sampling_support_changed": 2, "native_full_exact_cases": 32, "native_sham_exact_cases": 32, "native_restore_exact_cases": 32, "native_to_a": {"local_changed": 32, "primary_token_changed": 6, "primary_support_changed": 21, "any_token_changed": 9, "any_support_changed": 27}, "native_to_b": {"local_changed": 32, "primary_token_changed": 6, "primary_support_changed": 21, "any_token_changed": 9, "any_support_changed": 28}}}
- attention: {"cases": 32, "strata": {"historical": 16, "control": 16}, "local_changed": 2, "logits_changed": 1, "primary_support_changed": 0, "primary_token_changed": 0, "any_sampling_cell_changed": 1, "diagnostics": {"fp32_local_changed": 32, "selected_logits_changed": 1, "any_sampling_token_changed": 1, "any_sampling_support_changed": 1, "native_full_exact_cases": 32, "native_sham_exact_cases": 32, "native_restore_exact_cases": 32, "native_to_a": {"local_changed": 32, "primary_token_changed": 8, "primary_support_changed": 20, "any_token_changed": 12, "any_support_changed": 28}, "native_to_b": {"local_changed": 32, "primary_token_changed": 8, "primary_support_changed": 20, "any_token_changed": 13, "any_support_changed": 29}, "active_partitions_min": 3, "active_partitions_max": 19}}

Partition/bridge follow-up: attention chunks of 8 with at least 3 active partitions per case.
FP32 partials and pre-cast outputs are retained; same-input chunk-32 controls are saved.
Full-shape native operator outputs, native sham and restore controls matched exactly.
Native-to-recipe sampling comparisons are distinct from the A/B reduction-order comparisons.
The legacy any_sampling_cell_changed field counts token changes only; diagnostic support counts are explicit.

These are explicit diagnostic FP32 reduction recipes on historical inputs.
They are not evidence of native cuBLAS split-K or a native attention merge policy.
No historical mechanism attribution, deployment rate, fault or attack is established.
Every model replay started from the full token checkpoint with empty KV and recomputed all descendants.
The full M3 mechanism gate remains pending; zero propagated effects are valid negative results.
