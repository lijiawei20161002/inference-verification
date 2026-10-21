# M1 event-shape dispatch validation

Finished: 2026-10-21T00:11:42.368238+00:00

- Sequences: 64; positions: 8192.
- Historical events reproduced: 49/49.
- Selected provider forwards: 113; full reference forwards: 64.
- Two independent profiled baseline/capture pairs matched kernel names, grid and block.
- Full trajectories matched the unprofiled controls and parent E0 outputs.
- Captured raw event arrays and KV states matched with and without profiling.
- M1 gate: passed_for_historical_evaluation_corpus.
- M3: not started.

This is historical discovery data on the pinned single-GPU HF execution path.
Profiling covers complete selected forwards, after regenerating the original prefix/KV state.
It does not prove equality on untraced calls or identify a reduction mechanism, fault or attack.
