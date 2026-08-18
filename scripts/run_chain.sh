#!/usr/bin/env bash
# Wait for the running cost-of-a-verdict job, then run the benign-shape floor at
# the same pool size. Serialized on purpose: cost_of_a_verdict measures GPU
# seconds per token, so nothing else may touch the card while it is timing.
set -u
cd /root/inference-verification
L=docs/results/logs
mkdir -p "$L"

while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[chain] cost_of_a_verdict pid $1 exited $(date -Is)"

IVGYM_PROMPTS=80 IVGYM_TOKENS=256 IVGYM_BOOT=400 \
    python -m experiments.exp_benign_shape_dprime_gpu \
    > "$L/benign_shape_dprime_n80_t256.log" 2>&1
echo "[chain] benign_shape_dprime exited $? $(date -Is)"
