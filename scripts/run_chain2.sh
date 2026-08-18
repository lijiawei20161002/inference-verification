#!/usr/bin/env bash
# Stages 3-5 of the cost/accuracy chain. Waits for run_chain.sh (cost_of_a_verdict
# -> benign_shape_dprime) and then:
#
#   3. re-analyse the sequential test on the real score arrays
#   4. rebuild the poster from all three fresh artifacts
#   5. the deep-pool run, which is the only way to answer the question stage 3
#      cannot: at an 80x256 pool every interesting cell needs a batch larger than
#      the 10% ceiling allows, so 19 of 22 come back "unresolvable". 400 prompts
#      over the one headline attack puts quant_4bit's token_difr / token_toploc /
#      activation_difr cells inside the ceiling. ~3.5 h on one H100.
#
# Stages 3-4 are CPU-only; stage 5 is the GPU again, which is why it runs last.
set -u
cd /root/inference-verification
L=docs/results/logs
mkdir -p "$L"

while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "[chain2] run_chain.sh pid $1 exited $(date -Is)"

# one worker per reachable cell: the weak cells simulate ~1e9 token draws each and
# there is no reason for the strong ones to queue behind them. 192 cores, 1.4 TB.
IVGYM_WORKERS=24 python -m experiments.exp_sequential_verdict \
    > "$L/sequential_verdict.log" 2>&1
echo "[chain2] sequential_verdict exited $? $(date -Is)"

python paper/make_cost_poster.py > "$L/cost_poster.log" 2>&1
echo "[chain2] make_cost_poster exited $? $(date -Is)"

IVGYM_PROMPTS=400 IVGYM_TOKENS=256 IVGYM_BOOT=400 \
    IVGYM_ATTACKS=quant_4bit IVGYM_TAG=deep400 \
    python -m experiments.exp_cost_of_a_verdict_gpu \
    > "$L/cost_of_a_verdict_deep400.log" 2>&1
echo "[chain2] cost_of_a_verdict deep400 exited $? $(date -Is)"

# MAXMULT=2 because the truncation search never needed more than ~1.5x b, and the
# ceiling has to cover the longest stream a design may draw: at 6x, a 400-prompt
# pool would still resolve nothing.
IVGYM_SOURCE=cost_of_a_verdict_deep400 IVGYM_MAXMULT=2 IVGYM_WORKERS=24 \
    python -m experiments.exp_sequential_verdict \
    > "$L/sequential_verdict_deep400.log" 2>&1
echo "[chain2] sequential_verdict deep400 exited $? $(date -Is)"
