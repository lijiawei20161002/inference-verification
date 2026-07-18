# Unified verification algorithm - robustness sweep

- config: 16 prompts x 96 tokens, batch=400, n_batches=200, value_fn=entropy, selective=[0.25]
- models attempted: 9, succeeded: 9

## Models

| tag | reference | params | proxy | proxy params | vocab |
|---|---|---|---|---|---|
| qwen3-0.6b | Qwen/Qwen3-0.6B | 0.60B | (noised-M) | - | 151936 |
| qwen3-1.7b | Qwen/Qwen3-1.7B | 1.72B | Qwen/Qwen3-0.6B | 0.60B | 151936 |
| qwen3-4b | Qwen/Qwen3-4B | 4.02B | Qwen/Qwen3-0.6B | 0.60B | 151936 |
| llama3.2-1b | unsloth/Llama-3.2-1B-Instruct | 1.24B | (noised-M) | - | 128256 |
| llama3.2-3b | unsloth/Llama-3.2-3B-Instruct | 3.21B | unsloth/Llama-3.2-1B-Instruct | 1.24B | 128256 |
| smollm2-360m | HuggingFaceTB/SmolLM2-360M-Instruct | 0.36B | HuggingFaceTB/SmolLM2-135M-Instruct | 0.13B | 49152 |
| smollm2-1.7b | HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.71B | HuggingFaceTB/SmolLM2-135M-Instruct | 0.13B | 49152 |
| pythia-410m | EleutherAI/pythia-410m | 0.41B | EleutherAI/pythia-160m | 0.16B | 50304 |
| pythia-1.4b | EleutherAI/pythia-1.4b | 1.41B | EleutherAI/pythia-160m | 0.16B | 50304 |

## Per-verifier AUC across ALL (model x attack) full-recompute cells

| verifier | mean AUC | min AUC | worst (model/attack) | cells |
|---|---|---|---|---|
| token_difr | 0.803 | 0.040 | smollm2-1.7b/kv_fp8 | 63 |
| cross_entropy | 0.546 | 0.000 | llama3.2-1b/adv_quant_temp | 63 |
| activation_difr | 0.738 | 0.000 | pythia-1.4b/seed_43 | 63 |
| token_toploc | 0.670 | 0.013 | pythia-410m/seed_43 | 63 |
| surface_stat | 0.477 | 0.005 | smollm2-360m/kv_fp8 | 63 |
| surface_rank | 0.587 | 0.001 | llama3.2-1b/adv_quant_temp | 63 |
| surface_tokens | 0.439 | 0.000 | llama3.2-3b/quant_4bit | 63 |
| accept_rate | 0.519 | 0.001 | qwen3-1.7b/temp_1.1 | 63 |

## Headline check: does `token_difr` catch every attack on every model?

| model | quant_4bit | kv_fp8 | temp_1.1 | seed_43 | bug_k2 | bug_k32 | adv_quant_temp | min |
|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b | 0.631 | 0.733 | 0.886 | 1.000 | 0.123 | 0.999 | 0.899 | **0.123** |
| qwen3-1.7b | 0.708 | 0.947 | 0.901 | 1.000 | 0.697 | 0.985 | 0.928 | **0.697** |
| qwen3-4b | 0.989 | 0.871 | 0.862 | 1.000 | 0.780 | 1.000 | 0.940 | **0.780** |
| llama3.2-1b | 0.999 | 0.362 | 0.902 | 1.000 | 0.686 | 0.973 | 1.000 | **0.362** |
| llama3.2-3b | 0.976 | 0.936 | 0.683 | 1.000 | 0.964 | 1.000 | 0.999 | **0.683** |
| smollm2-360m | 1.000 | 0.984 | 0.987 | 1.000 | 0.991 | 1.000 | 1.000 | **0.984** |
| smollm2-1.7b | 0.973 | 0.040 | 0.603 | 1.000 | 0.491 | 0.984 | 0.615 | **0.040** |
| pythia-410m | 0.146 | 0.086 | 0.859 | 1.000 | 0.264 | 0.882 | 0.421 | **0.086** |
| pythia-1.4b | 0.815 | 0.793 | 0.746 | 1.000 | 0.226 | 0.883 | 0.425 | **0.226** |

## Attack difficulty (mean `token_difr` AUC across models; lower = harder)

| attack | mean token_difr AUC | min | max |
|---|---|---|---|
| bug_k2 | 0.580 | 0.123 | 0.991 |
| kv_fp8 | 0.639 | 0.040 | 0.984 |
| adv_quant_temp | 0.803 | 0.421 | 1.000 |
| quant_4bit | 0.804 | 0.146 | 1.000 |
| temp_1.1 | 0.825 | 0.603 | 0.987 |
| bug_k32 | 0.967 | 0.882 | 1.000 |
| seed_43 | 1.000 | 1.000 | 1.000 |

## Information-directed selective recompute (token_difr, mean over models x attacks)

| budget | realized ratio | mean AUC | vs full |
|---|---|---|---|
| 1.0 (full) | 1.000 | 0.803 | - |
| 0.25 | 0.250 | 0.787 | -0.016 |
