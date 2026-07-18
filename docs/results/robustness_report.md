# Unified verification algorithm — robustness across families, sizes, attacks

Source: `/root/inference-verification/docs/results/robustness_sweep.json`  •  9 models succeeded.

## Model matrix

| model | ref params | proxy | proxy params | vocab |
|---|---|---|---|---|
| `qwen3-0.6b` | 0.60B | (noised-M) | — | 151936 |
| `qwen3-1.7b` | 1.72B | Qwen/Qwen3-0.6B | 0.60B | 151936 |
| `qwen3-4b` | 4.02B | Qwen/Qwen3-0.6B | 0.60B | 151936 |
| `llama3.2-1b` | 1.24B | (noised-M) | — | 128256 |
| `llama3.2-3b` | 3.21B | unsloth/Llama-3.2-1B-Instruct | 1.24B | 128256 |
| `smollm2-360m` | 0.36B | HuggingFaceTB/SmolLM2-135M-Instruct | 0.13B | 49152 |
| `smollm2-1.7b` | 1.71B | HuggingFaceTB/SmolLM2-135M-Instruct | 0.13B | 49152 |
| `pythia-410m` | 0.41B | EleutherAI/pythia-160m | 0.16B | 50304 |
| `pythia-1.4b` | 1.41B | EleutherAI/pythia-160m | 0.16B | 50304 |

## Headline: `token_difr` full-recompute AUC (every model × attack)

| model | quant_4bit | kv_fp8 | temp_1.1 | seed_43 | bug_k2 | bug_k32 | adv_quant_temp | **min** |
|---|---|---|---|---|---|---|---|---|
| `qwen3-0.6b` | 0.631 | 0.733 | 0.886 | 1.000 | 0.123 | 0.999 | 0.899 | **0.123** |
| `qwen3-1.7b` | 0.708 | 0.947 | 0.901 | 1.000 | 0.697 | 0.985 | 0.928 | **0.697** |
| `qwen3-4b` | 0.989 | 0.871 | 0.862 | 1.000 | 0.780 | 1.000 | 0.940 | **0.780** |
| `llama3.2-1b` | 0.999 | 0.362 | 0.902 | 1.000 | 0.686 | 0.973 | 1.000 | **0.362** |
| `llama3.2-3b` | 0.976 | 0.936 | 0.683 | 1.000 | 0.964 | 1.000 | 0.999 | **0.683** |
| `smollm2-360m` | 1.000 | 0.984 | 0.987 | 1.000 | 0.991 | 1.000 | 1.000 | **0.984** |
| `smollm2-1.7b` | 0.973 | 0.040 | 0.603 | 1.000 | 0.491 | 0.984 | 0.615 | **0.040** |
| `pythia-410m` | 0.146 | 0.086 | 0.859 | 1.000 | 0.264 | 0.882 | 0.421 | **0.086** |
| `pythia-1.4b` | 0.815 | 0.793 | 0.746 | 1.000 | 0.226 | 0.883 | 0.425 | **0.226** |

## Per-verifier detectability across ALL (model × attack) cells

Tier-1 = raw AUC; Tier-0 = max(AUC, 1−AUC).

| verifier | tier | mean | median | min | worst cell |
|---|---|---|---|---|---|
| `token_difr` | 1 | 0.803 | 0.928 | 0.040 | smollm2-1.7b/kv_fp8 |
| `cross_entropy` | 1 | 0.546 | 0.668 | 0.000 | llama3.2-1b/adv_quant_temp |
| `activation_difr` | 1 | 0.738 | 0.884 | 0.000 | pythia-1.4b/seed_43 |
| `token_toploc` | 1 | 0.670 | 0.748 | 0.013 | pythia-410m/seed_43 |
| `surface_stat` | 0 | 0.831 | 0.848 | 0.510 | smollm2-1.7b/bug_k2 |
| `surface_rank` | 0 | 0.791 | 0.815 | 0.527 | qwen3-1.7b/bug_k2 |
| `surface_tokens` | 0 | 0.755 | 0.762 | 0.500 | qwen3-0.6b/kv_fp8 |
| `accept_rate` | 0 | 0.815 | 0.860 | 0.507 | llama3.2-1b/seed_43 |

## Best detector per attack (mean detectability over models)

| attack | best verifier | mean detect. | 2nd best |
|---|---|---|---|
| `quant_4bit` | `activation_difr` | 1.000 | `accept_rate` (0.832) |
| `kv_fp8` | `activation_difr` | 0.997 | `surface_stat` (0.781) |
| `temp_1.1` | `surface_stat` | 0.882 | `accept_rate` (0.878) |
| `seed_43` | `token_difr` | 1.000 | `surface_stat` (0.878) |
| `bug_k2` | `accept_rate` | 0.821 | `surface_rank` (0.780) |
| `bug_k32` | `token_difr` | 0.967 | `token_toploc` (0.904) |
| `adv_quant_temp` | `activation_difr` | 1.000 | `surface_stat` (0.853) |

## Attack difficulty — mean `token_difr` AUC across models (lower = harder)

| attack | mean | min | max |
|---|---|---|---|
| `bug_k2` | 0.580 | 0.123 | 0.991 |
| `kv_fp8` | 0.639 | 0.040 | 0.984 |
| `adv_quant_temp` | 0.803 | 0.421 | 1.000 |
| `quant_4bit` | 0.804 | 0.146 | 1.000 |
| `temp_1.1` | 0.825 | 0.603 | 0.987 |
| `bug_k32` | 0.967 | 0.882 | 1.000 |
| `seed_43` | 1.000 | 1.000 | 1.000 |

## Size trend — mean `token_difr` AUC over attacks, by model (does size help?)

| model | params | mean token_difr | mean best-detector |
|---|---|---|---|
| `llama3.2-1b` | 1.24B | 0.846 | 0.984 |
| `llama3.2-3b` | 3.21B | 0.937 | 1.000 |
| `pythia-410m` | 0.41B | 0.523 | 0.973 |
| `pythia-1.4b` | 1.41B | 0.698 | 0.960 |
| `qwen3-0.6b` | 0.60B | 0.753 | 0.998 |
| `qwen3-1.7b` | 1.72B | 0.881 | 0.998 |
| `qwen3-4b` | 4.02B | 0.920 | 0.975 |
| `smollm2-360m` | 0.36B | 0.994 | 1.000 |
| `smollm2-1.7b` | 1.71B | 0.672 | 0.967 |

## Information-directed selective recompute vs full (`token_difr`)

Mean `token_difr` AUC over models, on the hard attacks ['quant_4bit', 'kv_fp8', 'bug_k2', 'adv_quant_temp']:

| tier | recompute | mean AUC |
|---|---|---|
| full | 100% | 0.707 |
| selective | 25% | 0.737 |
