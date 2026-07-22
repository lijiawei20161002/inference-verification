"""Systematic performance-vs-cost sweep: run `exp_cost_curve_gpu` for every
(family, size) so the mega figure (`plot_mega_cost_curve`) can lay the SAME
curve out across five model families and their size ladders.

Five families, each a size ladder whose SMALLEST member doubles as the shared
same-family cheap proxy for its larger siblings (a smaller model of the same
tokenizer is exactly what the black-box `surface_stat` verifier needs):

    qwen     Qwen3-0.6B  ->  1.7B  ->  4B
    llama    Llama-3.2-1B -> 3.2-3B -> 3.1-8B
    smollm2  SmolLM2-135M -> 360M  -> 1.7B
    pythia   Pythia-160M  -> 410M  -> 1.4B
    gpt2     GPT2-124M    -> 355M  -> 774M

Each model is run in its OWN subprocess (a fresh CUDA context per model, and a
clean place to free its weights), and the HF cache is PRUNED between runs so the
sweep fits the small local disk: within a family only the shared proxy plus the
current reference model are ever on disk at once, and the whole family cache is
dropped before the next family. This is why the sweep shells out instead of
looping in-process.

Run:  python -m experiments.run_cost_curve_sweep
Env:  IVGYM_PROMPTS / IVGYM_TOKENS / IVGYM_BATCH / IVGYM_NBATCH pass through to
      every per-model run (defaults match the repo's robustness sweep:
      16 x 96 tok, batch 400). IVGYM_KEEP_CACHE=1 disables cache pruning (only
      safe with plenty of disk).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (family, [HF ids smallest -> largest]). Smallest = the shared cheap proxy.
FAMILIES: list[tuple[str, list[str]]] = [
    ("qwen",    ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]),
    ("llama",   ["unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct",
                 "unsloth/Meta-Llama-3.1-8B-Instruct"]),
    ("smollm2", ["HuggingFaceTB/SmolLM2-135M-Instruct",
                 "HuggingFaceTB/SmolLM2-360M-Instruct",
                 "HuggingFaceTB/SmolLM2-1.7B-Instruct"]),
    ("pythia",  ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
                 "EleutherAI/pythia-1.4b"]),
    ("gpt2",    ["gpt2", "gpt2-medium", "gpt2-large"]),
]

# Proven robustness-sweep config, so numbers cross-validate against the existing
# docs/results/robustness_sweep.json (e.g. qwen3-0.6b token_difr quant ~ 0.63).
DEFAULTS = {"IVGYM_PROMPTS": "16", "IVGYM_TOKENS": "96", "IVGYM_BATCH": "400",
            "IVGYM_NBATCH": "2000"}


def hf_cache_dir(hf_id: str) -> Path:
    """The HF hub cache directory for a repo id, e.g. Qwen/Qwen3-4B ->
    ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = home / "hub" if home.name != "hub" else home
    return hub / ("models--" + hf_id.replace("/", "--"))


def prune(hf_id: str):
    if os.environ.get("IVGYM_KEEP_CACHE") == "1":
        return
    d = hf_cache_dir(hf_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        print(f"  [disk] pruned {d.name}", flush=True)


def disk_free_gb() -> float:
    return shutil.disk_usage(str(ROOT)).free / 1e9


def run_one(hf_id: str, proxy: str | None) -> bool:
    env = dict(os.environ)
    for k, v in DEFAULTS.items():
        env.setdefault(k, v)
    env["IVGYM_M"] = hf_id
    if proxy:
        env["IVGYM_PROXY"] = proxy
    else:
        env.pop("IVGYM_PROXY", None)
    label = hf_id + (f"  (proxy {proxy})" if proxy else "  (no proxy)")
    print(f"\n{'='*72}\n>>> {label}   [disk free {disk_free_gb():.1f} GB]\n{'='*72}",
          flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "experiments.exp_cost_curve_gpu"],
                       cwd=str(ROOT), env=env)
    ok = r.returncode == 0
    print(f"<<< {label}  ->  {'OK' if ok else 'FAILED (rc=%d)' % r.returncode}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok


def main():
    t0 = time.time()
    done, failed = [], []
    for family, ladder in FAMILIES:
        proxy = ladder[0]  # smallest = shared same-family cheap proxy
        for i, hf_id in enumerate(ladder):
            use_proxy = None if i == 0 else proxy
            ok = run_one(hf_id, use_proxy)
            (done if ok else failed).append(hf_id)
            # prune the reference model right away unless it's this family's proxy
            if hf_id != proxy:
                prune(hf_id)
        prune(proxy)  # family finished -> drop its shared proxy too

    print(f"\n{'#'*72}\nSWEEP DONE in {time.time()-t0:.0f}s  |  "
          f"{len(done)} ok, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
    print(f"results in docs/results/cost_curve/  |  disk free {disk_free_gb():.1f} GB")


if __name__ == "__main__":
    main()
