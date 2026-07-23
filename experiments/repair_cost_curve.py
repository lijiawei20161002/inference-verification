"""Repair pass for the cost-curve sweep.

`/workspace` (the default HF cache here) is a network filesystem that throws
intermittent `OSError: [Errno 5]` on large download writes, so some models in
`run_cost_curve_sweep` fail mid-download and leave a STALE result on disk (the
old batch-400 file with no `selective` block). This script finds every model
whose result is missing or stale under the current sound config and re-runs just
those, forcing the HF cache onto LOCAL disk (reliable writes) and pruning each
model right after so the small local disk never overflows.

A result is FRESH iff config.batch == BATCH, config.n_batches == NBATCH, and --
for any model that has a smaller same-family sibling (a proxy) -- the `selective`
block is present. Run:  python -m experiments.repair_cost_curve
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "cost_curve"
LOCAL_HF = Path("/root/hfcache")            # local overlay disk, reliable writes

# Same ladders as run_cost_curve_sweep (smallest = shared same-family proxy).
FAMILIES = [
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

BATCH = int(os.environ.get("IVGYM_BATCH", 96))
NBATCH = int(os.environ.get("IVGYM_NBATCH", 4000))


def result_path(hf_id: str) -> Path:
    from ivgym.model_registry import identity
    return RESULTS / (identity(hf_id).label.lower().replace(" ", "-") + ".json")


def is_fresh(hf_id: str, has_proxy: bool) -> bool:
    p = result_path(hf_id)
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    c = d.get("config", {})
    if c.get("batch") != BATCH or c.get("n_batches") != NBATCH:
        return False
    if has_proxy and not d.get("selective"):
        return False
    return True


def hf_cache_dir(hf_id: str, home: Path) -> Path:
    hub = home / "hub"
    return hub / ("models--" + hf_id.replace("/", "--"))


def run_one(hf_id: str, proxy: str | None) -> bool:
    env = dict(os.environ)
    env["HF_HOME"] = str(LOCAL_HF)             # local disk: reliable writes
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env.setdefault("IVGYM_PROMPTS", "16")
    env.setdefault("IVGYM_TOKENS", "96")
    env["IVGYM_BATCH"] = str(BATCH)
    env["IVGYM_NBATCH"] = str(NBATCH)
    env["IVGYM_M"] = hf_id
    if proxy:
        env["IVGYM_PROXY"] = proxy
    else:
        env.pop("IVGYM_PROXY", None)
    print(f"\n{'='*72}\n>>> REPAIR {hf_id}"
          f"{f'  (proxy {proxy})' if proxy else '  (no proxy)'}\n{'='*72}", flush=True)
    r = subprocess.run([sys.executable, "-m", "experiments.exp_cost_curve_gpu"],
                       cwd=str(ROOT), env=env)
    ok = r.returncode == 0
    print(f"<<< REPAIR {hf_id} -> {'OK' if ok else 'FAILED rc=%d' % r.returncode}", flush=True)
    return ok


def main():
    LOCAL_HF.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for family, ladder in FAMILIES:
        proxy = ladder[0]
        for i, hf_id in enumerate(ladder):
            has_proxy = i != 0
            if is_fresh(hf_id, has_proxy):
                print(f"=== fresh, skip {hf_id} ===", flush=True)
                continue
            # up to 2 attempts (network FS can still hiccup even locally on first touch)
            ok = run_one(hf_id, proxy if has_proxy else None) or \
                 run_one(hf_id, proxy if has_proxy else None)
            (done if ok else failed).append(hf_id)
            # prune this model's local cache (keep proxy until family done)
            if has_proxy:
                shutil.rmtree(hf_cache_dir(hf_id, LOCAL_HF), ignore_errors=True)
        shutil.rmtree(hf_cache_dir(proxy, LOCAL_HF), ignore_errors=True)
    print(f"\n#### REPAIR DONE: {len(done)} ok, {len(failed)} failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
