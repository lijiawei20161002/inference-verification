"""Controlled performance-probe pilot, not a calibrated provider detector.

Uses pinned local weights, real text, BF16 SDPA, and explicit KV-cache policies.
CUDA event spans include GPU idle gaps between host launches; they are not
summed kernel time or measurements of DRAM bytes. No CPU copies in local decode.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import gzip
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_MODEL_PATH = Path("/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots") / REVISION


@dataclasses.dataclass(frozen=True)
class Case:
    suite: str
    block: int
    length: int
    batch: int
    condition: str = "full"
    outputs: int = 128
    cache: str = "static"
    mode: str = "replay"
    instrument: str = "events"
    repetition: int = 0
    execution: str = "graph"

    @property
    def key(self):
        return "_".join(str(x) for x in dataclasses.astuple(self))


def layout(length, condition, window=2048, suffix=128):
    if length <= 0 or window <= 0 or suffix <= 0:
        raise ValueError("Lengths must be positive")
    if condition not in {"full", "truncate", "cache_off", "cache_cold", "cache_hit"}:
        raise ValueError(condition)
    effective = min(length, window) if condition == "truncate" else length
    prefix = max(0, effective - suffix) if condition.startswith("cache_") else 0
    return {"effective": effective, "offset": length - effective, "prefix": prefix}


def build_plan(suite, blocks, session=0):
    cases = []
    if suite == "smoke":
        for length in (512, 4096):
            for arm in ("full", "truncate"):
                cases.append(Case("p1", 99, length, 1, arm, outputs=8))
        for arm in ("cache_off", "cache_cold", "cache_hit"):
            cases.append(Case("p4", 99, 4096, 1, arm, outputs=8))
        cases.append(Case("free", 99, 512, 1, outputs=8, mode="greedy"))
    elif suite == "preflight":
        for arm in ("full", "truncate"):
            cases.append(Case("preflight", 98, 16384, 8, arm))
        cases.append(Case("preflight", 98, 8192, 4, "cache_hit"))
    elif suite == "pilot":
        # Across three fresh processes: 7+7+6 technical repetitions per shape.
        for rep in range((7, 7, 6)[session]):
            for length in (512, 8192):
                for batch in (1, 4):
                    for instrument in ("plain", "events"):
                        cases.append(Case("p0", session, length, batch, outputs=16,
                                          instrument=instrument, repetition=rep))
        for block in blocks:
            for length in (512, 2048, 8192, 16384):
                for batch in (1, 4, 8):
                    for arm in ("full", "truncate"):
                        cases.append(Case("p1", block, length, batch, arm))
            for length in (512, 8192):
                cases.append(Case("p1_control", block, length, 1, cache="dynamic", execution="eager"))
                cases.append(Case("p1_control", block, length, 1, execution="eager"))
            for length in (2048, 8192):
                for batch in (1, 4):
                    for arm in ("cache_off", "cache_cold", "cache_hit", "truncate"):
                        cases.append(Case("p4", block, length, batch, arm))
        for length in (512, 8192):
            for arm in ("full", "truncate"):
                cases.append(Case("free", session, length, 1, arm, mode="greedy"))
    elif suite == "profile":
        for length in (512, 8192):
            for batch in (1, 4):
                cases.append(Case("profile", 0, length, batch, outputs=5))
        cases.append(Case("profile", 0, 8192, 1, outputs=5, cache="dynamic", execution="eager"))
        cases.append(Case("profile", 0, 8192, 1, outputs=5, execution="eager"))
    else:
        raise ValueError(suite)
    # Shuffle pairs as units, then shuffle arm order inside each pair.
    pairs = collections.defaultdict(list)
    for c in cases:
        pair = (c.suite, c.block, c.length, c.batch, c.outputs, c.repetition)
        pairs[pair].append(c)
    rng = random.Random(20261019 + session)
    groups = list(pairs.values())
    rng.shuffle(groups)
    ordered = []
    for group in groups:
        rng.shuffle(group)
        ordered.extend(group)
    if len({c.key for c in ordered}) != len(ordered):
        raise ValueError("Duplicate case identifiers")
    return ordered


def allocate_cache(model, batch, capacity, kind="static"):
    import torch
    from transformers import DynamicCache, StaticCache
    if kind == "dynamic":
        return DynamicCache(config=model.config)
    cache = StaticCache(config=model.config, max_cache_len=capacity)
    cache.early_initialization(
        batch, model.config.num_key_value_heads, model.config.head_dim,
        model.dtype, next(model.parameters()).device,
    )
    return cache


def forward_range(model, ids, cache, start, end, offset=0, chunk=256):
    import torch
    if not (0 <= start < end <= ids.shape[1]) or chunk < 1:
        raise ValueError("Invalid prefill range")
    logits = None
    for lo in range(start, end, chunk):
        hi = min(end, lo + chunk)
        positions = torch.arange(lo, hi, device=ids.device)
        out = model(input_ids=ids[:, lo:hi], past_key_values=cache,
                    cache_position=positions, position_ids=(positions + offset)[None, :],
                    use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1, :]
        del out
    return logits


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_snapshot():
    fields = "timestamp,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem"
    proc = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv"],
                          capture_output=True, text=True)
    return {"returncode": proc.returncode, "csv": proc.stdout, "stderr": proc.stderr}


class Engine:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, window=2048, chunk=256):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.window = window
        self.chunk = chunk
        self.model_path = Path(model_path).resolve()
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        fixture = ROOT / "experiments/data/specdec_longtext.txt"
        self.fixture_hash = file_hash(fixture)
        self.text_ids = self.tokenizer(fixture.read_text(), add_special_tokens=False).input_ids
        self.metadata = {
            "model": MODEL, "revision": self.model_path.name,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16", "attention": self.model.config._attn_implementation,
            "cache": "static preallocated outside local prefill timer; dynamic as control",
            "weight_bytes": sum(p.numel() * p.element_size() for p in self.model.parameters()),
            "kv_bytes_per_token_per_sequence": 2 * self.model.config.num_hidden_layers
                * self.model.config.num_key_value_heads * self.model.config.head_dim * 2,
            "fixture_sha256": self.fixture_hash,
            "sampler": "frozen text continuation or greedy argmax; no top-p sampling",
            "eos_policy": "fixed output budget, continue after EOS",
            "graph_capture": "outside local timers; repeated first decode position is overwritten before measurement",
            "model_config_sha256": file_hash(self.model_path / "config.json"),
            "truncation": "remove early prompt before prefill; preserve absolute RoPE positions; no rolling eviction",
            "cache_hit": "prefix primed outside request timer; priming time reported separately",
            "timing": "event spans include host enqueue gaps; profiler runs are separate",
        }

    def inputs(self, case):
        torch = self.torch
        rows, offsets = [], []
        total = case.length + case.outputs
        for row in range(case.batch):
            offset = (case.block * 997 + row * 503 + 17) % len(self.text_ids)
            offsets.append(offset)
            repeated = self.text_ids * ((total + offset) // len(self.text_ids) + 1)
            rows.append(repeated[offset:offset + total])
        raw = torch.tensor(rows, dtype=torch.long)
        digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
        ids = raw.to("cuda")
        return ids[:, :case.length], ids[:, case.length:], digest, offsets

    def warmup(self):
        for _ in range(2):
            self.run(Case("warmup", 0, 512, 1, outputs=8))

    def run(self, case, emit=None, trace_path=None):
        torch = self.torch
        with torch.inference_mode():
            return self._run(case, emit, trace_path)

    def _run(self, case, emit, trace_path):
        torch = self.torch
        if case.outputs < 2:
            raise ValueError("Need at least two outputs")
        spec = layout(case.length, case.condition, self.window)
        started = time.perf_counter()
        ids, continuation, prompt_hash, offsets = self.inputs(case)
        ids = ids[:, spec["offset"]:]
        effective = spec["effective"]
        capacity = effective + case.outputs - 1
        estimate = self.metadata["kv_bytes_per_token_per_sequence"] * capacity * case.batch
        free_bytes, _ = torch.cuda.mem_get_info()
        if estimate * 2 + 2 * 1024**3 > free_bytes:
            return {**dataclasses.asdict(case), "case_id": case.key, "status": "memory_precheck_skip",
                    "estimated_kv_bytes": estimate, "free_bytes": free_bytes}
        torch.cuda.reset_peak_memory_stats()
        cache = allocate_cache(self.model, case.batch, capacity, case.cache)
        # Positions and frozen inputs are allocated before timing.
        decode_pos = [torch.tensor([effective + i], device="cuda") for i in range(case.outputs - 1)]
        abs_pos = [p + spec["offset"] for p in decode_pos]
        replay = [continuation[:, i:i+1].contiguous() for i in range(case.outputs - 1)]
        predicted = torch.empty((case.batch, case.outputs), dtype=torch.long, device="cuda")
        instrument = case.instrument == "events"
        pre_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        step_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(case.outputs - 1)] if instrument else []
        # Initialize event objects before measured work.
        for event in (*pre_events, *(e for pair in step_events for e in pair)):
            event.record()
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - started) * 1000
        priming_ms = 0.0
        prefix = spec["prefix"]
        if case.condition == "cache_hit" and prefix:
            t = time.perf_counter()
            forward_range(self.model, ids, cache, 0, prefix, spec["offset"], self.chunk)
            torch.cuda.synchronize()
            priming_ms = (time.perf_counter() - t) * 1000
        t0 = time.perf_counter()
        if instrument:
            pre_events[0].record()
        if prefix and case.condition != "cache_hit":
            forward_range(self.model, ids, cache, 0, prefix, spec["offset"], self.chunk)
        first_logits = forward_range(
            self.model, ids, cache, prefix, effective, spec["offset"], self.chunk)
        token = first_logits.argmax(-1, keepdim=True)
        predicted[:, 0].copy_(token[:, 0])
        if instrument:
            pre_events[1].record()
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000
        if emit:
            emit(0, token[:, 0].tolist())
        graph_setup_ms = 0.0
        graph = None
        graph_output = None
        if case.execution == "graph":
            if case.cache != "static" or emit:
                raise ValueError("Graph mode requires a static cache and local timing")
            graph_started = time.perf_counter()
            graph_input = (replay[0] if case.mode == "replay" else token).clone()
            graph_pos = decode_pos[0].clone()
            graph_abs = abs_pos[0][None, :].clone()
            graph_output = torch.empty_like(token)
            def graph_step():
                value = self.model(input_ids=graph_input, past_key_values=cache,
                    cache_position=graph_pos, position_ids=graph_abs,
                    use_cache=True, logits_to_keep=1)
                graph_output.copy_(value.logits[:, -1, :].argmax(-1, keepdim=True))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    graph_step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_step()
            torch.cuda.synchronize()
            graph_setup_ms = (time.perf_counter() - graph_started) * 1000
        profiler = None
        if trace_path:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True)
        ctx = profiler if profiler else contextlib.nullcontext()
        with ctx:
            t1 = time.perf_counter()
            for i in range(case.outputs - 1):
                if instrument:
                    step_events[i][0].record()
                if graph:
                    graph_input.copy_(replay[i] if case.mode == "replay" else token)
                    graph_pos.copy_(decode_pos[i])
                    graph_abs.copy_(abs_pos[i][None, :])
                    graph.replay()
                    token = graph_output
                else:
                    out = self.model(
                        input_ids=replay[i] if case.mode == "replay" else token,
                        past_key_values=cache, cache_position=decode_pos[i],
                        position_ids=abs_pos[i][None, :], use_cache=True, logits_to_keep=1,
                    )
                    token = out.logits[:, -1, :].argmax(-1, keepdim=True)
                    del out
                predicted[:, i + 1].copy_(token[:, 0])
                if instrument:
                    step_events[i][1].record()
                if emit:
                    emit(i + 1, token[:, 0].tolist())
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - t1) * 1000
        actual_cache_bytes = sum(layer.keys.numel() * layer.keys.element_size()
                                 + layer.values.numel() * layer.values.element_size()
                                 for layer in cache.layers)
        result = {
            **dataclasses.asdict(case), "case_id": case.key, "status": "complete",
            "prompt_sha256": prompt_hash, "fixture_offsets": offsets, **spec,
            "capacity": capacity, "cache_allocated_bytes": actual_cache_bytes,
            "estimated_kv_bytes": estimate, "setup_ms": setup_ms, "priming_ms": priming_ms, "graph_setup_ms": graph_setup_ms,
            "prefill_wall_ms": prefill_ms, "decode_wall_ms": decode_ms,
            "decode_ms_per_step": decode_ms / (case.outputs - 1),
            "decode_tokens_per_second": case.batch * (case.outputs - 1) / (decode_ms / 1000),
            "prefill_cuda_span_ms": pre_events[0].elapsed_time(pre_events[1]) if instrument else None,
            "decode_cuda_spans_ms": [a.elapsed_time(b) for a, b in step_events],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "predicted_token_ids": predicted.cpu().tolist(),
            "first_logits_sha256": hashlib.sha256(first_logits.float().cpu().numpy().tobytes()).hexdigest(),
            "cuda_synchronization": "prefill boundary and end of decode; HTTP emission copies each step",
        }
        if profiler:
            path = Path(trace_path)
            raw_path = path.with_suffix("")
            profiler.export_chrome_trace(str(raw_path))
            with raw_path.open("rb") as src, gzip.open(path, "wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            raw_path.unlink()
            kernels = collections.defaultdict(lambda: {"count": 0, "duration_us": 0.0})
            for event in profiler.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    kernels[event.name]["count"] += 1
                    kernels[event.name]["duration_us"] += event.device_time_total
            result["kernels"] = dict(sorted(kernels.items(), key=lambda x: -x[1]["duration_us"]))
            result["trace_sha256"] = file_hash(path)
            result["trace_file"] = path.name
        if graph is not None:
            del graph, graph_step, graph_input, graph_pos, graph_abs, graph_output
        del cache, ids, continuation, predicted, first_logits, token
        # This is deliberately outside measurement, between independent cases.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return result


def manifest(engine, args):
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    paths = [Path(__file__), ROOT / "experiments/exp_performance_probe_http.py",
             ROOT / "experiments/analyze_performance_probes.py", ROOT / "experiments/run_performance_pilot.py"]
    return {**engine.metadata, "args": dict(vars(args)), "source_commit": proc.stdout.strip(),
            "source_status": status.stdout, "source_sha256": {str(p.relative_to(ROOT)): file_hash(p)
                    for p in paths if p.exists()},
            "gpu_before": gpu_snapshot(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "counter_status": os.environ.get("IVGYM_COUNTER_STATUS", "not_tested_by_this_run"),
            "result_scope": "mechanism pilot; eight text blocks across three process sessions; no deployment FPR calibration"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "preflight", "pilot", "profile"], default="smoke")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, choices=range(3), default=0)
    parser.add_argument("--blocks", default="0,1,2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    engine = Engine(args.model_path)
    plan = build_plan(args.suite, [int(x) for x in args.blocks.split(",")], args.session)
    (args.output / "plan.json").write_text(json.dumps([dataclasses.asdict(c) for c in plan], indent=2))
    info = manifest(engine, args)
    info["args"]["output"] = str(args.output)
    (args.output / "manifest.json").write_text(json.dumps(info, indent=2))
    subprocess.run([str(Path(os.sys.executable)), "-m", "pip", "list", "--format=json"],
                   stdout=(args.output / "packages.json").open("w"), check=True)
    engine.warmup()
    started = time.perf_counter()
    with (args.output / "measurements.jsonl").open("x") as f:
        for index, case in enumerate(plan):
            before = gpu_snapshot()
            try:
                trace_path = args.output / (case.key + ".json.gz") if args.suite == "profile" else None
                result = engine.run(case, trace_path=trace_path)
            except Exception as exc:
                result = {**dataclasses.asdict(case), "case_id": case.key, "status": "failed",
                          "error": repr(exc)}
                f.write(json.dumps(result) + "\n")
                f.flush()
                raise
            result.update({"session": args.session, "order": index, "gpu_before": before,
                           "gpu_after": gpu_snapshot()})
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(json.dumps({"completed": index + 1, "total": len(plan), "case": case.key,
                              "status": result["status"],
                              "prefill_ms": result.get("prefill_wall_ms"),
                              "decode_ms_per_step": result.get("decode_ms_per_step"),
                              "elapsed_s": time.perf_counter() - started}), flush=True)
    info.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "elapsed_seconds": time.perf_counter() - started, "gpu_after": gpu_snapshot(),
                 "measurement_sha256": file_hash(args.output / "measurements.jsonl")})
    (args.output / "manifest.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
