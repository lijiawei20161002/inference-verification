"""One isolated E0 worker using the archived generation and reference functions.

Baseline has no observation hooks. Capture observes actual layer arguments, KV
outputs, logits, masks, and random draws without replacing model arithmetic.
Full output hashes, including hidden states, are compared by the campaign.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from unittest.mock import patch

import numpy as np

from experiments.analyze_real_quantization import digest, load_run, write_json
from experiments.numerical.frozen_logits import array_hash, bf16_bits, decision, filter_detail
from ivgym import sampling

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/results/real_quantization_l40s_20261018/pilot"


def tensor_meta(t):
    return dict(dtype=str(t.dtype), shape=list(t.shape), stride=list(t.stride()),
                device=str(t.device))


def tensor_array(t):
    import torch
    # View is a bitcast; never cast BF16 values before preserving their bytes.
    x = t.detach()
    return (x.view(torch.uint16) if x.dtype == torch.bfloat16 else x).cpu().numpy().copy()


class ProfiledCall:
    """Trace complete selected forwards after naturally reconstructing their state."""

    def __init__(self, model, role, selected, directory, records):
        self.model, self.role, self.selected = model, role, set(selected)
        self.directory, self.records, self.step = directory, records, 0

    def __call__(self, *args, **kwargs):
        import torch

        step = self.step
        self.step += 1
        if self.role == "provider" and step not in self.selected:
            return self.model(*args, **kwargs)
        name = f"dispatch_{self.role}_{step}.json"
        inputs = [tensor_meta(x) for x in args if hasattr(x, "shape")]
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as trace:
            out = self.model(*args, **kwargs)
        trace.export_chrome_trace(str(self.directory / name))
        self.records.append(dict(role=self.role, step=step, file=name, inputs=inputs))
        return out


class Observer:
    def __init__(self, directory, selected):
        self.directory, self.selected = directory, set(selected)
        self.role, self.position = "provider", -1
        self.draws, self.masks, self.raw, self.states = {}, {}, {}, {}
        self.operations, self.state_arrays = [], {}
        self.active = False
        self.noise_index = 0

    def noise(self, original, vocab, seed):
        result = original(vocab, seed)
        pos = self.noise_index
        self.noise_index += 1
        self.position = pos
        if pos in self.selected:
            self.draws[self.role, pos] = result.copy()
        return result

    def filter(self, original, kind, *args):
        result = original(*args)
        if self.position in self.selected:
            key = self.role, self.position, kind
            mask = result > sampling.NEG_INF / 2
            if key in self.masks and not np.array_equal(mask, self.masks[key]):
                raise RuntimeError("Repeated same-position sampler filters disagree")
            self.masks[key] = mask.copy()
        return result

    def layer_arguments(self, module, args, kwargs):
        if not self.active:
            return
        for name in ("attention_mask", "position_ids", "cache_position"):
            value = kwargs.get(name)
            if value is not None:
                self.state_arrays[name] = tensor_array(value)
        self.operations.append(dict(module="model.layers.0", inputs=[
            tensor_meta(x) for x in args if hasattr(x, "shape")],
            kwargs={k: tensor_meta(v) for k, v in kwargs.items()
                    if hasattr(v, "shape")}))

    def operation(self, name, module, args, output):
        if self.active:
            self.operations.append(dict(module=name, type=type(module).__name__,
                inputs=[tensor_meta(x) for x in args if hasattr(x, "shape")],
                output=tensor_meta(output) if hasattr(output, "shape") else None))

    def wrapped_model(self, model, role, row):
        observer = self

        class ModelCall:
            step = 0

            def __call__(self, *args, **kwargs):
                observer.active = role == "reference" or self.step in observer.selected
                observer.operations, observer.state_arrays = [], {}
                out = model(*args, **kwargs)
                if observer.active:
                    arrays = observer.state_arrays
                    if role == "provider":
                        observer.raw[role, self.step] = tensor_array(out.logits[0, -1])
                    else:
                        start = len(row["prompt_tokens"]) - 1
                        for pos in observer.selected:
                            observer.raw[role, pos] = tensor_array(out.logits[0, start + pos])
                    cache_meta = []
                    for i, layer in enumerate(out.past_key_values.layers):
                        for key in ("keys", "values"):
                            value = getattr(layer, key)
                            arrays[f"kv_{i}_{key}"] = tensor_array(value)
                            cache_meta.append(dict(layer=i, kind=key, **tensor_meta(value)))
                    arrays["input_ids"] = tensor_array(args[0])
                    tag = f"{role}_{self.step}"
                    np.savez(observer.directory / f"{tag}_state.npz", **arrays)
                    observer.states[tag] = dict(
                        file=f"{tag}_state.npz", sha256=digest(observer.directory / f"{tag}_state.npz"),
                        kv_lineage="sequential cache after this forward" if role == "provider"
                                   else "single full-prefix forward including final claimed token",
                        cache=cache_meta, operations=observer.operations,
                        logits=tensor_meta(out.logits),
                        hidden=tensor_meta(out.hidden_states[-1]))
                self.step += 1
                observer.active = False
                return out

        return ModelCall()

    def hooks(self, model):
        handles = [model.model.layers[0].register_forward_pre_hook(
            self.layer_arguments, with_kwargs=True)]
        for name, module in model.named_modules():
            if type(module).__name__ in ("Linear", "Qwen3RMSNorm", "Qwen3Attention"):
                handles.append(module.register_forward_hook(
                    lambda m, a, o, n=name: self.operation(n, m, a, o)))
        return handles


def source_snapshot(output):
    names = [p.relative_to(ROOT) for p in (ROOT / "ivgym").rglob("*.py")]
    names += [p.relative_to(ROOT) for p in (ROOT / "experiments/numerical").glob("*.py")]
    names += [Path("experiments") / name for name in (
        "exp_real_quantization_gpu.py", "analyze_real_quantization.py",
        "exp_frozen_replay_gpu.py", "exp_benign_shape_dprime_gpu.py")]
    names += [Path("scripts/run_research_job.py"), Path("planning/NEXT_EXPERIMENTS.md")]
    hashes = {}
    for name in names:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        hashes[str(name)] = digest(target)
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--phase", choices=("baseline", "capture"), required=True)
    parser.add_argument("--selection-from", type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=128)
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile", action="store_true")
    profiling.add_argument("--profile-events", action="store_true",
                           help="Trace each selected provider forward and full reference forward")
    args = parser.parse_args(argv)
    if args.phase == "capture" and args.selection_from is None:
        parser.error("Capture requires a completed baseline for event selection")
    plan, old_arrays = load_run(args.archive)
    if not 1 <= args.limit <= plan["evaluation"] or not 1 <= args.tokens <= plan["tokens"]:
        parser.error("Invalid discovery subset")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="ivgym-paired-bf16-v1", status="running",
                    phase=args.phase, profiling=args.profile or args.profile_events,
                    profile_mode="selected_forwards" if args.profile_events else
                                 "full_sequence" if args.profile else "none",
                    injected_noise=False,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    pid=os.getpid(), source_sha256=source_snapshot(args.output))
    started = time.monotonic()
    write_json(args.output / "manifest.json", manifest)
    try:
        import torch
        from huggingface_hub import snapshot_download
        from experiments import exp_real_quantization_gpu as legacy
        from experiments.exp_frozen_replay_gpu import provenance
        from scripts.run_research_job import gpu_uuid, host_lock

        torch.set_num_threads(1)
        snapshot = Path(snapshot_download(plan["model"], revision=plan["revision"],
                                          local_files_only=True))
        manifest["environment"] = provenance()
        manifest["environment"]["python"] = platform.python_version()
        manifest["environment"]["software"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        original = json.loads((args.archive / "manifest.json").read_text())
        model_hashes = {}
        for name, expected in original["model_assets_sha256"].items():
            if not (snapshot / name).is_file():
                # The model's README and .gitattributes are not runtime inputs.
                if name not in (".gitattributes", "README.md"):
                    raise ValueError(f"Missing archived model asset: {name}")
                continue
            model_hashes[name] = digest(snapshot / name)
            if model_hashes[name] != expected:
                raise ValueError(f"Model asset differs from archive: {name}")
        manifest["model_assets_sha256"] = model_hashes
        corpus = json.loads((args.archive / "corpus.json").read_text())
        rows = corpus[plan["calibration"]:plan["calibration"] + args.limit]
        with gzip.open(args.archive / "capture_bf16.jsonl.gz", "rt") as f:
            old_captures = [json.loads(line) for line in f]
        selection = None
        if args.selection_from:
            bm = json.loads((args.selection_from / "manifest.json").read_text())
            if bm["status"] != "complete":
                raise ValueError("Selection baseline is incomplete")
            selection = json.loads((args.selection_from / "sequences.json").read_text())
            if [r["prompt_id"] for r in selection] != [r["prompt_id"] for r in rows]:
                raise ValueError("Selection baseline corpus differs")
            if bm["plan"]["tokens"] != args.tokens:
                raise ValueError("Selection baseline token count differs")
        manifest["plan"] = dict(
            archive=str(args.archive), archive_manifest_sha256=digest(args.archive / "manifest.json"),
            model=plan["model"], revision=plan["revision"], prompt_ids=[r["prompt_id"] for r in rows],
            tokens=args.tokens, discovery_only=True, sampling=plan["sampling"],
            original_filter_arithmetic={"provider": "float32", "reference": "float64"},
            native_logits="torch.bfloat16", no_model_arithmetic_replacement=True,
            control_position="first position without an archived cap for each sequence",
            selection_from=str(args.selection_from) if args.selection_from else None,
            dispatch_scope="selected provider forwards and complete reference forward"
                           if args.profile_events else "profiled subset only" if args.profile
                           else "module metadata; kernel tracing in separate controls")
        write_json(args.output / "corpus.json", rows)
        write_json(args.output / "manifest.json", manifest)
        with ExitStack() as stack:
            uuid = gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            root = Path(tempfile.gettempdir())
            stack.enter_context(host_lock(root / f"ivgym-research-{os.getuid()}.lock", shared=True))
            stack.enter_context(host_lock(root / f"ivgym-research-{os.getuid()}-{uuid}.lock"))
            active = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
            ], text=True).strip()
            if active:
                raise RuntimeError(f"GPU already in use by process IDs: {active}")
            reference_model, _ = legacy.load_model(str(snapshot), "bf16")
            provider, _ = legacy.load_model(str(snapshot), "bf16")
            proj = sampling.projection(123, 32, provider.config.hidden_size)
            results, event_rows = [], []
            for idx, row in enumerate(rows):
                pid = row["prompt_id"]
                historical = set(np.flatnonzero(old_arrays["bf16/token_difr"][pid, :args.tokens] == 30).tolist())
                control = next((i for i in range(args.tokens) if i not in historical), 0)
                selected = historical | {control}
                if selection is not None:
                    selected |= set(selection[idx]["capped_positions"])
                directory = args.output / f"prompt_{pid}"
                directory.mkdir()
                observer = Observer(directory, selected) if args.phase == "capture" else None
                handles = []
                with ExitStack() as instrument:
                    if observer:
                        for model in (provider, reference_model):
                            handles.extend(observer.hooks(model))
                        for handle in handles:
                            instrument.callback(handle.remove)
                        original_noise = legacy.gumbel_noise
                        instrument.enter_context(patch.object(legacy, "gumbel_noise",
                            lambda v, s: observer.noise(original_noise, v, s)))
                        for name, kind in (("apply_top_k", "top_k_mask"), ("apply_top_p", "support_mask")):
                            original_filter = getattr(sampling, name)
                            instrument.enter_context(patch.object(sampling, name,
                                lambda *a, fn=original_filter, k=kind: observer.filter(fn, k, *a)))
                    profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True) if args.profile else nullcontext()
                    provider_call = observer.wrapped_model(provider, "provider", row) if observer else provider
                    reference_call = observer.wrapped_model(reference_model, "reference", row) if observer else reference_model
                    dispatch_records = []
                    if args.profile_events:
                        provider_call = ProfiledCall(provider_call, "provider", selected,
                                                     directory, dispatch_records)
                        reference_call = ProfiledCall(reference_call, "reference", selected,
                                                      directory, dispatch_records)
                    with profiler as trace:
                        claimed, fps, served = legacy.generate(provider_call, row, args.tokens, proj)
                        ref_logits, ref_hidden = legacy.reference(reference_call, row, claimed)
                    if args.profile:
                        trace.export_chrome_trace(str(directory / "dispatch.json"))
                    if args.profile_events:
                        write_json(directory / "dispatch_index.json", dict(
                            prompt_id=pid, selected_positions=sorted(selected), calls=dispatch_records))
                    if observer:
                        observer.role, observer.noise_index, observer.position = "reference", 0, -1
                    scores = legacy.score(row, claimed, fps, ref_logits, ref_hidden, proj)
                # Exact widening checks also validate baseline output representation.
                pb, rb = bf16_bits(served), bf16_bits(ref_logits)
                hashes = {name: array_hash(arr) for name, arr in {
                    "provider_logits": served, "reference_logits": ref_logits,
                    "fingerprints": fps, "reference_hidden": ref_hidden,
                    **scores}.items()}
                old_claimed = old_captures[pid]["claimed_tokens"]
                capped = np.flatnonzero(scores["token_difr"] == 30).tolist()
                record = dict(prompt_id=pid, sampling_seed=row["sampling_seed"],
                              claimed_tokens=claimed, hashes=hashes, capped_positions=capped,
                              score_values=scores["token_difr"].tolist(),
                              historical=[dict(event_id=f"bf16:{pid}:{p}", position=p,
                                  prefix_and_token_match=claimed[:p + 1] == old_claimed[:p + 1],
                                  capped=bool(scores["token_difr"][p] == 30),
                                  reproduced=bool(claimed[:p + 1] == old_claimed[:p + 1]
                                                  and scores["token_difr"][p] == 30))
                                          for p in sorted(historical)])
                if observer:
                    write_json(directory / "states.json", observer.states)
                    for pos in sorted(selected):
                        if not (np.array_equal(observer.raw["provider", pos], pb[pos])
                                and np.array_equal(observer.raw["reference", pos], rb[pos])):
                            raise RuntimeError("Native logits do not match exact widened bits")
                        g = observer.draws["provider", pos]
                        rg = observer.draws["reference", pos]
                        if not np.array_equal(g, rg):
                            raise RuntimeError("Actual provider/reference Gumbel mismatch")
                        arrays = dict(provider_bf16=observer.raw["provider", pos],
                                      reference_bf16=observer.raw["reference", pos],
                                      provider_gumbel=g, reference_gumbel=rg,
                                      claimed_token=np.array(claimed[pos]),
                                      prefix_tokens=np.array(row["prompt_tokens"] + claimed[:pos]))
                        for role, values in (("provider", served[pos]),
                                             ("reference", ref_logits[pos].astype(np.float64))):
                            _, detail = filter_detail(values, 50, .95)
                            for key, value in detail.items():
                                if key.endswith("_mask"):
                                    actual = observer.masks[role, pos, key]
                                    if not np.array_equal(value, actual):
                                        raise RuntimeError("Captured sampler mask cannot be replayed")
                                    value = actual
                                arrays[f"{role}_{key}"] = value
                        file = directory / f"event_{pos}.npz"
                        np.savez(file, **arrays)
                        diagnosis, _ = decision(ref_logits[pos].astype(np.float64), g, claimed=claimed[pos])
                        if diagnosis["capped_score"] != scores["token_difr"][pos]:
                            raise RuntimeError("Raw and capped score disagree")
                        event_rows.append(dict(
                            event_id=f"{args.output.name}:{pid}:{pos}",
                            historical_event_id=f"bf16:{pid}:{pos}" if pos in historical else None,
                            historical_prefix_and_token_match=claimed[:pos + 1] == old_claimed[:pos + 1],
                            prompt_id=pid, position=pos, control=pos == control,
                            current_cap=pos in capped, evidence=str(file.relative_to(args.output)),
                            evidence_sha256=digest(file), state_metadata=f"prompt_{pid}/states.json",
                            provider_state=f"provider_{pos}", reference_state="reference_0",
                            diagnosis=diagnosis))
                results.append(record)
                write_json(args.output / "sequences.json", results)
                write_json(args.output / "events.json", event_rows)
                print(json.dumps(dict(phase=args.phase, sequences=len(results), total=len(rows),
                                      prompt_id=pid, capped=len(capped),
                                      elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                del served, ref_logits, ref_hidden, pb, rb, observer
            manifest["summary"] = dict(
                sequences=len(results), positions=len(results) * args.tokens,
                capped_positions=sum(len(r["capped_positions"]) for r in results),
                historical_events=sum(len(r["historical"]) for r in results),
                historical_reproduced=sum(e["reproduced"] for r in results for e in r["historical"]),
                captured_events=len(event_rows))
        manifest["artifacts_sha256"] = {
            str(p.relative_to(args.output)): digest(p)
            for p in sorted(args.output.rglob("*")) if p.is_file()
            and p.name != "manifest.json" and "source" not in p.relative_to(args.output).parts}
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
