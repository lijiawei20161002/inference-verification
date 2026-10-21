"""M3 pilot worker: explicit operator recipes and full-state model recomputation.

Every model replay starts from immutable token IDs with an empty KV cache.
Only one selected operation row is replaced. Ancestors and all descendants are
recomputed, including every layer's KV state and later fixed-prefix positions.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
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

from experiments.analyze_real_quantization import digest, write_json
from experiments.numerical.capture_bf16 import tensor_array
from experiments.numerical.dispatch_validation import kernel_launches
from experiments.numerical.frozen_logits import array_hash, widen_bf16
from experiments.numerical.mechanism_ops import calculate, sampling_matrix, validate_controls
from experiments.numerical.mechanism_diagnostics import native_reference, partition_coverage
from experiments.numerical.run_e0_campaign import now, status

ROOT = Path(__file__).resolve().parents[2]
MODES = ("capture", "a", "b", "sham_a", "sham_b", "forward", "reverse")


def read(path):
    return json.loads(Path(path).read_text())


def tensor_hash(tensor):
    return array_hash(tensor_array(tensor))


def operand_hashes(operands):
    return {key: tensor_hash(value) for key, value in operands.items()}


def tensor_layouts(tensors):
    return {key: dict(dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()))
            for key, value in tensors.items()}


def save_tensors(directory, stem, tensors):
    np.savez(directory / f"{stem}.npz", **{key: tensor_array(value) for key, value in tensors.items()})
    write_json(directory / f"{stem}.json", tensor_layouts(tensors))


def load_tensors(directory, stem, device):
    import torch

    meta = read(directory / f"{stem}.json")
    result = {}
    with np.load(directory / f"{stem}.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(meta):
            raise ValueError("Tensor inventory differs")
        for key, detail in meta.items():
            value = torch.from_numpy(arrays[key].copy())
            if detail["dtype"] == "torch.bfloat16":
                if value.dtype != torch.uint16:
                    raise ValueError("Invalid BF16 storage")
                value = value.view(torch.bfloat16)
            if str(value.dtype) != detail["dtype"] or list(value.shape) != detail["shape"]:
                raise ValueError("Tensor metadata differs")
            if "stride" in detail:
                restored = torch.empty_strided(detail["shape"], detail["stride"], dtype=value.dtype, device=device)
                restored.copy_(value)
                result[key] = restored
            else:
                result[key] = value.to(device)
    return result


def verify_artifacts(directory):
    manifest = read(directory / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Replay source is incomplete")
    for name, sha in manifest["artifacts_sha256"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"Replay artifact changed: {name}")
    return manifest


class OperationHook:
    def __init__(self, model, case, recipe, operands=None, replacements=None, mode="capture"):
        self.model, self.case, self.recipe = model, case, recipe
        self.operands, self.replacements, self.mode = operands, replacements or {}, mode
        self.expected = operand_hashes(operands) if operands is not None else None
        self.expected_layouts = tensor_layouts(operands) if operands is not None else None
        self.calls, self.native, self.full_native = 0, None, None

    def process(self, operands, output):
        import torch

        self.calls += 1
        if self.calls != 1:
            raise ValueError("Target operation called more than once")
        hashes = operand_hashes(operands)
        if self.expected is not None and hashes != self.expected:
            raise ValueError("Operation operands drifted during model replay")
        if self.expected_layouts is not None and tensor_layouts(operands) != self.expected_layouts:
            raise ValueError("Operation operand layout drifted during model replay")
        if self.operands is None:
            self.operands = operands
        self.native = output.clone()
        if self.mode == "capture":
            return output
        if self.mode == "native_sham":
            result = self.replacements["native_replay"].clone()
            if not torch.equal(result, output):
                raise ValueError("Native replay differs at intervention site")
            return result
        side = "a" if self.mode in ("a", "sham_a", "forward", "native_restore") else "b"
        result = calculate(self.case["family"], operands, self.recipe,
                           "ascending" if side == "a" else "descending")
        if not torch.equal(result, self.replacements[side]):
            raise ValueError("Standalone and in-model operator replay differ")
        if self.mode in ("sham_a", "sham_b", "forward", "reverse"):
            target = "b" if self.mode == "forward" else "a" if self.mode == "reverse" else side
            result = self.replacements[target].clone()
        if self.mode == "native_restore":
            result = self.replacements["native_replay"].clone()
        if result.shape != output.shape or result.dtype != output.dtype:
            raise ValueError("Replacement changes shape or dtype")
        return result

    def __enter__(self):
        from transformers.models.qwen3 import modeling_qwen3 as qwen

        idx = self.case["token_index"]
        layer = self.model.model.layers[self.case["layer"]]
        self.stack = ExitStack()
        if self.case["family"] == "gemm":
            module = layer.mlp.down_proj
            if module.bias is not None:
                raise ValueError("This pilot requires a bias-free down projection")

            def hook(module, inputs, output):
                operands = dict(x=inputs[0][:, idx:idx + 1].detach().contiguous(),
                                weight=module.weight.detach().contiguous())
                if self.recipe.get("diagnostics"):
                    operands["full_x"] = inputs[0].detach()
                    self.full_native = output.detach().clone()
                replacement = self.process(operands, output[:, idx:idx + 1])
                if self.mode == "capture":
                    return output
                result = output.clone()
                result[:, idx:idx + 1] = replacement
                return result

            handle = module.register_forward_hook(hook)
            self.stack.callback(handle.remove)
        else:
            original = qwen.eager_attention_forward

            def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
                output, weights = original(module, query, key, value, attention_mask,
                                           scaling=scaling, dropout=dropout, **kwargs)
                if module is not layer.self_attn:
                    return output, weights
                if dropout or module.training or attention_mask is None:
                    raise ValueError("Unsupported attention execution contract")
                if scaling != self.recipe["attention_scale"]:
                    raise ValueError("Attention scale differs")
                operands = dict(query=query[:, :, idx:idx + 1].detach().contiguous(),
                                key=qwen.repeat_kv(key, module.num_key_value_groups).detach().contiguous(),
                                value=qwen.repeat_kv(value, module.num_key_value_groups).detach().contiguous(),
                                mask=attention_mask[:, :, idx:idx + 1, :key.shape[-2]].detach().contiguous())
                if self.recipe.get("diagnostics"):
                    operands.update(full_query=query.detach(), full_mask=attention_mask.detach().contiguous())
                    self.full_native = output.detach().clone()
                replacement = self.process(operands, output[:, idx:idx + 1])
                if self.mode == "capture":
                    return output, weights
                result = output.clone()
                result[:, idx:idx + 1] = replacement
                return result, weights

            self.stack.enter_context(patch.object(qwen, "eager_attention_forward", wrapped))
        return self

    def __exit__(self, *exc):
        self.stack.close()
        if exc[0] is None and self.calls != 1:
            raise ValueError("Target operation was not observed")


def forward(model, input_ids, index, prompt_length, expected_parent=None):
    import torch

    with torch.inference_mode():
        # No cache from another execution can enter this call.
        out = model(input_ids, use_cache=True, output_hidden_states=True)
        logits = tensor_array(out.logits)
        if expected_parent is not None:
            sliced = out.logits[0, prompt_length - 1:prompt_length - 1 + 128].float().cpu().numpy()
            if array_hash(sliced) != expected_parent["hashes"]["reference_logits"]:
                raise ValueError("Native reference logits differ from parent M1")
            hidden = out.hidden_states[-1][0, prompt_length - 1:prompt_length - 1 + 128].float().cpu().numpy()
            if array_hash(hidden) != expected_parent["hashes"]["reference_hidden"]:
                raise ValueError("Native reference hidden states differ from parent M1")
        state = dict(logits=array_hash(logits), hidden=tensor_hash(out.hidden_states[-1]),
                     prefix_logits=array_hash(logits[:, :index]),
                     kv=[{name: tensor_hash(getattr(layer, name)) for name in ("keys", "values")}
                         for layer in out.past_key_values.layers])
        selected = logits[0, index].copy()
    del out
    return state, selected


def trace_recipe(directory, family, operands, recipe, order, expected):
    import torch

    path = directory / f"{order}_trace.json"
    # Explicit warm-up precedes profiling; profiled output must remain exact.
    with torch.inference_mode():
        warm = calculate(family, operands, recipe, order)
        if not torch.equal(warm, expected):
            raise ValueError("Operator repeat drift")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as trace:
            actual = calculate(family, operands, recipe, order)
        if not torch.equal(actual, expected):
            raise ValueError("Profiling changed the operator output")
    trace.export_chrome_trace(str(path))
    launches = kernel_launches(path)
    return dict(kernels=len(launches), launches_sha256=hashlib.sha256(json.dumps(launches).encode()).hexdigest())


def metrics(a, b):
    av, bv = tensor_array(a), tensor_array(b)
    af, bf = a.float().cpu().numpy(), b.float().cpu().numpy()
    return dict(changed=int(np.count_nonzero(av != bv)), elements=int(av.size),
                max_abs=float(np.max(np.abs(af - bf))), rms=float(np.sqrt(np.mean((af - bf).astype(np.float64)**2))))


def collect_diagnostics(directory, case, recipe, operands, full_native, replacements):
    import torch

    family = case["family"]
    coverage = None
    if family == "attention":
        coverage = partition_coverage(operands["mask"].float().cpu().numpy(), recipe["attention_chunk"],
                                      recipe["minimum_active_partitions"])
    with torch.inference_mode():
        details = {}
        for side, order in (("a", "ascending"), ("b", "descending")):
            output, details[side] = calculate(family, operands, recipe, order, return_details=True)
            if not torch.equal(output, replacements[side]):
                raise ValueError("FP32 instrumentation changed the local output")
            save_tensors(directory, f"fp32_{side}", details[side])
        common = set(details["a"]) - {"precast", "numerator", "denominator"}
        if any(not torch.equal(details["a"][k], details["b"][k]) for k in common):
            raise ValueError("Reduction orders did not use identical partials")
        replay = native_reference(family, operands, recipe)
        if not torch.equal(replay, full_native):
            raise ValueError("Full-shape native operator bridge differs")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True) as trace:
            traced = native_reference(family, operands, recipe)
        if not torch.equal(traced, full_native):
            raise ValueError("Profiling changed the native bridge")
        trace.export_chrome_trace(str(directory / "native_bridge_trace.json"))
        launches = kernel_launches(directory / "native_bridge_trace.json")
        save_tensors(directory, "native_full", dict(native=full_native, replay=replay))
        selected = replay[:, case["token_index"]:case["token_index"] + 1].clone()
        diagnostics = dict(coverage=coverage, operand_layouts=tensor_layouts(operands),
                           native_full_exact=True, native_full_hashes=operand_hashes(dict(native=full_native, replay=replay)),
                           native_dispatch=dict(kernels=len(launches), launches_sha256=hashlib.sha256(json.dumps(launches).encode()).hexdigest()),
                           fp32_hashes={side: operand_hashes(values) for side, values in details.items()},
                           fp32_difference=metrics(details["a"]["precast"], details["b"]["precast"]),
                           partials_exact=True)
        if family == "attention":
            previous_recipe = dict(recipe, attention_chunk=32)
            previous = {side: calculate(family, operands, previous_recipe, order)
                        for side, order in (("a", "ascending"), ("b", "descending"))}
            save_tensors(directory, "chunk32_control", previous)
            diagnostics["chunk32_control"] = dict(hashes=operand_hashes(previous),
                coverage=partition_coverage(operands["mask"].float().cpu().numpy(), 32, minimum=1),
                local_difference=metrics(previous["a"], previous["b"]),
                to_chunk8={side: metrics(previous[side], replacements[side]) for side in ("a", "b")})
    return diagnostics, selected


def run(args):
    import torch
    from huggingface_hub import snapshot_download
    from experiments.exp_real_quantization_gpu import load_model
    from experiments.exp_frozen_replay_gpu import provenance
    from scripts.run_research_job import gpu_uuid, host_lock

    output, plan = args.output.resolve(), read(args.plan)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(schema="ivgym-m3-worker-v1", status="running", pid=os.getpid(),
                    started_utc=now(), plan_sha256=digest(args.plan),
                    replay_from=str(args.replay_from) if args.replay_from else None)
    write_json(output / "manifest.json", manifest)
    try:
        parent = Path(plan["parent_m1"])
        pm = read(parent / "capture/manifest.json")
        if digest(parent / "summary.json") != plan["parent_summary_sha256"]:
            raise ValueError("Parent M1 summary changed")
        for name, sha in pm["source_sha256"].items():
            if name.startswith("experiments/numerical/") or name == "planning/NEXT_EXPERIMENTS.md":
                continue
            if digest(ROOT / name) != sha:
                raise ValueError(f"Production source changed: {name}")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        env = provenance()
        env["software"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        env["python"] = platform.python_version()
        for key in ("device", "gpu_uuid", "driver", "software", "python", "cuda_runtime",
                    "tf32_matmul", "tf32_cudnn", "torch_threads"):
            if env[key] != pm["environment"][key]:
                raise ValueError(f"Environment changed: {key}")
        manifest["environment"] = env
        snapshot = Path(snapshot_download(pm["plan"]["model"], revision=pm["plan"]["revision"], local_files_only=True))
        for name, sha in pm["model_assets_sha256"].items():
            if digest(snapshot / name) != sha:
                raise ValueError(f"Model asset changed: {name}")
        manifest["model_assets_sha256"] = pm["model_assets_sha256"]
        if args.replay_from:
            previous = verify_artifacts(args.replay_from)
            if previous["plan_sha256"] != manifest["plan_sha256"]:
                raise ValueError("Replay plan differs")
        records = []
        corpus = {r["prompt_id"]: r for r in read(parent / "capture/corpus.json")}
        rows = {r["prompt_id"]: r for r in read(parent / "capture/sequences.json")}
        for name in ("corpus.json", "sequences.json", "events.json"):
            if digest(parent / "capture" / name) != pm["artifacts_sha256"][name]:
                raise ValueError("Parent data integrity failure")
        with ExitStack() as stack:
            lock_root = Path(tempfile.gettempdir())
            stack.enter_context(host_lock(lock_root / f"ivgym-research-{os.getuid()}.lock", shared=True))
            uuid = gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            stack.enter_context(host_lock(lock_root / f"ivgym-research-{os.getuid()}-{uuid}.lock"))
            active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if active:
                raise RuntimeError(f"GPU already in use: {active}")
            model, _ = load_model(str(snapshot), "bf16")
            if len(model.model.layers) != 28 or model.config._attn_implementation != "eager":
                raise ValueError("Unexpected model architecture/backend")
            torch.cuda.reset_peak_memory_stats()
            for number, original_case in enumerate(plan["cases"]):
                if shutil.disk_usage(output).free < 12 * 1024**3:
                    raise RuntimeError("Stopping with less than 12 GiB free; preserve evidence")
                case = dict(original_case)
                row, historical = corpus[case["prompt_id"]], rows[case["prompt_id"]]
                plen = len(row["prompt_tokens"])
                case["token_index"] = plen - 1 + case["position"]
                directory = output / case["case_id"]
                directory.mkdir()
                input_ids = torch.tensor([row["prompt_tokens"] + historical["claimed_tokens"]], device="cuda")
                event_path = parent / "capture" / case["evidence"]
                if digest(event_path) != case["evidence_sha256"]:
                    raise ValueError("Selected event changed")
                with np.load(event_path, allow_pickle=False) as event:
                    noise = event["reference_gumbel"].copy()
                    expected_bits = event["reference_bf16"].copy()
                states, logits = {}, {}
                states["native"], logits["native"] = forward(model, input_ids, case["token_index"], plen, historical)
                if not np.array_equal(logits["native"], expected_bits):
                    raise ValueError("Selected native event differs from parent")
                state_path = parent / "capture" / f"prompt_{case['prompt_id']}/reference_0_state.npz"
                relative = str(state_path.relative_to(parent / "capture"))
                if digest(state_path) != pm["artifacts_sha256"][relative]:
                    raise ValueError("Parent KV evidence changed")
                with np.load(state_path, allow_pickle=False) as saved:
                    expected_kv = [{name: array_hash(saved[f"kv_{i}_{name}"]) for name in ("keys", "values")}
                                   for i in range(28)]
                if states["native"]["kv"] != expected_kv:
                    raise ValueError("Native KV state differs from parent")
                recipe = dict(plan["recipe"], attention_scale=model.model.layers[case["layer"]].self_attn.scaling)
                with OperationHook(model, case, recipe) as observer:
                    states["capture"], logits["capture"] = forward(model, input_ids, case["token_index"], plen)
                operands, native = observer.operands, observer.native
                if args.replay_from:
                    old_dir = args.replay_from / case["case_id"]
                    saved_operands = load_tensors(old_dir, "operands", "cuda")
                    if operand_hashes(operands) != operand_hashes(saved_operands):
                        raise ValueError("Saved operands do not match reconstructed model state")
                    operands = saved_operands
                with torch.inference_mode():
                    replacements = {side: calculate(case["family"], operands, recipe, order)
                                    for side, order in (("a", "ascending"), ("b", "descending"))}
                    if any(not torch.isfinite(t).all() for t in replacements.values()):
                        raise ValueError("Nonfinite local intervention")
                diagnostics, native_replay = None, None
                if recipe.get("diagnostics"):
                    diagnostics, native_replay = collect_diagnostics(
                        directory, case, recipe, operands, observer.full_native, replacements)
                save_tensors(directory, "operands", operands)
                save_tensors(directory, "local", dict(native=native, **replacements))
                traces = {side: trace_recipe(directory, case["family"], operands, recipe, order, replacements[side])
                          for side, order in (("a", "ascending"), ("b", "descending"))}
                for mode in MODES[1:]:
                    with OperationHook(model, case, recipe, operands, replacements, mode):
                        states[mode], logits[mode] = forward(model, input_ids, case["token_index"], plen)
                if diagnostics is not None:
                    for mode in ("native_sham", "native_restore"):
                        with OperationHook(model, case, recipe, operands,
                                           dict(replacements, native_replay=native_replay), mode):
                            states[mode], logits[mode] = forward(model, input_ids, case["token_index"], plen)
                        if states[mode] != states["native"]:
                            raise ValueError(f"Native bridge state control failed: {mode}")
                controls = validate_controls(states, case["layer"])
                if diagnostics is not None:
                    controls.update(native_sham_exact=True, native_restore_exact=True)
                matrix = sampling_matrix(widen_bf16(logits["a"]), widen_bf16(logits["b"]), noise)
                primary = next(c for c in matrix if c["dtype"] == "float32" and c["convention"] == "legacy"
                               and c["mode"] == "top_k_top_p")
                record = dict(case=case, recipe=recipe, operands=operand_hashes(operands),
                              local_hashes=operand_hashes(dict(native=native, **replacements)),
                              local_difference=metrics(replacements["a"], replacements["b"]),
                              native_bridge={side: metrics(native, t) for side, t in replacements.items()},
                              states=states, controls=controls, dispatch=traces, sampling=matrix,
                              primary_token_changed=primary["token_changed"], primary_support_delta=primary["support_delta"],
                              explanation="controlled_reduction_order_effect" if primary["token_changed"]
                              else "no_primary_sampling_effect", historical_attribution="not_established",
                              conformance="allowed_variation_under_explicit_experimental_contract")
                if diagnostics is not None:
                    diagnostics["selected_logits_changed"] = not np.array_equal(logits["a"], logits["b"])
                    diagnostics["native_sampling"] = {side: sampling_matrix(
                        widen_bf16(logits["native"]), widen_bf16(logits[side]), noise) for side in ("a", "b")}
                    record["diagnostics"] = diagnostics
                if args.replay_from:
                    if read(old_dir / "record.json") != record:
                        raise ValueError("Independent saved-operand replay differs")
                    saved_local = load_tensors(old_dir, "local", "cuda")
                    if operand_hashes(saved_local) != record["local_hashes"]:
                        raise ValueError("Saved local outputs do not replay")
                    if diagnostics is not None:
                        for side in ("a", "b"):
                            saved_detail = load_tensors(old_dir, f"fp32_{side}", "cuda")
                            if operand_hashes(saved_detail) != diagnostics["fp32_hashes"][side]:
                                raise ValueError("Saved FP32 intermediates do not replay")
                write_json(directory / "record.json", record)
                np.savez(directory / "logits.npz", **logits, gumbel=noise,
                         input_ids=input_ids.cpu().numpy())
                records.append(record)
                write_json(output / "records.json", records)
                progress = dict(completed=number + 1, total=len(plan["cases"]), case_id=case["case_id"],
                                elapsed_seconds=time.monotonic() - started,
                                local_changed=record["local_difference"]["changed"],
                                primary_token_changed=record["primary_token_changed"])
                status(output / "progress.json", **progress)
                print(json.dumps(progress), flush=True)
                del operands, replacements, native, observer
            manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest.update(status="complete", cases=len(records),
                        comparison_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest())
        manifest["artifacts_sha256"] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                                       if p.is_file() and p.name not in ("manifest.json", "progress.json")}
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
