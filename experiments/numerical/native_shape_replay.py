"""One isolated native-shape worker, including original provider-path controls."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
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

import numpy as np

from experiments.analyze_real_quantization import digest, write_json
from experiments.numerical.capture_bf16 import tensor_array
from experiments.numerical.dispatch_validation import kernel_launches
from experiments.numerical.frozen_logits import array_hash, widen_bf16
from experiments.numerical.mechanism_diagnostics import native_reference
from experiments.numerical.mechanism_ops import sampling_matrix, validate_controls
from experiments.numerical.mechanism_replay import (
    forward, metrics, operand_hashes, read, save_tensors, tensor_hash,
    tensor_layouts, verify_artifacts,
)
from experiments.numerical.native_shape import (
    NativeShapeHook, effective_names, shape_operands, shape_output,
)
from experiments.numerical.run_e0_campaign import now, status

ROOT = Path(__file__).resolve().parents[2]


def provider_forward(model, input_ids, prompt_length, position, observer=None):
    """Rebuild the actual sequential provider KV, teacher forcing its saved tokens."""
    import torch

    prefix = []
    out = None
    with torch.inference_mode():
        for step in range(position + 1):
            ids = input_ids[:, :prompt_length] if step == 0 else input_ids[:, prompt_length + step - 1:prompt_length + step]
            kwargs = {} if step == 0 else dict(past_key_values=out.past_key_values)
            context = observer if step == position and observer is not None else nullcontext()
            with context:
                out = model(ids, use_cache=True, output_hidden_states=True, **kwargs)
            selected = tensor_array(out.logits[0, -1])
            if step < position:
                prefix.append(array_hash(selected))
        state = dict(logits=tensor_hash(out.logits), hidden=tensor_hash(out.hidden_states[-1]),
                     prefix_logits=prefix,
                     kv=[{name: tensor_hash(getattr(layer, name)) for name in ("keys", "values")}
                         for layer in out.past_key_values.layers])
    return state, selected


def parent_kv(parent, manifest, case, role):
    step = 0 if role == "reference" else case["position"]
    name = f"prompt_{case['prompt_id']}/{role}_{step}_state.npz"
    file = parent / "capture" / name
    if digest(file) != manifest["artifacts_sha256"][name]:
        raise ValueError("Parent state evidence changed")
    with np.load(file, allow_pickle=False) as saved:
        return [{key: array_hash(saved[f"kv_{i}_{key}"]) for key in ("keys", "values")} for i in range(28)]


def trace(directory, name, function, expected):
    import torch

    with torch.inference_mode():
        if not torch.equal(function(), expected):
            raise ValueError("Native operator warm-up differs")
        # Keep the complete asynchronous launch inside the profiling interval.
        # A CPU launch event alone is not evidence that a GPU kernel was traced.
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            actual = function()
            torch.cuda.synchronize()
        if not torch.equal(actual, expected):
            raise ValueError("Native operator profiling changes output")
    file = directory / f"{name}_trace.json"
    profile.export_chrome_trace(str(file))
    launches = kernel_launches(file)
    return dict(kernels=len(launches), launches_sha256=hashlib.sha256(json.dumps(launches).encode()).hexdigest())


def primary(cells):
    return next(c for c in cells if (c["dtype"], c["convention"], c["mode"]) ==
                ("float32", "legacy", "top_k_top_p"))


def run(args):
    import torch
    from huggingface_hub import snapshot_download
    from experiments.exp_real_quantization_gpu import load_model
    from experiments.exp_frozen_replay_gpu import provenance
    from scripts.run_research_job import gpu_uuid, host_lock

    output, plan = args.output.resolve(), read(args.plan)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(schema="ivgym-native-shape-worker-v1", status="running", pid=os.getpid(),
                    started_utc=now(), plan_sha256=digest(args.plan))
    write_json(output / "manifest.json", manifest)
    try:
        parent = Path(plan["parent_m1"])
        if digest(parent / "summary.json") != plan["parent_summary_sha256"]:
            raise ValueError("Parent M1 summary changed")
        pm = read(parent / "capture/manifest.json")
        if digest(parent / "capture/manifest.json") != plan["parent_capture_manifest_sha256"]:
            raise ValueError("Parent M1 capture changed")
        for name, sha in pm["source_sha256"].items():
            if name.startswith("experiments/numerical/") or name == "planning/NEXT_EXPERIMENTS.md":
                continue
            if digest(ROOT / name) != sha:
                raise ValueError(f"Production source changed: {name}")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        environment = provenance()
        environment["python"] = platform.python_version()
        environment["software"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        for key in ("device", "gpu_uuid", "driver", "software", "python", "cuda_runtime", "tf32_matmul", "tf32_cudnn", "torch_threads"):
            if environment[key] != pm["environment"][key]:
                raise ValueError(f"Environment changed: {key}")
        manifest["environment"] = environment
        snapshot = Path(snapshot_download(pm["plan"]["model"], revision=pm["plan"]["revision"], local_files_only=True))
        for name, sha in pm["model_assets_sha256"].items():
            if digest(snapshot / name) != sha:
                raise ValueError(f"Model asset changed: {name}")
        manifest["model_assets_sha256"] = pm["model_assets_sha256"]
        for name in ("corpus.json", "sequences.json", "events.json"):
            if digest(parent / "capture" / name) != pm["artifacts_sha256"][name]:
                raise ValueError("Parent data integrity failure")
        corpus = {r["prompt_id"]: r for r in read(parent / "capture/corpus.json")}
        rows = {r["prompt_id"]: r for r in read(parent / "capture/sequences.json")}
        records, bundles = [], []
        with ExitStack() as stack:
            root = Path(tempfile.gettempdir())
            stack.enter_context(host_lock(root / f"ivgym-research-{os.getuid()}.lock", shared=True))
            uuid = gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
            stack.enter_context(host_lock(root / f"ivgym-research-{os.getuid()}-{uuid}.lock"))
            active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if active:
                raise RuntimeError(f"GPU already in use: {active}")
            model, _ = load_model(str(snapshot), "bf16")
            if len(model.model.layers) != 28 or model.config._attn_implementation != "eager":
                raise ValueError("Unexpected model architecture or attention backend")
            torch.cuda.reset_peak_memory_stats()
            for number, original in enumerate(plan["cases"]):
                if shutil.disk_usage(output).free < 12 * 1024**3:
                    raise RuntimeError("Evidence reached the free disk reserve")
                case = dict(original)
                row, historical = corpus[case["prompt_id"]], rows[case["prompt_id"]]
                plen = len(row["prompt_tokens"])
                case.update(token_index=plen - 1 + case["position"], prompt_length=plen)
                directory = output / case["case_id"]
                directory.mkdir()
                input_ids = torch.tensor([row["prompt_tokens"] + historical["claimed_tokens"]], device="cuda")
                event_file = parent / "capture" / case["evidence"]
                if digest(event_file) != case["evidence_sha256"]:
                    raise ValueError("Selected event changed")
                with np.load(event_file, allow_pickle=False) as event:
                    noise = event["reference_gumbel"].copy()
                    expected = {role: event[f"{role}_bf16"].copy() for role in ("provider", "reference")}
                    if not np.array_equal(noise, event["provider_gumbel"]):
                        raise ValueError("Actual provider and reference RNG differ")
                states, logits = {}, {}
                states["native"], logits["native"] = forward(model, input_ids, case["token_index"], plen, historical)
                if (not np.array_equal(logits["native"], expected["reference"])
                        or states["native"]["kv"] != parent_kv(parent, pm, case, "reference")):
                    raise ValueError("Reference path differs from parent evidence")
                recipe = dict(diagnostics=True, attention_scale=model.model.layers[case["layer"]].self_attn.scaling)
                with NativeShapeHook(model, case, recipe) as observer:
                    states["capture"], logits["capture"] = forward(model, input_ids, case["token_index"], plen)
                operands, native_full = observer.operands, observer.full_native
                provider_case = dict(case, token_index=plen - 1 if case["position"] == 0 else 0)
                provider_state, logits["provider"] = provider_forward(model, input_ids, plen, case["position"])
                provider_observer = NativeShapeHook(model, provider_case, recipe)
                captured_provider, provider_logits = provider_forward(model, input_ids, plen, case["position"], provider_observer)
                if (provider_state != captured_provider or not np.array_equal(provider_logits, logits["provider"])
                        or not np.array_equal(logits["provider"], expected["provider"])
                        or provider_state["kv"] != parent_kv(parent, pm, case, "provider")):
                    raise ValueError("Provider capture or parent continuity failed")
                provider_operands = provider_observer.operands
                provider_layouts = tensor_layouts(provider_operands)
                idx = case["token_index"]
                with torch.inference_mode():
                    reconstructed = native_reference(case["family"], operands, recipe)
                    provider_reconstructed = native_reference(case["family"], provider_operands, recipe)
                    if not torch.equal(reconstructed, native_full) or not torch.equal(provider_reconstructed, provider_observer.full_native):
                        raise ValueError("Full-shape native operator reconstruction failed")
                    replacements = dict(a=reconstructed[:, idx:idx + 1].clone(),
                                        b=shape_output(case["family"], operands, recipe, provider_layouts, case))
                    query_only = shape_output(case["family"], operands, recipe, provider_layouts, case, query_only=True)
                    alternate = shape_operands(case["family"], operands, provider_layouts, case)
                    names = effective_names(case["family"])
                    effective_equal = all(torch.equal(alternate[k], provider_operands[k]) for k in names)
                    if effective_equal and not torch.equal(replacements["b"], provider_observer.native):
                        raise ValueError("Identical provider inputs/layouts did not reproduce native output")
                save_tensors(directory, "operands", operands)
                save_tensors(directory, "provider_operands", provider_operands)
                save_tensors(directory, "local", dict(**replacements, query_only=query_only,
                             provider=provider_observer.native, native_full=native_full,
                             provider_full=provider_observer.full_native))
                dispatch = dict(
                    reference=trace(directory, "reference", lambda: native_reference(case["family"], operands, recipe), native_full),
                    alternate=trace(directory, "alternate", lambda: shape_output(case["family"], operands, recipe, provider_layouts, case), replacements["b"]),
                    provider=trace(directory, "provider", lambda: native_reference(case["family"], provider_operands, recipe), provider_observer.full_native),
                )
                for mode in ("a", "b", "sham_a", "sham_b", "forward", "reverse"):
                    with NativeShapeHook(model, case, recipe, operands, replacements, mode,
                                         provider_layouts=provider_layouts):
                        states[mode], logits[mode] = forward(model, input_ids, idx, plen)
                controls = validate_controls(states, case["layer"])
                if states["a"] != states["native"]:
                    raise ValueError("Native-output replacement changes model state")
                controls.update(native_sham_exact=True, provider_capture_exact=True,
                                provider_parent_exact=True, native_operator_reconstruction_exact=True)
                matrix = sampling_matrix(widen_bf16(logits["a"]), widen_bf16(logits["b"]), noise)
                cell = primary(matrix)
                before = sampling_matrix(widen_bf16(logits["native"]), widen_bf16(logits["provider"]), noise)
                after = sampling_matrix(widen_bf16(logits["b"]), widen_bf16(logits["provider"]), noise)
                bridge = dict(effective_operands_exact=effective_equal,
                              alternate_provider_output_exact=torch.equal(replacements["b"], provider_observer.native),
                              alternate_provider_logits_exact=np.array_equal(logits["b"], logits["provider"]),
                              primary_disagreement_removed=primary(before)["token_changed"] and not primary(after)["token_changed"],
                              primary_disagreement_introduced=not primary(before)["token_changed"] and primary(after)["token_changed"],
                              original_primary_disagreement=primary(before)["token_changed"],
                              native_to_provider_sampling=before, alternate_to_provider_sampling=after,
                              interpretation="Intervention-supported influence only; upstream operands and KV may differ")
                record = dict(case=case, recipe=recipe, operands=operand_hashes(operands),
                              provider_operands=operand_hashes(provider_operands),
                              provider_layouts=provider_layouts, states=states, provider_state=provider_state,
                              controls=controls, dispatch=dispatch, sampling=matrix,
                              local_difference=metrics(replacements["a"], replacements["b"]),
                              query_only_difference=metrics(replacements["a"], query_only),
                              masked_suffix_difference=metrics(query_only, replacements["b"]),
                              selected_logits_changed=not np.array_equal(logits["a"], logits["b"]),
                              primary_token_changed=cell["token_changed"], primary_support_delta=cell["support_delta"],
                              historical_bridge=bridge, historical_root_cause="not_established",
                              conformance="allowed_native_execution_variation",
                              contrast="Observed provider geometry and layout on reference effective operands; native BF16 arithmetic")
                write_json(directory / "record.json", record)
                np.savez(directory / "logits.npz", **logits, gumbel=noise, input_ids=input_ids.cpu().numpy())
                shutil.copy2(event_file, directory / "parent_event.npz")
                bundle = dict(schema="ivgym-native-shape-witness-v1", case=case, recipe=recipe,
                              environment=environment, model=pm["plan"]["model"], revision=pm["plan"]["revision"],
                              model_assets_sha256=pm["model_assets_sha256"], plan_sha256=digest(args.plan),
                              predicate="Exact local, descendant-state and sampler replay on the saved token checkpoint",
                              historical_root_cause="not_established", discovery_only=True,
                              artifacts_sha256={p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()})
                write_json(directory / "bundle.json", bundle)
                bundles.append(dict(case_id=case["case_id"], file=f"{case['case_id']}/bundle.json", sha256=digest(directory / "bundle.json")))
                records.append(record)
                write_json(output / "records.json", records)
                write_json(output / "witnesses.json", bundles)
                progress = dict(completed=number + 1, total=len(plan["cases"]), case_id=case["case_id"],
                                elapsed_seconds=time.monotonic() - started, primary_token_changed=cell["token_changed"])
                status(output / "progress.json", **progress)
                print(json.dumps(progress), flush=True)
                del observer, provider_observer, operands, provider_operands, alternate, replacements, native_full
            manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest.update(status="complete", cases=len(records),
                        comparison_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                        artifacts_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*'))
                                          if p.is_file() and p.name not in ("manifest.json", "progress.json")})
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
