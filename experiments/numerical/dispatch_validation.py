"""CPU checks for selected-forward dispatch coverage, shapes and repeatability."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.analyze_real_quantization import digest


def selected_positions(row, tokens):
    historical = {e["position"] for e in row["historical"]}
    control = next((p for p in range(tokens) if p not in historical), 0)
    return historical | set(row["capped_positions"]) | {control}


def expected_calls(rows, tokens):
    result = set()
    for row in rows:
        pid = row["prompt_id"]
        result.update((pid, "provider", p) for p in selected_positions(row, tokens))
        result.add((pid, "reference", 0))
    return result


def load_calls(directory, rows, tokens):
    """Reject missing/duplicate/extra calls, including silently truncated traces."""
    expected = expected_calls(rows, tokens)
    calls = {}
    corpus = json.loads((directory / "corpus.json").read_text())
    lengths = {r["prompt_id"]: len(r["prompt_tokens"]) for r in corpus}
    if set(lengths) != {r["prompt_id"] for r in rows}:
        raise ValueError("Dispatch corpus differs")
    for file in sorted(directory.glob("prompt_*/dispatch_index.json")):
        index = json.loads(file.read_text())
        pid = index["prompt_id"]
        if file.parent.name != f"prompt_{pid}":
            raise ValueError("Dispatch prompt identity differs")
        row = next((r for r in rows if r["prompt_id"] == pid), None)
        if row is None or index["selected_positions"] != sorted(selected_positions(row, tokens)):
            raise ValueError("Dispatch selection differs")
        for call in index["calls"]:
            key = pid, call["role"], call["step"]
            if key not in expected or key in calls:
                raise ValueError(f"Unexpected or duplicate dispatch call: {key}")
            name = f"dispatch_{key[1]}_{key[2]}.json"
            if call["file"] != name:
                raise ValueError("Unexpected dispatch artifact path")
            length = lengths[pid] + tokens if key[1] == "reference" else lengths[pid] if key[2] == 0 else 1
            if (len(call["inputs"]) != 1 or call["inputs"][0]["shape"] != [1, length]
                    or call["inputs"][0]["dtype"] != "torch.int64"):
                raise ValueError(f"Unexpected forward input shape/dtype: {key}")
            trace = file.parent / name
            if not trace.is_file():
                raise ValueError(f"Missing dispatch trace: {key}")
            calls[key] = dict(**call, path=trace)
    if set(calls) != expected:
        raise ValueError(f"Incomplete dispatch coverage: {len(calls)}/{len(expected)}")
    return calls


def kernel_launches(path):
    data = json.loads(Path(path).read_text())
    events = sorted((e for e in data["traceEvents"] if e.get("cat") == "kernel"),
                    key=lambda e: e["ts"])
    if not events:
        raise ValueError(f"Trace has no GPU kernels: {path}")
    launches = []
    for event in events:
        args = event.get("args", {})
        grid, block = args.get("grid"), args.get("block")
        if (not isinstance(grid, list) or not isinstance(block, list)
                or len(grid) != 3 or len(block) != 3
                or any(type(n) is not int or n < 1 for n in grid + block)):
            raise ValueError(f"Missing or invalid kernel launch dimensions: {path}")
        launches.append([event["name"], grid, block])
    return launches


def dispatch_signature(directory, rows, tokens):
    calls = load_calls(directory, rows, tokens)
    result = []
    for key, call in sorted(calls.items()):
        launches = kernel_launches(call["path"])
        result.append(dict(prompt_id=key[0], role=key[1], step=key[2], inputs=call["inputs"],
                           kernels=len(launches),
                           launches_sha256=hashlib.sha256(json.dumps(launches).encode()).hexdigest(),
                           trace=str(call["path"].relative_to(directory)),
                           trace_sha256=digest(call["path"])))
    return result


def compare_dispatch(left, right):
    def index(rows):
        result = {(r["prompt_id"], r["role"], r["step"]): r for r in rows}
        if not result or len(result) != len(rows):
            raise ValueError("Empty or duplicate dispatch signature")
        return result
    a, b = index(left), index(right)
    if set(a) != set(b):
        raise ValueError("Dispatch signatures cover different calls")
    failures = []
    for key in sorted(a):
        if any(a[key][field] != b[key][field] for field in ("inputs", "kernels", "launches_sha256")):
            failures.append(dict(prompt_id=key[0], role=key[1], step=key[2]))
    return dict(exact=not failures, calls=len(a),
                kernels=sum(r["kernels"] for r in left), failed_calls=failures)


def compare_parent(old_manifest, old_rows, new_manifest, new_rows):
    """Allow diagnostic-tool changes, but never silently change model/library inputs."""
    for field in ("model_assets_sha256",):
        if old_manifest[field] != new_manifest[field]:
            raise ValueError(f"Parent {field} differs")
    for name, value in old_manifest["source_sha256"].items():
        if name.startswith("experiments/numerical/") or name == "planning/NEXT_EXPERIMENTS.md":
            continue
        if new_manifest["source_sha256"].get(name) != value:
            raise ValueError(f"Production source differs from parent: {name}")
    for key in ("model", "revision", "tokens", "sampling", "archive_manifest_sha256"):
        if old_manifest["plan"][key] != new_manifest["plan"][key]:
            raise ValueError(f"Parent plan {key} differs")
    for key in ("device", "gpu_uuid", "driver", "software", "cuda_runtime", "python",
                "tf32_matmul", "tf32_cudnn", "torch_threads"):
        if old_manifest["environment"][key] != new_manifest["environment"][key]:
            raise ValueError(f"Parent environment {key} differs")
    previous = {r["prompt_id"]: r for r in old_rows}
    if not new_rows or any(previous.get(r["prompt_id"]) != r for r in new_rows):
        raise ValueError("Outputs differ from the parent E0 campaign")
    return dict(exact=True, sequences=len(new_rows),
                scope="same GPU/environment/production sources; diagnostic sources updated")
