"""Controlled vLLM HTTP pilot on one idle GPU; raw SSE, cache and provenance evidence.

Client concurrency is not the actual scheduler batch. This script produces
descriptive mechanism data, not a calibrated detector. Run with a separate
pinned vLLM Python environment. GPU server configurations run sequentially.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import gzip
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

from scripts.run_research_job import host_lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/"
                     "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
CONFIGS = ("graph_off", "graph_on", "eager_off")
SOURCE_FILES = ("experiments/exp_vllm_serving.py", "experiments/analyze_vllm_serving.py",
                "experiments/requirements_vllm_serving.txt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def command(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def gpu_snapshot():
    return command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.used,"
                    "utilization.gpu,temperature.gpu,power.draw,clocks.current.sm",
                    "--format=csv"])


@dataclasses.dataclass(frozen=True)
class Group:
    block: int
    length: int
    concurrency: int
    arm: str
    outputs: int = 128

    @property
    def key(self):
        return f"b{self.block}_l{self.length}_c{self.concurrency}_{self.arm}_o{self.outputs}"


def build_plan(config, session, suite="pilot"):
    if config not in CONFIGS:
        raise ValueError(config)
    if suite == "smoke":
        return [Group(0, 512, 1, "full", 16), Group(0, 16384, 4, "full", 16),
                Group(0, 16384, 4, "truncate", 16), Group(0, 8192, 1, "cold", 16),
                Group(0, 8192, 1, "hit", 16)]
    arms = ("cold", "hit") if config == "graph_on" else ("full", "truncate")
    lengths = (2048, 8192) if config == "graph_on" else (512, 2048, 8192, 16384)
    plan = [Group(block, length, concurrency, arm)
            for block in (2*session, 2*session+1)
            for length in lengths for concurrency in (1, 4) for arm in arms]
    random.Random(20261021 + session*17 + CONFIGS.index(config)).shuffle(plan)
    return plan


def prompt_ids(corpus, block, row, length):
    offset = (block*997 + row*503 + 17) % len(corpus)
    repeated = corpus * ((offset + length)//len(corpus) + 1)
    return repeated[offset:offset+length]


def make_payload(corpus, group, row, prime=False):
    ids = prompt_ids(corpus, group.block, row, group.length)
    if prime:
        ids = ids[:-128]
    salt = hashlib.sha256(f"ivgym-public-fixture:{group.block}:{row}".encode()).hexdigest()
    payload = {
        "model": "ivgym-qwen3-1.7b", "prompt": ids, "max_tokens": 1 if prime else group.outputs,
        "temperature": 0, "seed": 20261021, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True}, "return_token_ids": True,
        "cache_salt": salt, "add_special_tokens": False,
    }
    if group.arm == "truncate":
        payload["truncate_prompt_tokens"] = 2048
        payload["truncation_side"] = "left"
    return payload


def summarize_frames(frames, expected_ids, outputs):
    tokens, arrivals, event_times, usage = [], [], [], None
    first_event_tokens = 0
    returned_prompt = None
    finished = []
    for frame in frames:
        data = frame["data"]
        if data.get("usage") is not None:
            usage = data["usage"]
        for choice in data.get("choices", []):
            assert choice["index"] == 0
            if choice.get("prompt_token_ids") is not None:
                returned_prompt = choice["prompt_token_ids"]
            ids = choice.get("token_ids") or []
            if ids:
                if not event_times:
                    first_event_tokens = len(ids)
                tokens.extend(ids)
                arrivals.extend([frame["received_ns_since_send"]/1e6]*len(ids))
                event_times.append(frame["received_ns_since_send"]/1e6)
            if choice.get("finish_reason"):
                finished.append(choice["finish_reason"])
    assert returned_prompt == expected_ids, "Returned prompt does not match the intervention"
    assert len(tokens) == outputs, (len(tokens), outputs)
    assert usage is not None and usage["prompt_tokens"] == len(expected_ids), usage
    assert usage["completion_tokens"] == outputs, usage
    assert finished == ["length"], finished
    assert len(arrivals) >= 1
    gaps = [b-a for a, b in zip(event_times, event_times[1:])]
    remaining = len(tokens)-first_event_tokens
    post_first = (arrivals[-1]-arrivals[0])/remaining if remaining else None
    details = usage.get("prompt_tokens_details") or {}
    return {
        "predicted_token_ids": tokens, "effective_prompt_tokens": len(returned_prompt),
        "returned_prompt_sha256": hashlib.sha256(json.dumps(returned_prompt).encode()).hexdigest(),
        "ttft_ms": arrivals[0], "last_token_ms": arrivals[-1],
        "post_first_ms_per_token": post_first, "content_events": len(event_times),
        "event_gap_ms_median": statistics.median(gaps) if gaps else None,
        "coalesced_token_count": len(tokens)-len(event_times),
        "cached_tokens": details.get("cached_tokens", 0), "usage": usage,
    }


def stream_request(port, payload, barrier, request_id):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    connection.connect()
    connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    body = json.dumps(payload).encode()
    barrier.wait(timeout=60)
    send_start = time.perf_counter_ns()
    connection.request("POST", "/v1/completions", body=body,
                       headers={"Content-Type": "application/json", "X-Request-Id": request_id})
    sent = time.perf_counter_ns()
    response = connection.getresponse()
    if response.status != 200:
        detail = response.read().decode()
        connection.close()
        raise RuntimeError(f"HTTP {response.status}: {detail}")
    frames, done = [], False
    for line in response:
        received = time.perf_counter_ns()
        line = line.strip()
        if not line:
            continue
        assert line.startswith(b"data: "), line[:100]
        if line == b"data: [DONE]":
            done = True
            break
        data = json.loads(line[6:])
        if "error" in data:
            raise RuntimeError(data["error"])
        frames.append({"received_ns_since_send": received-sent, "data": data})
    connection.close()
    assert done
    expected = payload["prompt"]
    if payload.get("truncate_prompt_tokens"):
        expected = expected[-payload["truncate_prompt_tokens"]:]
    result = summarize_frames(frames, expected, payload["max_tokens"])
    return {"request_id": request_id, "status": "complete", "payload": payload,
            "send_start_ns": send_start, "send_completed_ns": sent,
            "send_ms": (sent-send_start)/1e6, "wire_request_bytes": len(body),
            "frames": frames, **result}


def request_group(port, corpus, group, prime=False):
    barrier = threading.Barrier(group.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=group.concurrency) as pool:
        futures = [pool.submit(stream_request, port, make_payload(corpus, group, row, prime),
                               barrier, group.key+f"_r{row}"+("_prime" if prime else ""))
                   for row in range(group.concurrency)]
        return [future.result(timeout=360) for future in futures]


def local_api(port, path, method="GET"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {body}")
        return body
    finally:
        connection.close()


def server_command(python, model, port, config):
    cmd = [str(python), "-m", "vllm.entrypoints.openai.api_server",
           "--model", str(model), "--served-model-name", "ivgym-qwen3-1.7b",
           "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
           "--max-model-len", "16512", "--max-num-seqs", "8",
           "--max-num-batched-tokens", "2048", "--kv-cache-memory-bytes", str(12*1024**3),
           "--gpu-memory-utilization", "0.8", "--block-size", "16",
           "--attention-backend", "FLASH_ATTN", "--enable-chunked-prefill",
           "--enable-prompt-tokens-details", "--no-enable-log-requests", "--seed", "20261021"]
    cmd += ["--enable-prefix-caching" if config == "graph_on" else "--no-enable-prefix-caching"]
    if config == "eager_off":
        cmd += ["--enforce-eager"]
    return cmd


def stop_server(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def run_server(args, root, config, session, corpus):
    root.mkdir(parents=True, exist_ok=False)
    plan = build_plan(config, session, args.suite)
    write_json(root/"plan.json", [dataclasses.asdict(group) for group in plan])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cmd = server_command(args.server_python, args.model_path, port, config)
    overrides = dict(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
               HF_HOME="/workspace/.cache/huggingface", VLLM_CACHE_ROOT="/workspace/.cache/vllm-round2",
               TORCHINDUCTOR_CACHE_DIR="/workspace/.cache/vllm-round2/inductor",
               TRITON_CACHE_DIR="/workspace/.cache/vllm-round2/triton",
               MAX_JOBS="8", VLLM_LOGGING_LEVEL="INFO", VLLM_SERVER_DEV_MODE="1")
    env = dict(os.environ, **overrides)
    info = {"config": config, "session": session, "command": cmd, "started_utc": utc(),
            "gpu_before": gpu_snapshot(), "status": "starting", "environment": overrides}
    write_json(root/"manifest.json", info)
    start = time.perf_counter()
    with (root/"server.log").open("x") as logfile:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logfile,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        info["server_pid"] = process.pid
        write_json(root/"manifest.json", info)
        try:
            deadline = time.monotonic()+1200
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {root/'server.log'}")
                try:
                    local_api(port, "/health")
                    break
                except (OSError, http.client.HTTPException, RuntimeError):
                    if time.monotonic()>deadline:
                        raise TimeoutError("Server readiness timed out")
                    time.sleep(1)
            info["startup_s"] = time.perf_counter()-start
            info["status"] = "running"
            info["models"] = json.loads(local_api(port, "/v1/models"))
            write_json(root/"manifest.json", info)
            # Warm all modes using separate, excluded requests.
            with gzip.open(root/"warmup.jsonl.gz", "xt") as warm:
                for group in (Group(99,512,1,"full",16), Group(99,8192,4,"full",16)):
                    for row in request_group(port, corpus, group):
                        warm.write(json.dumps(row)+"\n")
            with gzip.open(root/"requests.jsonl.gz", "xt") as raw, \
                 gzip.open(root/"primers.jsonl.gz", "xt") as primers, \
                 gzip.open(root/"metrics.jsonl.gz", "xt") as metrics, \
                 (root/"groups.jsonl").open("x") as groups:
                for order, group in enumerate(plan):
                    before_gpu = gpu_snapshot()
                    if config == "graph_on":
                        local_api(port, "/reset_prefix_cache", "POST")
                    priming_ms = 0.0
                    if group.arm == "hit":
                        priming_start = time.perf_counter()
                        for row in request_group(port, corpus, group, prime=True):
                            primers.write(json.dumps(row)+"\n")
                        primers.flush()
                        priming_ms = (time.perf_counter()-priming_start)*1000
                    before = local_api(port, "/metrics")
                    group_start = time.perf_counter()
                    measured = request_group(port, corpus, group)
                    wall_ms = (time.perf_counter()-group_start)*1000
                    after = local_api(port, "/metrics")
                    for row_index, row in enumerate(measured):
                        row.update(dataclasses.asdict(group))
                        row.update({"config":config, "session":session, "group_order":order,
                                    "row_index":row_index, "group_id":group.key})
                        if config != "graph_on" or group.arm == "cold":
                            assert row["cached_tokens"] == 0, row["cached_tokens"]
                        if group.arm == "hit":
                            assert 0 < row["cached_tokens"] <= group.length-128, row["cached_tokens"]
                        raw.write(json.dumps(row)+"\n")
                    raw.flush()
                    metrics.write(json.dumps({"group_id":group.key,"before":before,"after":after})+"\n")
                    metrics.flush()
                    summary = {"group_id":group.key, **dataclasses.asdict(group), "order":order,
                               "config":config, "session":session, "status":"complete",
                               "priming_ms":priming_ms, "wall_ms":wall_ms, "gpu_before":before_gpu,
                               "gpu_after":gpu_snapshot(),
                               "median_ttft_ms":statistics.median(r["ttft_ms"] for r in measured),
                               "median_post_first_ms_per_token":statistics.median(
                                   r["post_first_ms_per_token"] for r in measured),
                               "cached_tokens":[r["cached_tokens"] for r in measured],
                               "request_ids":[r["request_id"] for r in measured]}
                    groups.write(json.dumps(summary)+"\n");groups.flush()
                    print(json.dumps({"config":config,"session":session,"completed":order+1,
                                      "total":len(plan),"group":group.key,
                                      "ttft_ms":summary["median_ttft_ms"],
                                      "decode_ms_per_token":summary["median_post_first_ms_per_token"]}),flush=True)
            info["status"] = "complete"
        except BaseException as error:
            info["status"] = "failed"
            info["error"] = repr(error)
            raise
        finally:
            stop_server(process)
            info["finished_utc"] = utc()
            info["elapsed_s"] = time.perf_counter()-start
            info["server_returncode_after_shutdown"] = process.returncode
            info["gpu_after"] = gpu_snapshot()
            info["hashes"] = {str(p.relative_to(root)):sha(p) for p in root.iterdir()
                              if p.is_file() and p.name!="manifest.json"}
            write_json(root/"manifest.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke","pilot"), default="pilot")
    parser.add_argument("--server-python", type=Path,
                        default=Path("/workspace/.venvs/ivgym-vllm-0.22.0/bin/python"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    status = command(["git","status","--porcelain"])["stdout"].strip()
    if args.suite=="pilot" and status:
        raise RuntimeError("Commit the source before the formal pilot")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        snapshot = args.output / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    fixture = ROOT/"experiments/data/specdec_longtext.txt"
    corpus = tokenizer(fixture.read_text(), add_special_tokens=False).input_ids
    write_json(args.output/"corpus_token_ids.json",corpus)
    package_code = "import importlib.metadata,json;print(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())))"
    server_packages = command([str(args.server_python),"-c",package_code])
    assert server_packages["returncode"] == 0, server_packages
    write_json(args.output/"server_packages.json",json.loads(server_packages["stdout"]))
    state = {"status":"running","suite":args.suite,"started_utc":utc(),"jobs":[],
             "source_commit":command(["git","rev-parse","HEAD"])["stdout"].strip(),
             "source_status":status,"source_sha256":{name:sha(ROOT/name) for name in SOURCE_FILES},
             "fixture_sha256":sha(fixture),"model_revision":args.model_path.name,
             "model_config_sha256":sha(args.model_path/"config.json"),
             "scope":"loopback production engine mechanism pilot; C is not B; no calibrated detector",
             "clock":"client monotonic time after sending request; connection establishment excluded",
             "truncation":"vLLM preprocessing keeps last 2048 prompt tokens; positions restart at zero",
             "generation":"greedy, fixed 128 output tokens, ignore EOS; no frozen-continuation replay",
             "cache":"APC off or explicit reset; hit primes L-128 input tokens outside measured request",
             "configs":"graph modes use default compiler/CUDA-Graph optimizations; eager disables both",
             "gpu_before":gpu_snapshot()}
    write_json(args.output/"queue.json",state)
    sessions = (0,) if args.suite=="smoke" else range(3)
    jobs = []
    for session in sessions:
        configs = ["graph_on"] if args.suite=="smoke" else list(CONFIGS)
        if args.suite!="smoke":
            configs = configs[session:]+configs[:session]
        jobs.extend((session,config) for config in configs)
    with host_lock(Path(tempfile.gettempdir())/f"ivgym-research-{os.getuid()}.lock"):
        active = command(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"])
        if active["returncode"] or active["stdout"].strip():
            raise RuntimeError(f"GPU is not idle: {active}")
        try:
            for session,config in jobs:
                entry={"session":session,"config":config,"status":"running"}
                state["jobs"].append(entry);write_json(args.output/"queue.json",state)
                result=run_server(args,args.output/f"session_{session}_{config}",config,session,corpus)
                entry.update({"status":result["status"],"elapsed_s":result["elapsed_s"]})
                write_json(args.output/"queue.json",state)
            state["status"]="complete"
        except BaseException as error:
            state["status"]="failed";state["error"]=repr(error)
            if state["jobs"]:
                state["jobs"][-1]["status"]="failed"
            raise
        finally:
            state["finished_utc"]=utc()
            write_json(args.output/"queue.json",state)


if __name__ == "__main__":
    main()
