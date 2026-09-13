"""Exercise the independent API over real HTTP with separate H100 model processes.

Reuses prior token captures and clock traces, and makes fresh live token/clock
requests against a test provider. Reports scoring/integration evidence, not a
new calibrated operating point.
"""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx
import numpy as np


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(url, process):
    for _ in range(240):
        if process.poll() is not None:
            raise RuntimeError("child process stopped; inspect run logs")
        try:
            if httpx.get(url, timeout=1, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError("startup timeout")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", default="runs/service-h100")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    provider_port, service_port = port(), port()
    base = f"http://127.0.0.1:{service_port}"
    config = dict(
        models={
            "qwen3-0.6b": dict(
                repository="Qwen/Qwen3-0.6B",
                revision=Path(args.snapshot).name,
                snapshot=str(Path(args.snapshot).resolve()),
                dtype="bfloat16",
                attention="eager",
                device="cuda",
            )
        },
        allowed_provider_urls=[f"http://127.0.0.1:{provider_port}/v1"],
        max_reference_tokens=16384,
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    logs = [(out / "provider.log").open("w"), (out / "service.log").open("w")]
    procs = []
    try:
        provider = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.service_demo_provider",
                "--snapshot",
                args.snapshot,
                "--port",
                str(provider_port),
            ],
            stdout=logs[0],
            stderr=subprocess.STDOUT,
        )
        procs.append(provider)
        service = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ivgym",
                "serve",
                "--config",
                str(out / "config.json"),
                "--port",
                str(service_port),
            ],
            stdout=logs[1],
            stderr=subprocess.STDOUT,
        )
        procs.append(service)
        wait(f"http://127.0.0.1:{provider_port}", provider)
        wait(base + "/health", service)
        results = {}

        def post(name, body):
            r = httpx.post(base + "/v1/verify", json=body, timeout=180, trust_env=False)
            assert r.status_code == 200, (name, r.text)
            result = r.json()
            results[name] = result
            print(
                name,
                result["outcome"],
                {k: v.get("score") for k, v in result["verifiers"].items()},
                flush=True,
            )
            return result

        for label in ["honest", "temperature_2", "real_nf4"]:
            path = Path("runs/audit-h100") / label / "block-0000" / "records.json"
            rows = json.loads(path.read_text())
            captures = [
                dict(
                    prompt_token_ids=r["request"]["prompt_token_ids"],
                    output_token_ids=r["capture"]["payload"]["output_token_ids"],
                    prompt_id=i,
                )
                for i, r in enumerate(rows)
            ]
            result = post(
                "prior_" + label,
                dict(
                    model="qwen3-0.6b",
                    verifiers=["cross_entropy", "token_toploc"],
                    captures=captures,
                ),
            )
            assert result["evidence_provenance"] == "client_reported"
            assert result["reference"]["repository"] == "Qwen/Qwen3-0.6B"
        assert (
            results["prior_temperature_2"]["verifiers"]["cross_entropy"]["score"]
            > results["prior_honest"]["verifiers"]["cross_entropy"]["score"]
        )
        old = json.loads(Path("docs/results/slope_verifier_window.json").read_text())
        cells = {c["label"]: c for c in old["cells"]}
        low = cells["probe_lo"]["itl_ms"][:64]
        for label in ["honest_hi", "window_512"]:
            high = cells[label]["itl_ms"][:64]
            r = post(
                "prior_clock_" + label,
                dict(
                    model="Qwen/Qwen3-1.7B",
                    verifiers=["clock_slope"],
                    timing_pairs=[
                        dict(
                            short_context_tokens=256,
                            long_context_tokens=32768,
                            short_itl_ms=low,
                            long_itl_ms=high,
                            unit="device_token",
                        )
                    ],
                ),
            )
            expected = float(np.mean(np.asarray(low) - np.asarray(high)))
            assert r["verifiers"]["clock_slope"]["score"] == expected
            assert r["costs"]["reference_prefill_tokens"] == 0
        assert (
            results["prior_clock_window_512"]["verifiers"]["clock_slope"]["score"]
            > results["prior_clock_honest_hi"]["verifiers"]["clock_slope"]["score"]
        )
        live = dict(
            model="qwen3-0.6b",
            target=dict(
                base_url=f"http://127.0.0.1:{provider_port}/v1",
                model="test-provider",
                sampler_contract="ivgym_numpy_gumbel_v1",
            ),
            verifiers=["token_difr", "cross_entropy", "token_toploc", "clock_slope"],
            prompts=[
                "The capital of France is",
                "Explain the purpose of a hash table:",
            ],
            clock_probes=dict(
                short_prompt="Explain the number 42.",
                long_prompt="The city recorded 42 visits today. " * 128,
                pairs=2,
            ),
            max_tokens=16,
            include_token_scores=True,
        )
        r = post("live_independent_provider", live)
        assert r["evidence_provenance"] == "service_observed"
        assert all(
            r["verifiers"][name].get("score") is not None for name in live["verifiers"]
        )
        assert r["verifiers"]["clock_slope"]["details"]["unit"] == "token"
        # Only the independent operator provisions the honest null, through the CLI.
        control = {
            **live,
            "verifiers": ["token_difr"],
            "clock_probes": None,
            "prompts": [
                "Write an imaginative beginning to a story about a lighthouse:"
            ],
            "max_tokens": 8,
            "include_token_scores": False,
        }
        (out / "control-request.json").write_text(json.dumps(control))
        with (out / "calibration.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ivgym",
                    "calibrate",
                    "--config",
                    str(out / "config.json"),
                    "--request",
                    str(out / "control-request.json"),
                    "--blocks",
                    "19",
                    "--provenance",
                    "Operator-controlled separate HF process; integration test only",
                    "--out",
                    str(out / "calibration.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        service.terminate()
        service.wait(timeout=30)
        config["calibration_files"] = [str((out / "calibration.json").resolve())]
        (out / "config.json").write_text(json.dumps(config, indent=2))
        service = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ivgym",
                "serve",
                "--config",
                str(out / "config.json"),
                "--port",
                str(service_port),
            ],
            stdout=logs[1],
            stderr=subprocess.STDOUT,
        )
        procs.append(service)
        wait(base + "/health", service)
        honest = post("live_calibrated_honest", control)
        assert honest["outcome"] == "no_deviation_detected", honest
        attack = {**control, "target": {**control["target"], "model": "temperature-2"}}
        deviating = post("live_calibrated_temperature", attack)
        assert deviating["outcome"] == "inconsistent", deviating
        queued = httpx.post(
            base + "/v1/verifications",
            json={
                **live,
                "verifiers": ["token_toploc"],
                "clock_probes": None,
                "prompts": live["prompts"][:1],
            },
            timeout=30,
            trust_env=False,
        )
        assert queued.status_code == 202, queued.text
        for _ in range(240):
            job = httpx.get(
                base + queued.json()["status_url"], timeout=10, trust_env=False
            ).json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.25)
        assert job["status"] == "succeeded", job
        report = dict(
            purpose="Independent service integration and parity with prior experimental detectors",
            provider_process_id=provider.pid,
            verifier_process_id=service.pid,
            cases=results,
            queued_job=job,
            note="Prior captures and live four-detector checks are diagnostic without matching calibration. A 19-block operator-owned control demonstrates live DiFR decisions on one honest and one temperature-deviating request. This is an integration test, not an FPR or power study.",
        )
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"All {len(results) + 1} service integration cases passed.", flush=True)
    finally:
        for proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
