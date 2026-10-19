"""Run the L40S mechanism pilot sequentially with a host lock and raw logs."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from scripts.run_research_job import host_lock

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise RuntimeError("Commit the implementation before running the archived pilot")
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
             OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0")
    jobs=[
        ("preflight",["experiments.exp_performance_probe_gpu","--suite","preflight"]),
        ("profile",["experiments.exp_performance_probe_gpu","--suite","profile"]),
    ]
    for session,blocks in enumerate(("0,1,2","3,4,5","6,7")):
        jobs.append((f"session_{session}",["experiments.exp_performance_probe_gpu",
            "--suite","pilot","--session",str(session),"--blocks",blocks]))
    for session in range(3):
        jobs.append((f"http_{session}",["experiments.exp_performance_probe_http","--session",str(session)]))
    state={"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
           "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"jobs":[]}
    def save():
        (args.output/"queue.json").write_text(json.dumps(state,indent=2))
    with host_lock(Path(tempfile.gettempdir())/f"ivgym-research-{os.getuid()}.lock"):
        active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid",
                                        "--format=csv,noheader"],text=True).strip()
        if active:
            raise RuntimeError("GPU already has active compute processes: "+active)
        probe=["/usr/local/cuda/bin/ncu","--metrics",
               "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum",
               "--launch-count","1",sys.executable,"-c",
               'import torch; x=torch.ones(1024,device="cuda"); x.add_(1); torch.cuda.synchronize()']
        with (args.output/"counter_permission.log").open("w") as log:
            status=subprocess.run(probe,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        log=(args.output/"counter_permission.log").read_text()
        env["IVGYM_COUNTER_STATUS"]=("available" if status==0 else
            "ERR_NVGPUCTRPERM" if "ERR_NVGPUCTRPERM" in log else f"probe_failed_exit_{status}")
        state["counter_status"]=env["IVGYM_COUNTER_STATUS"]
        save()
        for name,job in jobs:
            command=[sys.executable,"-m",*job,"--output",str(args.output/name)]
            entry={"name":name,"command":command,"status":"running"}
            state["jobs"].append(entry);save()
            print("START "+name,flush=True)
            start=time.perf_counter()
            with (args.output/(name+".log")).open("x") as out:
                process=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
            entry.update({"returncode":process.returncode,"elapsed_s":time.perf_counter()-start,
                          "status":"complete" if process.returncode==0 else "failed"})
            save()
            print(json.dumps(entry),flush=True)
            if process.returncode:
                raise SystemExit(process.returncode)
        command=[sys.executable,"-m","experiments.analyze_performance_probes",str(args.output),
                 "--output",str(args.output/"summary.json")]
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        state["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        state["status"]="complete";save()

if __name__=="__main__":
    main()
