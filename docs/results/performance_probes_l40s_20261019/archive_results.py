"""Archive completed measurements after revalidating both experiment suites."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("--supplement",type=Path,required=True)
    args=p.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    queue=json.loads((args.source/"queue.json").read_text())
    assert queue["status"]=="complete"
    subprocess.run([sys.executable,"-m","experiments.analyze_performance_probes",
                    str(args.source),"--output",str(args.source/"recomputed_summary.json")],check=True)
    original=json.loads((args.source/"summary.json").read_text())
    recomputed=json.loads((args.source/"recomputed_summary.json").read_text())
    assert original==recomputed
    assert original["status_counts"]=={"complete":524}
    assert len(original["http"])==42
    assert original["checks"]=={"negative_control_pairs":48,"prefix_cache_pairs":32}
    for verified in original["verified_files"]:
        assert verified["source_status"]==""
        assert verified["source_commit"]==queue["source_commit"]
    supplement=[json.loads(x) for x in (args.supplement/"measurements.jsonl").read_text().splitlines()]
    assert len(supplement)==96
    assert all(r["status"]=="complete" for r in supplement)
    shutil.copytree(args.source,args.destination)
    shutil.copytree(args.supplement,args.destination/"sdpa_representation")
    shutil.copy2(__file__,args.destination/"archive_results.py")
    print(json.dumps({"destination":str(args.destination),"main_rows":524,"http_requests":42,
                      "supplement_rows":96,"recomputed_summary":"exact_match"}))

if __name__=="__main__":
    main()
