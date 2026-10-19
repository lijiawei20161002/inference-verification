"""Supplementary causal control: equivalent SDPA representations, identical Q/K/V.

No mask and an all-True mask denote the same fully visible key range for a single
decode query. Compare native GQA, explicit KV repetition and all-True masking.
All timings include repetition when needed. Kernel profiling is separate.
"""
from __future__ import annotations
import argparse, collections, gzip, hashlib, json, random, subprocess, time
from pathlib import Path
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa


def repeat_kv(x):
    b,h,l,d=x.shape
    return x[:,:,None,:,:].expand(b,h,2,l,d).reshape(b,h*2,l,d)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    arms=["native_gqa","repeat_no_mask","repeat_all_true_mask"]
    plan=[(block,batch,length) for block in range(8) for batch in (1,4) for length in (512,8192)]
    random.Random(20261020).shuffle(plan)
    info={"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"cuda":torch.version.cuda,
          "shapes":{"query_heads":16,"kv_heads":8,"head_dim":128,"queries":1},
          "blocks":8,"replays_per_timing":100,"dtype":"bfloat16",
          "scope":"single-layer SDPA mechanism control; randomized inputs; not an LLM detector",
          "equivalence":"all keys visible; is_causal=False; None and all-True Boolean mask",
          "planned_error_tolerance":{"atol":0.005,"rtol":0.02},
          "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    (args.output/"plan.json").write_text(json.dumps({"plan":plan,"arms":arms,"metadata":info},indent=2))
    with torch.inference_mode(), (args.output/"measurements.jsonl").open("x") as f:
        for block,batch,length in plan:
            torch.manual_seed(20261020+block*100+batch+length)
            q=torch.randn(batch,16,1,128,device="cuda",dtype=torch.bfloat16)
            k=torch.randn(batch,8,length,128,device="cuda",dtype=torch.bfloat16)
            v=torch.randn_like(k)
            mask=torch.ones((1,1,1,length),device="cuda",dtype=torch.bool)
            # FP32 arithmetic reference with explicit softmax; outside timing.
            kr,vr=repeat_kv(k).float(),repeat_kv(v).float()
            ref=((q.float() @ kr.transpose(-1,-2))/128**0.5).softmax(-1) @ vr
            order=list(arms);random.Random(20261020+block*100+batch+length).shuffle(order)
            for arm in order:
                def compute():
                    if arm=="native_gqa":
                        return sdpa(q,k,v,is_causal=False,enable_gqa=True)
                    return sdpa(q,repeat_kv(k),repeat_kv(v),
                        attn_mask=mask if arm=="repeat_all_true_mask" else None,is_causal=False)
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        out=compute()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out=compute()
                graph.replay()  # Capture records operations; replay materializes its output.
                torch.cuda.synchronize()
                torch.testing.assert_close(out.float(),ref,atol=0.005,rtol=0.02)
                maximum=float((out.float()-ref).abs().max())
                means=[]
                for rep in range(5):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(100):
                        graph.replay()
                    end.record();end.synchronize()
                    means.append(begin.elapsed_time(end)/100)
                row={"block":block,"batch":batch,"length":length,"arm":arm,
                    "ms_per_call":means,"max_abs_error_vs_fp32":maximum,"status":"complete"}
                if block==0 and batch==4 and length==8192:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(4):
                            graph.replay()
                        torch.cuda.synchronize()
                    kernels=collections.defaultdict(lambda:{"count":0,"duration_us":0.0})
                    for e in prof.events():
                        if e.device_type==torch.autograd.DeviceType.CUDA:
                            kernels[e.name]["count"]+=1
                            kernels[e.name]["duration_us"]+=e.device_time_total
                    row["kernels"]=dict(kernels)
                    trace=args.output/(arm+".json")
                    prof.export_chrome_trace(str(trace))
                    with gzip.open(str(trace)+".gz","wb") as dst:
                        dst.write(trace.read_bytes())
                    trace.unlink()
                f.write(json.dumps(row)+"\n");f.flush()
                print(json.dumps({k:row[k] for k in ("block","batch","length","arm","ms_per_call")}),flush=True)
                del graph,compute,out
            del q,k,v,mask,kr,vr,ref
            torch.cuda.empty_cache()
    info["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    (args.output/"manifest.json").write_text(json.dumps(info,indent=2))

if __name__=="__main__":
    main()
