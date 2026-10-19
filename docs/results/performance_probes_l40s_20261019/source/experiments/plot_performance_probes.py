"""Plot archived performance probes; bands are block ranges, not confidence intervals."""
from __future__ import annotations
import argparse, collections, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def plot(root, destination):
    destination.mkdir(parents=True,exist_ok=True)
    data=json.loads((root/"summary.json").read_text())
    colors={"full":"#2459A6","truncate":"#E17B32","cache_cold":"#2459A6","cache_hit":"#29946B"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharex=True)
    for ax,batch in zip(axes,(1,4,8)):
        for arm in ("full","truncate"):
            gs=sorted([g for g in data["groups"] if g["suite"]=="p1" and g["batch"]==batch
                       and g["condition"]==arm],key=lambda g:g["length"])
            xs=[g["length"] for g in gs]
            ax.plot(xs,[g["decode_ms_per_step_median"] for g in gs],"o-",color=colors[arm],
                    label="Full context" if arm=="full" else "Keep last 2,048 prompt tokens")
            ax.fill_between(xs,[g["decode_ms_per_step_min"] for g in gs],
                            [g["decode_ms_per_step_max"] for g in gs],color=colors[arm],alpha=.15)
        ax.set_xscale("log",base=2);ax.set_xticks([512,2048,8192,16384],["512","2K","8K","16K"])
        ax.set_xlabel("Original prompt length");ax.set_title(f"Actual batch = {batch}")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Decode wall time / step (ms)")
    axes[1].legend(fontsize=8)
    fig.suptitle("L40S / Qwen3-1.7B / BF16 / static KV / CUDA Graph")
    fig.text(.5,.005,"Eight text blocks across three process sessions; shading = observed block range. No deployment calibration.",
             ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ("png","svg"):
        fig.savefig(destination/f"response_surface.{ext}",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    labels=[]
    for i,(length,batch) in enumerate([(2048,1),(2048,4),(8192,1),(8192,4)]):
        labels.append(f"L={length//1024}K\nB={batch}")
        for arm,dx in (("cache_cold",-.18),("cache_hit",.18)):
            g=next(g for g in data["groups"] if g["suite"]=="p4" and g["length"]==length
                   and g["batch"]==batch and g["condition"]==arm)
            for ax,field in zip(axes,("prefill_ms_median","decode_ms_per_step_median")):
                ax.bar(i+dx,g[field],width=.34,color=colors[arm],label=arm if i==0 else None)
    for ax in axes:
        ax.set_xticks(range(4),labels);ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("Prefill wall time (ms)");axes[1].set_ylabel("Decode wall time / step (ms)")
    axes[0].legend(["Cold prefix","Reused prefix"],fontsize=9)
    fig.suptitle("Explicit prefix KV reuse: prefill savings versus decode response")
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(destination/f"prefix_cache.{ext}",dpi=180)
    plt.close(fig)

    if data["http"]:
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        for i,(arm,chunk) in enumerate([("full",1),("full",8),("truncate",1),("truncate",8)]):
            rs=[r for r in data["http"] if r["length"]==8192 and r["batch"]==1
                and r["condition"]==arm and r["chunk"]==chunk]
            for ax,field in zip(axes,("ttft_ms","event_gap_ms_median")):
                ax.bar(i,statistics.median(r[field] for r in rs),color=colors[arm])
                ax.scatter([i]*len(rs),[r[field] for r in rs],s=10,color="black",alpha=.4)
        for ax in axes:
            ax.set_xticks(range(4),["Full\n1 token/event","Full\n8 tokens/event",
                                  "Truncated\n1 token/event","Truncated\n8 tokens/event"], fontsize=8)
            ax.grid(axis="y",alpha=.15)
        axes[0].set_ylabel("First content event (ms)");axes[1].set_ylabel("Median event gap (ms)")
        fig.suptitle("Loopback HTTP / eager HF / 32 outputs / B=1 / L=8K")
        fig.tight_layout()
        for ext in ("png","svg"):
            fig.savefig(destination/f"http_buffering.{ext}",dpi=180)
        plt.close(fig)

    profiles=rows(root/"profile/measurements.jsonl")
    kernel_rows=[]
    for r in profiles:
        categories=collections.defaultdict(float)
        for name,k in r["kernels"].items():
            category=("attention" if "fmha" in name or "flash_" in name else
                      "matrix_vector" if "gemvx" in name else
                      "tensorcore_gemm" if "wmma_tensorop" in name else
                      "concat" if "CatArray" in name else
                      "copy" if "direct_copy" in name else "other")
            categories[category]+=k["duration_us"]/1000/(r["outputs"]-1)
        kernel_rows.append({"case":r["case_id"],"length":r["length"],"batch":r["batch"],
                            "cache":r["cache"],"execution":r["execution"],
                            "kernel_ms_per_step":dict(categories)})
    (destination/"kernel_breakdown.json").write_text(json.dumps(kernel_rows,indent=2))
    fig,ax=plt.subplots(figsize=(10,4.1))
    cats=["matrix_vector","tensorcore_gemm","attention","copy","concat","other"]
    palette=["#2459A6","#7296CB","#29946B","#E17B32","#CE4C4C","#ADB5BD"]
    bottom=[0.0]*len(kernel_rows)
    for category,color in zip(cats,palette):
        values=[r["kernel_ms_per_step"].get(category,0) for r in kernel_rows]
        ax.bar(range(len(values)),values,bottom=bottom,label=category,color=color)
        bottom=[a+b for a,b in zip(bottom,values)]
    labels=[f"L={r['length']} B={r['batch']}\n{r['cache']} / {r['execution']}" for r in kernel_rows]
    ax.set_xticks(range(len(labels)),labels,fontsize=8)
    ax.set_ylabel("Summed CUDA kernel duration / step (ms)")
    ax.set_title("Separate profiled runs: kernel-name categories (4 decode steps)")
    ax.legend(ncol=3,fontsize=8);ax.grid(axis="y",alpha=.15)
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(destination/f"kernel_breakdown.{ext}",dpi=180)
    plt.close(fig)

    # Matplotlib SVG paths contain trailing spaces; normalize generated text.
    for name in ("response_surface", "prefix_cache", "http_buffering", "kernel_breakdown"):
        path = destination / (name + ".svg")
        if path.exists():
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();plot(args.root,args.output)
