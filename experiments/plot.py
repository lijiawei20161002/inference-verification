import os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); KS=[1,2,4,8,16,32,64,128]
R=torch.load(f"{EXP}/scores_proxy_0.5b.pt", weights_only=False)
def stat(recs,name,k):
    o=[]
    for r in recs:
        n=min(k,len(r["nll"]))
        if n==0: continue
        o.append(float(r["nll"][:n].mean()) if name=="ce" else
                 float(r["correct"][:n].mean()) if name=="agree" else
                 float(np.log1p(r["rank"][:n]).mean()))
    return np.array(o)
def auc(a,b):
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; v=roc_auc_score(y,np.r_[a,b]); return max(v,1-v)
def curve(cfg,s): return [auc(stat(R["honest"],s,k),stat(R[cfg],s,k)) for k in KS]

cross=[0.664,0.651,0.654,0.654,0.600,0.533,0.548,0.647]  # Qwen0.5B proxy vs SmolLM2-360M (logrank)
plt.figure(figsize=(7.5,5))
plt.plot(KS, curve("sub_0.5b","logrank"), "o-", lw=2, label="sub → Qwen 0.5B (same-tier)  ✓ cheap+good")
plt.plot(KS, cross,                         "s-", lw=2, label="sub → SmolLM2-360M (cross-family)  ~ useful")
plt.plot(KS, curve("sub_3b","logrank"),   "^--", lw=2, label="sub → Qwen 3B (still capable)  ✗")
plt.plot(KS, curve("quant_4bit","ce"),    "x--", lw=2, label="4-bit quant of 7B (subtle)  ✗")
plt.axhline(0.5,color="gray",ls=":"); plt.xscale("log",base=2); plt.xticks(KS,KS)
plt.ylim(0.45,1.0); plt.xlabel("verified tokens"); plt.ylabel("detection AUC")
plt.title("Cheap 0.5B proxy (6.5% of claimed-7B FLOP) as inference verifier")
plt.legend(fontsize=9,loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","docs","figures","fig_difr_summary.png"),dpi=130)
print("saved",os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","docs","figures","fig_difr_summary.png"))
