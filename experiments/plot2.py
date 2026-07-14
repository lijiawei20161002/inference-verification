import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
KS=[1,2,4,8,16,32,64,128]
same_base=[0.697,0.698,0.747,0.793,0.861,0.909,0.922,0.923]
same_05=  [0.742,0.734,0.777,0.845,0.935,0.981,0.988,0.991]
same_pan= [0.733,0.761,0.805,0.888,0.975,0.994,0.996,0.997]
cross_base=[0.666,0.651,0.654,0.654,0.603,0.535,0.548,0.646]
cross_pan=[0.698,0.671,0.657,0.706,0.711,0.778,0.845,0.899]
quant_pan=[0.420,0.406,0.426,0.433,0.436,0.462,0.434,0.462]

fig,ax=plt.subplots(1,2,figsize=(13,5),sharey=True)
ax[0].plot(KS,same_base,"o--",color="#888",lw=2,label="baseline: mean log-rank (0.5B)")
ax[0].plot(KS,same_05,"s-",color="#1f77b4",lw=2,label="LR features (0.5B proxy)")
ax[0].plot(KS,same_pan,"D-",color="#d62728",lw=2.5,label="LR features (0.5B+1.5B panel)")
ax[0].set_title("Same-tier substitution  (7B → 0.5B)")
ax[1].plot(KS,cross_base,"o--",color="#888",lw=2,label="baseline: mean log-rank (0.5B)")
ax[1].plot(KS,cross_pan,"D-",color="#d62728",lw=2.5,label="LR features (0.5B+1.5B panel)")
ax[1].plot(KS,quant_pan,"x:",color="#2ca02c",lw=2,label="4-bit quant (subtle) — panel, stays chance")
ax[1].set_title("Cross-family substitution  (7B → SmolLM2-360M)")
for a in ax:
    a.axhline(0.5,color="k",ls=":",alpha=0.5); a.set_xscale("log",base=2); a.set_xticks(KS); a.set_xticklabels(KS)
    a.set_ylim(0.4,1.0); a.set_xlabel("scored tokens"); a.grid(alpha=0.3); a.legend(fontsize=9,loc="lower right")
ax[0].set_ylabel("detection AUC (5-fold CV)")
fig.suptitle("Pushing the cheap proxy higher: richer features + 2-proxy panel  (panel = 27% of one 7B forward pass)",fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","docs","figures","fig_difr_detect.png"),dpi=130); print("saved")
