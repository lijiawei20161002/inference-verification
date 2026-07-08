import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

T=torch.load(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"),"robust_temp_results.pt"),weights_only=False)
Sk=torch.load(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"),"robust_stack_results.pt"),weights_only=False)
temps=[r[0] for r in T["table"]]; ce=[r[2] for r in T["table"]]; lr=[r[3] for r in T["table"]]
hon=T["hon_nll"]; nll=[r[1] for r in T["table"]]
# temperature whose mean-NLL matches honest (adversarial operating point)
match_t=temps[int(np.argmin([abs(n-hon) for n in nll]))]

fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(temps,ce,"o-",color="#888",lw=2,label="mean cross-entropy detector")
ax[0].plot(temps,lr,"D-",color="#d62728",lw=2.5,label="LR feature panel (0.5B+1.5B)")
ax[0].axvline(match_t,color="k",ls="--",alpha=0.6)
ax[0].annotate("attacker tunes T here\n(matches honest mean-NLL)",xy=(match_t,0.66),
               xytext=(match_t+0.15,0.72),fontsize=9,arrowprops=dict(arrowstyle="->"))
ax[0].axhline(0.5,color="k",ls=":",alpha=0.5)
ax[0].set_xlabel("attacker's 0.5B sampling temperature"); ax[0].set_ylabel("detection AUC")
ax[0].set_ylim(0.45,1.02); ax[0].set_title("Adaptive attacker: tune temperature to hide\n(honest = 7B @ T=1.0)")
ax[0].legend(loc="lower right",fontsize=9); ax[0].grid(alpha=0.3)

labels=["teacher-forced\nagreement (7B)","stacks\ndistinguishable\n(CV-AUC)","false-pos on\nalt stack\n(5% target)","detect cheat\nunder alt stack\n(AUC)"]
vals=[Sk["tf_agree"],Sk["aucB"],Sk["fpr_alt"],Sk["aucD"]]
colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
b=ax[1].bar(range(4),vals,color=colors)
for i,v in enumerate(vals): ax[1].text(i,v+0.02,f"{v:.2f}",ha="center",fontsize=10)
ax[1].axhline(0.5,color="k",ls=":",alpha=0.5); ax[1].axhline(0.05,color="green",ls=":",alpha=0.5)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels,fontsize=8.5); ax[1].set_ylim(0,1.1)
ax[1].set_title("Benign cross-stack (bf16 sdpa/batch48 -> eager/batch8)")
fig.suptitle("Robustness: adaptive-temperature attack (left) and benign stack shift (right)",fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","docs","figures","fig_difr_robustness.png"),dpi=130); print("saved")
