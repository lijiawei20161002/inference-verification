import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); KS=[1,2,4,8,16,32,64,128]
L=lambda p:torch.load(f"{EXP}/{p}",weights_only=False)
F05=L("feats_proxy_0.5b.pt"); F15=L("feats_proxy_1.5b.pt")
C05=L("feats_cross_proxy_0.5b.pt"); C15=L("feats_cross_proxy_1.5b.pt")

def feat(r,k):
    n=min(k,len(r["nll"]))
    nll=r["nll"][:n].astype(float); rk=r["rank"][:n].astype(float); en=r["ent"][:n].astype(float)
    lr=np.log1p(rk)
    return [nll.mean(),nll.std(),nll.max(),lr.mean(),
            (rk==0).mean(),(rk<=2).mean(),(rk>=8).mean(),en.mean()]

def matrix(recs,k):
    return np.array([feat(r,k) for r in recs if r is not None and len(r["nll"])>0])

def cv_auc(neg_recs_list,pos_recs_list,k):
    # each *_list is a list of feature-source recs (one per proxy) aligned by index
    negs=[matrix(r,k) for r in neg_recs_list]; poss=[matrix(r,k) for r in pos_recs_list]
    Xn=np.hstack(negs); Xp=np.hstack(poss)
    X=np.vstack([Xn,Xp]); y=np.r_[np.zeros(len(Xn)),np.ones(len(Xp))]
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    s=cross_val_predict(clf,X,y,cv=cv,method="decision_function")
    return roc_auc_score(y,s)

def single_stat(neg,pos,k,which):
    def g(recs):
        o=[]
        for r in recs:
            if r is None or len(r["nll"])==0: continue
            n=min(k,len(r["nll"]))
            o.append(np.log1p(r["rank"][:n]).mean() if which=="logrank" else r["nll"][:n].mean())
        return np.array(o)
    a,b=g(neg),g(pos); v=roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[a,b]); return max(v,1-v)

def panel_auc(tier,k):
    # 0.5B+1.5B two-proxy panel: honest 7B vs the given served tier, 5-fold CV AUC.
    return cv_auc([F05["honest"],F15["honest"]],[F05[tier],F15[tier]],k)

# Served-config metadata for the capability-gradient view. `params` = served model's
# parameter count; `ratio` = how many× smaller than the claimed Qwen2.5-7B (7.62B).
# quant_4bit keeps the 7B weights (ratio 1×) but at 4-bit — the subtle, low-savings attack.
P_CLAIMED = 7.62e9
SERVED = {
    "sub_0.5b":   dict(params=0.494e9, label="Qwen2.5-0.5B"),
    "sub_1.5b":   dict(params=1.54e9,  label="Qwen2.5-1.5B"),
    "sub_3b":     dict(params=3.09e9,  label="Qwen2.5-3B"),
    "quant_4bit": dict(params=7.62e9,  label="Qwen2.5-7B nf4"),
}
for _c in SERVED.values():
    _c["ratio"] = P_CLAIMED/_c["params"]

def row(name,fn):
    print(f"  {name:34s}"+"".join(f"{fn(k):7.3f}" for k in KS))

def main():
    print("tokens:"+" "*29+"".join(f"{k:>7}" for k in KS))
    print("\n[SAME-FAMILY]  honest 7B  vs  substitution → smaller Qwen2.5")
    row("baseline: log-rank (0.5B)",      lambda k: single_stat(F05["honest"],F05["sub_0.5b"],k,"logrank"))
    row("LR features (0.5B proxy)",        lambda k: cv_auc([F05["honest"]],[F05["sub_0.5b"]],k))
    row("LR panel (0.5B+1.5B) → 0.5B",     lambda k: panel_auc("sub_0.5b",k))
    row("LR panel (0.5B+1.5B) → 1.5B",     lambda k: panel_auc("sub_1.5b",k))
    row("LR panel (0.5B+1.5B) → 3B",       lambda k: panel_auc("sub_3b",k))

    print("\n[CROSS-FAMILY]  honest 7B  vs  substitution → SmolLM2-360M")
    row("baseline: log-rank (0.5B)",      lambda k: single_stat(C05["honest"],C05["smol360"],k,"logrank"))
    row("LR features (0.5B proxy)",        lambda k: cv_auc([C05["honest"]],[C05["smol360"]],k))
    row("LR panel (0.5B+1.5B)",            lambda k: cv_auc([C05["honest"],C15["honest"]],[C05["smol360"],C15["smol360"]],k))

    print("\n[SANITY]  honest 7B  vs  4-bit quant of 7B  (should stay ~chance)")
    row("LR panel (0.5B+1.5B)",            lambda k: panel_auc("quant_4bit",k))

    # Dose-response: panel detectability vs. how far the served model is downgraded.
    print("\n[CAPABILITY GRADIENT]  panel AUC vs served-model size  (honest 7B = negative)")
    print(f"  {'served':16s}{'params':>9}{'× smaller':>11}{'AUC@32':>9}{'AUC@128':>9}")
    for tier,meta in SERVED.items():
        print(f"  {meta['label']:16s}{meta['params']/1e9:8.2f}B{meta['ratio']:10.1f}×"
              f"{panel_auc(tier,32):9.3f}{panel_auc(tier,128):9.3f}")

if __name__=="__main__":
    main()
