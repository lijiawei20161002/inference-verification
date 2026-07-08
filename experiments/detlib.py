import importlib.util
import numpy as np, torch
from transformers import AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

spec=importlib.util.spec_from_file_location("score2",os.path.join(os.path.dirname(os.path.abspath(__file__)),"score2.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); qtok=S.qtok

def score_proxy(pid, prompt_ids, comp_ids):
    m=AutoModelForCausalLM.from_pretrained(pid,dtype=torch.bfloat16,device_map="cuda").eval()
    r=S.score(m,prompt_ids,comp_ids); del m; torch.cuda.empty_cache(); return r

def feats(recs):
    X=[]
    for r in recs:
        if not r or len(r["nll"])==0: continue
        nll=r["nll"].astype(float); rk=r["rank"].astype(float); en=r["ent"].astype(float); lr=np.log1p(rk)
        X.append([nll.mean(),nll.std(),nll.max(),lr.mean(),(rk==0).mean(),(rk<=2).mean(),(rk>=8).mean(),en.mean()])
    return np.array(X)

def panel(*recs_lists): return np.hstack([feats(r) for r in recs_lists])

def cv_auc(Xneg, Xpos):
    X=np.vstack([Xneg,Xpos]); y=np.r_[np.zeros(len(Xneg)),np.ones(len(Xpos))]
    s=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),
                        X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="decision_function")
    return roc_auc_score(y,s)

def ce_auc(neg, pos):
    a=np.array([r["nll"].mean() for r in neg if r and len(r["nll"])>0])
    b=np.array([r["nll"].mean() for r in pos if r and len(r["nll"])>0])
    v=roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[a,b]); return max(v,1-v)

def load_feats():
    return (torch.load(f"{EXP}/feats_proxy_0.5b.pt",weights_only=False),
            torch.load(f"{EXP}/feats_proxy_1.5b.pt",weights_only=False))
