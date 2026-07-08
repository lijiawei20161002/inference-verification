import os, json, importlib.util
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

spec=importlib.util.spec_from_file_location("score2",os.path.join(os.path.dirname(os.path.abspath(__file__)),"score2.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); B=48; qtok=S.qtok
prompts=json.load(open(f"{EXP}/prompts.json"))
texts=[qtok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) for p in prompts]

def gen(model_id, temp):
    tok=AutoTokenizer.from_pretrained(model_id,padding_side="left")
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.bfloat16,device_map="cuda").eval()
    P,C=[],[]
    for i in range(0,len(texts),B):
        ch=texts[i:i+B]
        enc=tok(ch,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        torch.manual_seed(0)
        with torch.no_grad():
            g=m.generate(**enc,max_new_tokens=128,do_sample=True,temperature=temp,top_p=1.0,top_k=0,
                         pad_token_id=tok.pad_token_id)
        pl=enc["input_ids"].shape[1]
        for b in range(len(ch)):
            real=enc["input_ids"][b][enc["attention_mask"][b].bool()].tolist()
            comp=g[b][pl:].tolist()
            if tok.eos_token_id in comp: comp=comp[:comp.index(tok.eos_token_id)+1]
            P.append(real); C.append(comp)
    del m; torch.cuda.empty_cache()
    return P,C

# 1) honest provider: real 7B at temperature 1.0
hp,hc=gen("Qwen/Qwen2.5-7B-Instruct",1.0)
# 2) attacker: 0.5B at a sweep of temperatures
TEMPS=[0.3,0.5,0.7,0.9,1.1,1.3,1.5]
cheat={t:gen("Qwen/Qwen2.5-0.5B-Instruct",t) for t in TEMPS}

# score everything under both proxies (rich features)
def score_all(pid):
    m=AutoModelForCausalLM.from_pretrained(pid,dtype=torch.bfloat16,device_map="cuda").eval()
    out={"honest":S.score(m,hp,hc)}
    for t in TEMPS: out[t]=S.score(m,cheat[t][0],cheat[t][1])
    del m; torch.cuda.empty_cache(); return out
R05=score_all("Qwen/Qwen2.5-0.5B-Instruct"); R15=score_all("Qwen/Qwen2.5-1.5B-Instruct")

def mean_nll(recs):
    return np.mean([r["nll"].mean() for r in recs if r and len(r["nll"])>0])
def feats(recs):
    X=[]
    for r in recs:
        if not r or len(r["nll"])==0: continue
        nll=r["nll"].astype(float); rk=r["rank"].astype(float); en=r["ent"].astype(float); lr=np.log1p(rk)
        X.append([nll.mean(),nll.std(),nll.max(),lr.mean(),(rk==0).mean(),(rk<=2).mean(),(rk>=8).mean(),en.mean()])
    return np.array(X)
def auc_ce(neg,pos):
    a=np.array([r["nll"].mean() for r in neg if r and len(r["nll"])>0])
    b=np.array([r["nll"].mean() for r in pos if r and len(r["nll"])>0])
    v=roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[a,b]); return max(v,1-v)
def auc_lr(neg_list,pos_list):
    Xn=np.hstack([feats(r) for r in neg_list]); Xp=np.hstack([feats(r) for r in pos_list])
    X=np.vstack([Xn,Xp]); y=np.r_[np.zeros(len(Xn)),np.ones(len(Xp))]
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    s=cross_val_predict(clf,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="decision_function")
    return roc_auc_score(y,s)

hon_nll=mean_nll(R05["honest"])
print(f"honest 7B@T1.0 mean-NLL under 0.5B proxy = {hon_nll:.3f}\n")
print(f"{'0.5B temp':>10}{'cheat meanNLL':>15}{'AUC mean-CE':>13}{'AUC LR panel':>14}")
best_t=None; best_gap=1e9
for t in TEMPS:
    cn=mean_nll(R05[t]); gap=abs(cn-hon_nll)
    ce=auc_ce(R05["honest"],R05[t]); lr=auc_lr([R05["honest"],R15["honest"]],[R05[t],R15[t]])
    print(f"{t:>10}{cn:>15.3f}{ce:>13.3f}{lr:>14.3f}")
    if gap<best_gap: best_gap=gap; best_t=t
print(f"\nADVERSARIAL CHOICE: attacker picks 0.5B temp={best_t} (mean-NLL closest to honest, gap={best_gap:.3f})")
print(f"  -> mean-CE detector AUC = {auc_ce(R05['honest'],R05[best_t]):.3f}   (attacker's target: collapse this)")
print(f"  -> LR panel   detector AUC = {auc_lr([R05['honest'],R15['honest']],[R05[best_t],R15[best_t]]):.3f}")
torch.save({"temps":TEMPS,"hon_nll":float(hon_nll),"best_t":best_t,
            "table":[(t,float(mean_nll(R05[t])),float(auc_ce(R05['honest'],R05[t])),
                      float(auc_lr([R05['honest'],R15['honest']],[R05[t],R15[t]]))) for t in TEMPS]},
           f"{EXP}/robust_temp_results.pt")
