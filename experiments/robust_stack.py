import os, json, importlib.util
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

spec=importlib.util.spec_from_file_location("score2",os.path.join(os.path.dirname(os.path.abspath(__file__)),"score2.py"))
S=importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); B=48; qtok=S.qtok
prompts=json.load(open(f"{EXP}/prompts.json"))
texts=[qtok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) for p in prompts]

# honest provider on a BENIGN different stack: bf16 + EAGER attention + smaller batch
# (verifier calibrated on bf16 + sdpa + batch 48). fp16 is excluded: it overflows/breaks Qwen2.5.
ABATCH=8
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",padding_side="left")
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",dtype=torch.bfloat16,
                                       attn_implementation="eager",device_map="cuda").eval()
ap,ac=[],[]
for i in range(0,len(texts),ABATCH):
    ch=texts[i:i+ABATCH]; enc=tok(ch,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g=m.generate(**enc,max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id)
    pl=enc["input_ids"].shape[1]
    for b in range(len(ch)):
        real=enc["input_ids"][b][enc["attention_mask"][b].bool()].tolist()
        comp=g[b][pl:].tolist()
        if tok.eos_token_id in comp: comp=comp[:comp.index(tok.eos_token_id)+1]
        ap.append(real); ac.append(comp)
del m; torch.cuda.empty_cache()

# --- benign-noise level, measured the DiFR way (teacher-forced), not free-running ---
orig=torch.load(f"{EXP}/gen_honest.pt",weights_only=False)
# free-running positional match (harsh: one flip cascades) -- reported for contrast only
fr_match=fr_tot=0
for a,o in zip(ac,orig["completion_ids"]):
    n=min(len(a),len(o)); fr_tot+=n; fr_match+=sum(1 for j in range(n) if a[j]==o[j])
# teacher-forced next-token agreement of the eager-7B on the sdpa-7B's own tokens
mm=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",dtype=torch.bfloat16,
                                        attn_implementation="eager",device_map="cuda").eval()
tf=S.score(mm,orig["prompt_ids"],orig["completion_ids"]); del mm; torch.cuda.empty_cache()
tf_agree=float(np.mean([(r["rank"]==0).mean() for r in tf if r and len(r["rank"])>0]))
print(f"benign stack noise (bf16 sdpa->eager, batch48->8):")
print(f"  teacher-forced next-token agreement (7B) = {100*tf_agree:.2f}%   <- the meaningful metric")
print(f"  free-running positional match            = {100*fr_match/fr_tot:.2f}%   (low: one flip cascades)\n")

# score honest_alt under both proxies (verifier stack unchanged: bf16+sdpa)
def sc(pid,P,C):
    mm=AutoModelForCausalLM.from_pretrained(pid,dtype=torch.bfloat16,device_map="cuda").eval()
    r=S.score(mm,P,C); del mm; torch.cuda.empty_cache(); return r
A05=sc("Qwen/Qwen2.5-0.5B-Instruct",ap,ac); A15=sc("Qwen/Qwen2.5-1.5B-Instruct",ap,ac)

F05=torch.load(f"{EXP}/feats_proxy_0.5b.pt",weights_only=False)
F15=torch.load(f"{EXP}/feats_proxy_1.5b.pt",weights_only=False)
def feats(recs):
    X=[]
    for r in recs:
        if not r or len(r["nll"])==0: continue
        nll=r["nll"].astype(float); rk=r["rank"].astype(float); en=r["ent"].astype(float); lr=np.log1p(rk)
        X.append([nll.mean(),nll.std(),nll.max(),lr.mean(),(rk==0).mean(),(rk<=2).mean(),(rk>=8).mean(),en.mean()])
    return np.array(X)
def panel(a,b): return np.hstack([feats(a),feats(b)])

Xh   = panel(F05["honest"],   F15["honest"])     # honest, calibration stack (bf16+sdpa+batch48)
Xalt = panel(A05,            A15)                # honest, benign different stack (bf16+eager+batch8)
Xche = panel(F05["sub_0.5b"], F15["sub_0.5b"])   # cheat (0.5B substitution)

# (B) can a classifier even tell the two benign honest stacks apart? cross-validated -> want ~0.5
from sklearn.model_selection import cross_val_predict, StratifiedKFold
yB=np.r_[np.zeros(len(Xh)),np.ones(len(Xalt))]; XB=np.vstack([Xh,Xalt])
sB=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)),XB,yB,
                     cv=StratifiedKFold(5,shuffle=True,random_state=0),method="decision_function")
aucB=roc_auc_score(yB,sB)
print(f"(B) CV-AUC honest(sdpa) vs honest(eager)    = {max(aucB,1-aucB):.3f}   (near 0.5 = stacks indistinguishable)")

# (C) deployed detector trained on honest(bf16) vs cheat; false-positive rate on honest(alt stack)
det=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(
    np.vstack([Xh,Xche]),np.r_[np.zeros(len(Xh)),np.ones(len(Xche))])
thr=np.quantile(det.decision_function(Xh),0.95)     # threshold set for 5% FPR on calibration honest
fpr_alt=float((det.decision_function(Xalt)>thr).mean())
print(f"(C) threshold @5% FPR on calibration honest -> FPR on honest(alt stack) = {100*fpr_alt:.1f}%")

# (D) capability detection under provider stack mismatch: honest(alt) vs cheat
yD=np.r_[np.zeros(len(Xalt)),np.ones(len(Xche))]
aucD=roc_auc_score(yD,det.decision_function(np.vstack([Xalt,Xche])))
print(f"(D) AUC honest(alt stack) vs cheat (0.5B)   = {aucD:.3f}   (capability signal survives stack shift)")
torch.save({"tf_agree":tf_agree,"free_running_match":fr_match/fr_tot,"aucB":float(max(aucB,1-aucB)),
            "fpr_alt":fpr_alt,"aucD":float(aucD)},f"{EXP}/robust_stack_results.pt")
