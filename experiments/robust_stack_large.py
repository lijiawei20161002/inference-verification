import json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import detlib as D
import os
EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); B=16; qtok=D.qtok
prompts=json.load(open(f"{EXP}/prompts.json"))
texts=[qtok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) for p in prompts]

# LARGER benign stack gap: honest 7B in float32 (higher precision, legitimate) vs verifier's bf16 calibration
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",padding_side="left")
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",dtype=torch.float32,device_map="cuda").eval()
ap,ac=[],[]
for i in range(0,len(texts),B):
    ch=texts[i:i+B]; enc=tok(ch,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g=m.generate(**enc,max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id)
    pl=enc["input_ids"].shape[1]
    for b in range(len(ch)):
        real=enc["input_ids"][b][enc["attention_mask"][b].bool()].tolist()
        comp=g[b][pl:].tolist()
        if tok.eos_token_id in comp: comp=comp[:comp.index(tok.eos_token_id)+1]
        ap.append(real); ac.append(comp)
del m; torch.cuda.empty_cache()

orig=torch.load(f"{EXP}/gen_honest.pt",weights_only=False)
# teacher-forced next-token agreement of fp32-7B on bf16-7B's own tokens (benign noise level)
mm=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",dtype=torch.float32,device_map="cuda").eval()
tf=D.S.score(mm,orig["prompt_ids"],orig["completion_ids"]); del mm; torch.cuda.empty_cache()
tf_agree=float(np.mean([(r["rank"]==0).mean() for r in tf if r and len(r["rank"])>0]))
print(f"LARGER stack gap: honest fp32 vs bf16 calibration")
print(f"  teacher-forced next-token agreement (7B) = {100*tf_agree:.2f}%\n")

A05=D.score_proxy("Qwen/Qwen2.5-0.5B-Instruct",ap,ac); A15=D.score_proxy("Qwen/Qwen2.5-1.5B-Instruct",ap,ac)
F05,F15=D.load_feats()
Xh=D.panel(F05["honest"],F15["honest"]); Xalt=D.panel(A05,A15); Xche=D.panel(F05["sub_0.5b"],F15["sub_0.5b"])

aucB=D.cv_auc(Xh,Xalt)
print(f"(B) CV-AUC honest(bf16) vs honest(fp32)     = {max(aucB,1-aucB):.3f}   (near 0.5 = indistinguishable)")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
det=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(
    np.vstack([Xh,Xche]),np.r_[np.zeros(len(Xh)),np.ones(len(Xche))])
thr=np.quantile(det.decision_function(Xh),0.95)
fpr=float((det.decision_function(Xalt)>thr).mean())
from sklearn.metrics import roc_auc_score
aucD=roc_auc_score(np.r_[np.zeros(len(Xalt)),np.ones(len(Xche))],det.decision_function(np.vstack([Xalt,Xche])))
print(f"(C) FPR on honest(fp32) at 5% threshold     = {100*fpr:.1f}%")
print(f"(D) AUC honest(fp32) vs cheat (0.5B)        = {aucD:.3f}")
torch.save({"tf_agree":tf_agree,"aucB":float(max(aucB,1-aucB)),"fpr":fpr,"aucD":float(aucD)},
           f"{EXP}/robust_stack_large_results.pt")
