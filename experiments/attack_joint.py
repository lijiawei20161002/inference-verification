import json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import detlib as D
import os
EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); B=48; qtok=D.qtok
prompts=json.load(open(f"{EXP}/prompts.json"))
texts=[qtok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) for p in prompts]

# honest reference = 7B greedy (already scored)
F05,F15=D.load_feats(); Xh=D.panel(F05["honest"],F15["honest"])

# attacker sweeps ALL sampling knobs it controls on the 0.5B, seeking the config the
# defender's full-feature panel separates LEAST well (joint evasion, not just mean-CE).
TEMPS=[0.6,0.8,1.0]; TOPPS=[0.85,0.92,1.0]
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",padding_side="left")
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",dtype=torch.bfloat16,device_map="cuda").eval()
def gen(temp,topp):
    P,C=[],[]
    for i in range(0,len(texts),B):
        ch=texts[i:i+B]; enc=tok(ch,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        torch.manual_seed(0)
        with torch.no_grad():
            g=m.generate(**enc,max_new_tokens=128,do_sample=True,temperature=temp,top_p=topp,top_k=0,
                         pad_token_id=tok.pad_token_id)
        pl=enc["input_ids"].shape[1]
        for b in range(len(ch)):
            real=enc["input_ids"][b][enc["attention_mask"][b].bool()].tolist()
            comp=g[b][pl:].tolist()
            if tok.eos_token_id in comp: comp=comp[:comp.index(tok.eos_token_id)+1]
            P.append(real); C.append(comp)
    return P,C
sweeps={(t,tp):gen(t,tp) for t in TEMPS for tp in TOPPS}
del m; torch.cuda.empty_cache()

results=[]
for (t,tp),(P,C) in sweeps.items():
    r05=D.score_proxy("Qwen/Qwen2.5-0.5B-Instruct",P,C)
    r15=D.score_proxy("Qwen/Qwen2.5-1.5B-Instruct",P,C)
    Xc=D.panel(r05,r15)
    ce=D.ce_auc(F05["honest"],r05); pan=D.cv_auc(Xh,Xc)
    results.append((t,tp,ce,pan)); print(f"  temp={t} top_p={tp}:  mean-CE AUC={ce:.3f}  panel AUC={pan:.3f}",flush=True)

best=min(results,key=lambda r:r[3])   # attacker-optimal = min panel AUC for defender
print(f"\nATTACKER-OPTIMAL sampling config (lowest defender panel AUC): temp={best[0]} top_p={best[1]}")
print(f"  -> mean-CE detector AUC = {best[2]:.3f}")
print(f"  -> LR panel   detector AUC = {best[3]:.3f}  (if still high, no sampling-based evasion works)")
torch.save({"grid":results,"best":best},f"{EXP}/attack_joint_results.pt")
