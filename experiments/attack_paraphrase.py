import json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import detlib as D
import os
EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); B=48; qtok=D.qtok
prompts=json.load(open(f"{EXP}/prompts.json"))

# attacker's raw cheap answers (0.5B greedy)
sub=torch.load(f"{EXP}/gen_sub_0.5b.pt",weights_only=False)
raw=[qtok.decode(c,skip_special_tokens=True) for c in sub["completion_ids"]]

# attacker paraphrases its own cheap output with the SAME 0.5B (stays 0.5B-tier, still cheap)
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",padding_side="left")
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",dtype=torch.bfloat16,device_map="cuda").eval()
para=[]
for i in range(0,len(raw),B):
    ch=raw[i:i+B]
    msgs=[[{"role":"user","content":f"Rewrite the following text using different wording but keeping the same meaning. Output only the rewrite.\n\n{a}"}] for a in ch]
    txt=[tok.apply_chat_template(mm,tokenize=False,add_generation_prompt=True) for mm in msgs]
    enc=tok(txt,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g=m.generate(**enc,max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id)
    for b in range(len(ch)):
        para.append(tok.decode(g[b][enc["input_ids"].shape[1]:],skip_special_tokens=True))
del m; torch.cuda.empty_cache()

# served completion = paraphrase, conditioned on the ORIGINAL user prompt (Qwen template)
prompt_ids=[qtok(qtok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True),
                 add_special_tokens=False)["input_ids"] for p in prompts]
comp_ids=[qtok(t,add_special_tokens=False)["input_ids"][:128] for t in para]

P05=D.score_proxy("Qwen/Qwen2.5-0.5B-Instruct",prompt_ids,comp_ids)
P15=D.score_proxy("Qwen/Qwen2.5-1.5B-Instruct",prompt_ids,comp_ids)
F05,F15=D.load_feats()

Xh=D.panel(F05["honest"],F15["honest"])
Xpar=D.panel(P05,P15)
Xraw=D.panel(F05["sub_0.5b"],F15["sub_0.5b"])
print("PARAPHRASE ATTACK: honest 7B  vs  0.5B answer paraphrased by 0.5B")
print(f"  detection AUC, raw 0.5B (no paraphrase)  panel = {D.cv_auc(Xh,Xraw):.3f}")
print(f"  detection AUC, paraphrased 0.5B          panel = {D.cv_auc(Xh,Xpar):.3f}")
print(f"  detection AUC, paraphrased 0.5B      mean-CE   = {D.ce_auc(F05['honest'],P05):.3f}")
torch.save({"auc_raw":float(D.cv_auc(Xh,Xraw)),"auc_para":float(D.cv_auc(Xh,Xpar))},
           f"{EXP}/attack_paraphrase_results.pt")
