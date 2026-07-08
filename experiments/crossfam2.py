import os, json, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
score2 = importlib.util.spec_from_file_location("score2",os.path.join(os.path.dirname(os.path.abspath(__file__)),"score2.py"))
S = importlib.util.module_from_spec(score2); score2.loader.exec_module(S)

EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); B=48
SERVED="HuggingFaceTB/SmolLM2-360M-Instruct"
qtok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
prompts=json.load(open(f"{EXP}/prompts.json"))

# generate cross-family served text
stok=AutoTokenizer.from_pretrained(SERVED,padding_side="left")
if stok.pad_token is None: stok.pad_token=stok.eos_token
sm=AutoModelForCausalLM.from_pretrained(SERVED,dtype=torch.bfloat16,device_map="cuda").eval()
served=[]
for i in range(0,len(prompts),B):
    ch=prompts[i:i+B]
    txt=[stok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True) for p in ch]
    enc=stok(txt,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g=sm.generate(**enc,max_new_tokens=128,do_sample=False,pad_token_id=stok.pad_token_id)
    for b in range(len(ch)):
        served.append(stok.decode(g[b][enc["input_ids"].shape[1]:],skip_special_tokens=True))
del sm; torch.cuda.empty_cache()
print("served generated")

# build Qwen-tokenized prompt_ids + completion_ids for honest and smol360
hon=torch.load(f"{EXP}/gen_honest.pt",weights_only=False)
prompt_ids=hon["prompt_ids"]; honest_cids=hon["completion_ids"]
smol_cids=[qtok(t,add_special_tokens=False)["input_ids"][:128] for t in served]

for pname,mid in [("proxy_0.5b","Qwen/Qwen2.5-0.5B-Instruct"),("proxy_1.5b","Qwen/Qwen2.5-1.5B-Instruct")]:
    model=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16,device_map="cuda").eval()
    out={"honest":S.score(model,prompt_ids,honest_cids),
         "smol360":S.score(model,prompt_ids,smol_cids)}
    torch.save(out,f"{EXP}/feats_cross_{pname}.pt"); print("saved",pname)
    del model; torch.cuda.empty_cache()
