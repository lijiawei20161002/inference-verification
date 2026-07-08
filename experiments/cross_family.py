import os, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data")
PROXY = "Qwen/Qwen2.5-0.5B-Instruct"       # cheap verifier (Qwen family)
SERVED = "HuggingFaceTB/SmolLM2-360M-Instruct"  # cross-family cheap-tier substitution
KS = [1, 2, 4, 8, 16, 32, 64, 128]

qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
prompts = json.load(open(os.path.join(EXP, "prompts.json")))

# 1) honest completions as TEXT (decode stored Qwen-7B ids)
hon = torch.load(os.path.join(EXP, "gen_honest.pt"), weights_only=False)
honest_text = [qtok.decode(c, skip_special_tokens=True) for c in hon["completion_ids"]]

# 2) generate cross-family served completions as TEXT
stok = AutoTokenizer.from_pretrained(SERVED, padding_side="left")
if stok.pad_token is None: stok.pad_token = stok.eos_token
smodel = AutoModelForCausalLM.from_pretrained(SERVED, dtype=torch.bfloat16, device_map="cuda").eval()
served_text = []
B = 48
t0 = time.time()
for i in range(0, len(prompts), B):
    chunk = prompts[i:i+B]
    txt = [stok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True) for p in chunk]
    enc = stok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = smodel.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=stok.pad_token_id)
    for b in range(len(chunk)):
        served_text.append(stok.decode(g[b][enc["input_ids"].shape[1]:], skip_special_tokens=True))
print(f"generated cross-family served in {time.time()-t0:.0f}s")
del smodel; torch.cuda.empty_cache()

# 3) score BOTH under the Qwen-0.5B proxy, consistent Qwen chat template + completion text
proxy = AutoModelForCausalLM.from_pretrained(PROXY, dtype=torch.bfloat16, device_map="cuda").eval()
pad_id = qtok.pad_token_id or qtok.eos_token_id

@torch.no_grad()
def score(completions):
    recs = []
    for i in range(0, len(prompts), B):
        pc = prompts[i:i+B]; cc = completions[i:i+B]
        pref = [qtok.apply_chat_template([{"role":"user","content":p}], tokenize=False, add_generation_prompt=True) for p in pc]
        pids = [qtok(x, add_special_tokens=False)["input_ids"] for x in pref]
        cids = [qtok(x, add_special_tokens=False)["input_ids"][:128] for x in cc]
        full = [p+c for p,c in zip(pids,cids)]
        L = max(len(f) for f in full)
        ids = torch.full((len(full),L), pad_id); att = torch.zeros((len(full),L),dtype=torch.long)
        for b,f in enumerate(full):
            ids[b,:len(f)]=torch.tensor(f); att[b,:len(f)]=1
        logits = proxy(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
        for b in range(len(full)):
            pl=len(pids[b]); cl=len(cids[b])
            if cl==0:
                recs.append(None); continue
            idx=torch.arange(pl-1,pl+cl-1,device=logits.device)
            lg=logits[b,idx].float(); tgt=torch.tensor(cids[b],device=logits.device)
            tl=lg[torch.arange(cl),tgt].unsqueeze(-1)
            rank=(lg>tl).sum(-1)
            recs.append(dict(rank=rank.cpu().numpy(),
                             nll=(-torch.log_softmax(lg,-1)[torch.arange(cl),tgt]).cpu().numpy(),
                             correct=(rank==0).cpu().numpy()))
    return recs

H = score(honest_text); S = score(served_text)

def stat(recs, name, k):
    out=[]
    for r in recs:
        if r is None or len(r["rank"])==0: continue
        n=min(k,len(r["rank"]))
        if name=="ce": out.append(float(r["nll"][:n].mean()))
        elif name=="agree": out.append(float(r["correct"][:n].mean()))
        else: out.append(float(np.log1p(r["rank"][:n]).mean()))
    return np.array(out)

def auc(a,b):
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; s=np.r_[a,b]
    v=roc_auc_score(y,s); return max(v,1-v)

print(f"\nCROSS-FAMILY: honest=Qwen2.5-7B  vs  served={SERVED}")
print(f"proxy = Qwen2.5-0.5B  (de-confounds identity: different family/tokenizer)")
print("  tokens: " + "".join(f"{k:>7}" for k in KS))
for name in ["ce","agree","logrank"]:
    print(f"  {name:8s}" + "".join(f"{auc(stat(H,name,k),stat(S,name,k)):7.3f}" for k in KS))
