import os, sys, json, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data"); B=48
CONFIGS=["honest","sub_3b","sub_1.5b","sub_0.5b","quant_4bit"]
PROXIES={"proxy_0.5b":"Qwen/Qwen2.5-0.5B-Instruct","proxy_1.5b":"Qwen/Qwen2.5-1.5B-Instruct"}
qtok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
PAD=qtok.pad_token_id or qtok.eos_token_id

@torch.no_grad()
def score(model, prompt_ids, comp_ids):
    recs=[]
    for i in range(0,len(prompt_ids),B):
        pb=prompt_ids[i:i+B]; cb=comp_ids[i:i+B]
        full=[p+c for p,c in zip(pb,cb)]; L=max(len(f) for f in full)
        ids=torch.full((len(full),L),PAD); att=torch.zeros((len(full),L),dtype=torch.long)
        for b,f in enumerate(full):
            ids[b,:len(f)]=torch.tensor(f); att[b,:len(f)]=1
        logits=model(input_ids=ids.cuda(),attention_mask=att.cuda()).logits
        for b in range(len(full)):
            pl=len(pb[b]); cl=len(cb[b])
            if cl==0: recs.append(None); continue
            idx=torch.arange(pl-1,pl+cl-1,device=logits.device)
            lg=logits[b,idx].float(); tgt=torch.tensor(cb[b],device=logits.device)
            lp=torch.log_softmax(lg,-1); p=lp.exp()
            nll=-lp[torch.arange(cl),tgt]
            tl=lg[torch.arange(cl),tgt].unsqueeze(-1)
            rank=(lg>tl).sum(-1)
            ent=-(p*lp).sum(-1)
            recs.append(dict(nll=nll.cpu().numpy().astype(np.float32),
                             rank=rank.cpu().numpy().astype(np.int32),
                             ent=ent.cpu().numpy().astype(np.float32)))
        print(f"  {i+len(full)}",end=" ",flush=True)
    print()
    return recs

def main(pname):
    model=AutoModelForCausalLM.from_pretrained(PROXIES[pname],dtype=torch.bfloat16,device_map="cuda").eval()
    out={}
    for cfg in CONFIGS:
        d=torch.load(f"{EXP}/gen_{cfg}.pt",weights_only=False)
        print(f"[{pname}] {cfg}")
        out[cfg]=score(model,d["prompt_ids"],d["completion_ids"])
    torch.save(out,f"{EXP}/feats_{pname}.pt")
    print("saved",pname)

if __name__=="__main__": main(sys.argv[1])
