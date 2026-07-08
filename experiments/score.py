import os, sys, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data")
BATCH = int(os.environ.get("SBATCH", "48"))
CONFIGS = ["honest", "sub_3b", "sub_1.5b", "sub_0.5b", "quant_4bit"]
PROXIES = {
    "proxy_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "proxy_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

@torch.no_grad()
def score_config(model, seqs_prompt, seqs_comp, pad_id):
    """Teacher-force prompt+completion; return per-prompt per-token nll/correct/rank over completion positions."""
    out = []
    for i in range(0, len(seqs_prompt), BATCH):
        pb = seqs_prompt[i:i+BATCH]
        cb = seqs_comp[i:i+BATCH]
        full = [p + c for p, c in zip(pb, cb)]
        maxlen = max(len(f) for f in full)
        ids = torch.full((len(full), maxlen), pad_id, dtype=torch.long)
        att = torch.zeros((len(full), maxlen), dtype=torch.long)
        for b, f in enumerate(full):
            ids[b, :len(f)] = torch.tensor(f)
            att[b, :len(f)] = 1
        ids = ids.cuda(); att = att.cuda()
        logits = model(input_ids=ids, attention_mask=att).logits  # [B,L,V]
        for b in range(len(full)):
            plen = len(pb[b]); clen = len(cb[b])
            # position t predicts token t+1; completion token c[j] sits at index plen+j,
            # predicted by logits at index plen+j-1
            idx = torch.arange(plen-1, plen+clen-1, device=ids.device)
            lg = logits[b, idx].float()                      # [clen, V]
            tgt = torch.tensor(cb[b], device=ids.device)     # [clen]
            logp = torch.log_softmax(lg, dim=-1)
            nll = -logp[torch.arange(clen), tgt]
            tgt_logit = lg[torch.arange(clen), tgt].unsqueeze(-1)
            rank = (lg > tgt_logit).sum(-1)                  # 0 == argmax
            correct = (rank == 0)
            out.append(dict(nll=nll.cpu().numpy().astype(np.float32),
                            rank=rank.cpu().numpy().astype(np.int32),
                            correct=correct.cpu().numpy()))
        print(f"  scored {i+len(full)}", flush=True)
    return out

def main(proxy_name):
    mid = PROXIES[proxy_name]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    pad_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    result = {}
    t0 = time.time()
    for cfg in CONFIGS:
        d = torch.load(os.path.join(EXP, f"gen_{cfg}.pt"))
        print(f"[{proxy_name}] scoring {cfg} ({time.time()-t0:.0f}s)")
        result[cfg] = score_config(model, d["prompt_ids"], d["completion_ids"], pad_id)
    torch.save(result, os.path.join(EXP, f"scores_{proxy_name}.pt"))
    print(f"[{proxy_name}] saved.")

if __name__ == "__main__":
    main(sys.argv[1])
