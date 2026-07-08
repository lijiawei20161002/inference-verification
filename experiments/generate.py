import os, sys, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data")
os.makedirs(EXP, exist_ok=True)
N_PROMPTS = int(os.environ.get("N_PROMPTS", "300"))
MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
BATCH = int(os.environ.get("BATCH", "48"))

# Shared tokenizer/chat-template for the Qwen family (all served configs + proxies share it).
TOK_ID = "Qwen/Qwen2.5-7B-Instruct"

CONFIGS = {
    "honest":   dict(model="Qwen/Qwen2.5-7B-Instruct",   quant=None),
    "sub_3b":   dict(model="Qwen/Qwen2.5-3B-Instruct",   quant=None),
    "sub_1.5b": dict(model="Qwen/Qwen2.5-1.5B-Instruct", quant=None),
    "sub_0.5b": dict(model="Qwen/Qwen2.5-0.5B-Instruct", quant=None),
    "quant_4bit": dict(model="Qwen/Qwen2.5-7B-Instruct", quant="nf4"),
}

def build_prompts(tok):
    p = os.path.join(EXP, "prompts.json")
    if os.path.exists(p):
        return json.load(open(p))
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for r in ds:
        if r["input"].strip():
            continue
        prompts.append(r["instruction"].strip())
        if len(prompts) >= N_PROMPTS:
            break
    json.dump(prompts, open(p, "w"))
    print(f"built {len(prompts)} prompts")
    return prompts

def main(cfg_name):
    cfg = CONFIGS[cfg_name]
    tok = AutoTokenizer.from_pretrained(TOK_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(tok)

    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda")
    if cfg["quant"] == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kw)
    model.eval()

    # chat-templated prompt strings
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
             tokenize=False, add_generation_prompt=True) for p in prompts]

    out_prompt_ids, out_completion_ids = [], []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for b in range(len(chunk)):
            # strip left padding from the prompt portion
            real = enc["input_ids"][b][enc["attention_mask"][b].bool()].tolist()
            comp = gen[b][plen:].tolist()
            # trim trailing pad/eos beyond first eos
            if tok.eos_token_id in comp:
                comp = comp[:comp.index(tok.eos_token_id)+1]
            out_prompt_ids.append(real)
            out_completion_ids.append(comp)
        print(f"[{cfg_name}] {i+len(chunk)}/{len(texts)}  {time.time()-t0:.0f}s", flush=True)

    torch.save({"prompt_ids": out_prompt_ids, "completion_ids": out_completion_ids,
                "config": cfg_name, "model": cfg["model"]},
               os.path.join(EXP, f"gen_{cfg_name}.pt"))
    lens = [len(c) for c in out_completion_ids]
    print(f"[{cfg_name}] done. mean completion len {sum(lens)/len(lens):.1f}")

if __name__ == "__main__":
    main(sys.argv[1])
