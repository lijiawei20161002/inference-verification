"""
Pillar-A claims on a REAL LLM forward pass (Qwen3-0.6B, bf16, single A100).

Same input tokens for batch-element 0, embedded in batches of varying size B (all
sequences equal length, full attention mask => element 0 cannot attend to neighbors).
Any change in element-0's deep hidden state across B is therefore pure GEMM-shape /
reduction-order non-invariance, propagated through all transformer layers.

  - within-condition: repeated forward, same shape -> identical fingerprint
  - non-invariance:   fingerprint (and L2) of element-0 hidden state vs batch size B
  - verifier checksum: SHA-256 of (last layer, last token) hidden state -> pass/fail

Writes results/llm_results.json.
"""
import os, json, hashlib, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/bitexact/hf")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
dev, dt = "cuda", torch.bfloat16


def sha(t):
    b = t.detach().cpu().flatten().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def main():
    os.makedirs("results", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)            # noqa: F841 (kept for completeness)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dt, attn_implementation="eager").to(dev).eval()

    SEQ = 128
    torch.manual_seed(0)
    V = model.config.vocab_size
    row0 = torch.randint(0, V, (1, SEQ), device=dev)
    filler = torch.randint(0, V, (64, SEQ), device=dev)

    def elem0_hidden(B):
        ids = torch.cat([row0, filler[: B - 1]], dim=0)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        return out.hidden_states[-1][0, -1]              # last layer, elem 0, last token

    # within-condition
    within = len({sha(elem0_hidden(1)) for _ in range(5)})

    # non-invariance vs batch size
    Bs = [1, 2, 4, 8, 16, 32]
    ref_h, ref_v = None, None
    rows = []
    for B in Bs:
        v1 = elem0_hidden(B); v2 = elem0_hidden(B)
        repro = sha(v1) == sha(v2)
        if ref_h is None:
            ref_h, ref_v = sha(v1), v1.clone()
        rows.append({
            "B": B, "reproducible": repro,
            "fingerprint": sha(v1)[:16],
            "same_as_b1": sha(v1) == ref_h,
            "l2_vs_b1": (v1.float() - ref_v.float()).norm().item(),
        })

    res = {
        "model": MODEL, "dtype": "bfloat16", "attn": "eager", "seq_len": SEQ,
        "device": torch.cuda.get_device_name(0),
        "within_condition_distinct": within,
        "batch_sweep": rows,
        "verifier_checksum": sha(elem0_hidden(1)),
    }
    with open("results/llm_results.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"model {MODEL} on {res['device']}")
    print(f"within-condition (5 runs): distinct = {within} "
          f"({'BIT-IDENTICAL' if within == 1 else 'DIVERGED'})")
    print(" B   repro  same_as_b1  l2_vs_b1     fingerprint")
    for r in rows:
        print(f"{r['B']:2d}   {str(r['reproducible']):5s}  {str(r['same_as_b1']):5s}      "
              f"{r['l2_vs_b1']:.4e}  {r['fingerprint']}")
    print(f"verifier checksum: {res['verifier_checksum'][:32]}...")
    print("wrote results/llm_results.json")


if __name__ == "__main__":
    main()
