import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)),"difr_data")
ATTACKS = ["sub_3b", "sub_1.5b", "sub_0.5b", "quant_4bit"]
KS = [1, 2, 4, 8, 16, 32, 64, 128]
STATS = ["ce", "agree", "logrank"]
PARAMS = {"proxy_0.5b": 0.494e9, "proxy_1.5b": 1.54e9}
P_CLAIMED = 7.62e9  # Qwen2.5-7B

def per_prompt_stat(records, stat, k):
    """mean of stat over first k completion tokens, per prompt."""
    vals = []
    for r in records:
        n = min(k, len(r["nll"]))
        if n == 0:
            continue
        if stat == "ce":
            vals.append(float(r["nll"][:n].mean()))
        elif stat == "agree":
            vals.append(float(r["correct"][:n].mean()))
        elif stat == "logrank":
            vals.append(float(np.log1p(r["rank"][:n]).mean()))
    return np.array(vals)

def auc_oriented(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    a = roc_auc_score(y, s)
    return max(a, 1 - a)  # discriminative power regardless of sign

for proxy in ["proxy_0.5b", "proxy_1.5b"]:
    R = torch.load(os.path.join(EXP, f"scores_{proxy}.pt"), weights_only=False)
    print("\n" + "=" * 78)
    print(f"PROXY = {proxy}   ({PARAMS[proxy]/1e9:.2f}B params, "
          f"{100*PARAMS[proxy]/P_CLAIMED:.1f}% of claimed 7B per-token FLOP)")
    print("=" * 78)
    for attack in ATTACKS:
        # best statistic = highest AUC at k=128
        best_stat = max(STATS, key=lambda s: auc_oriented(
            per_prompt_stat(R["honest"], s, 128), per_prompt_stat(R[attack], s, 128)))
        print(f"\n  honest vs {attack}   [best stat: {best_stat}]")
        header = "    tokens:  " + "".join(f"{k:>7}" for k in KS)
        print(header)
        for stat in STATS:
            row = []
            for k in KS:
                neg = per_prompt_stat(R["honest"], stat, k)
                pos = per_prompt_stat(R[attack], stat, k)
                row.append(auc_oriented(neg, pos))
            mark = " *" if stat == best_stat else "  "
            print(f"    {stat:8s}{mark}" + "".join(f"{v:7.3f}" for v in row))
