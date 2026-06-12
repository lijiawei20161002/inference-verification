"""
Bit-exact inference verification — PoC on a single NVIDIA A100.

Reproduces the Pillar-A claims of "Bit-Exact AI Inference Verification Without
Performance Tradeoffs" (arXiv 2606.00279v1) on real GPU kernels, with NO
determinism flags set (so no performance tradeoff):

  E1  Within-condition bit-exactness   identical conditions  => identical BITS
  E2  Non-invariance (root cause)       reduction ORDER changes => different BITS
  E3  Batch-size non-invariance         fixed input, varying batch/seq shape
                                        => bits change with shape, each shape reproducible
  E4  Genuine non-determinism           float atomicAdd (scatter_add) => run-to-run drift

Tensors are fingerprinted with SHA-256 of their raw bytes (the paper's "hash a
deep tensor" idea): bit-identical  <=>  identical hash.

Outputs JSON results to results/results.json for plotting by plots.py.
"""
import os, json, hashlib, argparse, torch

def sha(t):
    """SHA-256 of a tensor's raw bytes — bf16/scalar-safe bitcast through uint8."""
    b = t.detach().cpu().flatten().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()

def l2(a, b):
    return (a.float() - b.float()).norm().item()


def env_info():
    return {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "determinism_flags_set": False,
    }


# ---------------------------------------------------------------------------
def e1_within_condition(dev):
    """Same GEMM, same shape, repeated 10x -> expect ONE distinct hash per dtype."""
    out = {}
    for name, dt in (("float32", torch.float32), ("float16", torch.float16),
                     ("bfloat16", torch.bfloat16)):
        torch.manual_seed(0)
        A = torch.randn(2048, 4096, device=dev, dtype=dt)
        B = torch.randn(4096, 4096, device=dev, dtype=dt)
        hashes = []
        for _ in range(10):
            Y = A @ B; torch.cuda.synchronize()
            hashes.append(sha(Y))
        out[name] = {"runs": len(hashes), "distinct": len(set(hashes)),
                     "bit_identical": len(set(hashes)) == 1}
    return out


def e2_non_associativity(dev):
    """Sum the SAME floats in different orders -> different bits (the root cause).

    Uses a wide-dynamic-range vector (mixed magnitudes) so that float absorption /
    cancellation makes the rounding genuinely order-dependent — exactly the regime
    that tensor-core reduction trees and Split-K tilings differ in.
    """
    torch.manual_seed(0)
    n = 1 << 20
    x = torch.randn(1, n, device=dev, dtype=torch.float32)
    idx = torch.randperm(n, device=dev)[:2048]                  # inject cancelling spikes
    x[0, idx[:1024]] += 1e9
    x[0, idx[1024:]] -= 1e9                                      # net contribution ~0...
    s_tree = x.sum()                                            # ...if order didn't matter
    s_lr = x.cumsum(1)[0, -1]                                   # sequential left -> right
    s_rl = x.flip(1).cumsum(1)[0, -1]                           # sequential right -> left
    vals = {"tree": s_tree, "L->R": s_lr, "R->L": s_rl}
    return {
        "values": {k: float(v.item()) for k, v in vals.items()},
        "hashes": {k: sha(v)[:12] for k, v in vals.items()},
        "distinct_bit_patterns": len({sha(v) for v in vals.values()}),
    }


def e3_batch_seq_noninvariance(dev):
    """
    Fix element-0's inputs; vary batch size B and contraction length K (a proxy
    for sequence length in the attention/FFN GEMMs). For each (B, K) cell:
      - reproducible?  run twice, compare hashes
      - same bits as B=1?  compare element-0 output hash to the B=1 reference
    Returns a grid for heatmaps.
    """
    Bs = [1, 2, 4, 8, 16, 32, 48, 64]
    Ks = [512, 1024, 2048, 4096, 8192]
    N = 4096
    dt = torch.bfloat16
    grid_diff = []   # 1 if bits differ from B=1 reference, else 0
    grid_repro = []  # 1 if reproducible (run twice identical), else 0
    grid_l2 = []
    for K in Ks:
        torch.manual_seed(K)
        W = torch.randn(K, N, device=dev, dtype=dt)
        row0 = torch.randn(1, K, device=dev, dtype=dt)
        filler = torch.randn(max(Bs), K, device=dev, dtype=dt)
        ref_h, ref_v, drow, rrow, lrow = None, None, [], [], []
        for B in Bs:
            batch = torch.cat([row0, filler[: B - 1]], dim=0)
            o1 = (batch @ W)[0]; torch.cuda.synchronize()
            o2 = (batch @ W)[0]; torch.cuda.synchronize()
            if ref_h is None:
                ref_h, ref_v = sha(o1), o1.clone()
            rrow.append(0 if sha(o1) == sha(o2) else 1)        # 0 = reproducible
            drow.append(0 if sha(o1) == ref_h else 1)          # 0 = same as B=1
            lrow.append(l2(o1, ref_v))
        grid_diff.append(drow); grid_repro.append(rrow); grid_l2.append(lrow)
    return {"batch_sizes": Bs, "K_values": Ks,
            "diff_from_b1": grid_diff, "nonreproducible": grid_repro, "l2": grid_l2}


def e4_genuine_nondeterminism(dev):
    """float scatter_add / index_add (atomicAdd) repeated -> run-to-run drift."""
    torch.manual_seed(0)
    src = torch.randn(1 << 22, device=dev, dtype=torch.float32)
    idx = torch.randint(0, 1024, (1 << 22,), device=dev)
    out = {}
    for op in ("scatter_add", "index_add"):
        hashes, spreads = [], []
        base = None
        for _ in range(10):
            acc = torch.zeros(1024, device=dev, dtype=torch.float32)
            if op == "scatter_add":
                acc.scatter_add_(0, idx, src)
            else:
                acc.index_add_(0, idx, src)
            torch.cuda.synchronize()
            hashes.append(sha(acc))
            if base is None:
                base = acc.clone()
            spreads.append(l2(acc, base))
        out[op] = {"runs": len(hashes), "distinct": len(set(hashes)),
                   "deterministic": len(set(hashes)) == 1,
                   "max_l2_drift": max(spreads)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = {"env": env_info()}
    print(f"device: {results['env']['device']}  torch {results['env']['torch']}  "
          f"cuda {results['env']['cuda']}  (no determinism flags)\n")

    print("[E1] within-condition bit-exactness ...")
    results["E1_within_condition"] = e1_within_condition(dev)
    for k, v in results["E1_within_condition"].items():
        print(f"     {k:9s} distinct={v['distinct']}/10  bit_identical={v['bit_identical']}")

    print("[E2] non-associativity (root cause) ...")
    results["E2_non_associativity"] = e2_non_associativity(dev)
    print(f"     distinct bit patterns from identical math = "
          f"{results['E2_non_associativity']['distinct_bit_patterns']}")

    print("[E3] batch x seqlen non-invariance grid ...")
    results["E3_noninvariance_grid"] = e3_batch_seq_noninvariance(dev)
    g = results["E3_noninvariance_grid"]
    print(f"     grid {len(g['K_values'])}x{len(g['batch_sizes'])}; "
          f"nonreproducible cells = {sum(sum(r) for r in g['nonreproducible'])}")

    print("[E4] genuine non-determinism (atomicAdd) ...")
    results["E4_genuine_nondeterminism"] = e4_genuine_nondeterminism(dev)
    for k, v in results["E4_genuine_nondeterminism"].items():
        print(f"     {k:11s} distinct={v['distinct']}/10  deterministic={v['deterministic']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
