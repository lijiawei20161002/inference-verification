"""Detection AUC vs recomputation ratio: proxy-q tie-triaged selective recompute
vs random subsampling, on subtle real quantization.

Produces docs/figures/fig_tie_triage_pareto.png -- the concrete "shrink how often
recompute fires" curve. For each bit-width the verifier audits (recomputes M on)
only a fraction rho of tokens; it either picks that rho by the client-owned proxy
q's near-tie score (triaged) or at random. Detection AUC is a batch-means
bootstrap of the max TV(served, p*) over the audited positions, honest vs quant,
with +/-1 std error bands across bootstrap seeds.

Same box-friendly setup as exp_real_quant_triage.py: M=Qwen3-1.7B (true p*),
proxy=Qwen3-0.6B (q), deterministic per-channel int-n weight-only fake-quant on
GPU (no bitsandbytes), logits streamed to disk memmaps.

    .venv/bin/python -m experiments.exp_tie_triage_pareto            # full run
    IVGYM_REPLOT=1 .venv/bin/python -m experiments.exp_tie_triage_pareto  # re-plot from cache
Env: IVGYM_M, IVGYM_PROXY, IVGYM_PROMPTS(24), IVGYM_TOKENS(96), IVGYM_BATCH(48),
     IVGYM_NBATCH(300), IVGYM_BITS("8,6,5"), IVGYM_BOOT(10).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym.backends.hf_gpu import DEFAULT_PROMPTS
from ivgym.metrics import roc_auc

M_NAME = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY_NAME = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N = int(os.environ.get("IVGYM_PROMPTS", 24))
T = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 48))
N_BATCH = int(os.environ.get("IVGYM_NBATCH", 300))
BITS = [int(b) for b in os.environ.get("IVGYM_BITS", "8,6,5").split(",")]
BOOT = int(os.environ.get("IVGYM_BOOT", 10))
MAX_PROMPT = 32
BENIGN_SIGMA = 0.02
GEN_TEMP = 0.8
ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("IVGYM_SCRATCH", ROOT / "experiments" / "_ttp"))
NPZ = ROOT / "experiments" / "difr_data" / "tie_triage_pareto.npz"
FIG = ROOT / "docs" / "figures" / "fig_tie_triage_pareto.png"
_EPS = 1e-12


def log(m): print(m, flush=True)


def softmax(z):
    z = z.astype(np.float32); z = z - z.max(); e = np.exp(z); return e / e.sum()


def tv(p, q): return float(0.5 * np.abs(p - q).sum())


# --------------------------------------------------------------------------- GPU
def fake_quant_(model, bits, torch):
    import torch.nn as nn
    lvl = 2 ** (bits - 1) - 1
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            W = mod.weight.data
            scale = W.abs().amax(dim=1, keepdim=True) / lvl + 1e-8
            mod.weight.data = (torch.clamp(torch.round(W / scale), -lvl - 1, lvl) * scale).to(W.dtype)


def load(name, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True).eval()
    return tok, mdl


def teacher_force(model, full_ids, L, n_tokens, torch):
    with torch.no_grad():
        out = model(full_ids)
    return out.logits[0, L - 1:L - 1 + n_tokens].float().cpu().numpy()


def compute_features(torch):
    """Run the models; return per-position scalar arrays flattened to [N*T]."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, M = load(M_NAME, torch)
    V = int(M.config.vocab_size)
    log(f"loaded M ({sum(p.numel() for p in M.parameters())/1e9:.2f}B, V={V}) [{time.time()-t0:.0f}s]")
    pstar = np.memmap(SCRATCH / "pstar.f16", mode="w+", dtype=np.float16, shape=(N, T, V))
    qmm = np.memmap(SCRATCH / "q.f16", mode="w+", dtype=np.float16, shape=(N, T, V))
    full_ids_all, L_all = [], []
    for n in range(N):
        text = DEFAULT_PROMPTS[n % len(DEFAULT_PROMPTS)]
        pids = tok(text, return_tensors="pt").input_ids[:, :MAX_PROMPT].to("cuda")
        L = pids.shape[1]; cur, claimed, past = pids, [], None
        with torch.no_grad():
            out = M(cur, use_cache=True); past = out.past_key_values; last = out.logits[0, -1]
            for _ in range(T):
                probs = torch.softmax(last.float() / GEN_TEMP, dim=-1)
                tokid = int(torch.multinomial(probs, 1).item()); claimed.append(tokid)
                step = torch.tensor([[tokid]], device="cuda", dtype=pids.dtype)
                out = M(step, past_key_values=past, use_cache=True)
                past = out.past_key_values; last = out.logits[0, -1]
        full = torch.cat([pids, torch.tensor([claimed], device="cuda", dtype=pids.dtype)], 1)
        pstar[n] = teacher_force(M, full, L, T, torch).astype(np.float16)
        full_ids_all.append(full); L_all.append(L)
    pstar.flush(); del M; torch.cuda.empty_cache()
    log(f"generated + p* cached [{time.time()-t0:.0f}s]")

    _, P = load(PROXY_NAME, torch)
    if int(P.config.vocab_size) != V:
        raise SystemExit("proxy vocab mismatch")
    tieness = np.zeros((N, T)); tv_hon = np.zeros((N, T))
    rng = np.random.default_rng(0)
    for n in range(N):
        q_rows = teacher_force(P, full_ids_all[n], L_all[n], T, torch)
        qmm[n] = q_rows.astype(np.float16)
        for t in range(T):
            qq = softmax(q_rows[t]); ps = softmax(np.asarray(pstar[n, t]))
            benign = softmax(np.asarray(pstar[n, t], np.float32)
                             + rng.normal(0, BENIGN_SIGMA, V).astype(np.float32))
            top2 = np.partition(qq, -2)[-2:]
            tieness[n, t] = -float(top2[1] - top2[0])
            tv_hon[n, t] = tv(benign, ps)
    qmm.flush(); del P; torch.cuda.empty_cache()
    log(f"proxy q + tieness cached [{time.time()-t0:.0f}s]")

    tv_faith = {}
    for bits in BITS:
        _, Mq = load(M_NAME, torch); fake_quant_(Mq, bits, torch)
        tvf = np.zeros((N, T))
        for n in range(N):
            ph_rows = teacher_force(Mq, full_ids_all[n], L_all[n], T, torch)
            for t in range(T):
                tvf[n, t] = tv(softmax(ph_rows[t]), softmax(np.asarray(pstar[n, t])))
        del Mq; torch.cuda.empty_cache()
        tv_faith[bits] = tvf.reshape(-1)
        log(f"  bits={bits}: meanTV={tvf.mean():.4f} spars(>0.05)={float((tvf>0.05).mean()):.2f} "
            f"[{time.time()-t0:.0f}s]")

    for f in ["pstar.f16", "q.f16"]:
        (SCRATCH / f).unlink(missing_ok=True)
    data = {"tv_hon": tv_hon.reshape(-1), "tieness": tieness.reshape(-1),
            "bits": np.array(BITS)}
    for bits in BITS:
        data[f"tvf_{bits}"] = tv_faith[bits]
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **data)
    log(f"saved features -> {NPZ}")
    return data


# ------------------------------------------------------------ AUC sweep (numpy)
def auc_curve(tv_hon, tv_att, tieness, rhos, mode, boot=BOOT, agg="mean"):
    """Mean +/- std detection AUC over `boot` bootstrap seeds, for each rho.

    Statistic = aggregation of TV(served, p*) over the AUDITED positions of a
    batch. `agg="mean"` (default) is the mean divergence over audited tokens: it
    does not saturate the instant one corrupted token is seen (unlike `max`), so
    it exposes the full triage-vs-random gap across the recompute ratio."""
    Mn = len(tv_hon)
    mean = np.zeros(len(rhos)); std = np.zeros(len(rhos))
    fagg = (lambda v: v.mean()) if agg == "mean" else (lambda v: v.max())
    for i, rho in enumerate(rhos):
        k = max(1, int(round(rho * BATCH)))
        seed_aucs = []
        for s in range(boot):
            r = np.random.default_rng(1000 + s)
            h = np.empty(N_BATCH); a = np.empty(N_BATCH)
            for b in range(N_BATCH):
                idx = r.choice(Mn, size=BATCH, replace=False)
                if mode == "triage":
                    sel = idx[np.argsort(-tieness[idx])[:k]]
                else:
                    sel = r.choice(idx, size=k, replace=False)
                h[b] = fagg(tv_hon[sel]); a[b] = fagg(tv_att[sel])
            seed_aucs.append(roc_auc(h, a))
        mean[i] = np.mean(seed_aucs); std[i] = np.std(seed_aucs)
    return mean, std


def cost_to_reach(rhos, mean_auc, target=0.95):
    """Smallest recompute ratio whose mean AUC >= target (linear-interp)."""
    for i in range(len(rhos)):
        if mean_auc[i] >= target:
            if i == 0:
                return rhos[0]
            x0, x1 = rhos[i - 1], rhos[i]; y0, y1 = mean_auc[i - 1], mean_auc[i]
            return float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-9))
    return None


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bits_list = [int(b) for b in data["bits"]]
    tv_hon = data["tv_hon"]; tieness = data["tieness"]
    # log-spaced recompute ratios from 1/BATCH up to full audit
    rhos = np.unique(np.round(np.geomspace(1.0 / BATCH, 1.0, 22) * BATCH) / BATCH)
    TARGET = 0.95

    fig, axes = plt.subplots(1, len(bits_list), figsize=(4.9 * len(bits_list), 4.6),
                             sharey=True)
    if len(bits_list) == 1:
        axes = [axes]
    C_TRI, C_RND, C_FULL = "#2ca02c", "#d1902f", "#1f77b4"
    summary = []

    for ax, bits in zip(axes, bits_list):
        tvf = data[f"tvf_{bits}"]
        mean_tv = float(tvf.mean()); spars = float((tvf > 0.05).mean())
        m_tri, s_tri = auc_curve(tv_hon, tvf, tieness, rhos, "triage")
        m_rnd, s_rnd = auc_curve(tv_hon, tvf, tieness, rhos, "random")

        ax.axhline(1.0, ls=":", color=C_FULL, lw=1.3, zorder=1,
                   label="full recompute (AUC 1.0)")
        ax.axhline(TARGET, ls="--", color="#888", lw=0.8, zorder=1)
        ax.fill_between(rhos, m_tri - s_tri, m_tri + s_tri, color=C_TRI, alpha=0.18, zorder=2)
        ax.fill_between(rhos, m_rnd - s_rnd, m_rnd + s_rnd, color=C_RND, alpha=0.18, zorder=2)
        ax.plot(rhos, m_tri, "-o", color=C_TRI, ms=4, lw=2.1, zorder=4,
                label="triaged (proxy-q ties)")
        ax.plot(rhos, m_rnd, "--s", color=C_RND, ms=3.2, lw=1.8, zorder=3,
                label="random subsample")

        c_tri = cost_to_reach(rhos, m_tri, TARGET)
        c_rnd = cost_to_reach(rhos, m_rnd, TARGET)
        factor = (c_rnd / c_tri) if (c_tri and c_rnd) else float("nan")
        summary.append((bits, mean_tv, c_tri, c_rnd, factor))
        if c_tri and c_rnd:
            ax.axvspan(c_tri, c_rnd, color="#bbb", alpha=0.20, zorder=0)
            ax.annotate(f"{factor:.1f}x fewer\nrecomputes\nto AUC {TARGET}",
                        xy=(np.sqrt(c_tri * c_rnd), TARGET), xytext=(0.16, 0.70),
                        fontsize=9, color="#222", ha="left",
                        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9))
        ax.set_xscale("log")
        ax.set_title(f"{bits}-bit weight-quant   (mean TV={mean_tv:.3f})\n"
                     f"{spars*100:.0f}% of tokens perturbed >0.05", fontsize=10)
        ax.set_xlabel("recomputation ratio  (fraction of tokens re-run on M)", fontsize=9.5)
        ax.set_xlim(1.0 / BATCH, 1.0); ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    axes[0].set_ylabel("detection AUC  (honest vs quant)", fontsize=10)
    fig.suptitle("Spend recompute where the proxy says it matters: tie-triaged selective recompute "
                 "vs random\nProxy-q tie-ness ranks quant-corrupted tokens (Spearman≈0.73), so a few "
                 "audited tokens suffice.\n"
                 f"M=Qwen3-1.7B, proxy=Qwen3-0.6B; statistic=mean TV over audited tokens; "
                 f"band=±1 std / {BOOT} seeds",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    log(f"wrote figure -> {FIG}")

    log(f"\nrecompute ratio to reach AUC >= {TARGET}:")
    log(f"  {'bits':>4}{'meanTV':>8}{'triaged':>10}{'random':>9}{'saving':>9}")
    for bits, mtv, ct, cr, fac in summary:
        log(f"  {bits:>4}{mtv:>8.4f}{(ct or float('nan')):>10.3f}"
            f"{(cr or float('nan')):>9.3f}{fac:>8.1f}x")


def main():
    if os.environ.get("IVGYM_REPLOT") == "1" and NPZ.exists():
        log(f"re-plotting from {NPZ}")
        data = dict(np.load(NPZ))
    else:
        import torch
        torch.manual_seed(0)
        data = compute_features(torch)
    render(data)


if __name__ == "__main__":
    main()
