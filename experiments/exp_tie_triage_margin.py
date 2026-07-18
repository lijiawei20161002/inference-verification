"""Sharper tie-triage Pareto using the SPARSE token-flip signal (Token-DiFR margin).

Companion to exp_tie_triage_pareto.py (which used the denser TV signal). The
Token-DiFR margin is nonzero ONLY where the provider's seed-synced Gumbel winner
differs from the verifier's -- i.e. only at genuine near-tie positions a
forward-pass corruption actually flips. That signal is far SPARSER than per-token
TV, so proxy-q tie-triage -- which is built to find exactly those near-tie
positions -- gives a much bigger recompute saving than on TV.

Faithful setup (same box constraints as exp_real_quant_triage.py):
  * M = Qwen3-1.7B (bf16) = true reference p*; served token = M's Gumbel-max draw.
  * quant provider = M with deterministic per-channel int-n weight-only fake-quant;
    served token = the QUANTIZED model's Gumbel-max draw under the SAME shared
    Gumbel noise at that position. The verifier recomputes M and scores the
    Token-DiFR margin z[M-winner] - z[claimed]; honest ~ 0, quant > 0 only at flips.
    Set IVGYM_QUANT=nf4 to swap the fake-quant for REAL bitsandbytes NF4 4-bit
    weights (needs the optional `bitsandbytes` dep); the bit sweep collapses to a
    single "nf4" panel and the flip-rate is confirmed on true 4-bit.
  * proxy = Qwen3-0.6B -> tie-ness triage (harness.token_values "tie_margin").

Selection uses `ivgym.harness.select_triaged` -- the SAME primitive the first-class
`harness.verify(budget<1)` tier uses -- so the figure exercises the shipped path.

    .venv/bin/python -m experiments.exp_tie_triage_margin
    IVGYM_REPLOT=1 .venv/bin/python -m experiments.exp_tie_triage_margin   # re-plot
Env: IVGYM_M, IVGYM_PROXY, IVGYM_PROMPTS(24), IVGYM_TOKENS(96), IVGYM_BATCH(64),
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
from ivgym.core import SamplingSpec
from ivgym.harness import select_triaged
from ivgym.verifiers import TokenDiFR
from ivgym.metrics import roc_auc
from ivgym.sampling import gumbel_max_sample, gumbel_noise, log_softmax, position_seed
from experiments import quantlib

M_NAME = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY_NAME = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N = int(os.environ.get("IVGYM_PROMPTS", 24))
T = int(os.environ.get("IVGYM_TOKENS", 96))
# the flip signal is SPARSE (few % of tokens), so a decision pool must hold enough
# tokens to contain flips; batch=256 makes full recompute saturate to AUC ~1 for
# every bit-width, so the recompute-ratio saving is well defined.
BATCH = int(os.environ.get("IVGYM_BATCH", 256))
N_BATCH = int(os.environ.get("IVGYM_NBATCH", 300))
BITS = [int(b) for b in os.environ.get("IVGYM_BITS", "8,6,5").split(",")]
SETTINGS = quantlib.quant_settings(BITS)   # bit ints, or ["nf4"] under IVGYM_QUANT=nf4
BOOT = int(os.environ.get("IVGYM_BOOT", 10))
MAX_PROMPT = 32
BENIGN_SIGMA = float(os.environ.get("IVGYM_BENIGN", 0.01))
ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("IVGYM_SCRATCH", ROOT / "experiments" / "_ttm"))
# A real-NF4 run writes its own artifacts so it never clobbers the fake-quant
# figure/cache (suffix is empty in the default fake-quant mode).
_QSUF = "" if quantlib.QUANT_MODE == "fake" else f"_{quantlib.QUANT_MODE}"
NPZ = ROOT / "experiments" / "difr_data" / f"tie_triage_margin{_QSUF}.npz"
FIG = ROOT / "docs" / "figures" / f"fig_tie_triage_margin{_QSUF}.png"
# Gumbel-max over the FULL vocab (no top-k/top-p): the margin then comes purely
# from Gumbel-argmax flips, with no top-k/top-p boundary "filtered-out" (delta_max)
# spikes -- which are a benign-noise artifact, not a quant signal, and hit honest
# and quant alike. Honest margins stay marginal (a benign flip barely wins);
# quant flips are decisive (larger logit change -> larger margin).
SPEC = SamplingSpec(top_k=None, top_p=None)     # temp=1.0, seed=42, no filtering
_TD = TokenDiFR()


def log(m): print(m, flush=True)


def softmax(z):
    z = z.astype(np.float32); z = z - z.max(); e = np.exp(z); return e / e.sum()


def benign(pid, pos, who, V):
    r = np.random.default_rng((pid, pos, who))
    return r.normal(0, BENIGN_SIGMA, V).astype(np.float32)


def margin(ref, srv, pid, pos, V):
    """Token-DiFR margin for a served token sampled from `srv` and scored under
    `ref` (both with independent benign noise), sharing the position's Gumbel."""
    g = gumbel_noise(V, position_seed(SPEC.seed, pid, pos))
    claimed = gumbel_max_sample(srv + benign(pid, pos, 1, V), SPEC.temperature, g,
                                SPEC.top_k, SPEC.top_p)
    # Token-DiFR is now a Tier-1 verifier; `score_token` is the per-token scorer
    # the driver calls on audited tokens (was `Defense.score(VerifyContext)`).
    return _TD.score_token(ref + benign(pid, pos, 2, V), g, claimed, SPEC)


# --------------------------------------------------------------------------- GPU
load = quantlib.load       # bf16 reference/proxy load (shared)


def teacher_force(model, full_ids, L, n_tokens, torch):
    with torch.no_grad():
        out = model(full_ids)
    return out.logits[0, L - 1:L - 1 + n_tokens].float().cpu().numpy()


def compute_features(torch):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, M = load(M_NAME, torch)
    V = int(M.config.vocab_size)
    log(f"loaded M ({sum(p.numel() for p in M.parameters())/1e9:.2f}B, V={V}) [{time.time()-t0:.0f}s]")
    pstar = np.memmap(SCRATCH / "pstar.f16", mode="w+", dtype=np.float16, shape=(N, T, V))
    full_ids_all, L_all = [], []
    m_hon = np.zeros((N, T))
    for n in range(N):
        text = DEFAULT_PROMPTS[n % len(DEFAULT_PROMPTS)]
        pids = tok(text, return_tensors="pt").input_ids[:, :MAX_PROMPT].to("cuda")
        L = pids.shape[1]; cur, claimed, past = pids, [], None
        with torch.no_grad():
            out = M(cur, use_cache=True); past = out.past_key_values; last = out.logits[0, -1]
            for pos in range(T):
                lg = last.float().cpu().numpy()
                g = gumbel_noise(V, position_seed(SPEC.seed, n, pos))
                tokid = gumbel_max_sample(lg + benign(n, pos, 1, V), SPEC.temperature, g,
                                          SPEC.top_k, SPEC.top_p)
                claimed.append(int(tokid))
                step = torch.tensor([[int(tokid)]], device="cuda", dtype=pids.dtype)
                out = M(step, past_key_values=past, use_cache=True)
                past = out.past_key_values; last = out.logits[0, -1]
        full = torch.cat([pids, torch.tensor([claimed], device="cuda", dtype=pids.dtype)], 1)
        ref_rows = teacher_force(M, full, L, T, torch)
        pstar[n] = ref_rows.astype(np.float16)
        for pos in range(T):
            m_hon[n, pos] = margin(ref_rows[pos], ref_rows[pos], n, pos, V)  # honest: srv=ref
        full_ids_all.append(full); L_all.append(L)
    pstar.flush(); del M; torch.cuda.empty_cache()
    log(f"honest continuation + honest margins [{time.time()-t0:.0f}s]  "
        f"(mean honest margin {m_hon.mean():.4f})")

    _, P = load(PROXY_NAME, torch)
    if int(P.config.vocab_size) != V:
        raise SystemExit("proxy vocab mismatch")
    tieness = np.zeros((N, T))
    for n in range(N):
        q_rows = teacher_force(P, full_ids_all[n], L_all[n], T, torch)
        for t in range(T):
            qq = softmax(q_rows[t]); top2 = np.partition(qq, -2)[-2:]
            tieness[n, t] = -float(top2[1] - top2[0])
    del P; torch.cuda.empty_cache()
    log(f"proxy tie-ness [{time.time()-t0:.0f}s]")

    m_quant = {}
    for setting in SETTINGS:
        _, Mq = quantlib.make_quant(M_NAME, setting, torch)
        mq = np.zeros((N, T)); flips = 0
        for n in range(N):
            ph_rows = teacher_force(Mq, full_ids_all[n], L_all[n], T, torch)
            for pos in range(T):
                v = margin(np.asarray(pstar[n, pos], np.float32), ph_rows[pos], n, pos, V)
                mq[n, pos] = v
                flips += int(v > 1e-6)
        del Mq; torch.cuda.empty_cache()
        m_quant[str(setting)] = mq.reshape(-1)
        log(f"  {quantlib.quant_label(setting)}: flip-rate={flips/(N*T):.3f}  "
            f"mean margin={mq.mean():.4f} [{time.time()-t0:.0f}s]")

    (SCRATCH / "pstar.f16").unlink(missing_ok=True)
    # `settings` stored as strings so a real-NF4 run ("nf4") round-trips through the
    # npz alongside numeric fake-quant bit widths ("8","6","5").
    data = {"m_hon": m_hon.reshape(-1), "tieness": tieness.reshape(-1),
            "settings": np.array([str(s) for s in SETTINGS])}
    for setting in SETTINGS:
        data[f"mq_{setting}"] = m_quant[str(setting)]
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **data)
    log(f"saved -> {NPZ}")
    return data


# ------------------------------------------------------------ AUC sweep (numpy)
def auc_curve(m_hon, m_att, tieness, rhos, mode, boot=BOOT):
    Mn = len(m_hon); mean = np.zeros(len(rhos)); std = np.zeros(len(rhos))
    for i, rho in enumerate(rhos):
        aucs = []
        for s in range(boot):
            r = np.random.default_rng(1000 + s)
            h = np.empty(N_BATCH); a = np.empty(N_BATCH)
            for b in range(N_BATCH):
                idx = r.choice(Mn, size=BATCH, replace=False)
                if mode == "triage":
                    sel = idx[select_triaged(tieness[idx], rho)]     # shipped primitive
                else:
                    k = max(1, int(round(rho * BATCH))); sel = r.choice(idx, size=k, replace=False)
                h[b] = m_hon[sel].mean(); a[b] = m_att[sel].mean()
            aucs.append(roc_auc(h, a))
        mean[i] = np.mean(aucs); std[i] = np.std(aucs)
    return mean, std


def cost_to_reach(rhos, m, target):
    for i in range(len(rhos)):
        if m[i] >= target:
            if i == 0:
                return rhos[0]
            x0, x1, y0, y1 = rhos[i - 1], rhos[i], m[i - 1], m[i]
            return float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-9))
    return None


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # `settings` is the current key (strings; supports "nf4"); fall back to the
    # legacy numeric "bits" key so old npz caches still re-plot.
    if "settings" in data:
        settings = [str(s) for s in data["settings"]]
    else:
        settings = [str(int(b)) for b in data["bits"]]
    m_hon = data["m_hon"]; tieness = data["tieness"]
    rhos = np.unique(np.round(np.geomspace(1.0 / BATCH, 1.0, 22) * BATCH) / BATCH)
    TARGET = 0.95
    C_TRI, C_RND, C_FULL = "#2ca02c", "#d1902f", "#1f77b4"

    fig, axes = plt.subplots(1, len(settings), figsize=(4.9 * len(settings), 4.6), sharey=True)
    if len(settings) == 1:
        axes = [axes]
    summary = []
    for ax, setting in zip(axes, settings):
        mq = data[f"mq_{setting}"]
        flip = float((mq > 1e-6).mean())
        m_tri, s_tri = auc_curve(m_hon, mq, tieness, rhos, "triage")
        m_rnd, s_rnd = auc_curve(m_hon, mq, tieness, rhos, "random")
        ax.axhline(1.0, ls=":", color=C_FULL, lw=1.3, label="full recompute (AUC 1.0)")
        ax.axhline(TARGET, ls="--", color="#888", lw=0.8)
        ax.fill_between(rhos, m_tri - s_tri, m_tri + s_tri, color=C_TRI, alpha=0.18)
        ax.fill_between(rhos, m_rnd - s_rnd, m_rnd + s_rnd, color=C_RND, alpha=0.18)
        ax.plot(rhos, m_tri, "-o", color=C_TRI, ms=4, lw=2.1, label="triaged (proxy-q ties)")
        ax.plot(rhos, m_rnd, "--s", color=C_RND, ms=3.2, lw=1.8, label="random subsample")
        ct = cost_to_reach(rhos, m_tri, TARGET); cr = cost_to_reach(rhos, m_rnd, TARGET)
        fac = (cr / ct) if (ct and cr) else float("nan")
        summary.append((setting, flip, ct, cr, fac))
        if ct and cr:
            ax.axvspan(ct, cr, color="#bbb", alpha=0.20)
            ax.annotate(f"{fac:.1f}x fewer\nrecomputes\nto AUC {TARGET}",
                        xy=(np.sqrt(ct * cr), TARGET), xytext=(0.16, 0.66),
                        fontsize=9, color="#222", ha="left",
                        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9))
        ax.set_xscale("log")
        ax.set_title(f"{quantlib.quant_label(setting)} weight-quant   (flip-rate {flip*100:.1f}%)",
                     fontsize=10)
        ax.set_xlabel("recomputation ratio  (fraction of tokens re-run on M)", fontsize=9.5)
        ax.set_xlim(1.0 / BATCH, 1.0); ax.set_ylim(0.5, 1.02); ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    axes[0].set_ylabel("detection AUC  (honest vs quant)", fontsize=10)
    fig.suptitle("Sparse token-flip signal sharpens the win: tie-triaged selective recompute vs random\n"
                 "Token-DiFR margin fires only at near-tie flips (few % of tokens); proxy-q ties "
                 "point straight at them.\n"
                 f"M=Qwen3-1.7B, proxy=Qwen3-0.6B; statistic=mean margin over audited tokens; "
                 f"band=±1 std / {BOOT} seeds", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    log(f"wrote figure -> {FIG}")
    log(f"\nrecompute ratio to reach AUC >= {TARGET}:")
    log(f"  {'quant':>5}{'flip%':>8}{'triaged':>10}{'random':>9}{'saving':>9}")
    for setting, flip, ct, cr, fac in summary:
        log(f"  {quantlib.quant_label(setting):>5}{flip*100:>7.1f}%{(ct or float('nan')):>10.3f}"
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
