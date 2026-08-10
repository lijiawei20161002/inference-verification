"""Does the *fixed* Token-DiFR hold across settings? A generalization sweep.

`exp_specdec_difr_gpu` established, on one configuration (Qwen2.5-1.5B target,
0.5B draft, K=4, T=1.0, int4 provider), that

  * Token-DiFR's honest null survives ordinary incremental decoding
    (match 0.988 cross-stack) but **collapses** under speculative decoding with a
    provider-local RNG (0.519) -- the benign shift is larger than the attack
    signal, so no threshold separates anything;
  * making the provider's speculation **seed-coupled** (common random numbers: the
    draft proposes `argmax(log q/T + G_t)` and is accepted iff it equals
    `argmax(log p/T + G_t)`) restores the null exactly (0.989) while keeping the
    int4 provider detectable (0.771).

That is a single cell. This experiment asks whether the fix *generalizes*, by
sweeping the knobs a deployment actually varies and reporting, for every cell, the
two numbers that decide it:

  d'_benign  -- honest speculative provider vs the honest incremental-decode
                baseline. This is the FALSE-POSITIVE side. The fix works only if
                it is ~0; under provider-local RNG it is the thing that breaks.
  d'_attack  -- cheating provider vs honest provider *in the same regime*. This is
                the POWER side, and it must not be paid for the fix.

Both are on `ivgym.signal.per_token_stats`' scale, so they are directly comparable
to `d'(quant_4bit) = 0.0775` and `d'(kv_fp8) = 0.015` elsewhere in the repo, and a
standardized pAUC @ FPR <= 0.5% is reported at a batch inside the 10% batch/pool
ceiling (`ivgym.harness.EvalConfig`).

Axes swept
  speculation depth K       1, 2, 4, 8
  temperature               0.4, 0.7, 1.0
  truncation                none, top-p 0.9, top-k 50   (the coupled Gumbel-argmax
                            is taken over the *filtered* support, so provider and
                            verifier must agree on set membership -- a real failure
                            mode for rank/support-based verification)
  model pair / family       Qwen2.5-1.5B<-0.5B, Qwen2.5-3B<-0.5B, Qwen3-1.7B<-0.6B,
                            and a cross-family draft (Qwen2.5-Coder-0.5B)
  provider forward shape    single-stream vs a batch of 4 (benign, and known from
                            `exp_specdec_shape_gpu` to move logits)
  verifier forward shape    batch 1 vs batch 4, same weights, cross-stack dtype
  provider deviation        clean, int8/int4/int3 RTN-g128 weights, fp8 KV cache,
                            temperature retune, wholesale model substitution,
                            and Medusa-style typical acceptance (a lossy cheat)

Per-token scores. Two, both computed by the verifier's replay:
  mismatch  = 1 - [replayed token == claimed token]   (what DiFR's match rate is)
  margin    = (log p(x*)/T + G(x*)) - (log p(claimed)/T + G(claimed)) >= 0, the
              post-Gumbel deficit of the claimed token -- the continuous statistic
              `verifiers.token_difr` actually aggregates. Reported alongside
              because a binary indicator throws away how badly a token missed.

Usage
    python -m experiments.exp_specdec_difr_sweep_gpu --smoke
    python -m experiments.exp_specdec_difr_sweep_gpu --out sweep
    python -m experiments.exp_specdec_difr_sweep_gpu --analyze-only --out sweep

Writes `docs/results/specdec_difr_<out>.jsonl` (one line per cell, appended as it
runs, so a partial sweep is still analyzable) and prints the table.
"""
from __future__ import annotations

import argparse, copy, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "docs" / "results"

import sys
sys.path.insert(0, str(ROOT))
from ivgym import metrics, signal                     # noqa: E402
from ivgym.harness import batch_means                 # noqa: E402
from experiments.specdec_common import load_prompts   # noqa: E402

DEV = "cuda"
DTYPE = torch.float16            # the provider's arithmetic
VDTYPE = torch.bfloat16          # the verifier's arithmetic (cross-stack channel)
SEED = 20260809                  # DiFR's shared sampling seed
OOS_MARGIN = 1e6                 # claimed token outside the verifier's support

PAIRS = {
    "q2.5-1.5b": dict(target="Qwen/Qwen2.5-1.5B-Instruct", draft="Qwen/Qwen2.5-0.5B-Instruct"),
    "q2.5-3b":   dict(target="Qwen/Qwen2.5-3B-Instruct",   draft="Qwen/Qwen2.5-0.5B-Instruct"),
    "q3-1.7b":   dict(target="Qwen/Qwen3-1.7B",            draft="Qwen/Qwen3-0.6B"),
    "q2.5-1.5b/coder-draft": dict(target="Qwen/Qwen2.5-1.5B-Instruct",
                                  draft="Qwen/Qwen2.5-Coder-0.5B-Instruct"),
}


# ---------------------------------------------------------------- seeded Gumbel
_GCACHE: dict[tuple[int, int], torch.Tensor] = {}


def gumbel(pos: int, V: int) -> torch.Tensor:
    """The shared Gumbel vector at absolute position `pos`. Provider and verifier
    derive it from (seed, pos) alone -- this is DiFR's synchronization channel."""
    key = (V, pos)
    g = _GCACHE.get(key)
    if g is None:
        gen = torch.Generator(device=DEV)
        gen.manual_seed(SEED * 1000003 + pos)
        u = torch.rand(V, generator=gen, device=DEV, dtype=torch.float32)
        g = -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_max(-1e-20))
        if len(_GCACHE) < 2048:
            _GCACHE[key] = g
    return g


def filter_(z: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Truncate a (already temperature-scaled) logit vector to the sampling support.

    Order is temperature -> top-k -> top-p, which is what serving stacks do; the
    verifier applies the identical rule to its own logits, so any disagreement about
    membership is genuine numeric disagreement, not a protocol difference."""
    if top_k:
        kth = torch.topk(z, min(top_k, z.shape[-1])).values[..., -1]
        z = z.masked_fill(z < kth, float("-inf"))
    if top_p < 1.0:
        s, idx = torch.sort(z, descending=True)
        p = F.softmax(s, dim=-1)
        keep = (p.cumsum(-1) - p) < top_p          # smallest set with cum >= top_p
        mask = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, idx, keep)
        z = z.masked_fill(~mask, float("-inf"))
    return z


def perturbed(lg: torch.Tensor, pos: int, T: float, top_k: int, top_p: float):
    z = filter_(lg.float() / T, top_k, top_p)
    lp = F.log_softmax(z, dim=-1)
    return lp + gumbel(pos, lp.shape[-1])


def gumbel_argmax(lg, pos, T, top_k=0, top_p=1.0) -> int:
    return int(torch.argmax(perturbed(lg, pos, T, top_k, top_p)).item())


# ------------------------------------------------------------ provider deviations
def fake_quant_(model, bits=4, group=128):
    """Symmetric round-to-nearest group quantization, in place. Genuine weight
    quantization (deterministic, sparse, heavy-tailed), not a logit perturbation."""
    qmax = 2 ** (bits - 1) - 1
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            W = mod.weight.data.float()
            O, I = W.shape
            if I % group:
                continue
            Wg = W.view(O, I // group, group)
            s = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
            mod.weight.data = (torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s
                               ).view(O, I).to(DTYPE)
            n += 1
    return n


def quantize_cache_(cache) -> None:
    """Round-trip every K/V tensor through fp8 e4m3 -- an fp8 KV cache, applied for
    real rather than modelled as noise."""
    for layer in cache.layers:
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if torch.is_tensor(t):
                setattr(layer, attr, t.to(torch.float8_e4m3fn).to(t.dtype))


# ----------------------------------------------------------------------- runners
class Runner:
    """(KV cache, next-token logits) for row 0 of a batch of `B` rows.

    `B > 1` fills rows 1..B-1 with unrelated random tokens of the same length and
    feeds them the same tokens. Rows cannot influence each other through attention,
    so anything the filler changes about row 0's logits is the batch-invariance
    problem -- the provider's *serving shape*, which a verifier does not share."""

    def __init__(self, model, B: int = 1, kv_fp8: bool = False, seed: int = 11):
        self.m, self.B, self.kv_fp8, self.seed = model, B, kv_fp8, seed
        self.cache = None
        self.next_logits = None
        self.len = 0

    def _rows(self, ids):
        if self.B == 1:
            return ids
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        hi = min(int(self.m.config.vocab_size), 30000)
        filler = torch.randint(1000, hi, (self.B - 1, ids.shape[1]), generator=g).to(ids.device)
        return torch.cat([ids, filler], 0)

    def prefill(self, ids):
        out = self.m(self._rows(ids), use_cache=True)
        self.cache = out.past_key_values
        if self.kv_fp8:
            quantize_cache_(self.cache)
        self.next_logits = out.logits[0, -1]
        self.len = ids.shape[1]

    def feed(self, toks):
        ids = torch.tensor([toks], device=DEV).expand(self.B, len(toks))
        out = self.m(ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        if self.kv_fp8:
            quantize_cache_(self.cache)
        self.len += len(toks)
        self.next_logits = out.logits[0, -1]
        return out.logits[0]

    def rewind_to(self, L, last_tok):
        self.cache.crop(L)
        self.len = L
        self.feed([last_tok])


def decode_exact(target, prompt_ids, N, T, tk, tp, B=1, kv_fp8=False, T_prov=None):
    """Plain incremental decode under seeded Gumbel-Max: DiFR's own null."""
    r = Runner(target, B=B, kv_fp8=kv_fp8)
    r.prefill(prompt_ids)
    P = prompt_ids.shape[1]
    toks, roles = [], []
    for i in range(N):
        toks.append(gumbel_argmax(r.next_logits, P + i, T_prov or T, tk, tp))
        roles.append("exact")
        r.feed([toks[-1]])
    return toks, roles, dict(n_rounds=N, n_drafted=0, n_accepted=0)


def decode_spec(target, draft, prompt_ids, N, T, tk, tp, K=4, mode="coupled",
                gen=None, B=1, kv_fp8=False, T_prov=None):
    """mode: 'coupled' (common random numbers -- the fix) | 'standard'
    (Leviathan/Chen rejection sampling with a provider-local RNG -- what real
    serving stacks do) | 'typical' (Medusa typical acceptance, deliberately lossy)."""
    Tp = T_prov or T
    rt = Runner(target, B=B, kv_fp8=kv_fp8)
    rd = Runner(draft, B=B)
    rt.prefill(prompt_ids)
    rd.prefill(prompt_ids)
    P = prompt_ids.shape[1]
    toks, roles = [], []
    n_rounds = n_drafted = n_accepted = 0
    t = P
    while len(toks) < N:
        d_toks, q_list = [], []
        for i in range(K):
            lg = rd.next_logits
            zq = filter_(lg.float() / Tp, tk, tp)
            q_list.append(F.softmax(zq, dim=-1))
            dt = (gumbel_argmax(lg, t + i, Tp, tk, tp) if mode == "coupled"
                  else int(torch.multinomial(q_list[-1], 1, generator=gen).item()))
            d_toks.append(dt)
            rd.feed([dt])

        lg_first = rt.next_logits
        out = rt.feed(d_toks)
        p_logits = [lg_first] + [out[i] for i in range(K)]
        n_rounds += 1
        n_drafted += K

        emitted, rr, rejected = [], [], None
        for i in range(K):
            zp = filter_(p_logits[i].float() / Tp, tk, tp)
            p = F.softmax(zp, dim=-1)
            dt = d_toks[i]
            if mode == "coupled":
                tgt = int(torch.argmax(F.log_softmax(zp, -1) + gumbel(t + i, p.shape[-1])).item())
                if tgt == dt:
                    emitted.append(dt); rr.append("accept")
                else:
                    rejected = i; emitted.append(tgt); rr.append("residual"); break
            elif mode == "typical":
                H = float(-(p * p.clamp_min(1e-12).log()).sum())
                if float(p[dt]) > min(0.09, 0.3 * math.exp(-H)):
                    emitted.append(dt); rr.append("accept")
                else:
                    rejected = i
                    emitted.append(int(torch.multinomial(p, 1, generator=gen).item()))
                    rr.append("residual"); break
            else:
                u = float(torch.rand(1, generator=gen, device=DEV))
                if u < float(p[dt]) / max(float(q_list[i][dt]), 1e-30):
                    emitted.append(dt); rr.append("accept")
                else:
                    rejected = i
                    resid = (p - q_list[i]).clamp_min(0)
                    resid = resid / resid.sum().clamp_min(1e-30)
                    emitted.append(int(torch.multinomial(resid, 1, generator=gen).item()))
                    rr.append("residual"); break
        if rejected is None:
            zp = filter_(p_logits[K].float() / Tp, tk, tp)
            b = (int(torch.argmax(F.log_softmax(zp, -1) + gumbel(t + K, zp.shape[-1])).item())
                 if mode == "coupled"
                 else int(torch.multinomial(F.softmax(zp, -1), 1, generator=gen).item()))
            emitted.append(b); rr.append("bonus")
        n_accepted += sum(1 for x in rr if x == "accept")

        m = len(emitted)
        toks += emitted
        roles += rr
        rt.rewind_to(t + m - 1, emitted[-1])
        rd.rewind_to(t + m - 1, emitted[-1])
        t += m
    return (toks[:N], roles[:N],
            dict(n_rounds=n_rounds, n_drafted=n_drafted, n_accepted=n_accepted))


# ---------------------------------------------------------------------- verifier
@torch.no_grad()
def replay(ref, prompt_ids, toks, T, tk, tp, B=1, seed=23):
    """One batched teacher-forced pass: DiFR's audit. Returns (mismatch, margin)."""
    full = torch.cat([prompt_ids, torch.tensor([toks], device=DEV)], dim=1)
    x = full
    if B > 1:
        g = torch.Generator(device="cpu").manual_seed(seed)
        hi = min(int(ref.config.vocab_size), 30000)
        filler = torch.randint(1000, hi, (B - 1, full.shape[1]), generator=g).to(DEV)
        x = torch.cat([full, filler], 0)
    out = ref(x)
    P = prompt_ids.shape[1]
    mism, marg = [], []
    for i, claimed in enumerate(toks):
        per = perturbed(out.logits[0, P - 1 + i], P + i, T, tk, tp)
        best = per.max()
        mism.append(int(int(torch.argmax(per).item()) != claimed))
        c = float(per[claimed])
        marg.append(OOS_MARGIN if not math.isfinite(c) else round(float(best) - c, 5))
    return mism, marg


# -------------------------------------------------------------------- cell table
BASE = dict(K=4, T=1.0, top_k=0, top_p=1.0, pbatch=1)


def cell_key(c):
    return (c["pair"], c["mode"], c["prov"], c["K"], c["T"], c["top_k"], c["top_p"], c["pbatch"])


def build_cells(pairs) -> list[dict]:
    out, seen = [], set()

    def add(**kw):
        c = dict(BASE, **kw)
        if cell_key(c) not in seen:
            seen.add(cell_key(c))
            out.append(c)

    for pair in pairs:                                  # every pair: the core 5
        for mode, prov in (("exact", "clean"), ("coupled", "clean"), ("standard", "clean"),
                           ("coupled", "int4"), ("standard", "int4")):
            add(pair=pair, mode=mode, prov=prov)
    p0 = pairs[0]
    for K in (1, 2, 8):                                 # speculation depth
        for prov in ("clean", "int4"):
            add(pair=p0, mode="coupled", prov=prov, K=K)
            add(pair=p0, mode="exact", prov=prov)       # (dedupes to the base exact)
    for T, tk, tp in ((0.4, 0, 1.0), (0.7, 0, 1.0), (1.0, 0, 0.9), (1.0, 50, 1.0)):
        for prov in ("clean", "int4"):                  # temperature x truncation
            add(pair=p0, mode="coupled", prov=prov, T=T, top_k=tk, top_p=tp)
            add(pair=p0, mode="exact", prov=prov, T=T, top_k=tk, top_p=tp)
    for prov in ("clean", "int4"):                      # provider serving shape
        add(pair=p0, mode="coupled", prov=prov, pbatch=4)
        add(pair=p0, mode="exact", prov=prov, pbatch=4)
    for prov in ("int8", "int3", "sub", "temp1.05", "kvfp8"):   # more deviations
        add(pair=p0, mode="coupled", prov=prov)
        add(pair=p0, mode="exact", prov=prov)
    add(pair=p0, mode="typical", prov="clean")          # a lossy-acceptance cheat
    return out


# --------------------------------------------------------------------- execution
def load(name, dtype):
    return AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="sdpa").to(DEV).eval()


def encode(tok, prompt):
    try:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids.to(DEV)


def run_pair(pair: str, cells: list[dict], args, out_path: Path, done: set):
    """Load one (target, draft) pair and run every cell that needs it."""
    spec = PAIRS[pair]
    todo = [c for c in cells if c["pair"] == pair and cell_key(c) not in done]
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(spec["target"])
    t0 = time.time()
    clean = load(spec["target"], DTYPE)
    draft = load(spec["draft"], DTYPE)
    ref_v = load(spec["target"], VDTYPE)      # cross-stack verifier: same weights
    print(f"[{pair}] loaded in {time.time()-t0:.0f}s", flush=True)

    provs: dict[str, torch.nn.Module] = {"clean": clean, "kvfp8": clean, "temp1.05": clean,
                                         "sub": draft}
    for bits in (3, 4, 8):
        tag = f"int{bits}"
        if any(c["prov"] == tag for c in todo):
            q = copy.deepcopy(clean)
            n = fake_quant_(q, bits=bits)
            print(f"[{pair}] {tag}: quantized {n} linears", flush=True)
            provs[tag] = q

    ids = [encode(tok, p) for p in load_prompts(args.nprompt)]

    for c in todo:
        t1 = time.time()
        provider = provs[c["prov"]]
        rec = dict(c, nprompt=args.nprompt, ntok=args.ntok,
                   mism=[], marg=[], mism_same=[], marg_same=[], mism_vb=[], marg_vb=[],
                   roles=[], rounds=0, drafted=0, accepted=0)
        T, tk, tp = c["T"], c["top_k"], c["top_p"]
        T_prov = T * 1.05 if c["prov"] == "temp1.05" else None
        kv8 = c["prov"] == "kvfp8"
        for pi, pid in enumerate(ids):
            gen = torch.Generator(device=DEV); gen.manual_seed(777 + 31 * pi)
            with torch.no_grad():
                if c["mode"] == "exact":
                    toks, roles, st = decode_exact(provider, pid, args.ntok, T, tk, tp,
                                                   B=c["pbatch"], kv_fp8=kv8, T_prov=T_prov)
                else:
                    toks, roles, st = decode_spec(provider, draft, pid, args.ntok, T, tk, tp,
                                                  K=c["K"], mode=c["mode"], gen=gen,
                                                  B=c["pbatch"], kv_fp8=kv8, T_prov=T_prov)
                m_x, g_x = replay(ref_v, pid, toks, T, tk, tp)            # cross-stack
                m_s, g_s = replay(clean, pid, toks, T, tk, tp)            # same-stack
                m_b, g_b = replay(ref_v, pid, toks, T, tk, tp, B=4)       # verifier batch 4
            rec["mism"] += m_x; rec["marg"] += g_x
            rec["mism_same"] += m_s; rec["marg_same"] += g_s
            rec["mism_vb"] += m_b; rec["marg_vb"] += g_b
            rec["roles"] += roles
            for k in ("rounds", "drafted", "accepted"):
                rec[k] += st["n_" + k]
        rec["secs"] = round(time.time() - t1, 1)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.add(cell_key(c))
        print(f"[{pair}] {c['mode']:8s} {c['prov']:9s} K={c['K']} T={c['T']} "
              f"tk={c['top_k']} tp={c['top_p']} pb={c['pbatch']} | "
              f"mismatch={np.mean(rec['mism']):.4f} (same-stack {np.mean(rec['mism_same']):.4f}) "
              f"| {rec['secs']:.0f}s", flush=True)

    del clean, draft, ref_v, provs
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------- analysis
MARGIN_CAP = 30.0    # nats; also what the out-of-support sentinel collapses to


def _cap(x):
    """Cap at a FIXED value, not at an honest percentile.

    `harness.evaluate` winsorizes at the honest 99.9th percentile, which is right
    for a score whose honest distribution is diffuse. It is wrong here: a coupled
    honest provider matches the replay at ~99% of positions, so the honest margin is
    *exactly 0* well past the 99.9th percentile, and percentile winsorization would
    clip both arms to zero and report d' = 0 for every attack. A fixed cap does the
    one job winsorization is needed for -- bounding the out-of-support sentinel."""
    return np.minimum(np.asarray(x, float), MARGIN_CAP)


def dprime(honest, dev):
    """d' on `signal.per_token_stats`' scale (no percentile winsorization; see
    `_cap`). `inf` when the honest arm has zero variance -- which is a real answer
    at small pools, not a number to average."""
    st = signal.per_token_stats(_cap(honest), _cap(dev), winsor_pct=None)
    if st["honest_sd"] < 1e-9:
        st["d_prime"] = float("inf") if st["attack_mean"] > st["honest_mean"] else 0.0
    return st


def pauc(honest, dev, batch, seed=0, n_batches=4000):
    rng = np.random.default_rng(seed)
    return metrics.partial_auc(batch_means(_cap(honest), batch, n_batches, rng),
                               batch_means(_cap(dev), batch, n_batches, rng),
                               max_fpr=0.005)


def analyze(path: Path, batch: int | None = None):
    recs = [json.loads(l) for l in open(path)]
    by = {cell_key(r): r for r in recs}
    pool = min(len(r["mism"]) for r in recs)
    b = batch or max(1, int(0.08 * pool))            # inside the 10% ceiling
    print(f"\n{len(recs)} cells, pool={pool} tokens/cell, batch={b} "
          f"({b/pool:.1%} of pool)\n")

    def setting(r):
        return (r["pair"], r["K"], r["T"], r["top_k"], r["top_p"], r["pbatch"])

    def get(pair, mode, prov, K, T, tk, tp, pb):
        return by.get((pair, mode, prov, K, T, tk, tp, pb))

    rows = []
    for r in recs:
        if r["prov"] == "clean" and r["mode"] != "typical":
            continue                                  # honest cells are the baselines
        pair, K, T, tk, tp, pb = setting(r)
        mode = r["mode"]
        # `typical` IS the deviation (a provider that claims seed-coupled
        # speculation and quietly runs lossy acceptance), so its null is the
        # coupled honest provider, not itself.
        honest = get(pair, "coupled" if mode == "typical" else mode,
                     "clean", K, T, tk, tp, pb)
        if honest is None:
            continue
        d_a = dprime(honest["marg"], r["marg"])["d_prime"]
        d_a_m = dprime(honest["mism"], r["mism"])["d_prime"]
        rows.append(dict(kind="attack", pair=pair, mode=mode, prov=r["prov"], K=K, T=T,
                         top_k=tk, top_p=tp, pbatch=pb,
                         honest_mismatch=float(np.mean(honest["mism"])),
                         mismatch=float(np.mean(r["mism"])),
                         d_margin=d_a, d_mismatch=d_a_m,
                         pauc=pauc(honest["marg"], r["marg"], b),
                         pauc_mismatch=pauc(honest["mism"], r["mism"], b)))

    for r in recs:                                    # benign: spec vs exact, honest
        if r["prov"] != "clean" or r["mode"] in ("exact", "typical"):
            continue
        pair, K, T, tk, tp, pb = setting(r)
        ex = get(pair, "exact", "clean", K, T, tk, tp, pb) or \
             get(pair, "exact", "clean", BASE["K"], T, tk, tp, pb)
        if ex is None:
            continue
        rows.append(dict(kind="benign", pair=pair, mode=r["mode"], prov="clean", K=K, T=T,
                         top_k=tk, top_p=tp, pbatch=pb,
                         honest_mismatch=float(np.mean(ex["mism"])),
                         mismatch=float(np.mean(r["mism"])),
                         d_margin=dprime(ex["marg"], r["marg"])["d_prime"],
                         d_mismatch=dprime(ex["mism"], r["mism"])["d_prime"],
                         pauc=pauc(ex["marg"], r["marg"], b),
                         pauc_mismatch=pauc(ex["mism"], r["mism"], b)))

    hdr = (f"{'setting':46s} {'kind':7s} {'prov':9s} {'mism_h':>7s} {'mism':>7s} "
           f"{'d_marg':>8s} {'d_mism':>8s} {'pAUC':>6s} {'pAUC_m':>7s}")
    print(hdr); print("-" * len(hdr))
    for x in sorted(rows, key=lambda x: (x["pair"], x["mode"], x["kind"], x["prov"],
                                         x["K"], x["T"], x["top_k"], x["top_p"], x["pbatch"])):
        s = (f"{x['pair']}/{x['mode']} K={x['K']} T={x['T']} tk={x['top_k']} "
             f"tp={x['top_p']} pb={x['pbatch']}")
        print(f"{s:46s} {x['kind']:7s} {x['prov']:9s} {x['honest_mismatch']:7.4f} "
              f"{x['mismatch']:7.4f} {x['d_margin']:8.3f} {x['d_mismatch']:8.3f} "
              f"{x['pauc']:6.3f} {x['pauc_mismatch']:7.3f}")

    summary = dict(pool=pool, batch=b, rows=rows,
                   cells=[{k: v for k, v in r.items()
                           if k not in ("mism", "marg", "mism_same", "marg_same",
                                        "mism_vb", "marg_vb", "roles")}
                          | dict(mismatch=float(np.mean(r["mism"])),
                                 mismatch_same=float(np.mean(r["mism_same"])),
                                 mismatch_vb4=float(np.mean(r["mism_vb"])),
                                 accept_rate=(r["accepted"] / r["drafted"]
                                              if r["drafted"] else None),
                                 tokens_per_round=(len(r["mism"]) / r["rounds"]
                                                   if r["rounds"] else None))
                          for r in recs])
    sp = path.with_name(path.stem + "_summary.json")
    sp.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {sp.relative_to(ROOT)}")
    return summary


# -------------------------------------------------- the transparency claim, direct
def identity(args):
    """Is seed-coupled speculation *output-identical* to plain seeded decoding?

    The sweep above shows the two are statistically indistinguishable to DiFR. This
    is the stronger and much cheaper statement: run both on the same prompt with the
    same seed and compare the token sequences directly. Three arms per (K, T):

      coupled vs exact   -- the claim. Any mismatch is numeric, not algorithmic.
      exact  vs exact    -- the rerun control: the same shape twice, which
                            `exp_specdec_shape_gpu` finds bitwise identical, so this
                            arm should be 1.000 and bounds the floor.
      standard vs exact  -- what a provider-local RNG does to the same comparison.

    Writes `docs/results/specdec_difr_identity.json`.
    """
    spec = PAIRS[args.pairs.split(",")[0]]
    tok = AutoTokenizer.from_pretrained(spec["target"])
    target, draft = load(spec["target"], DTYPE), load(spec["draft"], DTYPE)
    ids = [encode(tok, p) for p in load_prompts(args.nprompt)]
    grid = [(K, T) for K in (1, 2, 4, 8) for T in (0.7, 1.0)] + [(4, 0.4)]
    out = []
    for K, T in grid:
        for arm in ("coupled", "exact_rerun", "standard"):
            same, first_div, n = 0, [], 0
            for pi, pid in enumerate(ids):
                gen = torch.Generator(device=DEV); gen.manual_seed(777 + 31 * pi)
                with torch.no_grad():
                    ref_toks, _, _ = decode_exact(target, pid, args.ntok, T, 0, 1.0)
                    if arm == "exact_rerun":
                        alt, _, st = decode_exact(target, pid, args.ntok, T, 0, 1.0)
                    else:
                        alt, _, st = decode_spec(target, draft, pid, args.ntok, T, 0, 1.0,
                                                 K=K, mode=arm, gen=gen)
                eq = [int(a == b) for a, b in zip(ref_toks, alt)]
                same += sum(eq); n += len(eq)
                first_div.append(next((i for i, e in enumerate(eq) if not e), None))
            rec = dict(K=K, T=T, arm=arm, n=n, identical_frac=same / n,
                       n_seq=len(ids),
                       n_seq_fully_identical=sum(1 for d in first_div if d is None),
                       median_first_divergence=float(np.median(
                           [d if d is not None else args.ntok for d in first_div])))
            out.append(rec)
            print(f"K={K} T={T} {arm:12s} token-identical={rec['identical_frac']:.4f} "
                  f"seqs fully identical={rec['n_seq_fully_identical']}/{len(ids)}", flush=True)
    p = RES_DIR / "specdec_difr_identity.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprompt", type=int, default=16)
    ap.add_argument("--ntok", type=int, default=192)
    ap.add_argument("--pairs", default="q2.5-1.5b,q3-1.7b,q2.5-3b,q2.5-1.5b/coder-draft")
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--batch", type=int, default=0, help="eval batch; 0 = 8% of pool")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--identity", action="store_true",
                    help="run the token-identity arm instead of the sweep")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.nprompt, args.ntok, args.out = 2, 48, "sweepsmoke"
        args.pairs = "q2.5-1.5b"

    RES_DIR.mkdir(parents=True, exist_ok=True)
    if args.identity:
        identity(args)
        return
    out_path = RES_DIR / f"specdec_difr_{args.out}.jsonl"
    pairs = args.pairs.split(",")
    cells = build_cells(pairs)
    done = set()
    if out_path.exists():                              # resume
        for line in open(out_path):
            r = json.loads(line)
            if r.get("nprompt") == args.nprompt and r.get("ntok") == args.ntok:
                done.add(cell_key(r))
        print(f"resuming: {len(done)} cells already in {out_path.name}")

    if not args.analyze_only:
        print(f"{len(cells)} cells, {len(cells)-len(done)} to run "
              f"({args.nprompt} prompts x {args.ntok} tokens)", flush=True)
        for pair in pairs:
            run_pair(pair, cells, args, out_path, done)
    analyze(out_path, args.batch or None)


if __name__ == "__main__":
    main()
