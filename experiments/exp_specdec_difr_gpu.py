"""
Root-cause experiment: why Token-DiFR breaks under speculative decoding.

Token-DiFR (Karvonen et al., arXiv:2511.20621) audits a provider by re-running the
returned sequence through a trusted reference under *seed-synchronised Gumbel-Max*
sampling: token_t = argmax_x ( log p(x | prefix) / T + G_t(x) ), where G_t is a
Gumbel vector derived deterministically from (seed, position t).  An honest provider
reproduces the reference's token at >98% of positions; the residual mismatch is
benign floating-point nondeterminism.

This harness measures, on real models, what each provider-side decoding strategy
does to that match rate.

Provider regimes
  exact          : incremental KV-cache decode, seeded Gumbel-Max      (DiFR's null)
  spec_standard  : speculative decoding, Leviathan/Chen rejection sampling,
                   provider-local RNG (what real vLLM/SGLang/TRT-LLM do)
  spec_coupled   : speculative decoding with common random numbers -- the draft
                   proposes argmax(log q/T + G_t) and is accepted iff it equals
                   argmax(log p/T + G_t).  Lossless AND seed-consistent.
  spec_typical   : Medusa-style "typical acceptance" (deliberately lossy)
Target variants
  clean          : fp16 weights
  int4           : group-128 symmetric RTN 4-bit weight quantisation (the
                   corner-cutting DiFR is designed to catch)

The verifier always replays with clean fp16 weights in a single batched
teacher-forced forward pass, so the provider/verifier numeric mismatch is real
(cached incremental decode vs. batched prefill), not injected.

Writes `docs/results/specdec_difr_<out>.json` (`--out main`: all regimes at
T=1.0; `--out temps`: a temperature sweep). Figures come from
`python -m experiments.plot_specdec_difr`, which needs both artifacts.

    python -m experiments.exp_specdec_difr_gpu --out main
    python -m experiments.exp_specdec_difr_gpu --out temps --temps 0.1,0.3,0.6,1.0,1.4
"""

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "docs" / "results"

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda"
DTYPE = torch.float16
SEED = 20260809


# ---------------------------------------------------------------- seeded Gumbel
_GCACHE = {}


def gumbel(pos, V):
    """Deterministic Gumbel vector for absolute position `pos`. Shared by
    provider and verifier -- this is DiFR's seed-synchronisation channel."""
    key = pos
    g = _GCACHE.get(key)
    if g is None:
        gen = torch.Generator(device=DEV)
        gen.manual_seed(SEED * 1000003 + pos)
        u = torch.rand(V, generator=gen, device=DEV, dtype=torch.float32)
        g = -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_max(-1e-20))
        if len(_GCACHE) < 4096:
            _GCACHE[key] = g
    return g


def gumbel_argmax(logits, pos, T):
    lp = F.log_softmax(logits.float() / T, dim=-1)
    return int(torch.argmax(lp + gumbel(pos, lp.shape[-1])).item())


# ------------------------------------------------------------------ fake int4
def fake_quant_(model, bits=4, group=128):
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
            Wq = torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s
            mod.weight.data = Wq.view(O, I).to(DTYPE)
            n += 1
    return n


# ------------------------------------------------------------------ providers
class Runner:
    """Thin wrapper keeping (cache, logits-for-next-position) in sync."""

    def __init__(self, model):
        self.m = model
        self.cache = None
        self.next_logits = None
        self.len = 0  # number of positions currently in cache

    def prefill(self, ids):
        out = self.m(ids, use_cache=True)
        self.cache = out.past_key_values
        self.next_logits = out.logits[0, -1]
        self.len = ids.shape[1]

    def feed(self, toks):
        """toks: list[int] appended at positions self.len ... ; returns the
        logits produced at each fed position (i.e. predictions for len+1 ...)."""
        ids = torch.tensor([toks], device=DEV)
        out = self.m(ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.len += len(toks)
        self.next_logits = out.logits[0, -1]
        return out.logits[0]

    def rewind_to(self, L, last_tok):
        """Crop cache to L positions, then feed `last_tok` at position L."""
        self.cache.crop(L)
        self.len = L
        self.feed([last_tok])


def decode_exact(target, prompt_ids, N, T):
    r = Runner(target)
    r.prefill(prompt_ids)
    P = prompt_ids.shape[1]
    toks, meta = [], []
    for i in range(N):
        t = P + i
        x = gumbel_argmax(r.next_logits, t, T)
        toks.append(x)
        meta.append(dict(role="exact", off=0, acc_len=1))
        r.feed([x])
    return toks, meta, dict(n_rounds=N, n_drafted=0, n_accepted=0, sum_min=[], coupled_agree=[])


def _dists(lg, T):
    return F.softmax(lg.float() / T, dim=-1)


def decode_spec(target, draft, prompt_ids, N, T, K=4, mode="standard", gen=None):
    """mode: 'standard' (rejection sampling) | 'coupled' (common random numbers)
    | 'typical' (Medusa typical acceptance, lossy)."""
    rt, rd = Runner(target), Runner(draft)
    rt.prefill(prompt_ids)
    rd.prefill(prompt_ids)
    P = prompt_ids.shape[1]
    toks, meta = [], []
    n_drafted = n_accepted = n_rounds = 0
    sum_min, coupled_agree = [], []
    t = P
    while len(toks) < N:
        # ---- draft K tokens autoregressively from q
        d_toks, q_list = [], []
        for i in range(K):
            lg = rd.next_logits
            q = _dists(lg, T)
            q_list.append(q)
            if mode == "coupled":
                dt = gumbel_argmax(lg, t + i, T)
            else:
                dt = int(torch.multinomial(q, 1, generator=gen).item())
            d_toks.append(dt)
            rd.feed([dt])
        # ---- target verifies all K+1 positions in one pass
        lg_first = rt.next_logits
        out = rt.feed(d_toks)
        p_logits = [lg_first] + [out[i] for i in range(K)]  # p at t .. t+K
        n_rounds += 1
        n_drafted += K

        emitted, roles = [], []
        rejected_at = None
        for i in range(K):
            p = _dists(p_logits[i], T)
            q = q_list[i]
            dt = d_toks[i]
            sum_min.append(float(torch.minimum(p, q).sum()))
            if mode == "coupled":
                tgt_tok = gumbel_argmax(p_logits[i], t + i, T)
                ok = tgt_tok == dt
                coupled_agree.append(float(ok))
                if ok:
                    emitted.append(dt); roles.append("accept")
                else:
                    rejected_at = i
                    emitted.append(tgt_tok); roles.append("residual")
                    break
            elif mode == "typical":
                H = float(-(p * p.clamp_min(1e-12).log()).sum())
                thr = min(0.09, 0.3 * math.exp(-H))
                if float(p[dt]) > thr:
                    emitted.append(dt); roles.append("accept")
                else:
                    rejected_at = i
                    emitted.append(int(torch.multinomial(p, 1, generator=gen).item()))
                    roles.append("residual")
                    break
            else:
                u = float(torch.rand(1, generator=gen, device=DEV))
                if u < float(p[dt]) / max(float(q[dt]), 1e-30):
                    emitted.append(dt); roles.append("accept")
                else:
                    rejected_at = i
                    resid = (p - q).clamp_min(0)
                    resid = resid / resid.sum().clamp_min(1e-30)
                    emitted.append(int(torch.multinomial(resid, 1, generator=gen).item()))
                    roles.append("residual")
                    break
        if rejected_at is None:
            if mode == "coupled":
                b = gumbel_argmax(p_logits[K], t + K, T)
            else:
                b = int(torch.multinomial(_dists(p_logits[K], T), 1, generator=gen).item())
            emitted.append(b); roles.append("bonus")
        n_accepted += sum(1 for r_ in roles if r_ == "accept")

        m = len(emitted)
        for j, (x, r_) in enumerate(zip(emitted, roles)):
            toks.append(x)
            meta.append(dict(role=r_, off=j, acc_len=m))
        rt.rewind_to(t + m - 1, emitted[-1])
        rd.rewind_to(t + m - 1, emitted[-1])
        t += m
    return (toks[:N], meta[:N],
            dict(n_rounds=n_rounds, n_drafted=n_drafted, n_accepted=n_accepted,
                 sum_min=sum_min, coupled_agree=coupled_agree))


# ------------------------------------------------------------------- verifier
@torch.no_grad()
def replay(ref, prompt_ids, toks, T):
    """DiFR verification: one batched teacher-forced pass with clean weights."""
    full = torch.cat([prompt_ids, torch.tensor([toks], device=DEV)], dim=1)
    P = prompt_ids.shape[1]
    out = ref(full)
    rows = []
    for i, x in enumerate(toks):
        lg = out.logits[0, P - 1 + i]
        pos = P + i
        lp = F.log_softmax(lg.float() / T, dim=-1)
        pred = int(torch.argmax(lp + gumbel(pos, lp.shape[-1])).item())
        p = lp.exp()
        top1 = float(p.max())
        coll = float((p * p).sum())
        ent = float(-(p * lp).sum())
        rows.append(dict(match=int(pred == x), top1=top1, coll=coll, ent=ent,
                         p_emitted=float(p[x])))
    return rows


# ----------------------------------------------------------------------- main
PROMPTS = [
    "Explain why the sky is blue, in three sentences.",
    "Write a short poem about a lighthouse in winter.",
    "What are the trade-offs between TCP and UDP?",
    "Describe a recipe for a simple tomato pasta.",
    "Summarise the causes of the French Revolution.",
    "Give me three ideas for a weekend project in Python.",
    "How does a refrigerator work?",
    "Tell a very short story about a lost umbrella.",
    "Compare renting and buying a home.",
    "What is the difference between latency and throughput?",
    "Draft a friendly email declining a meeting invitation.",
    "Explain gradient descent to a high-school student.",
    "List some tips for keeping houseplants alive.",
    "What makes sourdough bread different from other bread?",
    "Describe the plot of a made-up detective novel.",
    "Why do cats purr?",
    "Explain the CAP theorem briefly.",
    "Write a haiku about debugging.",
    "What should I consider when buying a used bicycle?",
    "Outline an argument for teaching statistics before calculus.",
    "How do noise-cancelling headphones work?",
    "Give a gentle explanation of what a hash function is.",
    "Suggest a training plan for a first 10k run.",
    "What are common mistakes when writing unit tests?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprompt", type=int, default=24)
    ap.add_argument("--ntok", type=int, default=256)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", default="main", help="writes docs/results/specdec_difr_<out>.json")
    ap.add_argument("--temps", default="1.0")
    ap.add_argument("--regimes", default="exact,spec_standard,spec_coupled,spec_typical,"
                                         "q4_exact,q4_spec_standard,q4_spec_coupled")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TARGET)
    t0 = time.time()
    ref = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=DTYPE, attn_implementation="sdpa").to(DEV).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=DTYPE, attn_implementation="sdpa").to(DEV).eval()
    # The verifier is a *different implementation* of the same weights (bf16
    # accumulation vs the provider's fp16) -- this is the real cross-stack
    # benign-nondeterminism channel DiFR has to tolerate, not injected noise.
    # (fp16 `eager` attention overflows to NaN on Qwen2, so bf16 is the usable
    # honest cross-implementation knob here.)
    ref_v = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    print(f"loaded clean target+draft+verifier in {time.time()-t0:.0f}s", flush=True)

    need_q4 = any(r.startswith("q4") for r in args.regimes.split(","))
    q4 = None
    if need_q4:
        q4 = AutoModelForCausalLM.from_pretrained(TARGET, dtype=DTYPE).to(DEV).eval()
        print("int4 layers quantised:", fake_quant_(q4), flush=True)

    prompts = PROMPTS[: args.nprompt]
    def enc(p):
        r = tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(r):
            r = r["input_ids"]
        return r.to(DEV)

    ids = [enc(p) for p in prompts]

    records = []
    for T in [float(x) for x in args.temps.split(",")]:
        for regime in args.regimes.split(","):
            provider = q4 if regime.startswith("q4") else ref
            kind = regime.replace("q4_", "")
            t1 = time.time()
            for pi, pid in enumerate(ids):
                gen = torch.Generator(device=DEV); gen.manual_seed(777 + 31 * pi)
                with torch.no_grad():
                    if kind == "exact":
                        toks, meta, stats = decode_exact(provider, pid, args.ntok, T)
                    else:
                        toks, meta, stats = decode_spec(
                            provider, draft, pid, args.ntok, T, K=args.K,
                            mode=kind.replace("spec_", ""), gen=gen)
                    rows = replay(ref_v, pid, toks, T)          # cross-stack verifier
                    same = replay(ref, pid, toks, T)            # same-stack control
                for r, s in zip(rows, same):
                    r["match_same"] = s["match"]
                for j, (r, m) in enumerate(zip(rows, meta)):
                    records.append(dict(T=T, regime=regime, prompt=pi, pos=j, **r, **m))
                records[-1]["stats"] = stats
            print(f"T={T} {regime:18s} {time.time()-t1:6.0f}s "
                  f"match={np.mean([r['match'] for r in records if r['regime']==regime and r['T']==T]):.4f}",
                  flush=True)

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RES_DIR / f"specdec_difr_{args.out}.json"
    with open(out_path, "w") as f:
        json.dump(records, f)
    print("wrote", out_path.relative_to(ROOT), len(records), "rows")


if __name__ == "__main__":
    main()
