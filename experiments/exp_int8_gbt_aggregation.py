"""Is the int8 wall an AGGREGATION problem or an INFORMATION problem?

`docs/results/specdec_difr_sweep.md` S4 leaves int8 RTN-g128 weights at chance
(`d'` = 0.058 on the match rate, 0.073 on the margin; pAUC 0.518/0.537 at batch 245)
while int4 sits at 1.000. Two stories fit that:

  (A) AGGREGATION. The signal is in the per-token records but `mean(margin)` throws it
      away -- 99% of honest tokens are structural zeros, the effect lives in the shape
      of the 1% tail, and a mean over a zero-inflated variable is a weak statistic.
      Fix = a better aggregator. This is where a gradient-boosted tree earns its keep:
      it is the cheapest way to ask "is there ANY function of these per-token features
      that separates the arms?" without guessing the functional form first.

  (B) INFORMATION. The verifier's own bf16-vs-fp16 replay gap is a *systematic bias*
      of the same magnitude as int8's weight error, so the two are not separable in
      the cross-stack channel at all. Fix = change the measurement, not the statistic.

We settle it by fitting XGBoost on the per-token features the verifier actually has and
seeing how far it can get, then re-fitting with the numerically-matched replay channel
added. The gap between the two answers the question.

Feature sets (all per-token, from `docs/results/specdec_difr_sweep.jsonl`):

  difr     margin only                       -- what `verifiers.token_difr` aggregates
  xstack   margin, mismatch, and the SECOND cross-stack replay (verifier batch 4),
           + position + speculation role     -- DEPLOYABLE: needs only the verifier's
                                                own independent stack, replayed twice
  matched  xstack + the same-stack replay    -- needs the provider's exact numerics;
                                                an upper bound, not a protocol

Training a detector needs labelled deviant tokens, which looks circular -- but is not:
the verifier can RTN-quantize the claimed weights itself and generate its own labelled
pool with no provider cooperation. The `--train int4` arm tests exactly that transfer:
fit on a deviation the verifier can synthesize, score the one it cannot.

Grouped 4-fold CV by prompt (16 prompts x 192 tokens), so no token is scored by a model
that saw its own sequence. Reported at the sweep's batch 245 / pAUC @ FPR <= 0.5%.

Run:
    .venv/bin/python -m experiments.exp_int8_gbt_aggregation
    .venv/bin/python -m experiments.exp_int8_gbt_aggregation --dev kvfp8 --train int4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import metrics, signal  # noqa: E402

SWEEP = Path("docs/results/specdec_difr_sweep.jsonl")
OUT = Path("docs/results/int8_gbt_aggregation.json")

MARGIN_CAP = 30.0          # same fixed cap as the sweep's `_cap` (see its docstring)
BATCH = 245                # 8.0% of the 3 072-token pool, as in the sweep
N_BATCHES = 4000
ROLES = ("accept", "residual", "bonus")

BASE = dict(K=4, T=1.0, top_k=0, top_p=1.0, pbatch=1)

FEATURE_SETS = {
    "difr": ["marg"],
    "xstack": ["marg", "mism", "marg_vb", "mism_vb", "pos", "role_accept",
               "role_residual", "role_bonus"],
    "matched": ["marg", "mism", "marg_vb", "mism_vb", "pos", "role_accept",
                "role_residual", "role_bonus", "marg_same", "mism_same"],
}


# ----------------------------------------------------------------------- loading
def load_cells(path: Path, pair: str):
    """{provider tag -> record} for the sweep's base coupled setting."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if (r["pair"] == pair and r["mode"] == "coupled"
                and all(r[k] == v for k, v in BASE.items())):
            out[r["prov"]] = r
    return out


def frame(rec):
    """Per-token feature table + the prompt group id each token belongs to."""
    n = len(rec["mism"])
    ntok = rec["ntok"]
    cols = {}
    for k in ("mism", "mism_same", "mism_vb"):
        cols[k] = np.asarray(rec[k], float)
    for k in ("marg", "marg_same", "marg_vb"):
        cols[k] = np.minimum(np.asarray(rec[k], float), MARGIN_CAP)
    cols["pos"] = (np.arange(n) % ntok) / ntok
    roles = np.asarray(rec["roles"])
    for r in ROLES:
        cols["role_" + r] = (roles == r).astype(float)
    groups = np.arange(n) // ntok
    return cols, groups


def matrix(cols, names):
    return np.column_stack([cols[k] for k in names])


# ----------------------------------------------------------------------- scoring
def batch_means(x, rng, batch=BATCH, n_batches=N_BATCHES):
    return x[rng.integers(0, len(x), size=(n_batches, batch))].mean(1)


SCALING_BATCHES = (61, 122, 245, 500, 1000, 2000, 4000, 8000, 16000)


def evaluate(honest, dev, seed=0):
    """(d', pAUC) for a per-token score, on the sweep's scale."""
    st = signal.per_token_stats(honest, dev, winsor_pct=None)
    d = st["d_prime"]
    if st["honest_sd"] < 1e-12:
        d = float("inf") if st["attack_mean"] > st["honest_mean"] else 0.0
    rng = np.random.default_rng(seed)
    return d, metrics.partial_auc(batch_means(honest, rng), batch_means(dev, rng),
                                  max_fpr=0.005)


def batch_scaling(honest, dev, batches=SCALING_BATCHES, n_batches=20000, seed=0):
    """pAUC as a function of audit batch size -- the "how many tokens does this cost?"
    curve. Bootstrapped, so batches above the pool size assume the observed margin tail
    is representative of the population tail (see the doc's caveats)."""
    out = []
    for b in batches:
        rng = np.random.default_rng(seed)
        out.append(dict(batch=b, pauc=metrics.partial_auc(
            batch_means(honest, rng, b, n_batches), batch_means(dev, rng, b, n_batches),
            max_fpr=0.005)))
    return out


# -------------------------------------------------------------------------- fits
def fixed_statistics(h, a):
    """Hand-built aggregators, for the "did the tree find anything a human wouldn't?"
    column. Each is a per-token score whose null is calibrable from honest data.

    The monotone transforms of the margin are the cheap version of what a tree does to a
    single feature (bin it, give each bin its own weight); a convex transform is the
    right direction if the deviation fattens the tail rather than shifting the body."""
    return {
        "mean mismatch (match rate)": (h["mism"], a["mism"]),
        "mean margin (token_difr)": (h["marg"], a["marg"]),
        "mean sqrt(margin)": (np.sqrt(h["marg"]), np.sqrt(a["marg"])),
        "mean margin^2": (h["marg"] ** 2, a["marg"] ** 2),
        "mean margin^3": (h["marg"] ** 3, a["marg"] ** 3),
        "both cross-stack replays mismatch": ((h["mism"] > 0) * (h["mism_vb"] > 0) * 1.0,
                                              (a["mism"] > 0) * (a["mism_vb"] > 0) * 1.0),
        "min margin over 2 replays": (np.minimum(h["marg"], h["marg_vb"]),
                                      np.minimum(a["marg"], a["marg_vb"])),
        "margin x 1[2nd replay mismatch]": (h["marg"] * h["mism_vb"],
                                            a["marg"] * a["mism_vb"]),
        "* same-stack margin": (h["marg_same"], a["marg_same"]),
        "* same-stack margin^2": (h["marg_same"] ** 2, a["marg_same"] ** 2),
        "* cross margin x same-stack mism": (h["marg"] * h["mism_same"],
                                             a["marg"] * a["mism_same"]),
    }


def token_auc(x, y):
    """Full per-token AUC (tie-corrected), in numpy. `ivgym.metrics` only exposes the
    partial AUC, and the tie handling matters here: 98% of tokens are exact zeros in
    both arms, so a tie-blind AUC would be meaningless."""
    v = np.r_[x, y]
    lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    o = np.argsort(v, kind="mergesort")
    v, lab = v[o], lab[o]
    rank = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        rank[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n1 = lab.sum()
    return (rank[lab == 1].sum() - n1 * (n1 + 1) / 2) / ((len(lab) - n1) * n1)


def gbt_oof(h_cols, h_groups, t_cols, t_groups, d_cols, d_groups, names,
            n_folds=4, seed=0):
    """Grouped-CV out-of-fold GBT scores.

    Fits honest-vs-`t` (the TRAINING deviation) on the training prompts of each fold and
    scores the held-out prompts of honest and of `d` (the deviation under TEST). When
    t is d this is a same-deviation power ceiling; when it is not, it is a transfer test.
    Returns (honest scores, dev scores, gain-importance dict)."""
    import xgboost as xgb

    Xh, Xt, Xd = (matrix(c, names) for c in (h_cols, t_cols, d_cols))
    gs = np.unique(h_groups)
    folds = np.array_split(gs, n_folds)
    s_h = np.full(len(Xh), np.nan)
    s_d = np.full(len(Xd), np.nan)
    gain = {k: 0.0 for k in names}
    for f in folds:
        te = np.isin(h_groups, f)
        tr_h, tr_t = ~te, ~np.isin(t_groups, f)
        X = np.vstack([Xh[tr_h], Xt[tr_t]])
        y = np.r_[np.zeros(tr_h.sum()), np.ones(tr_t.sum())]
        # Native Booster API on purpose: xgboost's sklearn wrapper imports sklearn,
        # which this repo does not depend on (see ivgym/metrics.py).
        dtrain = xgb.DMatrix(X, label=y, feature_names=list(names))
        b = xgb.train(dict(objective="binary:logistic", eval_metric="logloss",
                           max_depth=3, eta=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                           seed=seed, nthread=4), dtrain, num_boost_round=300)
        # `output_margin=True` -- the RAW additive score, not the sigmoid. Averaging
        # probabilities squashes the tail that carries the evidence; the raw score is a
        # per-token log-likelihood ratio, and Neyman-Pearson says to SUM those.
        s_h[te] = b.predict(xgb.DMatrix(Xh[te], feature_names=list(names)),
                            output_margin=True)
        te_d = np.isin(d_groups, f)
        s_d[te_d] = b.predict(xgb.DMatrix(Xd[te_d], feature_names=list(names)),
                              output_margin=True)
        for k, v in b.get_score(importance_type="gain").items():
            gain[k] += v / n_folds
    assert not np.isnan(s_h).any() and not np.isnan(s_d).any()
    return s_h, s_d, gain


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    ap.add_argument("--pair", default="q2.5-1.5b")
    ap.add_argument("--dev", default="int8", help="deviation under test")
    ap.add_argument("--train", default=None,
                    help="deviation to TRAIN the GBT on (default: same as --dev)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        tr = args.train or args.dev
        suffix = args.dev if tr == args.dev else f"{args.dev}_from_{tr}"
        args.out = OUT.with_name(f"int8_gbt_aggregation_{suffix}.json")

    cells = load_cells(args.sweep, args.pair)
    missing = [t for t in ("clean", args.dev, args.train or args.dev) if t not in cells]
    if missing:
        sys.exit(f"missing provider cells {missing} in {args.sweep}")

    h_cols, h_groups = frame(cells["clean"])
    d_cols, d_groups = frame(cells[args.dev])
    t_cols, t_groups = frame(cells[args.train or args.dev])

    n = len(h_cols["mism"])
    print(f"\npair={args.pair} coupled {BASE}  dev={args.dev} "
          f"train={args.train or args.dev}\n{n} tokens/arm, batch {BATCH} "
          f"({BATCH/n:.1%} of pool), pAUC @ FPR <= 0.5%\n")

    # The diagnostic that explains everything below: how much does a SINGLE token tell
    # you? A GBT optimizes logloss at the token level, so this bounds what it can learn.
    print("per-token AUC (what a token-level classifier has to work with):")
    for k in ("mism", "marg", "mism_vb", "marg_vb", "mism_same", "marg_same"):
        print(f"  {k:10s} {token_auc(h_cols[k], d_cols[k]):.4f}")
    per_token = {k: float(token_auc(h_cols[k], d_cols[k]))
                 for k in ("mism", "marg", "mism_vb", "marg_vb", "mism_same", "marg_same")}

    rows = []
    print(f"\n{'aggregator':40s} {'honest':>10s} {'dev':>10s} {'d_prime':>9s} {'pAUC':>7s}")
    print("-" * 80)
    for name, (sh, sd) in fixed_statistics(h_cols, d_cols).items():
        d, p = evaluate(sh, sd)
        rows.append(dict(kind="fixed", name=name, honest=float(sh.mean()),
                         dev=float(sd.mean()), d_prime=d, pauc=p))
        print(f"{name:40s} {sh.mean():10.5f} {sd.mean():10.5f} {d:9.3f} {p:7.3f}")

    print()
    gains = {}
    for fs, names in FEATURE_SETS.items():
        sh, sd, gain = gbt_oof(h_cols, h_groups, t_cols, t_groups,
                               d_cols, d_groups, names)
        d, p = evaluate(sh, sd)
        gains[fs] = gain
        label = f"XGBoost [{fs}]"
        rows.append(dict(kind="gbt", name=label, features=names,
                         honest=float(sh.mean()), dev=float(sd.mean()),
                         d_prime=d, pauc=p, gain=gain))
        print(f"{label:40s} {sh.mean():10.5f} {sd.mean():10.5f} {d:9.3f} {p:7.3f}")

    # The cost curve: "at chance" is a statement about the batch, not about the deviation.
    scaling = {"cross-stack margin": batch_scaling(h_cols["marg"], d_cols["marg"]),
               "same-stack margin": batch_scaling(h_cols["marg_same"], d_cols["marg_same"])}
    print(f"\npAUC vs audit batch size:\n{'batch':>7s} " +
          " ".join(f"{k:>19s}" for k in scaling))
    for i, b in enumerate(SCALING_BATCHES):
        print(f"{b:7d} " + " ".join(f"{scaling[k][i]['pauc']:19.3f}" for k in scaling))

    print("\n(* = uses the same-stack replay: an upper bound, not a deployable protocol)")
    print("gain importance (mean over folds):")
    for fs, gain in gains.items():
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:5]
        print(f"  {fs:8s} " + "  ".join(f"{k}={v:.0f}" for k, v in top if v > 0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        dict(pair=args.pair, base=BASE, dev=args.dev, train=args.train or args.dev,
             n_tokens=n, batch=BATCH, per_token_auc=per_token, scaling=scaling,
             rows=rows), indent=1) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
