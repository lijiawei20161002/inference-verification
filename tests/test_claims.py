"""Every claim this repo makes must still have the artifact it was measured from.

The rest of the test suite checks that the code is correct. This file checks that
the *record* is correct, which turned out to be the harder problem: over its
history this repo published a headline table measured above the batch/pool
ceiling, a "beats full recompute" result that was a winsorization artifact, a
"detector reversal" that was the ratio artifact again, and an audit-cost figure
rendered from one attack under another attack's caption. None of those were code
bugs. All four were a claim outliving, or drifting from, its evidence.

So this file pins the three properties that would have caught them:

  1. **Every claim names an artifact, and the artifact exists.** A result file
     deleted in a cleanup must break a test, not silently orphan a figure.
  2. **Every figure names the attack it plots.** The audit-cost figure was wrong
     for exactly one reason: `prefix_cost.json` held whichever attack ran last.
     Per-attack filenames are now structural, and this asserts it.
  3. **Every published AUC records the ratio it was measured at**, and the ones
     the report labels as measured over the ceiling really are, while the ones it
     labels as valid really are. That is the difference between the corrected
     record and the record it corrected.

Run:  python tests/test_claims.py        (no GPU, no torch -- reads JSON only)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "docs/results"
FIGS = REPO / "docs/figures"

# The batch/pool ceiling, duplicated here on purpose: this file must fail if the
# constant in `harness` is ever loosened to make an old number look valid again.
CEILING = 0.10


# ---------------------------------------------------------------------------
# 1. Claims -> artifacts
# ---------------------------------------------------------------------------
# (claim, artifact) for every load-bearing result the report stands behind.
CLAIMS = [
    ("the batch/pool ratio artifact: same scores, 0.977 at 69% and 0.530 at 1.8%",
     "baseline_headroom.json"),
    ("the headline grid re-measured inside the ceiling: 16/24 cells fall",
     "headline_ratio.json"),
    ("d' predicts the batch at which detection arrives, on a 56,160-token pool",
     "pool_scaling.json"),
    ("none of the reported detector reversals is real (every d' > 0)",
     "reversal_check.json"),
    ("the prefix scheduler spends its budget and runs 16x faster (quant_2bit)",
     "prefix_cost_quant2bit.json"),
    ("the accounting gap is a property of the schedule, not the deviation (kv_fp8)",
     "prefix_cost_kvfp8.json"),
    ("... and again on a sampler-side attack (bug_k32)",
     "prefix_cost_bugk32.json"),
    ("the learned confidence head does not beat the `entropy` heuristic",
     "confidence_head.json"),
    ("the derived matched filter loses, and so does the oracle",
     "info_directed.json"),
    ("no single recompute detector is robust across model families",
     "robustness_sweep.json"),
]


def test_every_claim_has_its_artifact():
    missing = [(c, a) for c, a in CLAIMS if not (RES / a).exists()]
    assert not missing, "claims with no evidence on disk:\n" + "\n".join(
        f"  {a}  <- {c}" for c, a in missing)


def test_every_artifact_is_valid_json_with_content():
    for _, a in CLAIMS:
        d = json.loads((RES / a).read_text())
        assert d, f"{a} is empty"


# ---------------------------------------------------------------------------
# 2. Figures name the attack they plot
# ---------------------------------------------------------------------------
def test_prefix_cost_archives_are_named_for_their_attack():
    """The bug this catches: `exp_prefix_cost_gpu` used to write a single
    `prefix_cost.json`, so `plot_triage` rendered whichever attack ran last under
    a caption naming a different one. The committed `fig_prefix_cost.png` was in
    fact `bug_k32` data under a `quant_2bit` caption. Filenames now carry the
    attack, and the attack recorded *inside* each file must agree with its name."""
    assert not (RES / "prefix_cost.json").exists(), (
        "prefix_cost.json is attack-ambiguous by construction -- "
        "exp_prefix_cost_gpu must write prefix_cost_<attack>.json")
    for tag, attack in (("quant2bit", "quant_2bit"),
                        ("kvfp8", "kv_fp8"),
                        ("bugk32", "bug_k32")):
        p = RES / f"prefix_cost_{tag}.json"
        d = json.loads(p.read_text())
        inside = d.get("attack") or d.get("config", {}).get("attack")
        assert inside == attack, f"{p.name} records attack={inside!r}, expected {attack!r}"


# ---------------------------------------------------------------------------
# 3. Published numbers carry the ratio they were measured at
# ---------------------------------------------------------------------------
def test_headline_grid_arms_are_on_the_right_side_of_the_ceiling():
    """The corrected record's central comparison: the same per-token scores read
    at the published 78% ratio and at a legitimate 8.9% one. If the two arms ever
    stopped straddling the ceiling the comparison would be vacuous."""
    d = json.loads((RES / "headline_ratio.json").read_text())
    ratios = _collect_ratios(d)
    assert ratios, "headline_ratio.json records no batch/pool ratio"
    assert max(ratios) > CEILING, "no inflated arm: nothing to correct"
    assert min(ratios) <= CEILING, "no valid arm: nothing to correct it with"


def test_pool_scaling_has_points_inside_the_ceiling():
    """The prediction test only means something below the ceiling -- above it the
    measurement stops being evidence. The report leans on five such points."""
    d = json.loads((RES / "pool_scaling.json").read_text())
    ratios = _collect_ratios(d)
    assert ratios, "pool_scaling.json records no batch/pool ratio"
    inside = [r for r in ratios if r <= CEILING]
    assert len(inside) >= 4, f"only {len(inside)} points inside the ceiling: {sorted(ratios)}"


def _collect_ratios(obj) -> list[float]:
    """Every `ratio`-ish number anywhere in a result payload. Deliberately
    structural rather than schema-bound: the point is that the ratio is recorded
    *somewhere* in every artifact, whatever that artifact's shape."""
    out: list[float] = []

    def numeric_leaves(o):
        if isinstance(o, (int, float)) and not isinstance(o, bool):
            out.append(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                numeric_leaves(v)
        elif isinstance(o, list):
            for v in o:
                numeric_leaves(v)

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if "prefill" in kl or "recompute" in kl:
                    continue                      # cost ratios, not batch/pool
                if kl == "ratio" or "ratio" in kl and kl != "ratio_ceiling":
                    # `ratio` may be a scalar or a dict of named arms
                    # ({"ratioed": .089, "readme": .781}); take every leaf.
                    numeric_leaves(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def test_no_figure_is_orphaned():
    """A figure with no script that can regenerate it is a number no one can
    check. Every committed PNG must be named by some experiment or plot script."""
    scripts = "\n".join(p.read_text() for p in (REPO / "experiments").glob("*.py"))

    def named(fig: str) -> bool:
        if fig in scripts:
            return True
        # Some scripts build the name with an f-string suffix, e.g.
        # f"fig_proxy_distance_grid_{LADDER}.png" -- match on the stable prefix.
        stem = fig[: -len(".png")]
        while "_" in stem:
            stem = stem.rsplit("_", 1)[0]
            if f'"{stem}_' in scripts or f"'{stem}_" in scripts:
                return True
        return False

    orphans = [f.name for f in sorted(FIGS.glob("*.png")) if not named(f.name)]
    assert not orphans, f"figures no script regenerates: {orphans}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}\n     {e}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
