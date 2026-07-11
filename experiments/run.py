"""Run an attack x defense AUC sweep with *your own* strategies and backend --
no edits to the library required.

This is the no-edit extension path: write a Python file that registers custom
attacks/defenses (see `examples/custom_strategies.py`), point `--strategies` at
it, and the harness scores them against everything else.

Examples (the backend is a real model on a GPU; needs CUDA + transformers)
--------
    # built-in strategies on the default model (Qwen/Qwen3-0.6B)
    .venv/bin/python -m experiments.run

    # add your own strategies from a file, run them against the built-ins
    .venv/bin/python -m experiments.run --strategies examples/custom_strategies.py

    # only your strategies, larger batch
    .venv/bin/python -m experiments.run --strategies examples/custom_strategies.py \
        --attacks logit_spike --defenses topk_overlap token_difr --batch 2000
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness
from ivgym.backends import BACKENDS, make_backend
from ivgym.core import SamplingSpec


def load_strategies(paths: list[str]) -> None:
    """Import each user file so its `@register`-decorated attacks/defenses land
    in the registries. The file just needs to import from `ivgym.attacks` /
    `ivgym.defenses` and register; we exec it for its side effects."""
    for i, p in enumerate(paths):
        path = Path(p).resolve()
        if not path.exists():
            raise SystemExit(f"--strategies file not found: {path}")
        name = f"_ivgym_strategies_{i}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # Register before exec so @dataclass (which introspects sys.modules for
        # the defining module) works on classes defined in the loaded file.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"loaded strategies from {path}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="experiments.run",
        description="Attack x defense detection-AUC sweep with pluggable strategies and backend.",
    )
    ap.add_argument("--strategies", nargs="*", default=[],
                    help="Python files that register custom attacks/defenses.")
    ap.add_argument("--backend", default="hf_gpu", choices=BACKENDS,
                    help="Arena to run in (default: hf_gpu, a real model on a GPU).")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF model id for the hf_gpu backend.")
    ap.add_argument("--proxy-model", default=None,
                    help="HF model id for the cheap proxy that drives selective-recompute "
                         "triage (must share the reference's tokenizer/vocab, e.g. a smaller "
                         "same-family model). If omitted, the hf_gpu backend falls back to a "
                         "noised read of M for the tie signal.")
    ap.add_argument("--attacks", nargs="*", default=None,
                    help="Attack names to evaluate (default: all registered except 'honest').")
    ap.add_argument("--defenses", nargs="*", default=None,
                    help="Defense names to score with (default: all registered).")
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--n-batches", type=int, default=400)
    ap.add_argument("--selective", nargs="*", type=float, default=None, metavar="BUDGET",
                    help="Also score the proxy-triaged SELECTIVE-recompute tier "
                         "(ivgym.harness.verify_selective) at each of these recompute-budget "
                         "fractions, e.g. --selective 0.125 0.25 0.5. Each budget prints its own "
                         "AUC table (attack x defense) so the cost-aware tier is directly "
                         "comparable to full recompute. Assumes divergence-style defenses "
                         "(non-audited tokens take the 0.0 'no divergence' score).")
    ap.add_argument("--list", action="store_true",
                    help="List registered attacks/defenses and exit.")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    load_strategies(args.strategies)

    if args.list:
        print("attacks: ", ", ".join(sorted(attacks.all_attacks())))
        print("defenses:", ", ".join(sorted(defenses.all_defenses())))
        return

    attack_names = args.attacks
    if attack_names is None:
        attack_names = [n for n in attacks.all_attacks() if n != "honest"]
    defense_names = args.defenses or list(defenses.all_defenses())

    defs = [defenses.get(d) for d in defense_names]
    needs_act = any(d.needs_activation for d in defs)
    budgets = args.selective if args.selective is not None else []

    if args.backend == "hf_gpu":
        bkw = {"model_name": args.model}
        if args.proxy_model:
            bkw["proxy_model_name"] = args.proxy_model
        backend = make_backend(args.backend, **bkw)
    else:
        backend = make_backend(args.backend)
    spec = SamplingSpec()

    def gen(atk):
        return harness.generate_dataset(backend, atk, spec, args.prompts, args.tokens,
                                        record_activations=needs_act)

    def full_auc(honest_scores, attack_scores):
        res = harness.evaluate(honest_scores, attack_scores, defs, [args.batch],
                               n_batches=args.n_batches, winsor_pct=99.9)
        return {r.defense: r.auc for r in res}

    # --- honest reference (full + selective, computed while its cache is live) ---
    honest_seqs = gen(attacks.get("honest"))
    honest = harness.verify(backend, honest_seqs, spec, defs)
    # (defense_name, budget) -> honest TokenScores under that selective budget.
    honest_sel: dict[tuple[str, float], harness.TokenScores] = {}
    realized_ratio: dict[float, float] = {}
    if budgets:
        h_tri = harness.proxy_tie_scores(backend, honest_seqs, spec)
        for b in budgets:
            for d in defs:
                ts, ratio = harness.verify_selective(backend, honest_seqs, spec, d, b, triage=h_tri)
                honest_sel[(d.name, b)] = ts
                realized_ratio[b] = ratio

    # --- sweep attacks, accumulating AUCs (keep numbers, not backend reads) ------
    full_rows: dict[str, dict[str, float]] = {}
    sel_rows: dict[float, dict[str, dict[str, float]]] = {b: {} for b in budgets}
    for aname in attack_names:
        seqs = gen(attacks.get(aname))
        full_rows[aname] = full_auc(honest, harness.verify(backend, seqs, spec, defs))
        if budgets:
            a_tri = harness.proxy_tie_scores(backend, seqs, spec)
            for b in budgets:
                row = {}
                for d in defs:
                    a_ts, _ = harness.verify_selective(backend, seqs, spec, d, b, triage=a_tri)
                    r = harness.evaluate(honest_sel[(d.name, b)], a_ts, [d], [args.batch],
                                         n_batches=args.n_batches, winsor_pct=99.9)[0]
                    row[d.name] = r.auc
                sel_rows[b][aname] = row

    # --- report -----------------------------------------------------------------
    header = f"{'attack':>14} | " + " ".join(f"{d:>16}" for d in defense_names)

    def print_table(title, rows_by_attack):
        print(f"\n{title}\n" + header)
        print("-" * len(header))
        for aname in attack_names:
            row = " ".join(f"{rows_by_attack[aname][d]:>16.4f}" for d in defense_names)
            print(f"{aname:>14} | {row}")

    print_table(f"backend={args.backend}  full-recompute AUC @ batch={args.batch} "
                f"tokens (FPR target 1%)", full_rows)
    for b in budgets:
        print_table(f"selective recompute @ budget={b:g}  (realized recompute ratio "
                    f"{realized_ratio[b]:.3f})  AUC @ batch={args.batch} tokens", sel_rows[b])


if __name__ == "__main__":
    main()
