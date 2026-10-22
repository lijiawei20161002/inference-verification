"""Detached M3 native-shape witnesses with fresh repeats and independent checking."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from experiments.analyze_real_quantization import digest, write_json
from experiments.numerical.mechanism_ops import select_cases
from experiments.numerical.mechanism_replay import read, verify_artifacts
from experiments.numerical.native_shape import summarize
from experiments.numerical.run_e0_campaign import freeze_source, now, status
from experiments.numerical.run_mechanism_campaign import compare_records, TESTS

ROOT = Path(__file__).resolve().parents[2]


def launch(args):
    output, parent = args.output.resolve(), args.parent_m1.resolve()
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 20 * 1024**3:
        raise RuntimeError('Need 20 GiB free before the native-shape campaign')
    if (read(parent / 'status.json')['state'] != 'complete'
            or read(parent / 'summary.json')['m1_gate'] != 'passed_for_historical_evaluation_corpus'):
        raise ValueError('Parent M1 gate has not passed')
    for name, expected in read(parent / 'artifacts.json').items():
        if digest(parent / name) != expected:
            raise ValueError('Parent M1 artifact changed')
    pm = verify_artifacts(parent / 'capture')
    cases = select_cases(read(parent / 'capture/events.json'), args.cases_per_family)
    if args.repeats not in (2, 3):
        raise ValueError('Use two preflight repeats or three full repeats')
    if args.cases_per_family > 2:
        if args.repeats != 3 or args.preflight_from is None:
            raise ValueError('Full round needs three repeats and completed preflight')
        preflight = args.preflight_from.resolve()
        pf, ps = read(preflight / 'plan.json'), read(preflight / 'summary.json')
        if (read(preflight / 'status.json')['state'] != 'complete' or pf['cases_per_family'] != 2
                or pf['parent_m1'] != str(parent) or ps['independent_checker']['verified'] is not True):
            raise ValueError('Preflight did not validate this protocol and parent')
        for name, expected in read(preflight / 'source_inventory.json').items():
            if name.endswith('.py') and digest(ROOT / name) != expected:
                raise ValueError(f'Source changed after preflight: {name}')
    output.mkdir(parents=True, exist_ok=False)
    source = freeze_source(output)
    plan = dict(schema='ivgym-native-shape-plan-v1', created_utc=now(), parent_m1=str(parent),
                parent_summary_sha256=digest(parent / 'summary.json'),
                parent_capture_manifest_sha256=digest(parent / 'capture/manifest.json'),
                cases=cases, cases_per_family=args.cases_per_family, repeats=args.repeats,
                preflight_from=str(args.preflight_from.resolve()) if args.preflight_from else None,
                discovery_only=True, injected_noise=False, max_seconds=args.max_seconds,
                selection='Unchanged deterministic historical/control selection and layers 0/14/27',
                primary_sampling='float32, legacy top-k 50 then top-p .95, actual saved Gumbel',
                contrast='Observed provider operator extent/layout on identical effective reference operands',
                attention_factors='Query extent/layout and fully masked KV suffix; query-only local control',
                checkpoint='Full original reference token IDs, empty KV, all descendants recomputed',
                provider_control='Rebuild sequential provider from saved tokens and empty KV; exact parent logits and KV',
                checker='Separate standalone implementation of native operators, model intervention and sampler',
                m3_full_gate_automatic=False, historical_root_cause='not_established')
    write_json(output / 'plan.json', plan)
    shutil.copy2(source / 'experiments/numerical/check_native_witness.py', output / 'check_native_witness.py')
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.update(HF_HOME=args.hf_home, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_ENABLE_HF_TRANSFER='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', '-m', 'experiments.numerical.run_native_campaign', '--run', '--output', str(output)]
    status(output / 'status.json', state='starting', stage='cpu_checks')
    with (output / 'supervisor.log').open('x') as log:
        process = subprocess.Popen(command, cwd=source, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (output / 'supervisor.pid').write_text(f'{process.pid}\n')
    write_json(output / 'launch.json', dict(pid=process.pid, command=command, source=str(source),
               launched_utc=now(), separate_os_session=True, stdin='DEVNULL', log='supervisor.log'))
    print(json.dumps(dict(pid=process.pid, output=str(output), status=str(output / 'status.json'))))


def run(args):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def interrupted(signum, frame):
        raise RuntimeError(f'Supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    output = args.output.resolve()
    plan, started = read(output / 'plan.json'), time.monotonic()
    child, stage = None, 'integrity_checks'

    def execute(name, command):
        nonlocal child, stage
        stage = name
        print(f'{now()} START {name}', flush=True)
        with (output / f'{name}.log').open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            while True:
                elapsed = time.monotonic() - started
                progress = output / name / 'progress.json'
                status(output / 'status.json', state='running', stage=name, supervisor_pid=os.getpid(),
                       worker_pid=child.pid, elapsed_seconds=elapsed,
                       progress=read(progress) if progress.exists() else None, log=str(output / f'{name}.log'))
                if elapsed > plan['max_seconds']:
                    raise TimeoutError('Frozen campaign time budget exceeded')
                if shutil.disk_usage(output).free < 12 * 1024**3:
                    raise RuntimeError('Campaign reached free-space reserve')
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
        child = None
        if code:
            raise RuntimeError(f'{name} exited {code}; inspect retained log')
        print(f'{now()} DONE {name}', flush=True)

    try:
        for name, expected in read(output / 'source_inventory.json').items():
            if digest(ROOT / name) != expected:
                raise ValueError('Frozen source integrity failure')
        execute('cpu_checks', [sys.executable, '-m', 'pytest', '-q', *TESTS, 'tests/test_native_shape.py'])
        expected, comparisons = None, []
        case_ids = [c['case_id'] for c in plan['cases']]
        for index in range(plan['repeats']):
            name = f'repeat_{index + 1}'
            execute(name, [sys.executable, '-u', '-m', 'experiments.numerical.native_shape_replay',
                           '--plan', str(output / 'plan.json'), '--output', str(output / name)])
            manifest = verify_artifacts(output / name)
            records = read(output / name / 'records.json')
            if (manifest['cases'] != len(case_ids) or manifest['comparison_sha256'] !=
                    hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()):
                raise ValueError('Worker case count or record digest differs')
            if expected is None:
                expected = records
            comparisons.append(dict(worker=name, **compare_records(expected, records, case_ids)))
            write_json(output / 'comparisons.json', comparisons)
        index_file = output / 'repeat_1/witnesses.json'
        execute('independent_checker', [sys.executable, '-u', str(output / 'check_native_witness.py'),
                 '--index', str(index_file), '--expected-sha256', digest(index_file),
                 '--output', str(output / 'independent_review.json')])
        review = read(output / 'independent_review.json')
        if not review['verified'] or review['cases'] != len(case_ids) or [r['case_id'] for r in review['results']] != case_ids:
            raise ValueError('Independent checker coverage incomplete')
        summary = dict(schema='ivgym-native-shape-summary-v1', finished_utc=now(), discovery_only=True,
                       families=summarize(expected), cases_per_family=plan['cases_per_family'],
                       independent_repeats=plan['repeats'], comparisons=comparisons,
                       independent_checker=dict(verified=True, cases=review['cases'],
                            checker_sha256=review['checker_sha256'], index_sha256=review['index_sha256']),
                       elapsed_seconds=time.monotonic() - started,
                       historical_root_cause='not_established', native_kernel_internals='not_inferred',
                       m3_gate='native_shape_round_complete_full_mechanism_gate_pending',
                       peak_cuda_allocated_bytes=max(read(output / c['worker'] / 'manifest.json')['peak_cuda_allocated_bytes'] for c in comparisons))
        write_json(output / 'summary.json', summary)
        (output / 'report.md').write_text('\n'.join([
            '# M3 native execution shape and independent witnesses', '',
            f"Finished: {summary['finished_utc']}", '',
            f"Cases per family: {plan['cases_per_family']}; independent repeats: {plan['repeats']}.",
            'Native operator and full descendant replay passed the separate checker.',
            'Provider and reference selected logits and KV matched the parent M1 evidence.', '',
            *[f'- {family}: {json.dumps(values)}' for family, values in summary['families'].items()], '',
            'The intervention uses actual native BF16 arithmetic on the reference effective operands,',
            'with the observed provider operator shape and layout. Attention additionally removes',
            'a fully masked KV suffix; a query-only local control is retained.',
            'A successful token reconciliation demonstrates influence in the reference replay.',
            'It does not establish a unique historical root cause or a native reduction mechanism.',
            'This is historical discovery data; technical repeats are not independent samples.',
            'The full M3 gate remains pending. M4 is deferred.', ''
        ]))
        write_json(output / 'artifacts.json', {name: digest(output / name) for name in
                   ('plan.json', 'source_inventory.json', 'comparisons.json', 'independent_review.json',
                    'check_native_witness.py', 'summary.json', 'report.md')})
        status(output / 'status.json', state='complete', stage='complete', supervisor_pid=os.getpid(),
               elapsed_seconds=time.monotonic() - started, summary=str(output / 'summary.json'))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status(output / 'status.json', state='failed', stage=stage, supervisor_pid=os.getpid(),
               elapsed_seconds=time.monotonic() - started, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent-m1', type=Path)
    parser.add_argument('--preflight-from', type=Path)
    parser.add_argument('--cases-per-family', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--hf-home', default='/workspace/.cache/huggingface')
    parser.add_argument('--max-seconds', type=int, default=14400)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--detach', action='store_true')
    mode.add_argument('--run', action='store_true')
    arguments = parser.parse_args()
    if arguments.detach and arguments.parent_m1 is None:
        parser.error('--detach requires --parent-m1')
    launch(arguments) if arguments.detach else run(arguments)
