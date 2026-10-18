"""Run the archived H100 capture on A100 with a smoke and preserved provenance."""
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[1]
SOURCE = DEST / 'source'
BUNDLE = DEST / 'h100-replay-transfer'
CAPTURE_HASH = '53985023de9acea63b76eaaa1c2183f4cc94917b95e49bf59597f6a6881e5bba'
ENV = dict(os.environ, HF_HOME=str(ROOT / 'runs/huggingface'),
           HF_HUB_CACHE=str(ROOT / 'runs/huggingface/hub'), HF_HUB_OFFLINE='1',
           TRANSFORMERS_OFFLINE='1', HF_HUB_ENABLE_HF_TRANSFER='0',
           CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    state = dict(status='running', started_utc=stamp(), jobs=[],
                 source_commit=subprocess.check_output(
                     ['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(),
                 environment={k: ENV[k] for k in ('HF_HOME', 'HF_HUB_CACHE',
                     'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_ENABLE_HF_TRANSFER',
                     'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'TOKENIZERS_PARALLELISM', 'PYTHONUNBUFFERED')})
    queue = DEST / 'queue.json'
    if queue.exists():
        raise FileExistsError(queue)

    def run(name, args):
        command = [sys.executable, '-m', 'experiments.exp_frozen_replay_gpu', *args]
        row = dict(name=name, command=command, cwd=str(SOURCE), status='running', started_utc=stamp())
        state['jobs'].append(row)
        write(queue, state)
        print('Starting', name, flush=True)
        start = time.monotonic()
        with (DEST / f'{name}.log').open('x') as log:
            p = subprocess.run(command, cwd=SOURCE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        row.update(status='complete' if p.returncode == 0 else 'failed',
                   returncode=p.returncode, elapsed_seconds=time.monotonic()-start, finished_utc=stamp())
        write(queue, state)
        print('Finished', name, 'returncode', p.returncode, flush=True)
        if p.returncode:
            raise RuntimeError(f'{name} failed')

    try:
        capture = BUNDLE / 'capture/capture.json.gz'
        assert digest(capture) == CAPTURE_HASH
        env = json.loads((BUNDLE / 'environment.json').read_text())
        assert {p: importlib.metadata.version(p) for p in env['packages']} == env['packages']
        baseline = json.loads((BUNDLE / 'h100/manifest.json').read_text())
        for path, expected in baseline['provenance']['source_sha256'].items():
            assert digest(SOURCE / path) == expected, path
        assert digest(BUNDLE / 'h100/scores.npz') == baseline['scores_sha256']
        with gzip.open(capture, 'rt') as f:
            data = json.load(f)
        smoke = copy.deepcopy(data)
        smoke.update(n_prompts=4, n_tokens=16,
                     derived_from_capture_sha256=CAPTURE_HASH,
                     derivation='First four sequences and first sixteen positions; no regenerated tokens')
        smoke['sequences'] = smoke['sequences'][:4]
        for row in smoke['sequences']:
            row['claimed_tokens'] = row['claimed_tokens'][:16]
            row['fingerprints'] = row['fingerprints'][:16]
        smoke_path = DEST / 'smoke-capture.json.gz'
        with gzip.open(smoke_path, 'xt') as f:
            json.dump(smoke, f, allow_nan=False)
        state['smoke_capture_sha256'] = digest(smoke_path)
        write(queue, state)
        (DEST / 'nvidia-before.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))
        for name, path, sha in (('smoke-replay', smoke_path, digest(smoke_path)),
                                ('a100-replay', capture, CAPTURE_HASH)):
            run(name, ['replay', '--node', 'a100-cross-gpu-20261018', '--capture', str(path),
                       '--capture-sha256', sha, '--bootstrap', '400', '--output', str(DEST / name)])
        run('compare', ['compare', '--baseline', str(BUNDLE / 'h100'),
                       '--alternate', str(DEST / 'a100-replay'), '--bootstrap', '400',
                       '--output', str(DEST / 'comparison.json')])
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished_utc'] = stamp()
        write(queue, state)
        (DEST / 'nvidia-after.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))


if __name__ == '__main__':
    main()
