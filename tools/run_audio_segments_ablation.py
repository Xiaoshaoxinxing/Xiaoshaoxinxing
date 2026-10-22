#!/usr/bin/env python3
"""Run the registered four-segment audio experiment; training only."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mer_omni.protocol import sha256

REPORT = ROOT/'reports/audio_segments_ablation_20261017'
PYTHON = ROOT/'.venv/bin/python'


def write(name, value):
    p = REPORT/name
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def status(stage, **fields):
    value = dict(stage=stage, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **fields)
    write('status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify(plan, *, smoke=True):
    for path, digest in {**plan['source_sha256'], **plan['protected_sha256']}.items():
        if sha256(ROOT/path) != digest:
            raise RuntimeError(f'Experiment file changed after registration: {path}')
    if not plan['tests_passed'] or sha256(REPORT/'tests.log') != plan['tests_log_sha256']:
        raise RuntimeError('Test receipt does not match registration')
    if smoke:
        result = json.loads((REPORT/'smoke.json').read_text())
        if (not result['passed'] or result['optimizer_updates'] != 0 or
                result['plan_sha256'] != sha256(REPORT/'plan.json') or
                result['trainable_parameters'] != plan['parameter_budget']['candidate_trainable']):
            raise RuntimeError('Real-model preflight is missing or inconsistent')


def validate_completed(plan):
    output, control = ROOT/plan['output'], ROOT/plan['control']
    actual = [json.loads(line) for line in (output/'train.jsonl').open()]
    baseline = [json.loads(line) for line in (control/'train.jsonl').open()]
    if len(actual) != plan['optimizer_steps'] or len(actual) != len(baseline):
        raise RuntimeError('Training update count mismatch')
    fields = ('step', 'sample_ids', 'sample_indices', 'lr', 'samples_this_step')
    if any(any(a[k] != b[k] for k in fields) for a, b in zip(actual, baseline)):
        raise RuntimeError('Training order or schedule differs from the control')
    if not all(math.isfinite(r['loss']) and math.isfinite(r['nll']) for r in actual):
        raise RuntimeError('Nonfinite training loss')
    coverage = json.loads((output/'coverage_epoch_0001.json').read_text())
    if not coverage['complete'] or coverage['unique_samples'] != plan['expected_samples']:
        raise RuntimeError('Incomplete training coverage')
    checkpoint = output/f"checkpoint_step_{plan['optimizer_steps']:07d}.pt"
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                mean_last_100_loss=sum(r['loss'] for r in actual[-100:])/100,
                mean_last_1000_loss=sum(r['loss'] for r in actual[-1000:])/1000,
                sample_order_and_learning_rate_match_control=True, all_losses_finite=True,
                unique_samples=coverage['unique_samples'])


def worker(plan, resume=None):
    verify(plan)
    output = ROOT/plan['output']
    with (REPORT/'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume and output.exists() and any(output.iterdir()):
            raise FileExistsError('Output already contains a run; explicit resume required')
        command = [str(PYTHON), '-B', '-u', str(ROOT/'tools/mer_omni.py'), '--config',
                   str(ROOT/plan['train_config']), 'train', '--output', str(output),
                   '--device', 'cuda:0', '--log-every', '10', '--checkpoint-every', '1000']
        if resume:
            checkpoint = Path(resume).expanduser().resolve()
            if checkpoint.parent != output or not checkpoint.is_file():
                raise ValueError('Resume must use a checkpoint from this registered experiment')
            command += ['--resume', str(checkpoint)]
        environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        child = None
        try:
            with (REPORT/'training.log').open('a' if resume else 'x') as handle:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT)
                status('training', child_pid=child.pid, total_steps=plan['optimizer_steps'],
                       output=str(output), training_log=str(REPORT/'training.log'), evaluation_scheduled=False)
                code = child.wait()
            if code:
                raise RuntimeError(f'Training exited with code {code}; inspect training.log')
            completion = validate_completed(plan)
            verify(plan)
            write('completion.json', completion)
            status('training_complete', **completion, evaluation_scheduled=False)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            status('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def launch(plan):
    verify(plan)
    output = ROOT/plan['output']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to start over an existing run')
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (REPORT/'workflow.log').open('x') as handle:
        child = subprocess.Popen([str(PYTHON), '-B', '-u', str(Path(__file__).resolve()), '--worker'],
                                 cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    receipt = dict(pid=child.pid, started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_sha256=sha256(REPORT/'plan.json'))
    write('launch.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--worker', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.resume and not args.worker:
        parser.error('--resume is only accepted with --worker')
    plan = json.loads((REPORT/'plan.json').read_text())
    if args.check:
        verify(plan)
        print('Registered sources, full tests and real-model preflight verified.', flush=True)
    elif args.launch:
        launch(plan)
    else:
        worker(plan, args.resume)
