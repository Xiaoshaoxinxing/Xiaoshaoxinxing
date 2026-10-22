#!/usr/bin/env python3
"""Run the registered native-mean AV experiment; training only, never auto-evaluate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mer_omni.protocol import sha256

REPORT = ROOT/'reports/native_av_ablation_20261016'
OUTPUT = ROOT/'runs/mer_omni_native_av_epoch01'
PYTHON = ROOT/'.venv/bin/python'


def write(name, value):
    p = REPORT/name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(p)


def status(stage, **fields):
    write('status.json',dict(stage=stage,pid=os.getpid(),
                            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**fields))


def verify(plan, *, smoke=True):
    for path,digest in {**plan['source_sha256'],**plan['protected_sha256']}.items():
        if sha256(ROOT/path) != digest:
            raise RuntimeError(f'实验文件与登记哈希不一致：{path}')
    if not plan['tests_passed'] or sha256(REPORT/'tests.log') != plan['tests_log_sha256']:
        raise RuntimeError('离线测试结果与登记版本不一致')
    if smoke:
        result = json.loads((REPORT/'smoke.json').read_text())
        if (not result['passed'] or result['optimizer_updates'] != 0
                or result['plan_sha256'] != sha256(REPORT/'plan.json')):
            raise RuntimeError('GPU 预检未通过或与实验登记不一致')


def validate_completed(plan):
    actual = [json.loads(s) for s in (OUTPUT/'train.jsonl').read_text().splitlines()]
    control = [json.loads(s) for s in (ROOT/plan['control']/'train.jsonl').read_text().splitlines()]
    if len(actual) != plan['optimizer_steps'] or len(actual) != len(control):
        raise RuntimeError('训练步数与预定实验不一致')
    keys = ('step','sample_ids','sample_indices','lr','samples_this_step')
    if any(any(a[k]!=c[k] for k in keys) for a,c in zip(actual,control)):
        raise RuntimeError('样本顺序、覆盖或学习率与独立音视频对照不同')
    coverage = json.loads((OUTPUT/'coverage_epoch_0001.json').read_text())
    if not coverage['complete'] or coverage['unique_samples'] != plan['expected_samples']:
        raise RuntimeError('训练未完整覆盖原始数据')
    checkpoint = OUTPUT/f"checkpoint_step_{plan['optimizer_steps']:07d}.pt"
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                mean_last_100_loss=sum(r['loss'] for r in actual[-100:])/100,
                sample_order_and_learning_rate_match_control=True,unique_samples=coverage['unique_samples'])


def main(args):
    plan = json.loads((REPORT/'plan.json').read_text())
    verify(plan)
    if args.check:
        print('原生音视频对照：文件、离线测试、GPU 预检均通过；尚未启动训练。',flush=True)
        return
    # Keep this lock through the child and completion validation; a second launch
    # cannot start another training process for the same output directory.
    with (REPORT/'workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.resume and OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise FileExistsError('实验输出已有内容；必须明确指定本实验检查点恢复')
        command = [str(PYTHON),'-B','-u',str(ROOT/'tools/mer_omni.py'),'--config',str(ROOT/plan['train_config']),
                   'train','--output',str(OUTPUT),'--device','cuda:0','--log-every','10','--checkpoint-every','1000']
        if args.resume:
            cp = Path(args.resume).expanduser().resolve()
            if cp.parent != OUTPUT or not cp.is_file():
                raise ValueError('只能从本实验输出目录中的检查点恢复')
            command += ['--resume',str(cp)]
        env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        status('starting',command=command,evaluation_scheduled=False)
        child = None
        try:
            child = subprocess.Popen(command,cwd=ROOT,env=env)
            status('training',child_pid=child.pid,total_steps=plan['optimizer_steps'],
                   training_log=str(OUTPUT/'train.jsonl'),evaluation_scheduled=False)
            code = child.wait()
            if code:
                raise RuntimeError(f'训练退出码 {code}')
            completion = validate_completed(plan)
            verify(plan)
            write('completion.json',completion)
            status('training_complete',**completion,evaluation_scheduled=False)
            print(json.dumps(completion,ensure_ascii=False),flush=True)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            status('failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume')
    options = parser.parse_args()
    if options.check and options.resume:
        parser.error('--check and --resume cannot be combined')
    main(options)
