#!/usr/bin/env python3
"""User-started foreground training with visible progress, saved logs, and final evaluation."""
import argparse
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mer_omni.console import panel

REPORT = ROOT / 'reports/av_tokens_ablation_20261015'
OUTPUT = ROOT / 'runs/mer_omni_av_tokens_epoch01'
LOGS = ROOT / 'logs/training/mer_omni_av_tokens_epoch01'
PYTHON = ROOT / '.venv/bin/python'
EVAL_PYTHON = ROOT / '.venv-eval/bin/python'
CLI = ROOT / 'tools/mer_omni.py'
LOCKED = False


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(name, value):
    path = REPORT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def status(stage, **extra):
    write('status.json', dict(stage=stage, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                              pid=os.getpid(), **extra))


def verify(plan):
    for path, expected in plan['protected_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError(f'受保护的数据或旧实验结果发生变化：{path}')
    for path, expected in plan['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f'实验代码或配置与检查版本不一致：{path}')
    if not plan['tests_passed'] or sha(REPORT / 'tests.log') != plan['tests_log_sha256']:
        raise RuntimeError('测试报告与本次检查版本不一致')


def run(command, stage, *, dataset=None, total_samples=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    label = f'{dataset}.{stage}' if dataset else stage
    logfile = LOGS / f'{label}.{time.strftime("%Y%m%d-%H%M%S")}.log'
    print(panel('当前阶段', [label, f'完整日志：{logfile}']), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', VLLM_HOST_IP='127.0.0.1',
               VLLM_CONFIG_ROOT=str(ROOT / '.cache/vllm'), XDG_CACHE_HOME=str(ROOT / '.cache'))
    tail, extracted = deque(maxlen=30), 0
    with logfile.open('x') as handle:
        child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        status(stage, child_pid=child.pid, dataset=dataset, log=str(logfile), total_steps=10443)
        try:
            for line in child.stdout:
                handle.write(line)
                handle.flush()
                tail.append(line)
                if stage == 'extraction':
                    if line.startswith(' Response:'):
                        extracted += 1
                        if extracted % 64 == 0 or extracted == total_samples:
                            print(f'{dataset} 关键词抽取：{extracted}/{total_samples}', flush=True)
                    elif line.startswith('INFO '):
                        print(line, end='', flush=True)
                elif line.strip() != 'Unused or unrecognized kwargs: return_tensors.':
                    print(line, end='', flush=True)
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            child.stdout.close()
    if code:
        if stage == 'extraction':
            print(''.join(tail), flush=True)
        raise RuntimeError(f'{label} 退出码 {code}，日志：{logfile}')


def validate_training():
    actual = [json.loads(s) for s in (OUTPUT / 'train.jsonl').read_text().splitlines()]
    control = [json.loads(s) for s in (ROOT / 'runs/mer_omni_subtitle_epoch01/train.jsonl').read_text().splitlines()]
    if len(actual) != 10443 or any(new['step'] != old['step'] or new['sample_ids'] != old['sample_ids'] or
                                   new['sample_indices'] != old['sample_indices'] or new['lr'] != old['lr']
                                   for new, old in zip(actual, control)):
        raise RuntimeError('训练步数、样本顺序或学习率与对照组不一致')
    coverage = json.loads((OUTPUT / 'coverage_epoch_0001.json').read_text())
    if not coverage['complete'] or coverage['unique_samples'] != 31327:
        raise RuntimeError('本轮没有完整覆盖原始训练样本')
    checkpoint = OUTPUT / 'checkpoint_step_0010443.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def smoke_verified():
    path = REPORT / 'smoke.json'
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    return (result.get('passed') is True and result.get('optimizer_updates') == 0
            and result.get('plan_sha256') == sha(REPORT / 'plan.json'))


def main(args):
    global LOCKED
    plan = json.loads((REPORT / 'plan.json').read_text())
    verify(plan)
    print(panel('独立音视频 token 对照实验', [
        '输入：融合 token + 音频 token + 视频 token + 原始字幕',
        '训练：31,327 条样本，1 轮，共 10,443 step；seed 42',
        '每 10 step 显示进度，每 1,000 step 保存检查点，训练连续进行',
        '训练后依次评测 MER2023、MER2024' if not args.train_only else '本次只训练，评测可稍后单独启动',
        f'输出：{OUTPUT}', '数据、旧检查点和旧评测结果校验通过。']), flush=True)
    if args.check:
        print('静态检查通过；GPU 预检' + ('已通过。' if smoke_verified() else '待运行，启动时会先执行。'), flush=True)
        print('未启动训练或评测。', flush=True)
        return
    with (REPORT / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        LOCKED = True
        if args.smoke_only or not smoke_verified():
            run([PYTHON, '-B', '-u', REPORT / 'smoke.py'], 'gpu_precheck')
            verify(plan)
            if not smoke_verified():
                raise RuntimeError('GPU 预检未生成与当前版本一致的通过记录；不会启动训练')
        if args.smoke_only:
            status('ready', formal_training_started=False)
            return
        if not args.eval_only:
            command = [PYTHON, '-B', '-u', CLI, '--config', ROOT / plan['train_config'],
                       'train', '--output', OUTPUT, '--device', 'cuda:0',
                       '--log-every', '10', '--checkpoint-every', '1000']
            if args.resume:
                checkpoint = Path(args.resume).expanduser().resolve()
                if checkpoint.parent != OUTPUT or not checkpoint.is_file():
                    raise ValueError('--resume 必须使用本实验输出目录中的检查点')
                # The trainer checks checkpoint/log alignment and restores all RNG/optimizer state.
                command.extend(['--resume', checkpoint])
            elif OUTPUT.exists() and any(OUTPUT.iterdir()):
                raise FileExistsError('输出目录已有内容；不会覆盖。请检查状态后明确使用 --resume 或 --eval-only。')
            run(command, 'training')
        checkpoint = validate_training()
        verify(plan)
        status('training_complete', checkpoint=str(checkpoint), total_steps=10443)
        if args.train_only:
            return
        results = []
        for dataset in plan['evaluation_datasets']:
            folder = OUTPUT / 'evaluation_step10443_affectgpt' / dataset
            if folder.exists():
                raise FileExistsError(f'评测目录已存在，不会覆盖：{folder}')
            count = 411 if dataset == 'MER2023' else 1169
            cfg = ROOT / plan['evaluation_config']
            run([PYTHON, '-B', '-u', CLI, '--config', cfg, 'infer', '--checkpoint', checkpoint,
                 '--dataset', dataset, '--output', folder, '--sequence-overflow', 'truncate_right'],
                'inference', dataset=dataset, total_samples=count)
            verify(plan)
            run([EVAL_PYTHON, '-B', '-u', CLI, '--config', cfg, 'extract', '--dataset', dataset,
                 '--predictions', folder / 'predictions.jsonl', '--output', folder / 'extraction.json',
                 '--gpu-memory-utilization', '0.36', '--max-model-len', '4096', '--max-num-seqs', '8',
                 '--single-process-store'], 'extraction', dataset=dataset, total_samples=count)
            run([EVAL_PYTHON, '-B', '-u', CLI, '--config', cfg, 'score', '--dataset', dataset,
                 '--predictions', folder / 'predictions.jsonl', '--extraction', folder / 'extraction.json',
                 '--output', folder / 'score.json'], 'scoring', dataset=dataset)
            new = json.loads((folder / 'score.json').read_text())
            old = json.loads((ROOT / plan['control'] / 'evaluation_step10443_affectgpt' / dataset / 'score.json').read_text())
            for key in ['train_manifest_sha256', 'generation_sha256', 'evaluator_sha256',
                        'reference_sha256', 'manifest_sha256', 'input_processing_sha256']:
                if new[key] != old[key]:
                    raise RuntimeError(f'{dataset} 评测口径不一致：{key}')
            result = dict(dataset=dataset, metric='hit_rate', subtitle_control=old['primary'],
                          independent_av=new['primary'], delta_percentage_points=100*(new['primary']-old['primary']))
            results.append(result)
            print(panel(f'{dataset} 结果', [f'字幕对照组：{100*old["primary"]:.2f}%',
                         f'独立音视频 token：{100*new["primary"]:.2f}%',
                         f'变化：{result["delta_percentage_points"]:+.2f} 个百分点']), flush=True)
            write('comparison.json', dict(complete=False, datasets=results))
            verify(plan)
        write('comparison.json', dict(complete=True, datasets=results, checkpoint=str(checkpoint),
                                      checkpoint_sha256=sha(checkpoint), protected_data_and_controls_unchanged=True))
        status('complete', comparison=str(REPORT / 'comparison.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='只验证检查结果及文件完整性，不启动训练')
    mode.add_argument('--eval-only', action='store_true', help='使用本实验已完成的最终检查点评测')
    mode.add_argument('--train-only', action='store_true', help='完成训练后停止，不自动评测')
    mode.add_argument('--smoke-only', action='store_true', help='只做真实模型前向/反向/生成检查，不更新参数')
    parser.add_argument('--resume', help='从本实验保存的检查点恢复；训练日志必须与该检查点对齐')
    args = parser.parse_args()
    if args.resume and (args.check or args.eval_only or args.smoke_only):
        parser.error('--resume 只用于继续训练')
    try:
        main(args)
    except KeyboardInterrupt:
        if LOCKED:
            status('interrupted')
        print('\n已中断；现有日志和已保存检查点保留。', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        if LOCKED:
            status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
