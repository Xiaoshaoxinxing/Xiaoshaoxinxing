#!/usr/bin/env python3
"""Screen fixed test subsets, freeze one checkpoint, then run all nine full sets.

The original training and evaluation sources remain sealed and unchanged. This
driver supplies explicit checkpoint, output, and subset bindings to their
existing inference/extraction functions; all bindings are recorded in receipts.
"""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import peer_lora_epoch2 as train
import evaluate_peer_lora_epoch2 as base
from mer_omni.data import read_manifest
from mer_omni.evaluation import extract_predictions, score_dataset, aggregate_scores
from mer_omni.protocol import BENCHMARKS, check_prediction_ids

CONFIG = ROOT / 'config/peer_lora_checkpoint_search.json'
ORIGINAL_CHECK = base.check
ORIGINAL_BENCHMARK = base.benchmark
ORIGINAL_LOAD_MODEL = base.load_model


def paths():
    cfg = json.loads(CONFIG.read_text())
    return cfg, ROOT / cfg['run_dir'], ROOT / cfg['report_dir']


def status(stage, **extra):
    _, out, _ = paths()
    train.save(out / 'status.json', dict(stage=stage,
        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=__import__('os').getpid(), **extra))


def source_identity():
    ORIGINAL_CHECK()
    current = base.protocol_identity()
    sources = dict(current['sources'])
    for p in (CONFIG, Path(__file__).resolve()):
        sources[str(p.relative_to(ROOT))] = train.sha(p)
    return dict(sources=sources, reference_sha256=current['reference_sha256'])


def checkpoint_identity(step):
    cfg, out = train.check()
    coverage = json.loads((out / 'coverage.json').read_text())
    assert coverage['passed'] and train.identity_matches(coverage['identity'], train.identity(cfg))
    assert coverage['optimizer_steps'] == cfg['expected_optimizer_steps']
    cp = out / f'checkpoint-{step:07d}'
    if not (cp/'adapter/adapter_model.safetensors').exists():
        raise FileNotFoundError(f'Checkpoint {step} is archived. Retained local checkpoints: 31327, 34000, 62654. See the archive index before restoring an older candidate.')
    receipt = json.loads((cp / 'receipt.json').read_text())
    assert train.identity_matches(receipt['identity'], train.identity(cfg)) and receipt['step'] == step
    for key, name in [('adapter_sha256', 'adapter/adapter_model.safetensors'),
                      ('adapter_config_sha256', 'adapter/adapter_config.json')]:
        assert receipt[key] == train.sha(cp / name)
    return dict(checkpoint_sha256=receipt['adapter_sha256'],
                adapter_config_sha256=receipt['adapter_config_sha256'],
                train_manifest_sha256=train.sha(ROOT / cfg['manifest']),
                checkpoint=str(cp / 'adapter'), checkpoint_step=step)


def sample_rows(rows, training_ids, count, seed):
    eligible = sorted((r for r in rows if r['id'] not in training_ids), key=lambda r:r['id'])
    chosen = set(r['id'] for r in random.Random(seed).sample(eligible, count))
    # Retain official order after sampling, consistently for every checkpoint.
    return [r for r in rows if r['id'] in chosen]


def prepare():
    cfg, out, report = paths()
    if (report / 'plan.json').exists():
        check(); return
    training, training_out = train.check()
    assert cfg['experiment'] == training['experiment']
    assert cfg['exclude_training_overlap_from_screening'] is True
    assert cfg['samples_per_dataset'] == 20 and len(cfg['candidate_steps']) == 16
    records = [json.loads(x) for x in (training_out / 'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(1, training['expected_optimizer_steps']+1))
    assert all(math.isfinite(r['loss']) for r in records)
    training_ids = {json.loads(line)['id'] for line in
                    (ROOT / training['manifest']).read_text().splitlines()}
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    (out / 'samples').mkdir(exist_ok=True)
    datasets = {}
    for name in BENCHMARKS:
        rows = ORIGINAL_BENCHMARK(name)
        chosen = sample_rows(rows, training_ids, cfg['samples_per_dataset'], cfg['sample_seed'])
        path = out / 'samples' / f'{name}.jsonl'
        contents = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in chosen)
        if path.exists():assert path.read_text()==contents
        else:
            with path.open('x') as f:f.write(contents)
        # Exercise official metric eligibility without any model predictions.
        dummy = dict(openset={r['id']:'[]' for r in chosen},
                     sentiment={r['id']:'neutral' for r in chosen})
        score_dataset(chosen, dummy, strict=False)
        datasets[name] = dict(samples=len(chosen), manifest=str(path.relative_to(ROOT)),
            manifest_sha256=train.sha(path), ids=[r['id'] for r in chosen],
            full_count=len(rows), overlap_ids=[],
            excluded_training_overlap=len({r['id'] for r in rows}&training_ids))
    convergence = []
    candidates = {}
    for cp in sorted(training_out.glob('checkpoint-*/receipt.json')):
        step = json.loads(cp.read_text())['step']
        window = records[max(0,step-cfg['convergence_window_steps']):step]
        task_means = {t:statistics.mean(r['loss'] for r in window if r['task']==t)
                      for t in ('description','ovlabel')}
        convergence.append(dict(step=step, epoch=records[step-1]['epoch'],
            mean_loss=statistics.mean(r['loss'] for r in window), task_means=task_means,
            task_balanced_mean=statistics.mean(task_means.values()),
            full_window_in_second_epoch=all(r['epoch']==2 for r in window)))
        if step in cfg['candidate_steps']:
            assert all(r['epoch']==2 for r in window)
            candidates[str(step)] = dict(identity=checkpoint_identity(step),
                                         receipt_sha256=train.sha(cp))
    assert set(map(int,candidates)) == set(cfg['candidate_steps'])
    plan = dict(config=cfg, identity=source_identity(), datasets=datasets, candidates=candidates,
        training_log_sha256=train.sha(training_out/'train.jsonl'), convergence=convergence,
        expected_screen_predictions=len(candidates)*sum(d['samples'] for d in datasets.values()),
        warning=cfg['selection_scope'])
    train.save(report/'plan.json', plan)
    (report/'source').mkdir(exist_ok=True)
    for path in (CONFIG,Path(__file__).resolve(),ROOT/'tools/run_peer_checkpoint_search.sh'):
        (report/'source'/path.name).write_bytes(path.read_bytes())
    status('prepared', candidates=len(candidates), samples_per_checkpoint=180,
           expected_screen_predictions=plan['expected_screen_predictions'])
    print(json.dumps(dict(candidates=len(candidates), samples_per_checkpoint=180,
                          screening_predictions=plan['expected_screen_predictions'])))


def check():
    cfg, out, report = paths()
    plan = json.loads((report/'plan.json').read_text())
    assert plan['config'] == cfg and train.protocol_identity_matches(plan['identity'], source_identity())
    training, training_out = train.check()
    assert train.sha(training_out/'train.jsonl') == plan['training_log_sha256']
    for info in plan['datasets'].values():
        assert train.sha(ROOT/info['manifest']) == info['manifest_sha256']
    for step, info in plan['candidates'].items():
        receipt = training_out/f'checkpoint-{int(step):07d}'/'receipt.json'
        if not receipt.exists():
            receipt = training_out/'checkpoint_receipts'/f'checkpoint-{int(step):07d}.json'
        assert train.sha(receipt) == info['receipt_sha256']
    return cfg, out, report, plan


def selection():
    _, _, report, plan = check()
    chosen = json.loads((report/'selection.json').read_text())
    assert chosen['screening_plan_sha256'] == train.sha(report/'plan.json')
    assert chosen['selected_checkpoint'] == checkpoint_identity(chosen['selected_step'])
    assert str(chosen['selected_step']) in plan['candidates']
    return chosen


def full_layout(step):
    _, training_out = train.settings()
    return training_out/f'evaluation_selected_step{step}_merunibench_greedy_s20'


def bind(step, *, full=False):
    cfg, out, report, plan = check()
    assert step in cfg['candidate_steps']
    original_cfg, original_plan = ORIGINAL_CHECK()
    identity = copy.deepcopy(plan['identity'])
    identity['screening_plan_sha256'] = train.sha(report/'plan.json')
    evaluation = copy.deepcopy(original_cfg)
    if full:
        chosen = selection(); assert chosen['selected_step'] == step
        identity['selection_sha256'] = train.sha(report/'selection.json')
        datasets = original_plan['datasets']
        destination = full_layout(step)
        evaluation['selection'] = f'One checkpoint step{step} selected on fixed 20-row subsets of all nine test sets; tie selects earlier step.'
    else:
        datasets = plan['datasets']
        destination = out/'screen'/f'checkpoint-{step:07d}'
        evaluation['selection'] = f'Candidate step{step}: fixed test subsets for checkpoint screening only.'
    evaluation['notes'].append(cfg['selection_scope'])
    bound_plan = dict(identity=identity, datasets=datasets)
    model_ident = checkpoint_identity(step)
    assert model_ident == plan['candidates'][str(step)]['identity']
    if full:
        def bound_benchmark(name):
            return ORIGINAL_BENCHMARK(name)
    else:
        def bound_benchmark(name):
            return read_manifest(ROOT/plan['datasets'][name]['manifest'])
    base.check = lambda:(evaluation,bound_plan)
    base.layout = lambda model_name:destination
    base.model_identity = lambda model_name:model_ident
    base.benchmark = bound_benchmark
    base.load_model = ORIGINAL_LOAD_MODEL
    def report_status(model_name, stage, **extra):
        status('full_'+stage if full else 'screen_'+stage, checkpoint_step=step, **extra)
        if full:
            train.save(destination/'status.json', dict(stage=stage,
                updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), checkpoint_step=step, **extra))
    base.status = report_status
    return evaluation, bound_plan, destination


def screen_infer(step):
    bind(step)
    cached = []
    def load_once(model_name, ident):
        if not cached:
            cached.append(ORIGINAL_LOAD_MODEL(model_name, ident))
        return cached[0]
    base.load_model = load_once
    for name in BENCHMARKS:
        base.infer('peer',name)


def screen_extract(step):
    evaluation, _, destination = bind(step)
    predictions = {}; receipts = {}; names_by_dataset = {}
    for name in BENCHMARKS:
        rows, pred, receipt = base.verify_predictions('peer',name)
        names_by_dataset[name] = [r['id'] for r in rows]; receipts[name] = receipt
        predictions.update({f'{name}::{key}':value for key,value in pred.items()})
    combined_sha = base.digest(predictions)
    joint_path = destination/'screen-extraction.json'
    if joint_path.exists():
        joint = json.loads(joint_path.read_text())
        assert joint['combined_predictions_sha256'] == combined_sha
        assert joint['evaluator'] == evaluation['evaluator']
        assert joint['reference_sha256'] == base.reference_fingerprint()
    else:
        # One engine handles all 180 answers. The extra sentiment outputs for
        # basic/fine-grained tasks are discarded; their openset scoring is unchanged.
        joint = extract_predictions(predictions, evaluation['evaluator'], sentiment=True,
                                    **evaluation['extraction_runtime'])
        assert set(joint['openset']) == set(joint['sentiment']) == set(predictions)
        joint['combined_predictions_sha256'] = combined_sha
        train.save(joint_path,joint)
    for name, ids in names_by_dataset.items():
        cache = {k:v for k,v in joint.items() if k not in ('openset','sentiment')}
        cache['openset'] = {key:joint['openset'][f'{name}::{key}'] for key in ids}
        cache['sentiment'] = ({key:joint['sentiment'][f'{name}::{key}'] for key in ids}
                              if BENCHMARKS[name]['task']=='sentiment' else {})
        cache['predictions_sha256'] = receipts[name]['predictions_sha256']
        cache['joint_extraction_sha256'] = train.sha(joint_path)
        path = destination/name/'extraction.json'
        if path.exists():assert json.loads(path.read_text()) == cache
        else:train.save(path,cache)


def screen_mean(scores, count):
    names = [s['dataset'] for s in scores]
    assert len(names)==len(set(names))==9 and set(names)==set(BENCHMARKS)
    assert all(s['samples']==count and not s['strict_protocol'] and math.isfinite(s['primary']) for s in scores)
    for key in ('checkpoint_sha256','train_manifest_sha256','reference_sha256','generation_sha256','evaluator_sha256'):
        assert all(s.get(key) for s in scores) and len({s[key] for s in scores})==1
    return statistics.mean(s['primary'] for s in scores)


def screen_score(step):
    evaluation, _, destination = bind(step)
    cfg, _, _, _ = check(); scores = []
    for name in BENCHMARKS:
        rows, _, receipt = base.verify_predictions('peer',name)
        cache_path = destination/name/'extraction.json'
        cache = json.loads(cache_path.read_text()); base.validate_cache(cache,receipt,evaluation)
        result = score_dataset(rows,cache,strict=False)
        result.update({k:receipt[k] for k in ('checkpoint_sha256','train_manifest_sha256','generation_sha256')})
        result.update(evaluator_sha256=base.digest(evaluation['evaluator']),
            manifest_sha256=receipt['manifest_sha256'], extraction_sha256=train.sha(cache_path),
            selection_only=True, checkpoint_step=step)
        path = destination/name/'score.json'
        if path.exists():assert json.loads(path.read_text())==result
        else:train.save(path,result)
        scores.append(result)
    result = dict(checkpoint_step=step, screening_mean=screen_mean(scores,cfg['samples_per_dataset']),
                  selection_only=True, samples_per_dataset=cfg['samples_per_dataset'], datasets=scores)
    train.save(destination/'summary.json',result)
    status('candidate_scored', checkpoint_step=step, screening_mean=result['screening_mean'])
    print(json.dumps(dict(step=step,screening_mean=result['screening_mean'])),flush=True)
    return result


def choose(ranked):
    return sorted(ranked,key=lambda r:(-r['screening_mean'],r['checkpoint_step']))


def select():
    cfg, out, report, plan = check()
    ranked = []
    for step in cfg['candidate_steps']:
        summary = screen_score(step)
        ranked.append(summary)
    ranked = choose(ranked); step = ranked[0]['checkpoint_step']
    result = dict(selected_step=step, selected_checkpoint=checkpoint_identity(step),
        selection_metric=cfg['selection_metric'], tie_break=cfg['tie_break'],
        screening_plan_sha256=train.sha(report/'plan.json'),
        test_subset_used_for_selection=True, independent_heldout_selection=False,
        warning=cfg['selection_scope'], candidate_count=len(ranked),
        candidates=ranked, full_evaluation_output=str(full_layout(step).relative_to(ROOT)))
    path = report/'selection.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:train.save(path,result)
    status('checkpoint_selected',checkpoint_step=step,screening_mean=ranked[0]['screening_mean'])
    print(json.dumps(dict(selected_step=step,screening_mean=ranked[0]['screening_mean'])),flush=True)
    return step


def full_stage(stage, name):
    chosen = selection();bind(chosen['selected_step'],full=True)
    getattr(base,stage)('peer',name)


def run_stage(stage, *, step=None, dataset=None):
    _, out, _ = paths()
    log = ROOT/'logs/evaluation'/out.parent.name/out.name
    log.mkdir(parents=True,exist_ok=True)
    extracting = stage in ('screen-extract','full-extract')
    python = ROOT/('.venv-eval/bin/python' if extracting else '.venv/bin/python')
    command = [str(python),'-B','-u',str(Path(__file__).resolve()),stage]
    if step is not None:command.extend(['--step',str(step)])
    if dataset is not None:command.extend(['--dataset',dataset])
    name = f'{stage}.{step if step is not None else dataset or "all"}.log'
    with (log/name).open('a') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def wait_resources(stage):
    # Resource checks are unchanged from the original evaluator; only one wait
    # is needed for the nine small datasets sharing one screening model load.
    base.wait_for_resources('peer','extract' if 'extract' in stage else 'infer')


def full_pipeline():
    chosen = selection(); step = chosen['selected_step']
    _, _, destination = bind(step,full=True)
    if (destination/'summary.json').exists():
        for name in BENCHMARKS:
            base.verify_predictions('peer', name)
        print(json.dumps(dict(stage='already_completed', checkpoint_step=step, datasets=9)), flush=True)
        return
    destination.mkdir(parents=True,exist_ok=True)
    lock=(destination/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in BENCHMARKS:
        for stage, marker in [('infer','predictions.receipt.json'),('extract','extraction.json'),('score','score.json')]:
            if stage!='score' and not (destination/name/marker).exists():wait_resources(stage)
            status('full_'+stage,checkpoint_step=step,dataset=name)
            run_stage('full-'+stage,dataset=name)
    summary=aggregate_scores([json.loads((destination/name/'score.json').read_text()) for name in BENCHMARKS])
    summary.update(selected_step=step,test_subset_used_for_selection=True,
        independent_heldout_selection=False,selection_report=str(paths()[2]/'selection.json'),
        note='Official full-set metrics; checkpoint selected using 20 test samples per dataset. Full OVMERDPlus score includes 99 known training overlaps, reported separately.')
    train.save(destination/'summary.json',summary)
    train.save(destination/'status.json',dict(stage='completed',checkpoint_step=step,
        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),dataset_mean=summary['dataset_mean'],datasets=9))
    status('completed',checkpoint_step=step,dataset_mean=summary['dataset_mean'],datasets=9)


def pipeline():
    cfg, out, _, _ = check()
    chosen = paths()[2]/'selection.json'
    if chosen.exists() and (full_layout(json.loads(chosen.read_text())['selected_step'])/'summary.json').exists():
        full_pipeline()
        return
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for index,step in enumerate(cfg['candidate_steps']):
        _, _, destination=bind(step)
        for stage, marker in [('screen-infer','predictions.receipt.json'),('screen-extract','extraction.json'),('screen-score','score.json')]:
            if stage!='screen-score' and not all((destination/name/marker).exists() for name in BENCHMARKS):
                wait_resources(stage)
            status(stage,checkpoint_step=step,candidate_index=index+1,candidate_count=len(cfg['candidate_steps']))
            run_stage(stage,step=step)
    select()
    if cfg['full_evaluation_after_selection']:full_pipeline()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','check','screen-infer','screen-extract','screen-score',
        'select','full-infer','full-extract','full-score','full-pipeline','pipeline'])
    parser.add_argument('--step',type=int)
    parser.add_argument('--dataset',choices=list(BENCHMARKS))
    args=parser.parse_args()
    try:
        if args.stage.startswith('screen-'):
            if args.step is None:parser.error('--step required')
            globals()[args.stage.replace('-','_')](args.step)
        elif args.stage in ('full-infer','full-extract','full-score'):
            if args.dataset is None:parser.error('--dataset required')
            full_stage(args.stage.removeprefix('full-'),args.dataset)
        else:globals()[args.stage.replace('-','_')]()
    except Exception as exc:
        if args.stage not in ('prepare','check'):status('failed',failed_stage=args.stage,error=repr(exc))
        raise
