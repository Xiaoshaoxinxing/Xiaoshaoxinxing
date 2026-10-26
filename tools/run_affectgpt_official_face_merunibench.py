#!/usr/bin/env python3
"""Use official AffectGPT code to screen seven checkpoints, then evaluate one."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/affectgpt_official_face_merunibench.json'
CFG = json.loads(CONFIG.read_text())
OUT = ROOT / 'runs' / CFG['experiment']
LOGS = ROOT / 'logs/evaluation' / CFG['experiment']
UPSTREAM = ROOT / CFG['upstream']
PYTHON = ROOT / '.venv-eval/bin/python'
MODEL_NAME = 'emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz'
WEIGHTS = OUT / 'checkpoints' / MODEL_NAME
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
    VLLM_CONFIG_ROOT=str(ROOT/'.cache/vllm'), XDG_CACHE_HOME=str(ROOT/'.cache'),
    MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'), NUMBA_CACHE_DIR=str(ROOT/'.cache/numba'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def status(stage, **extra):
    save(OUT/'status.json', dict(stage=stage, updated_at=datetime.now().astimezone().isoformat(),
                               pid=os.getpid(), **extra))


def entries():
    values = read(ROOT/CFG['checkpoint_directory']/'official-files.json')
    result = {int(Path(v['path']).name.split('_')[1]):v for v in values}
    assert sorted(result) == CFG['epochs']
    return result


def source_identity():
    provenance = read(UPSTREAM.parent/'source.json')
    assert provenance['commit'] == CFG['upstream_commit'] and provenance['source_edits'] == []
    for item in provenance['files']:
        assert sha(UPSTREAM/item['path']) == item['sha256'], item['path']
    paths = [CONFIG, Path(__file__).resolve(), UPSTREAM.parent/'source.json',ROOT/CFG['peer_screening_plan'],
             ROOT/'tools/download_affectgpt_face_official.py',ROOT/'tools/download_affectgpt_face_paper.py',
             ROOT/'tools/run_affectgpt_official_face_merunibench.sh']
    sample_plan = read(ROOT/CFG['peer_screening_plan'])
    for info in sample_plan['datasets'].values():
        path = ROOT/info['manifest']
        assert sha(path) == info['manifest_sha256']
        paths.append(path)
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def bootstrap():
    """Only local-path and import compatibility; original .py files stay unchanged."""
    sys.path.insert(0, str(UPSTREAM)); os.chdir(UPSTREAM)
    sys.modules['torchvision.transforms.functional_tensor'] = importlib.import_module(
        'torchvision.transforms._functional_tensor')
    config = importlib.import_module('config')
    assert Path(config.__file__).resolve() == UPSTREAM/'config.py'
    for value in vars(config).values():
        if isinstance(value, dict):
            for key, path in list(value.items()):
                if isinstance(path, str) and path.startswith('../dataset/'):
                    value[key] = str(ROOT/'data/shared'/path.removeprefix('../dataset/'))
                elif isinstance(path, str) and path.startswith('models/'):
                    value[key] = str(UPSTREAM/path)
    config.EMOTION_WHEEL_ROOT = str(UPSTREAM/'emotion_wheel')
    return importlib.import_module('inference_hybird'), importlib.import_module('evaluation')


def manifest(dataset):
    from mer_omni.data import read_manifest, verify_manifest_receipt
    from mer_omni.protocol import check_coverage
    path = ROOT/'data/manifests/benchmark'/f'{dataset}.jsonl'
    verify_manifest_receipt(path)
    rows = read_manifest(path); check_coverage(rows, dataset)
    return rows, path


def normalized(label, openset_parser=None):
    if openset_parser is not None:
        label = openset_parser(label)
    if isinstance(label,(list,tuple)):
        return sorted(normalized(value) for value in label)
    return {'happy':'joy','sad':'sadness','anger':'angry'}.get(label, label) if isinstance(label,str) else float(label)


def sample_rows(dataset):
    from mer_omni.data import read_manifest
    plan = read(ROOT/CFG['peer_screening_plan'])
    info = plan['datasets'][dataset]
    path = ROOT/info['manifest']
    assert sha(path) == info['manifest_sha256'] and info['samples'] == CFG['samples_per_dataset']
    rows = read_manifest(path)
    assert len(rows) == len({r['id'] for r in rows}) == CFG['samples_per_dataset']
    return rows


def patch_screen_dataset(official_infer,dataset):
    # The original inference and score functions still execute. Only the test
    # loader's returned IDs/ground truth are restricted to the registered subset.
    names = [r['id'] for r in sample_rows(dataset)]; wanted = set(names)
    cls = type(official_infer.get_name2cls(dataset))
    original_labels = cls.get_test_name2gt
    cls.read_test_names = lambda self:list(names)
    cls.get_test_name2gt = lambda self:{k:v for k,v in original_labels(self).items() if k in wanted}


def prepare():
    import numpy as np
    import torch
    if (OUT/'request.json').exists():
        check(); return
    OUT.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    identity = source_identity()
    infer, evaluate = bootstrap()
    datasets = {}
    for name in CFG['datasets']:
        cls = infer.get_name2cls(name); rows, path = manifest(name)
        names = cls.read_test_names(); labels = cls.get_test_name2gt()
        assert len(names) == len(set(names)) == len(rows)
        assert set(names) == set(labels) == {r['id'] for r in rows}
        full_by_id = {r['id']:r for r in rows}
        sampled = sample_rows(name)
        assert all(r == full_by_id[r['id']] for r in sampled)
        if evaluate.get_discrete_or_dimension_flag(name) == 'discrete':
            _, idx2emo = evaluate.get_emo2idx_idx2emo(cls)
            labels = {k:v if isinstance(v,str) else idx2emo[v] for k,v in labels.items()}
        inventory = []
        for row in rows:
            key = row['id']; sample = dict(name=key)
            parser = evaluate.string_to_list if name == 'OVMERDPlus' else None
            assert normalized(labels[key],parser) == normalized(row['label'],parser), (name,key,'label')
            assert cls.name2subtitle[key] == row['text'], (name,key,'subtitle')
            audio = Path(cls._get_audio_path(sample))
            assert audio.resolve() == Path(row['audio']).resolve()
            face = Path(cls._get_face_path(sample))
            array = np.load(face,mmap_mode='r',allow_pickle=False)
            assert array.ndim == 4 and array.shape[0] > 0 and array.shape[-1] == 3 and array.dtype == np.uint8
            inventory.append(dict(id=key,face=str(face),shape=list(array.shape),
                                  face_sha256=sha(face),audio=str(audio),audio_sha256=sha(audio)))
        inventory_path = OUT/'inventories'/f'{name}.json'
        save(inventory_path,inventory)
        datasets[name] = dict(samples=len(rows),manifest_sha256=sha(path),
            inventory_sha256=sha(inventory_path),official_test_order=names,screen_ids=[r['id'] for r in sampled])
        print('Verified dataset',name,len(rows),flush=True)
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    checkpoints = {}
    for epoch, entry in entries().items():
        path = ROOT/CFG['checkpoint_directory']/Path(entry['path']).name
        assert path.stat().st_size == entry['size'] and sha(path) == entry['lfs']['oid']
        cp = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert cp['epoch'] == epoch
        assert {d['face_or_frame'] for d in cp['config']['datasets'].values()} == {'multiface_audio_face_text'}
        assert cp['config']['model']['vis_processor']['train']['n_frms'] == 8
        assert all('video_llama_proj.'+x in cp['model'] for x in ('weight','bias'))
        link = WEIGHTS/path.name
        if link.is_symlink():assert link.resolve() == path.resolve()
        else:link.symlink_to(path)
        checkpoints[str(epoch)] = dict(path=str(path),sha256=entry['lfs']['oid'],bytes=entry['size'])
        del cp
    verifier_path = Path('/data/user02/AffectGPT/verify_model_integrity.py')
    spec = importlib.util.spec_from_file_location('face_backbone_integrity',verifier_path)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    verifier.REPORT_PATH = OUT/'backbone-integrity.json'
    assert verifier.verify() == 0 and read(verifier.REPORT_PATH)['all_passed']
    request = dict(config=CFG,identity=identity,datasets=datasets,checkpoints=checkpoints,
        backbone_integrity_sha256=sha(verifier.REPORT_PATH),
        expected_screen_predictions=7*9*CFG['samples_per_dataset'],
        expected_full_predictions=sum(x['samples'] for x in datasets.values()),
        created_at=datetime.now().astimezone().isoformat())
    save(OUT/'request.json',request)
    (OUT/'source').mkdir(exist_ok=True)
    for path in (CONFIG,Path(__file__).resolve()):
        (OUT/'source'/path.name).write_bytes(path.read_bytes())
    status('prepared',datasets=9,checkpoints=7,expected_screen_predictions=request['expected_screen_predictions'],
           expected_full_predictions=request['expected_full_predictions'])


def check():
    request = read(OUT/'request.json')
    assert request['config'] == CFG and request['identity'] == source_identity()
    assert sha(OUT/'backbone-integrity.json') == request['backbone_integrity_sha256']
    for name, info in request['datasets'].items():
        assert sha(ROOT/'data/manifests/benchmark'/f'{name}.jsonl') == info['manifest_sha256']
        assert sha(OUT/'inventories'/f'{name}.json') == info['inventory_sha256']
    for info in request['checkpoints'].values():
        assert Path(info['path']).stat().st_size == info['bytes'] and sha(info['path']) == info['sha256']
    return request


def selected():
    result = read(OUT/'selection.json')
    assert result['request_sha256'] == sha(OUT/'request.json')
    assert result['selected_epoch'] in CFG['epochs']
    assert result['checkpoint_sha256'] == read(OUT/'request.json')['checkpoints'][str(result['selected_epoch'])]['sha256']
    return result


def epochs(scope):
    return CFG['epochs'] if scope == 'screen' else [selected()['selected_epoch']]


def result_dir(dataset,scope):
    return OUT/scope/'output'/f'results-{dataset.lower()}'/MODEL_NAME


def prediction_path(dataset,epoch,scope):
    return result_dir(dataset,scope)/Path(entries()[epoch]['path']).with_suffix('.npz').name


def verify_predictions(dataset,scope):
    import numpy as np
    request = read(OUT/'request.json')
    expected = set(request['datasets'][dataset]['screen_ids' if scope=='screen' else 'official_test_order'])
    files = {}
    for epoch in epochs(scope):
        path = prediction_path(dataset,epoch,scope)
        with np.load(path,allow_pickle=True) as data:
            predictions = data['name2reason'].item()
        assert set(predictions) == expected and all(isinstance(v,str) for v in predictions.values())
        files[str(epoch)] = dict(path=str(path),sha256=sha(path),samples=len(predictions),
                                empty_answers=sum(not v.strip() for v in predictions.values()))
    return files


def infer(dataset,scope):
    import numpy as np
    import torch
    request = check(); original_infer,_ = bootstrap()
    if scope == 'screen':patch_screen_dataset(original_infer,dataset)
    receipt_path = OUT/scope/'receipts'/f'{dataset}.inference.json'
    if receipt_path.exists():
        assert read(receipt_path)['files'] == verify_predictions(dataset,scope); return
    names = request['datasets'][dataset]['screen_ids' if scope=='screen' else 'official_test_order']
    wanted = set(names)
    for row in read(OUT/'inventories'/f'{dataset}.json'):
        if row['id'] not in wanted:continue
        assert sha(row['face']) == row['face_sha256'] and sha(row['audio']) == row['audio_sha256']
    from my_affectgpt.models.affectgpt import AffectGPT
    from my_affectgpt.conversation.conversation_video import Chat
    load_original = AffectGPT.load_state_dict
    answer_original = Chat.answer_sample
    current = dict(epoch=None,completed=0)
    epoch_iterator = iter(epochs(scope))
    def load_verified(model,state_dict,*args,**kwargs):
        epoch = next(epoch_iterator); state = dict(state_dict)
        aliases = {f'video_llama_proj.{s}':f'affectgpt_proj.{s}' for s in ('weight','bias')}
        for old,new in aliases.items():
            assert old in state and new not in state
            state[new] = state.pop(old)
        required = {n for n,p in model.named_parameters() if p.requires_grad and not n.startswith('image_llama_proj.')}
        assert required and required <= state.keys()
        loaded = load_original(model,state,*args,**kwargs)
        assert not loaded.unexpected_keys and not required.intersection(loaded.missing_keys)
        parameters = dict(model.named_parameters())
        for name in required:
            value = state[name].to(dtype=parameters[name].dtype,device=parameters[name].device)
            assert torch.isfinite(value).all() and torch.equal(parameters[name].detach(),value),name
        save(OUT/scope/'loads'/dataset/f'epoch{epoch}.json',dict(epoch=epoch,
            checkpoint_sha256=request['checkpoints'][str(epoch)]['sha256'],aliases=aliases,
            active_trainable_tensors=len(required),all_active_values_verified=True))
        current.update(epoch=epoch,completed=0)
        random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed'])
        torch.cuda.manual_seed_all(CFG['seed'])
        status(scope+'_loading_checkpoint',dataset=dataset,checkpoint_epoch=epoch)
        return loaded
    def answer_progress(chat,*args,**kwargs):
        answer = answer_original(chat,*args,**kwargs)
        current['completed'] += 1
        status(scope+'_inference',dataset=dataset,checkpoint_epoch=current['epoch'],
               completed=current['completed'],total=len(names))
        return answer
    AffectGPT.load_state_dict = load_verified; Chat.answer_sample = answer_progress
    random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed'])
    torch.cuda.manual_seed_all(CFG['seed']);torch.set_num_threads(2)
    # Same generation arguments as the official README. Selection only changes
    # the supported checkpoint-range and dataset-loader arguments.
    checkpoint_options = ['inference.test_epochs=30-60','inference.skip_epoch=5'] if scope=='screen' else [f'inference.test_epoch={epochs(scope)[0]}']
    sys.argv = [str(UPSTREAM/'inference_hybird.py'),'--zeroshot','--dataset',dataset,
        '--cfg-path',str(UPSTREAM/CFG['yaml']),'--options',*checkpoint_options,
        f'inference.ckpt_root={WEIGHTS}',f'inference.base_root={OUT}/{scope}/output/results']
    with torch.no_grad():
        runpy.run_path(str(UPSTREAM/'inference_hybird.py'),run_name='__main__')
    files = verify_predictions(dataset,scope)
    save(receipt_path,dict(complete=True,files=files,identity=request['identity'],
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def verified_caches(dataset,scope):
    import numpy as np
    from mer_omni.protocol import BENCHMARKS
    names = set(read(OUT/'request.json')['datasets'][dataset]['screen_ids' if scope=='screen' else 'official_test_order'])
    caches = {}
    for epoch in epochs(scope):
        kinds = ['openset','openset-sentiment'] if BENCHMARKS[dataset]['task']=='sentiment' else ['openset']
        for kind in kinds:
            prediction = prediction_path(dataset,epoch,scope)
            path = prediction.with_name(prediction.stem+f'-{kind}.npz')
            with np.load(path,allow_pickle=True) as data:
                assert len(data['filenames']) == len(data['fileitems']) == len(names)
                assert set(data['filenames']) == names
            caches[f'{epoch}-{kind}'] = sha(path)
    return caches


def evaluator_runtime():
    import torch.distributed as dist
    import vllm
    from transformers import AutoTokenizer
    original = vllm.LLM
    class CheckedLLM:
        def __init__(self,*args,**kwargs):
            kwargs.update(CFG['extraction_runtime'])
            if not dist.is_initialized():
                dist.init_process_group('nccl',store=dist.HashStore(),rank=0,world_size=1)
            self.engine = original(*args,**kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model'],local_files_only=True)
        def generate(self,prompts,sampling_params,**kwargs):
            assert all(len(self.tokenizer.encode(p))+sampling_params.max_tokens <= CFG['extraction_runtime']['max_model_len'] for p in prompts)
            return self.engine.generate(prompts,sampling_params,**kwargs)
    vllm.LLM = CheckedLLM


def evaluate(dataset,scope):
    request = check()
    assert read(OUT/scope/'receipts'/f'{dataset}.inference.json')['files'] == verify_predictions(dataset,scope)
    destination = OUT/scope/'scores'/f'{dataset}.json'
    if destination.exists():
        assert read(destination)['cache_sha256'] == verified_caches(dataset,scope); return
    evaluator_runtime(); original_infer, official = bootstrap()
    if scope == 'screen':patch_screen_dataset(original_infer,dataset)
    status(scope+'_evaluation',dataset=dataset,checkpoints=len(epochs(scope)))
    best = official.main_zeroshot_scores(str(result_dir(dataset,scope)),debug=False,inter_print=True)
    caches = verified_caches(dataset,scope)
    scores = []
    for epoch in epochs(scope):
        values = official.main_zeroshot_scores(str(result_dir(dataset,scope)),debug=True,test_epochs=str(epoch),inter_print=False)
        assert all(math.isfinite(float(v)) for v in values)
        scores.append(dict(epoch=epoch,primary=float(values[0]),secondary=float(values[1]),tertiary=float(values[2]),
            checkpoint_sha256=request['checkpoints'][str(epoch)]['sha256']))
    winner = max(scores,key=lambda s:s['primary'])
    assert abs(winner['primary']-best[0]) < 1e-12
    count = CFG['samples_per_dataset'] if scope=='screen' else request['datasets'][dataset]['samples']
    save(destination,dict(dataset=dataset,samples=count,selection_only=scope=='screen',
        selected_epoch=winner['epoch'],primary=winner['primary'],per_checkpoint=scores,
        official_best=list(map(float,best)),cache_sha256=caches,identity=request['identity']))
    status(scope+'_dataset_completed',dataset=dataset,selected_epoch=winner['epoch'],primary=winner['primary'])


def rank_screen(scores):
    assert len(scores) == 9 and {s['dataset'] for s in scores} == set(CFG['datasets'])
    assert all(s['selection_only'] and s['samples']==CFG['samples_per_dataset'] for s in scores)
    ranking = []
    for score in scores:
        assert len(score['per_checkpoint']) == len(CFG['epochs'])
        assert {s['epoch'] for s in score['per_checkpoint']} == set(CFG['epochs'])
    for epoch in CFG['epochs']:
        values = [next(s for s in d['per_checkpoint'] if s['epoch']==epoch) for d in scores]
        assert len({s['checkpoint_sha256'] for s in values}) == 1
        assert all(math.isfinite(s['primary']) for s in values)
        ranking.append(dict(epoch=epoch,screening_mean=statistics.mean(s['primary'] for s in values),
                            checkpoint_sha256=values[0]['checkpoint_sha256']))
    return sorted(ranking,key=lambda s:(-s['screening_mean'],s['epoch']))


def select():
    request = check()
    scores = [read(OUT/'screen/scores'/f'{name}.json') for name in CFG['datasets']]
    for score in scores:
        assert score['identity'] == request['identity']
        assert score['cache_sha256'] == verified_caches(score['dataset'],'screen')
    ranking = rank_screen(scores); winner = ranking[0]
    assert winner['checkpoint_sha256'] == request['checkpoints'][str(winner['epoch'])]['sha256']
    result = dict(selected_epoch=winner['epoch'],checkpoint_sha256=winner['checkpoint_sha256'],
        screening_mean=winner['screening_mean'],ranking=ranking,request_sha256=sha(OUT/'request.json'),
        sample_plan_sha256=sha(ROOT/CFG['peer_screening_plan']),samples_per_dataset=CFG['samples_per_dataset'],
        test_subset_used_for_selection=True,independent_heldout_selection=False)
    path = OUT/'selection.json'
    if path.exists():assert read(path) == result
    else:save(path,result)
    status('checkpoint_selected',checkpoint_epoch=winner['epoch'],screening_mean=winner['screening_mean'])


def summarize():
    request = check(); _, official = bootstrap(); os.chdir(OUT/'full')
    chosen = selected()
    datasets = [read(OUT/'full/scores'/f'{name}.json') for name in CFG['datasets']]
    for score in datasets:
        assert not score['selection_only'] and score['selected_epoch']==chosen['selected_epoch']
        assert score['samples'] == request['datasets'][score['dataset']]['samples']
        assert score['cache_sha256'] == verified_caches(score['dataset'],'full')
    printed, average = official.func_return_scores_one(modelname=MODEL_NAME)
    assert abs(average-statistics.mean(s['primary'] for s in datasets)) < 1e-12
    peer = read(ROOT/CFG['dependency_summary'])
    peer_scores = {s['dataset']:s['primary'] for s in peer['datasets']}
    comparison = dict(peer_selected_step=peer['selected_step'],peer_mean=peer['dataset_mean'],
        face_mean=float(average),face_minus_peer_percentage_points=100*(average-peer['dataset_mean']),
        datasets=[dict(dataset=s['dataset'],face=s['primary'],peer=peer_scores[s['dataset']],
                       face_minus_peer_percentage_points=100*(s['primary']-peer_scores[s['dataset']])) for s in datasets],
        note='Both use one checkpoint selected on the same 180 test samples. Face uses official sampled generation; peer uses greedy generation. Different models and inference recipes remain.')
    save(OUT/'summary.json',dict(complete=True,benchmark='MER-UniBench',dataset_mean=float(average),
        selection=CFG['selection'],selected_epoch=chosen['selected_epoch'],datasets=datasets,
        screening=chosen,test_subset_used_for_selection=True,independent_heldout_selection=False,
        official_formatted_table=printed,comparison=comparison,notes=CFG['notes'],request_sha256=sha(OUT/'request.json')))
    status('completed',datasets=9,dataset_mean=float(average))


def dependency_ready(path):
    if not path.exists():return False
    from mer_omni.protocol import BENCHMARKS
    summary = read(path); scores = summary['datasets']
    assert len(scores) == 9 and {s['dataset'] for s in scores} == set(CFG['datasets'])
    assert all(s['strict_protocol'] and s['samples']==BENCHMARKS[s['dataset']]['count'] for s in scores)
    assert len({s['checkpoint_sha256'] for s in scores}) == 1
    assert abs(summary['dataset_mean']-statistics.mean(s['primary'] for s in scores)) < 1e-12
    assert summary['selected_step'] == 34000
    return True


def wait_resources(stage):
    limits = CFG['resource_limits']; stable = None
    while True:
        free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        host = int(memory['MemAvailable'].split()[0])//1024
        ready = free>=limits[stage+'_gpu_free_mib'] and host>=limits[stage+'_host_available_mib']
        stable = (stable or time.monotonic()) if ready else None
        if stable and time.monotonic()-stable>=30:return
        status('waiting_for_resources',next_stage=stage,gpu_free_mib=free,host_available_mib=host)
        time.sleep(10)


def pipeline():
    check()
    lock = (OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dependency = ROOT/CFG['dependency_summary']
    while not dependency_ready(dependency):
        state = read(dependency.parent/'status.json') if (dependency.parent/'status.json').exists() else {}
        status('waiting_for_peer_evaluation',dependency=str(dependency),peer_status=state)
        time.sleep(20)
    save(OUT/'dependency.json',dict(path=str(dependency),sha256=sha(dependency)))
    for scope in ('screen','full'):
        if scope=='full':select()
        for dataset in CFG['datasets']:
            for stage,kind,marker in [('infer','inference',OUT/scope/'receipts'/f'{dataset}.inference.json'),
                                      ('evaluate','evaluation',OUT/scope/'scores'/f'{dataset}.json')]:
                if not marker.exists():wait_resources(kind)
                status(scope+'_'+stage,dataset=dataset)
                with (LOGS/f'{scope}.{dataset}.{stage}.log').open('a') as log:
                    subprocess.run([str(PYTHON),'-B','-u',str(Path(__file__).resolve()),stage,'--dataset',dataset,'--scope',scope],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
    summarize()


def queue():
    OUT.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'setup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (OUT/'request.json').exists():
        status('preparing_assets',next_stage='wait_for_peer_evaluation')
        with (LOGS/'download.log').open('a') as log:
            subprocess.run([sys.executable,'-B','-u',str(ROOT/'tools/download_affectgpt_face_official.py')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        prepare()
    pipeline()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','check','infer','evaluate','select','summarize','pipeline','queue'])
    parser.add_argument('--dataset',choices=CFG['datasets'])
    parser.add_argument('--scope',choices=['screen','full'],default='screen')
    args = parser.parse_args()
    try:
        if args.stage in ('infer','evaluate'):
            if not args.dataset:parser.error('--dataset required')
            globals()[args.stage](args.dataset,args.scope)
        else:globals()[args.stage]()
    except BaseException as exc:
        if args.stage not in ('check','prepare'):status('failed',failed_stage=args.stage,error=repr(exc))
        raise
