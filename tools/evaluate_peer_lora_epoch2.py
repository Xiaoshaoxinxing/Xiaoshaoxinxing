#!/usr/bin/env python3
"""Nine-dataset official scoring for the Qwen LoRA SFT baseline."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import peer_lora_epoch2 as train
from mer_omni.protocol import BENCHMARKS, OV_QUESTION, check_coverage, check_prediction_ids
from mer_omni.data import read_manifest, verify_manifest_receipt
from mer_omni.evaluation import (score_dataset, aggregate_scores, reference_fingerprint,
                                 read_predictions, extract_predictions)

CONFIG = ROOT/'config/peer_lora_epoch2_evaluation.json'
PLAN = ROOT/'reports'/train.settings()[0]['experiment']/'evaluation-plan.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def layout(model_name):
    if model_name != 'peer':
        raise ValueError('Historical experiment A is archived; use the current SFT baseline')
    cfg, out = train.settings()
    return out/f"evaluation_step{cfg['expected_optimizer_steps']}_merunibench_greedy"


def status(model_name, stage, **extra):
    train.save(layout(model_name)/'status.json', dict(stage=stage, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                    pid=os.getpid(), **extra))


def protocol_identity():
    paths = ['config/peer_lora_epoch2_evaluation.json','tools/evaluate_peer_lora_epoch2.py',
             'mer_omni/evaluation.py','mer_omni/protocol.py','mer_omni/data.py',
             'mer_omni/reference/affectgpt.py',
             'tools/peer_lora_epoch2.py',train.settings()[0]['peer_source'],
             'models/Qwen2.5-7B-Instruct/config.json','models/Qwen2.5-7B-Instruct/tokenizer_config.json']
    return dict(sources={p:train.sha(ROOT/p) for p in paths}, reference_sha256=reference_fingerprint())


def prepare():
    if PLAN.exists():
        check()
        return
    cfg = json.loads(CONFIG.read_text())
    assert set(cfg['datasets']) == set(BENCHMARKS) and len(cfg['datasets']) == 9
    assert cfg['question'] == OV_QUESTION
    _, out = train.check()
    training = json.loads((out/'request.json').read_text())
    sets = {}
    for name in cfg['datasets']:
        p = ROOT/'data/manifests/benchmark'/f'{name}.jsonl'
        verify_manifest_receipt(p)
        rows = read_manifest(p); check_coverage(rows,name)
        sets[name] = dict(samples=len(rows), manifest_sha256=train.sha(p),
                          overlap_ids=training['benchmark_overlap_ids'][name])
    value = dict(config=cfg, identity=protocol_identity(), datasets=sets,
                 expected_total_samples=sum(x['samples'] for x in sets.values()),
                 train_manifest_sha256=training['identity'][training['config']['manifest']])
    if PLAN.exists() and json.loads(PLAN.read_text()) != value:
        raise ValueError('Evaluation plan already sealed with different contents')
    train.save(PLAN,value)
    print(json.dumps(dict(passed=True, datasets=len(sets), samples=value['expected_total_samples'])))


def check():
    cfg = json.loads(CONFIG.read_text()); plan = json.loads(PLAN.read_text())
    if not train.evaluation_configs_match(plan['config'], cfg) or not train.protocol_identity_matches(plan['identity'], protocol_identity()):
        raise ValueError('Evaluation protocol changed after sealing')
    for name, info in plan['datasets'].items():
        path = ROOT/'data/manifests/benchmark'/f'{name}.jsonl'
        if train.sha(path) != info['manifest_sha256']:
            raise ValueError('Benchmark manifest changed')
    return cfg, plan


def benchmark(name):
    p = ROOT/'data/manifests/benchmark'/f'{name}.jsonl'
    verify_manifest_receipt(p)
    rows = read_manifest(p); check_coverage(rows,name)
    return rows


def peer_prompt(peer, processor, row, cfg, cache, device, *, question=OV_QUESTION, audio_samples=None):
    # Whitelist before calling shared training prompt code; labels cannot affect generation.
    clean = {k:row[k] for k in ('id','text','video','audio')}
    clean['question'] = question
    audio_kwargs={}
    if audio_samples is not None and audio_samples>processor.feature_extractor.n_samples:
        hop=processor.feature_extractor.hop_length
        audio_kwargs=dict(audio_kwargs=dict(max_length=math.ceil(audio_samples/hop)*hop,truncation=False))
    prompt = processor.apply_chat_template(peer.conversation_for(clean, cache, cfg['max_frames']),
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', padding=False,**audio_kwargs)
    return {k:v.to(device) if hasattr(v,'to') else v for k,v in prompt.items()}


def model_identity(model_name):
    if model_name == 'peer':
        cfg,out = train.check()
        completion = json.loads((out/'completed.json').read_text())
        coverage = json.loads((out/'coverage.json').read_text())
        assert completion['global_step'] == cfg['expected_optimizer_steps'] and coverage['passed']
        assert train.identity_matches(coverage['identity'], train.identity(cfg))
        cp = out/f"checkpoint-{cfg['expected_optimizer_steps']:07d}"
        receipt = json.loads((cp/'receipt.json').read_text())
        assert train.identity_matches(receipt['identity'], train.identity(cfg)) and receipt['step'] == cfg['expected_optimizer_steps']
        assert receipt['adapter_sha256'] == train.sha(cp/'adapter/adapter_model.safetensors')
        assert receipt['adapter_config_sha256'] == train.sha(cp/'adapter/adapter_config.json')
        return dict(checkpoint_sha256=receipt['adapter_sha256'],
                    adapter_config_sha256=receipt['adapter_config_sha256'],
                    train_manifest_sha256=train.sha(ROOT/cfg['manifest']), checkpoint=str(cp/'adapter'))

    raise ValueError("Only the current peer SFT model is supported")


def load_model(model_name, ident):
    import torch
    if model_name == 'peer':
        from peft import PeftModel
        cfg,_=train.check(); peer=train.peer_module()
        proc=peer.Qwen2_5OmniProcessor.from_pretrained(ROOT/cfg['model_path'],max_pixels=cfg['max_pixels'])
        base=peer.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ROOT/cfg['model_path'],
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda')
        model=PeftModel.from_pretrained(base,ident['checkpoint'],is_trainable=False)
        return model.eval(),proc,cfg,peer

    raise ValueError("Only the current peer SFT model is supported")


def verify_predictions(model_name,name,ident=None):
    cfg,plan=check(); out=layout(model_name)/name
    ident=ident or model_identity(model_name)
    receipt=json.loads((out/'predictions.receipt.json').read_text())
    for k,v in ident.items():
        assert receipt[k] == v
    assert receipt['complete'] and receipt['dataset']==name
    assert receipt['generation_sha256']==digest(cfg['generation'])
    assert receipt['protocol_identity']==plan['identity']
    assert receipt['manifest_sha256']==plan['datasets'][name]['manifest_sha256']
    assert receipt['predictions_sha256']==train.sha(out/'predictions.jsonl')
    assert receipt['input_processing_sha256']==train.sha(out/'input-processing.json')
    pred=read_predictions(out/'predictions.jsonl'); rows=benchmark(name)
    check_prediction_ids(rows,pred)
    return rows,pred,receipt


def generation_budget(settings, prompt_tokens):
    budget=settings['max_new_tokens']
    if prompt_tokens+budget>32768:
        raise ValueError('Input exceeds registered context; no truncation')
    return budget


def infer(model_name,name):
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import GenerationConfig
    cfg,plan=check(); ident=model_identity(model_name); out=layout(model_name)/name
    if (out/'predictions.receipt.json').exists():
        verify_predictions(model_name,name,ident); return
    if out.exists():
        raise FileExistsError(f'Incomplete inference directory {out}; preserve it and use a new run identity')
    rows=benchmark(name); model,proc,training,peer=load_model(model_name,ident)
    assert proc.tokenizer.eos_token_id==cfg['generation']['eos_token_id']
    assert proc.tokenizer.pad_token_id==cfg['generation']['pad_token_id']
    out.mkdir(parents=True)
    train.save(out/'protocol.json',dict(config=cfg,identity=plan['identity'],model=ident,
                                      training_config=training,checkpoint_selection=cfg['selection']))
    torch.manual_seed(cfg['generation']['seed']); records=[]; predictions={}
    gen=GenerationConfig(**{k:v for k,v in cfg['generation'].items() if k not in ('seed','max_length')},use_cache=True)
    with (out/'predictions.jsonl').open('x') as stream, torch.inference_mode():
        for index,row in enumerate(rows):
            header=sf.info(row['audio'])
            samples_16k=math.ceil(header.frames*16000/header.samplerate)
            batch=peer_prompt(peer,proc,row,training,ROOT/'.cache/sft/evaluation'/layout(model_name).name/name,'cuda',audio_samples=samples_16k)
            model.base_model.model.rope_deltas=None
            length=batch['input_ids'].shape[1]
            remaining=generation_budget(cfg['generation'],length)
            generated=model.generate(**batch,generation_config=gen,max_new_tokens=remaining,use_audio_in_video=False)
            suffix=generated[0,length:].tolist(); eos={cfg['generation']['eos_token_id']}
            stop=next((i for i,token in enumerate(suffix) if token in eos),len(suffix))
            answer=proc.tokenizer.decode(suffix[:stop],skip_special_tokens=True).strip()
            if not answer:
                raise ValueError(f"Empty answer: {row['id']}")
            predictions[row['id']]=answer
            stream.write(json.dumps(dict(id=row['id'],prediction=answer),ensure_ascii=False)+'\n');stream.flush()
            records.append(dict(id=row['id'],prompt_tokens=length,generated_tokens=len(suffix),
                                ended_with_eos=stop<len(suffix),generation_budget=remaining,reached_budget=len(suffix)==remaining,
                                audio_seconds=header.frames/header.samplerate,audio_buffer_extended=header.frames/header.samplerate>300))
            status(model_name,'inference',dataset=name,completed=index+1,total=len(rows))
            print(json.dumps(records[-1]),flush=True)
            del batch,generated
    check_prediction_ids(rows,predictions)
    train.save(out/'input-processing.json',dict(records=records,truncated_samples=0,processed_samples=len(rows)))
    np.savez_compressed(out/'checkpoint_final.npz',name2reason=predictions)
    train.save(out/'predictions.receipt.json',dict(**ident,dataset=name,complete=True,
        protocol_identity=plan['identity'],generation=cfg['generation'],generation_sha256=digest(cfg['generation']),
        manifest_sha256=plan['datasets'][name]['manifest_sha256'],predictions_sha256=train.sha(out/'predictions.jsonl'),
        input_processing_sha256=train.sha(out/'input-processing.json'),
        accepted_release_overlap_ids=plan['datasets'][name]['overlap_ids']))


def extract(model_name,name):
    cfg,_=check(); out=layout(model_name)/name
    rows,pred,receipt=verify_predictions(model_name,name)
    if (out/'extraction.json').exists():
        cache=json.loads((out/'extraction.json').read_text())
        validate_cache(cache,receipt,cfg);return
    cache=extract_predictions(pred,cfg['evaluator'],sentiment=BENCHMARKS[name]['task']=='sentiment',**cfg['extraction_runtime'])
    cache['predictions_sha256']=receipt['predictions_sha256']
    train.save(out/'extraction.json',cache)


def validate_cache(cache,receipt,cfg):
    if (cache['predictions_sha256']!=receipt['predictions_sha256'] or
        cache['reference_sha256']!=reference_fingerprint() or cache['evaluator']!=cfg['evaluator']):
        raise ValueError('Extraction identity mismatch')


def score(model_name,name):
    cfg,_=check(); out=layout(model_name)/name
    rows,_,receipt=verify_predictions(model_name,name)
    cache=json.loads((out/'extraction.json').read_text());validate_cache(cache,receipt,cfg)
    result=score_dataset(rows,cache)
    result.update({k:receipt[k] for k in ('checkpoint_sha256','train_manifest_sha256','generation_sha256')})
    result.update(evaluator_sha256=digest(cfg['evaluator']),manifest_sha256=receipt['manifest_sha256'],
                  extraction_sha256=train.sha(out/'extraction.json'),accepted_release_overlap_ids=receipt['accepted_release_overlap_ids'])
    destination=out/'score.json'
    if destination.exists() and json.loads(destination.read_text())!=result:
        raise ValueError('Existing score differs; refusing overwrite')
    train.save(destination,result)
    if name=='OVMERDPlus':
        overlap=set(receipt['accepted_release_overlap_ids']); subsets={}
        for label,inside in [('overlap',True),('nonoverlap',False)]:
            selected=[r for r in rows if (r['id'] in overlap)==inside]
            ids={r['id'] for r in selected}
            assert len(selected)==(99 if inside else 433)
            subcache=dict(cache,openset={k:v for k,v in cache['openset'].items() if k in ids})
            subsets[label]=score_dataset(selected,subcache,strict=False)
        train.save(out/'overlap-subsets.json',dict(source_extraction_sha256=train.sha(out/'extraction.json'),
                   note='Subset diagnostics do not replace the official full-set score.',**subsets))


def wait_for_resources(model_name,stage):
    required_gpu=22528 if stage=='extract' else 24576
    required_host=24576 if stage=='extract' else 16384
    ready=None
    while True:
        free,host=train.resources()
        if free>=required_gpu and host>=required_host:
            ready=ready or time.monotonic()
            if time.monotonic()-ready>=30:return
        else:ready=None
        status(model_name,'waiting_for_resources',next_stage=stage,gpu_free_mib=free,host_available_mib=host,
               required_gpu_mib=required_gpu,required_host_mib=required_host)
        time.sleep(10)


def pipeline(model_name):
    cfg,_=check(); out=layout(model_name);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    log=ROOT/'logs/evaluation'/out.parent.name/out.name;log.mkdir(parents=True,exist_ok=True)
    for name in cfg['datasets']:
        for stage,marker in [('infer','predictions.receipt.json'),('extract','extraction.json'),('score','score.json')]:
            # Even on resume, each command verifies existing artifacts and never silently skips corrupt output.
            if stage!='score' and not (out/name/marker).exists():wait_for_resources(model_name,stage)
            status(model_name,stage,dataset=name)
            python=ROOT/('.venv-eval/bin/python' if stage=='extract' else '.venv/bin/python')
            command=[str(python),'-B','-u',str(Path(__file__).resolve()),stage,'--model',model_name,'--dataset',name]
            with (log/f'{name}.{stage}.log').open('a') as f:
                subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    scores=[json.loads((out/name/'score.json').read_text()) for name in cfg['datasets']]
    summary=aggregate_scores(scores)
    summary['note']='Official complete-set mean includes 99 overlapping OVMERDPlus samples; see separate subset diagnostics.'
    train.save(out/'summary.json',summary)
    status(model_name,'completed',dataset_mean=summary['dataset_mean'],datasets=9)



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','check','infer','extract','score','pipeline'])
    parser.add_argument('--model',choices=['peer'],default='peer')
    parser.add_argument('--dataset',choices=list(BENCHMARKS))
    args=parser.parse_args()
    try:
        if args.stage in ('prepare','check'):globals()[args.stage]()
        elif args.stage=='pipeline':pipeline(args.model)
        else:
            if args.dataset is None:parser.error('--dataset is required')
            globals()[args.stage](args.model,args.dataset)
    except Exception as exc:
        if args.stage not in ('prepare','check'):status(args.model,'failed',failed_stage=args.stage,error=repr(exc))
        raise
