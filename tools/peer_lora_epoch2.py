#!/usr/bin/env python3
"""Qwen LoRA SFT on the fixed, versioned MER-Caption+ targets.

The vendored training recipe is passed explicitly. Runtime support includes
torch2.4 TP compatibility, identities, exact RNG restoration, resource waiting,
and an isolated real forward/backward preflight.
"""
import argparse
import builtins
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/peer_lora_epoch2.json'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  NUMBA_CACHE_DIR=str(ROOT / '.cache/peer-lora-numba'))
sys.path.insert(0, str(ROOT / '.cache/peer-lora-python'))
sys.path.insert(0, str(ROOT))

from mer_omni.peer_source_migration import (
    source_identities_match as identity_matches,
    protocol_identities_match as protocol_identity_matches,
    training_configs_match as config_matches,
    evaluation_configs_match,
)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def settings():
    cfg = json.loads(CONFIG.read_text())
    return cfg, ROOT / cfg['output_dir']


def status(stage, **extra):
    _, out = settings()
    save(out / 'status.json', dict(stage=stage, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                  pid=os.getpid(), **extra))


def identity(cfg):
    paths = [str(CONFIG.relative_to(ROOT)), cfg['peer_source'], cfg['source_manifest'],
             cfg['source_plan'], cfg['manifest'], 'tools/peer_lora_epoch2.py',
             'requirements/peer-lora-extra.txt', 'models/Qwen2.5-Omni-3B/config.json',
             'models/Qwen2.5-Omni-3B/tokenizer.json', 'models/Qwen2.5-Omni-3B/chat_template.json',
             'models/Qwen2.5-Omni-3B/model.safetensors.index.json']
    return {p: sha(ROOT / p) for p in paths}


def prepare():
    from collections import Counter
    from tokenizers import Tokenizer
    from mer_omni.protocol import DESCRIPTION_QUESTION, OV_QUESTION
    cfg, out = settings()
    if (out / 'request.json').exists():
        check()
        return
    out.mkdir(parents=True, exist_ok=True)
    source = [json.loads(x) for x in (ROOT / cfg['source_manifest']).read_text().splitlines()]
    plan = json.loads((ROOT / cfg['source_plan']).read_text())
    assigned = {p['id']: p['qtype'] for p in plan}
    assert len(source) == len(assigned) == cfg['expected_samples_per_epoch']
    assert {r['id'] for r in source} == set(assigned)
    manifest = ROOT / cfg['manifest']
    rows = []
    for row in source:
        qtype = assigned[row['id']]
        assert row['dataset'] == 'MER-Caption+' and row['split'] == 'train'
        for k in ('video', 'audio'):
            if not Path(row[k]).is_file():
                raise FileNotFoundError(row[k])
        rows.append(dict(id=row['id'], task=qtype,
                         question=DESCRIPTION_QUESTION if qtype == 'description' else OV_QUESTION,
                         target=row['description'] if qtype == 'description' else
                         "The character's emotional state is " + ', '.join(row['openset']) + '.',
                         **{k: row[k] for k in ('text', 'video', 'audio')}))
    tokenizer = Tokenizer.from_file(str(ROOT / cfg['model_path'] / 'tokenizer.json'))
    lengths = [len(x.ids) for x in tokenizer.encode_batch([r['target'] for r in rows], add_special_tokens=False)]
    if max(lengths) > cfg['max_target_tokens']:
        raise ValueError('A targets would be truncated by peer token cap')
    with manifest.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    benchmarks = {}
    for p in sorted((ROOT / 'data/manifests/benchmark').glob('*.jsonl')):
        ids = {json.loads(s)['id'] for s in p.read_text().splitlines()}
        benchmarks[p.stem] = sorted(set(assigned) & ids)
    save(out / 'request.json', dict(config=cfg, identity=identity(cfg),
         samples=len(rows), targets=dict(Counter(r['task'] for r in rows)),
         max_target_tokens_actual=max(lengths), truncated_targets=0,
         expected_optimizer_steps=len(rows) * cfg['epochs'],
         benchmark_overlap_ids=benchmarks,
         data_provenance='A fixed targets approved by user; peer manifest unavailable'))
    (out / 'source').mkdir(exist_ok=True)
    for p in [cfg['peer_source'], 'tools/peer_lora_epoch2.py', str(CONFIG.relative_to(ROOT))]:
        (out / 'source' / Path(p).name).write_bytes((ROOT / p).read_bytes())
    status('prepared', total_steps=cfg['expected_optimizer_steps'])
    print(json.dumps(dict(samples=len(rows), steps=len(rows) * cfg['epochs'],
                         max_target_tokens=max(lengths), overlap_counts={k:len(v) for k,v in benchmarks.items()})))


def check():
    cfg, out = settings()
    request = json.loads((out / 'request.json').read_text())
    if not identity_matches(request['identity'], identity(cfg)) or not config_matches(request['config'], cfg):
        raise ValueError('Experiment identity changed; use a new experiment instead')
    return cfg, out


def resources():
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    host = int(mem['MemAvailable'].split()[0]) // 1024
    return free, host


def wait_resources(required_gpu_mib):
    ready_since = None
    while True:
        free, host = resources()
        if free >= required_gpu_mib and host >= 16384:
            ready_since = ready_since or time.monotonic()
            if time.monotonic() - ready_since >= 30:
                return
        else:
            ready_since = None
        status('waiting_for_resources', gpu_free_mib=free, host_available_mib=host,
               required_gpu_free_mib=required_gpu_mib, required_host_available_mib=16384)
        print(f'Waiting/stabilizing: GPU {free}/{required_gpu_mib} MiB, host {host}/16384 MiB', flush=True)
        time.sleep(10)


def peer_module():
    cfg, out = check()
    import torch
    from transformers import Qwen2_5OmniThinkerConfig
    from packaging.version import Version
    spec = importlib.util.spec_from_file_location('peer_original_train_sft', ROOT / cfg['peer_source'])
    peer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(peer)
    original_load = peer.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained

    def load(*args, **kwargs):
        config = Qwen2_5OmniThinkerConfig.from_pretrained(args[0], local_files_only=True)
        if Version(torch.__version__.split('+')[0]) < Version('2.5'):
            config.text_config.base_model_tp_plan = {}
        kwargs.update(config=config, local_files_only=True)
        return original_load(*args, **kwargs)

    peer.Qwen2_5OmniThinkerForConditionalGeneration = SimpleNamespace(from_pretrained=load)
    # Decord's default threads=0 can create hundreds of decoder threads on this host.
    import decord
    original_reader = decord.VideoReader
    def reader(*args, **kwargs):
        kwargs.setdefault('num_threads', 2)
        return original_reader(*args, **kwargs)
    decord.VideoReader = reader
    return peer


def checkpoint_with_rng(model, optimizer, output, epoch, next_index, global_step):
    import torch
    import numpy as np
    cfg, _ = settings()
    cp = output / f'checkpoint-{global_step:07d}'
    cp.mkdir(parents=True, exist_ok=True)
    if (cp/'receipt.json').exists():
        receipt = json.loads((cp/'receipt.json').read_text())
        if not identity_matches(receipt['identity'], identity(cfg)):
            raise ValueError('Checkpoint destination belongs to another experiment')
    model.save_pretrained(cp/'adapter')
    torch.save(dict(optimizer=optimizer.state_dict(), epoch=epoch, next_index=next_index,
                    global_step=global_step), cp/'state.tmp')
    (cp/'state.tmp').replace(cp/'state.pt')
    torch.save(dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all()), cp/'rng.tmp')
    (cp/'rng.tmp').replace(cp/'rng.pt')
    save(cp/'receipt.json', dict(identity=identity(cfg), step=global_step,
                               adapter_sha256=sha(cp/'adapter/adapter_model.safetensors'),
                               adapter_config_sha256=sha(cp/'adapter/adapter_config.json'),
                               state_sha256=sha(cp/'state.pt'), rng_sha256=sha(cp/'rng.pt')))
    # Advance latest only after adapter, optimizer, RNG and receipt are durable.
    (output/'latest.tmp').write_text(str(cp.resolve()))
    (output/'latest.tmp').replace(output/'latest')
    print(f'checkpoint={cp}', flush=True)


def preflight():
    import torch
    import numpy as np
    from peft import LoraConfig, get_peft_model
    from importlib.metadata import version
    cfg, out = check()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU preflight requires host CUDA access')
    peer = peer_module()
    free_mib,_ = resources()
    total_mib = torch.cuda.get_device_properties(0).total_memory/2**20
    torch.cuda.set_per_process_memory_fraction(min(.60,(free_mib-2048)/total_mib),0)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    proc = peer.Qwen2_5OmniProcessor.from_pretrained(ROOT / cfg['model_path'], max_pixels=cfg['max_pixels'])
    rows = peer.load_rows(ROOT / cfg['manifest'])
    order = list(range(len(rows))); random.Random(cfg['seed'] + 1).shuffle(order)
    # Longest conservative A context covers a long audio/text case without reading test sets.
    bounds = json.loads((ROOT / cfg['source_bounds']).read_text())
    longest = max(bounds, key=lambda x:max(x['upper_tokens'].values()))['id']
    selected = [rows[order[0]], next(r for r in rows if r['id'] == longest), max(rows, key=lambda r:len(r['target']))]
    model = peer.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        ROOT / cfg['model_path'], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation='sdpa').to('cuda')
    model.config.use_cache = False
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=.05, target_modules=r'.*model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',
        bias='none', task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
    params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _,p in params) == 3686400
    assert all('.self_attn.' in n and '.lora_' in n for n,_ in params)
    optimizer = torch.optim.AdamW([p for _,p in params], lr=cfg['lr'])
    records = []
    for row in selected:
        optimizer.zero_grad(set_to_none=True)
        inputs = peer.encode(proc, row, ROOT / cfg['frame_cache'], cfg['max_frames'], cfg['max_target_tokens'], 'cuda')
        assert 'pixel_values' in inputs and 'pixel_values_videos' not in inputs
        assert len(inputs['image_grid_thw']) == 2 and inputs['feature_attention_mask'].sum() > 0
        supervised = inputs['labels'][inputs['labels'] != -100]
        assert supervised[-1].item() == proc.tokenizer.convert_tokens_to_ids('<|im_end|>')
        assert proc.tokenizer.decode(supervised[:-1]) == row['target']
        from evaluate_peer_lora_epoch2 import peer_prompt
        prompt = peer_prompt(peer, proc, row, cfg, ROOT/cfg['frame_cache'], 'cuda', question=row['question'])
        prefix = inputs['labels'].eq(-100).sum().item()
        assert torch.equal(prompt['input_ids'], inputs['input_ids'][:, :prefix])
        assert all(torch.equal(prompt[k], inputs[k]) for k in prompt if k not in ('input_ids','attention_mask'))
        del prompt
        outputs = model(**inputs, use_cache=False)
        loss = outputs.loss
        assert torch.isfinite(loss)
        loss.backward()
        disconnected = [n for n,p in params if p.grad is None]
        assert not disconnected, disconnected
        assert sum(float(p.grad.float().norm()) for n,p in params if '.lora_B.' in n) > 0
        norm = torch.nn.utils.clip_grad_norm_([p for _,p in params], 1.0, error_if_nonfinite=True)
        optimizer.step()
        records.append(dict(id=row['id'], task=row['task'], tokens=inputs['input_ids'].shape[1],
                            supervised_tokens=supervised.numel(), loss=float(loss.detach()),
                            gradient_norm_before_clip=float(norm), image_grid=inputs['image_grid_thw'].tolist()))
        del inputs, outputs, loss
    optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(out / 'preflight_adapter')
    from safetensors.torch import load_file
    saved = load_file(str(out / 'preflight_adapter/adapter_model.safetensors'))
    from peft import get_peft_model_state_dict
    current = get_peft_model_state_dict(model)
    assert saved.keys() == current.keys()
    assert all(torch.equal(saved[k], current[k].detach().cpu()) for k in saved)
    # The normal SDPA kernels above match the peer training path, but their BF16
    # backward is not bitwise deterministic. Isolate the state-restoration check
    # with math SDPA; keep the peer's default backend for all formal training.
    import copy
    from peft import set_peft_model_state_dict
    from torch.nn.attention import SDPBackend, sdpa_kernel
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = torch.get_rng_state(); saved_cuda_rng = torch.cuda.get_rng_state_all()
    saved_python_rng = random.getstate(); saved_numpy_rng = np.random.get_state()
    def next_update():
        optimizer.zero_grad(set_to_none=True)
        batch = peer.encode(proc, selected[0], ROOT/cfg['frame_cache'], 2, 384, 'cuda')
        with sdpa_kernel(SDPBackend.MATH):
            loss = model(**batch, use_cache=False).loss
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _,p in params], 1.0, error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())
    expected_loss = next_update()
    expected_state = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    set_peft_model_state_dict(model, saved)
    optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(saved_rng); torch.cuda.set_rng_state_all(saved_cuda_rng)
    random.setstate(saved_python_rng); np.random.set_state(saved_numpy_rng)
    restored = get_peft_model_state_dict(model)
    assert all(torch.equal(v, restored[k].detach().cpu()) for k,v in saved.items())
    restored_optimizer = optimizer.state_dict()
    assert restored_optimizer['param_groups'] == saved_optimizer['param_groups']
    for key, state in saved_optimizer['state'].items():
        assert state.keys() == restored_optimizer['state'][key].keys()
        for name, value in state.items():
            actual = restored_optimizer['state'][key][name]
            assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
    assert torch.equal(saved_rng, torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(saved_cuda_rng,torch.cuda.get_rng_state_all()))
    replay_loss = next_update()
    assert expected_loss == replay_loss
    assert all(torch.equal(v, get_peft_model_state_dict(model)[k].detach().cpu()) for k,v in expected_state.items())
    peak = math.ceil(torch.cuda.max_memory_reserved() / 2**20)
    result = dict(passed=True, identity=identity(cfg), parameters=3686400, records=records,
                  adapter_roundtrip_exact=True, isolated_optimizer_steps=len(records)+2,
                  next_update_replay_exact=True, training_inference_prefix_equal=True,
                  replay_attention_backend='math SDPA only for the isolated replay check',
                  formal_training_attention_backend='peer default SDPA',
                  default_sdpa_bitwise_resume_guaranteed=False,
                  restored_weights_optimizer_rng_exact=True,
                  formal_training_starts_from_fresh_base=True, peak_reserved_mib=peak,
                  required_gpu_free_mib=max(12288, math.ceil(peak*1.15)+1536),
                  versions={k:version(k) for k in ['torch','transformers','peft','librosa','decord']})
    save(out / 'preflight.json', result)
    status('preflight_passed', **{k:result[k] for k in ('peak_reserved_mib','required_gpu_free_mib')})
    print(json.dumps(result), flush=True)


def train(resume=False):
    import torch
    import numpy as np
    cfg, out = check()
    pre = json.loads((out / 'preflight.json').read_text())
    if not pre['passed'] or not identity_matches(pre['identity'], identity(cfg)):
        raise ValueError('Preflight identity mismatch')
    if (out / 'completed.json').exists():
        raise FileExistsError('Training already completed')
    if (out / 'latest').exists() != resume:
        raise ValueError('Existing checkpoint requires explicit resume; fresh run must not overwrite')
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(min(.80, (pre['required_gpu_free_mib']-1024)/total_mib), 0)
    peer = peer_module()
    original_encode = peer.encode
    pending_rng = None
    if resume:
        cp = Path((out / 'latest').read_text().strip())
        receipt = json.loads((cp/'receipt.json').read_text())
        assert receipt['identity'] == pre['identity']
        for key, filename in [('adapter_sha256','adapter/adapter_model.safetensors'),('adapter_config_sha256','adapter/adapter_config.json'),('state_sha256','state.pt'),('rng_sha256','rng.pt')]:
            assert receipt[key] == sha(cp/filename)
        pending_rng = torch.load(cp / 'rng.pt', map_location='cpu', weights_only=False)
        records = [json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
        keep = [r for r in records if r['step'] <= receipt['step']]
        assert [r['step'] for r in keep] == list(range(1, receipt['step']+1))
        if len(keep) != len(records):
            (out/f'train.interrupted-{time.time_ns()}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in keep))

    def encode(*args, **kwargs):
        nonlocal pending_rng
        if pending_rng is not None:
            random.setstate(pending_rng['python']); np.random.set_state(pending_rng['numpy'])
            torch.set_rng_state(pending_rng['torch']); torch.cuda.set_rng_state_all(pending_rng['cuda'])
            pending_rng = None
        return original_encode(*args, **kwargs)

    def logprint(*args, **kwargs):
        builtins.print(*args, **kwargs)
        if args and isinstance(args[0], str) and args[0].startswith('{'):
            record = json.loads(args[0])
            if 'step' in record and 'loss' in record:
                with (out/'train.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                status('training', total_steps=cfg['expected_optimizer_steps'], **record)

    peer.encode = encode; peer.save_checkpoint = checkpoint_with_rng; peer.print = logprint
    sys.argv = [str(ROOT / cfg['peer_source'])]
    for key in ('manifest','model_path','output_dir','frame_cache','epochs','lr','seed','max_frames',
                'max_pixels','max_target_tokens','lora_r','lora_alpha','save_every','max_steps'):
        value = ROOT / cfg[key] if key in ('manifest','model_path','output_dir','frame_cache') else cfg[key]
        sys.argv.extend(['--' + key.replace('_','-'), str(value)])
    if resume:
        sys.argv.append('--resume')
    status('loading_for_training', total_steps=cfg['expected_optimizer_steps'])
    peer.main()
    if (out/'completed.json').exists():
        completion = json.loads((out/'completed.json').read_text())
        assert completion['global_step'] == cfg['expected_optimizer_steps']
        records = [json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
        assert [r['step'] for r in records] == list(range(1, cfg['expected_optimizer_steps']+1))
        rows = peer.load_rows(ROOT/cfg['manifest'])
        for epoch in range(1,cfg['epochs']+1):
            order = list(range(len(rows))); random.Random(cfg['seed']+epoch).shuffle(order)
            actual = [r for r in records if r['epoch']==epoch]
            assert [(r['id'],r['task']) for r in actual] == [(rows[i]['id'],rows[i]['task']) for i in order]
        save(out/'coverage.json', dict(passed=True, all_samples_exactly_twice=True,
                                      identity=pre['identity'], optimizer_steps=len(records)))
        status('completed', step=completion['global_step'], total_steps=cfg['expected_optimizer_steps'])
    else:
        status('stopped_resumable', checkpoint=(out/'latest').read_text().strip())


def supervise():
    cfg, out = check()
    logs = ROOT / 'logs/training' / cfg['experiment']
    logs.mkdir(parents=True, exist_ok=True)
    lock = (out/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from evaluate_peer_lora_epoch2 import check as check_evaluation
    check_evaluation()
    for stage in ('preflight','train'):
        if stage=='train' and (out/'completed.json').exists():
            if not (out/'coverage.json').exists():
                raise ValueError('Training completed but final coverage verification is missing')
            continue
        if stage == 'preflight' and (out/'preflight.json').exists():
            continue
        required = 13824 if stage == 'preflight' else json.loads((out/'preflight.json').read_text())['required_gpu_free_mib']
        wait_resources(required)
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), stage]
        if stage == 'train' and (out/'latest').exists():
            command.append('--resume')
        with (logs / (stage+'.log')).open('a') as f:
            result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            status('failed', failed_stage=stage, returncode=result.returncode, log=str(logs/(stage+'.log')))
            raise RuntimeError(f'{stage} failed; see {logs/(stage+".log")}')
    if not (out/'coverage.json').exists():
        print('Training stopped before completion; evaluation is not started.', flush=True)
        return
    command = [sys.executable, '-B', '-u', str(ROOT/'tools/search_peer_lora_checkpoints.py'), 'pipeline']
    with (logs/'evaluation-selected.log').open('a') as f:
        result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Selected SFT evaluation failed; see {logs/"evaluation-selected.log"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','check','preflight','train','supervise'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        if args.stage == 'train':
            train(args.resume)
        else:
            globals()[args.stage]()
    except Exception as exc:
        if args.stage not in ('check','prepare'):
            status('failed', failed_stage=args.stage, error=str(exc))
        traceback.print_exc()
        sys.exit(1)
