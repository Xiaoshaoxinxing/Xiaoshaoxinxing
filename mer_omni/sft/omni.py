"""Native Qwen loading and answer supervision shared by SFT experiments."""
import hashlib
import json
import math
from pathlib import Path
import random
import torch
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
ROOT = Path(__file__).resolve().parents[2]


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def select_rows(cfg):
    manifest = resolve(cfg['train_manifest'])
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    indices = random.Random(cfg['sample_seed']).sample(range(len(rows)), cfg['sample_count'])
    selected = []
    for index in indices:
        row = dict(rows[index])
        if row['split'] != 'train' or row['dataset'] != 'MER-Caption+':
            raise ValueError('Only MER-Caption+ training examples are allowed')
        if not row['openset'] or not all(isinstance(x, str) and x.strip() for x in row['openset']):
            raise ValueError('Invalid fixed labels')
        row.update(manifest_index=index, question=cfg['question'],
                   answer="The character's emotional state is " + ', '.join(row['openset']) + '.')
        for name in ('audio', 'video'):
            p = Path(row[name])
            row[name] = str(p if p.is_absolute() else (manifest.parent / p).resolve())
            if not Path(row[name]).is_file():
                raise FileNotFoundError(row[name])
        selected.append(row)
    if len({r['id'] for r in selected}) != cfg['sample_count']:
        raise ValueError('Duplicate sample IDs')
    return selected


def assistant_labels(full_ids, prompt_ids, eos_id):
    start = prompt_ids.shape[1]
    if not torch.equal(full_ids[:, :start], prompt_ids):
        raise ValueError('Native full conversation does not have the exact inference prompt prefix')
    ends = (full_ids[0, start:] == eos_id).nonzero().flatten()
    if len(ends) != 1 or int(ends[0]) == 0:
        raise ValueError('Expected one nonempty assistant answer and one im_end')
    stop = start + int(ends[0]) + 1
    labels = torch.full_like(full_ids, -100)
    labels[:, start:stop] = full_ids[:, start:stop]
    return labels, start, stop


def fingerprints(model):
    hashes = {k: hashlib.sha256() for k in ('audio_encoder', 'video_encoder', 'frozen_language', 'lora')}
    counts = {k: 0 for k in hashes}
    for name, p in model.named_parameters():
        group = ('lora' if p.requires_grad else 'audio_encoder' if name.startswith('audio_tower.') else
                 'video_encoder' if name.startswith('visual.') else 'frozen_language')
        hashes[group].update(name.encode())
        hashes[group].update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        counts[group] += p.numel()
    return {k: dict(sha256=h.hexdigest(), parameters=counts[k]) for k, h in hashes.items()}


def load_model(cfg, device):
    source = resolve(cfg['model_source'])
    config = Qwen2_5OmniThinkerConfig.from_pretrained(source, local_files_only=True)
    # Same local torch2.4/transformers4.52 compatibility adjustment as encoders/omni.py.
    if Version(torch.__version__.split('+')[0]) < Version('2.5'):
        config.text_config.base_model_tp_plan = {}
    model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        source, config=config, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation=cfg['train']['attn_implementation'], low_cpu_mem_usage=True,
        use_safetensors=True, output_loading_info=True)
    unexpected = [k for k in info.get('unexpected_keys', []) if not k.startswith(('talker.', 'token2wav.'))]
    if unexpected or any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Incomplete native Thinker load: ' + str(info))
    model.requires_grad_(False)
    lora = cfg['lora']
    model.model = get_peft_model(model.model, LoraConfig(
        r=lora['r'], lora_alpha=lora['alpha'], lora_dropout=lora['dropout'],
        target_modules=lora['target_modules'], bias='none'))
    for name, p in model.named_parameters():
        if p.requires_grad:
            if '.lora_' not in name or not name.startswith('model.'):
                raise ValueError('Unexpected trainable parameter: ' + name)
            p.data = p.data.float()
    if cfg['train']['gradient_checkpointing']:
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.model.config.use_cache = False
    return model.to(device), info


def move(batch, device):
    return {k: v.to(device=device, dtype=torch.bfloat16 if k == 'input_features' else v.dtype)
            for k, v in batch.items()}


def forward(model, batch):
    # Reset per-conversation RoPE state, particularly after autoregressive generation.
    model.rope_deltas = None
    return model(**batch, use_audio_in_video=False, use_cache=False)


def gradient_stats(model):
    groups = {k: dict(tensors=0, nonzero_tensors=0, norm_squared=0.0, absent=0) for k in ('lora_A', 'lora_B')}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise ValueError('Frozen parameter received gradients: ' + name)
            continue
        group = groups['lora_A' if '.lora_A.' in name else 'lora_B']
        group['tensors'] += 1
        if p.grad is None:
            group['absent'] += 1
            continue
        if not torch.isfinite(p.grad).all():
            raise ValueError('Nonfinite gradient: ' + name)
        norm = float(p.grad.float().norm())
        group['nonzero_tensors'] += int(norm > 0)
        group['norm_squared'] += norm**2
    for group in groups.values():
        group['norm'] = math.sqrt(group.pop('norm_squared'))
    if any(x['absent'] for x in groups.values()) or groups['lora_B']['norm'] == 0:
        raise ValueError('Disconnected adapter gradients')
    return groups
