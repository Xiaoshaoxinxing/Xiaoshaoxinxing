"""Native multimodal inputs, fixed targets and resumable SFT state."""
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from transformers import Qwen2_5OmniProcessor
from . import omni as native
from mer_omni.data import read_manifest, verify_manifest_receipt
from mer_omni.media import NativeMediaCollator, decode_video, decode_audio
from mer_omni.protocol import DESCRIPTION_QUESTION, OV_QUESTION
from mer_omni.training import learning_rate
ROOT = Path(__file__).resolve().parents[2]


def path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def save(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def plan_for(rows, seed):
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    rng = random.Random(seed)
    plan = [dict(index=i, id=rows[i]['id'], qtype=rng.choice(('description', 'ovlabel'))) for i in order]
    return plan, rng.getstate()


def groups_for(plan, count=3):
    return [plan[i:i+count] for i in range(0, len(plan), count)]


def target(row, qtype):
    question = DESCRIPTION_QUESTION if qtype == 'description' else OV_QUESTION
    answer = row['description'] if qtype == 'description' else "The character's emotional state is " + ', '.join(row['openset']) + '.'
    return question, answer


def conversation(row, question, cfg):
    return [dict(role='system', content=[dict(type='text', text=cfg['system_prompt'])]),
            dict(role='user', content=[dict(type='video', video=row['video']),
                                      dict(type='audio', audio=row['audio']),
                                      dict(type='text', text='The original subtitle is: ' + row['text'] + '\n' + question)])]


def render(processor, messages, generation_prompt):
    value = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    # This local Omni processor returns a one-element list for a single conversation.
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise TypeError('Expected a single rendered native conversation')
    return value


def media_options(cfg):
    return dict(source=cfg['model_source'], max_frames=cfg['media']['max_frames'],
                min_pixels=cfg['media']['min_pixels'], max_pixels=cfg['media']['max_pixels'],
                max_text_tokens=1024, audio_sample_rate=cfg['media']['sample_rate'])


def load_inputs(cfg):
    verify_manifest_receipt(path(cfg['train_manifest']))
    rows = read_manifest(path(cfg['train_manifest']))
    if len(rows) != 31327 or any(r['dataset'] != 'MER-Caption+' or r['split'] != 'train' for r in rows):
        raise ValueError('Expected the complete 31327-row MER-Caption+ training manifest')
    processor = Qwen2_5OmniProcessor.from_pretrained(path(cfg['model_source']), local_files_only=True)
    metadata = json.loads(path(cfg['length_metadata']).read_text())
    if metadata['manifest_sha256'] != native.sha(path(cfg['train_manifest'])):
        raise ValueError('Existing video metadata does not belong to the current manifest')
    by_id = {r['id']: r for r in metadata['records']}
    if set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Incomplete video metadata')
    return rows, processor, by_id


def identity(cfg, plan):
    sources = ['mer_omni/sft/native_inputs.py', 'mer_omni/sft/omni.py',
               'mer_omni/media.py', 'mer_omni/data.py', 'mer_omni/training.py', 'mer_omni/protocol.py']
    import transformers, peft
    return dict(config=cfg, manifest=native.sha(path(cfg['train_manifest'])), plan=digest(plan),
                sources={s: native.sha(path(s)) for s in sources},
                versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__))


def make_item(row, qtype, cfg, processor, metadata, *, compare_direct=False):
    # Exactly the same decoder, resizing and feature extractor as the old frames16 run.
    media = NativeMediaCollator(media_options(cfg), processor=processor)([row])
    question, answer = target(row, qtype)
    messages = conversation(row, question, cfg)
    prompt_text = render(processor, messages, True)
    full_text = render(processor, messages + [dict(role='assistant', content=answer)], False)
    count = metadata['video_decoded_frames']
    nframes = max(2, min(count, cfg['media']['max_frames'])) // 2 * 2
    fps = count / metadata['video_duration_seconds']
    effective_fps = (nframes - 1) * fps / (count - 1)
    spg = processor.video_processor.temporal_patch_size / effective_fps
    mel = int(media['feature_attention_mask'].sum())
    a_len = (((mel - 1)//2 + 1)-2)//2 + 1
    v_len = int(media['video_grid_thw'].prod()) // processor.video_processor.merge_size**2
    def tokenize(text):
        expanded = processor.replace_multimodal_special_tokens(
            [text], iter([a_len]), iter([]), iter(media['video_grid_thw']),
            video_second_per_grid=iter([spg]), use_audio_in_video=False,
            position_id_per_seconds=25, seconds_per_chunk=2)
        return dict(processor.tokenizer(expanded, padding=False, truncation=False, return_tensors='pt'))
    prompt, full = tokenize(prompt_text), tokenize(full_text)
    labels, start, stop = native.assistant_labels(full['input_ids'], prompt['input_ids'],
                                                processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
    if processor.tokenizer.decode(labels[labels != -100][:-1], skip_special_tokens=False) != answer:
        raise ValueError('Supervision differs from the exact original answer')
    common = {k: v for k, v in media.items() if k not in ('input_ids', 'text_attention_mask')}
    common['video_second_per_grid'] = [spg]
    # move() expects tensors. HF accepts this tensor exactly as the native processor's .to() path.
    common['video_second_per_grid'] = torch.tensor([spg], dtype=torch.float32)
    prompt.update(common)
    full.update(common, labels=labels)
    length = int(full['input_ids'].shape[1])
    if length > cfg['train']['max_length']:
        raise ValueError(f"{row['id']}: {length} exceeds context; no truncation/replacement")
    for token, expected in ((processor.audio_token, a_len), (processor.video_token, v_len)):
        if int((prompt['input_ids'] == processor.tokenizer.convert_tokens_to_ids(token)).sum()) != expected:
            raise ValueError('Media placeholder count mismatch')
    direct_checks = {}
    if compare_direct:
        frames = decode_video(row['video'], cfg['media']['max_frames'], cfg['media']['max_pixels'])
        waveform = decode_audio(row['audio'], cfg['media']['sample_rate'])
        direct = dict(processor(text=full_text, videos=[frames], audio=[waveform], return_tensors='pt',
            videos_kwargs=dict(min_pixels=cfg['media']['min_pixels'], max_pixels=cfg['media']['max_pixels'],
                               fps=effective_fps, use_audio_in_video=False),
            audio_kwargs=dict(sampling_rate=cfg['media']['sample_rate'], truncation=False,
                              max_length=math.ceil(len(waveform)/160)*160),
            text_kwargs=dict(padding=False, truncation=False)))
        for k, v in direct.items():
            direct_checks[k] = bool(torch.equal(torch.as_tensor(v), torch.as_tensor(full[k])))
        if not all(direct_checks.values()):
            raise ValueError(f'Original media + native template differs from direct processor: {direct_checks}')
    return dict(train=full, prompt=prompt, metadata=dict(id=row['id'], qtype=qtype, prompt_tokens=start,
                supervised_tokens=stop-start, total_tokens=length, audio_tokens=a_len, video_tokens=v_len,
                video_grid=media['video_grid_thw'].tolist(), direct_processor_equal=direct_checks))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s['cuda'] is not None:
        torch.cuda.set_rng_state_all(s['cuda'])


def trainable(model):
    return {n: p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def save_checkpoint(destination, model, optimizer, step, run_identity):
    temporary = destination.with_suffix('.tmp')
    torch.save(dict(format='native-omni-epoch1-v1', model=trainable(model), optimizer=optimizer.state_dict(),
                    step=step, identity=run_identity, rng=rng_state()), temporary)
    temporary.replace(destination)
    save(destination.with_suffix('.receipt.json'), dict(step=step, sha256=native.sha(destination), identity=run_identity))


def restore_checkpoint(source, model, optimizer, run_identity):
    receipt = json.loads(source.with_suffix('.receipt.json').read_text())
    if receipt['identity'] != run_identity or receipt['sha256'] != native.sha(source):
        raise ValueError('Checkpoint identity/hash mismatch')
    cp = torch.load(source, map_location='cpu', weights_only=False)
    if cp['identity'] != run_identity or set(cp['model']) != set(trainable(model)):
        raise ValueError('Checkpoint trainable tensors/identity mismatch')
    model.load_state_dict(cp['model'], strict=False)
    optimizer.load_state_dict(cp['optimizer'])
    restore_rng(cp['rng'])
    return cp['step']


def update(model, optimizer, items, settings, step, *, check_gradients=False):
    lr = learning_rate(settings, step)
    for g in optimizer.param_groups:
        g['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    counts = [int((it['train']['labels'][:, 1:] != -100).sum()) for it in items]
    if min(counts) < 1:
        raise ValueError('Empty supervised target')
    loss_value = 0.
    for item, count in zip(items, counts):
        batch = native.move(item['train'], next(model.parameters()).device)
        result = native.forward(model, batch)
        if not torch.isfinite(result.loss):
            raise FloatingPointError('Nonfinite training loss')
        weight = count/sum(counts)
        (result.loss * weight).backward()
        loss_value += float(result.loss.detach()) * weight
        del result, batch
    grads = native.gradient_stats(model) if check_gradients else None
    params = [p for p in model.parameters() if p.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(params, settings['grad_clip'], error_if_nonfinite=True))
    optimizer.step()
    return dict(loss=loss_value, lr=lr, supervised_tokens=sum(counts), gradient_norm_before_clip=norm,
                **({'gradients':grads} if grads is not None else {}))


def prompt_only(row, cfg, proc, metadata):
    # Whitelist inputs before rendering: benchmark labels cannot reach the model.
    row = {k: row[k] for k in ('id', 'text', 'video', 'audio')}
    media = NativeMediaCollator(media_options(cfg), processor=proc)([row])
    count = metadata['video_decoded_frames']
    nframes = max(2, min(count, cfg['media']['max_frames'])) // 2 * 2
    if count < 2:
        raise ValueError('Native frame timing requires at least two source frames')
    fps = count / metadata['video_duration_seconds']
    effective_fps = (nframes - 1) * fps / (count - 1)
    spg = proc.video_processor.temporal_patch_size / effective_fps
    mel = int(media['feature_attention_mask'].sum())
    audio_tokens = (((mel - 1) // 2 + 1) - 2) // 2 + 1
    video_tokens = int(media['video_grid_thw'].prod()) // proc.video_processor.merge_size**2
    text = render(proc, conversation(row, OV_QUESTION, cfg), True)
    expanded = proc.replace_multimodal_special_tokens(
        [text], iter([audio_tokens]), iter([]), iter(media['video_grid_thw']),
        video_second_per_grid=iter([spg]), use_audio_in_video=False,
        position_id_per_seconds=25, seconds_per_chunk=2)
    result = dict(proc.tokenizer(expanded, padding=False, truncation=False, return_tensors='pt'))
    result.update({k: v for k, v in media.items() if k not in ('input_ids', 'text_attention_mask')})
    result['video_second_per_grid'] = torch.tensor([spg], dtype=torch.float32)
    for token, expected in ((proc.audio_token, audio_tokens), (proc.video_token, video_tokens)):
        if int((result['input_ids'] == proc.tokenizer.convert_tokens_to_ids(token)).sum()) != expected:
            raise ValueError('Native media token count mismatch')
    return result, dict(id=row['id'], prompt_tokens=result['input_ids'].shape[1],
                        audio_tokens=audio_tokens, video_tokens=video_tokens,
                        video_second_per_grid=spg)
