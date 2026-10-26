"""Reusable Atsuko input, optimizer, calibration and checkpoint components.

No historical experiment is imported. A new SFT recipe must supply its own
configuration, training-only calibration and audit before running these helpers.
"""
import fcntl
from functools import lru_cache
import json
import math
from pathlib import Path
import time
import torch
from . import native_inputs as a
from . import native_inputs as native_evaluation
from mer_omni.native_atsuko_core import CoreNativeAtsuko, insert_affect_slots
from mer_omni.core_protocol import train_vocabulary, auxiliary_target, calibration_signature, apply_calibration
from mer_omni.calibration import set_calibration
from mer_omni.training import build_optimizer as original_optimizer, learning_rate
ROOT = Path(__file__).resolve().parents[2]


def identity(cfg, plan):
    result = a.identity(cfg, plan)
    sources = [Path(__file__), ROOT / 'mer_omni/native_atsuko.py', ROOT / 'mer_omni/native_atsuko_norm.py', ROOT / 'mer_omni/native_atsuko_core.py',
               ROOT / 'mer_omni/core_protocol.py', ROOT / 'mer_omni/sft/native_inputs.py',
               ROOT / 'mer_omni/model.py',
               ROOT / 'mer_omni/calibration.py', ROOT / 'mer_omni/encoders/omni.py',
               ROOT / cfg['bridge']['vocab_file']]
    sources += sorted((ROOT / 'mer_omni/layers').glob('*.py'))
    result['sources'].update({str(p.relative_to(ROOT)): a.native.sha(p) for p in sources})
    return result


def slot_id(proc, cfg):
    token = cfg['bridge']['placeholder']
    ids = proc.tokenizer(token, add_special_tokens=False)['input_ids']
    if token not in proc.tokenizer.all_special_tokens or len(ids) != 1 or ids[0] in (proc.tokenizer.unk_token_id, proc.tokenizer.pad_token_id,
            proc.tokenizer.convert_tokens_to_ids(proc.audio_token), proc.tokenizer.convert_tokens_to_ids(proc.video_token),
            proc.tokenizer.convert_tokens_to_ids('<|im_end|>')):
        raise ValueError('Fusion slot must be one existing non-media, non-EOS special token')
    return ids[0]


def make_item(row, qtype, cfg, proc, meta, *, compare_direct=False):
    original = a.make_item(row, qtype, cfg, proc, meta, compare_direct=compare_direct)
    token, end = slot_id(proc, cfg), proc.tokenizer.convert_tokens_to_ids('<|im_end|>')
    result = dict(metadata=dict(original['metadata']))
    subtitle = proc.tokenizer([row['text'] or ' '], padding=False, truncation=False, return_tensors='pt')['input_ids']
    for key in ('train', 'prompt'):
        result[key], pos = insert_affect_slots(original[key], token, end)
        result[key]['bypass_subtitle_ids'] = subtitle
        if not torch.equal(torch.cat((result[key]['input_ids'][:, :pos], result[key]['input_ids'][:, pos+4:]), 1), original[key]['input_ids']):
            raise ValueError('Fusion insertion changed native A tokens')
    result['train']['aux_target'] = auxiliary_target(row['openset'], vocabulary(str(a.path(cfg['bridge']['vocab_file']))))
    labels, start, stop = a.native.assistant_labels(result['train']['input_ids'], result['prompt']['input_ids'], end)
    if not torch.equal(labels, result['train']['labels']):
        raise ValueError('B3 answer masking differs from native assistant supervision')
    if not torch.equal(labels[labels != -100], original['train']['labels'][original['train']['labels'] != -100]):
        raise ValueError('B3 changed the target answer')
    if result['train']['input_ids'].shape[1] > cfg['train']['max_length']:
        raise ValueError('B3 context overflow; no truncation')
    result['metadata'].update(prompt_tokens=start, total_tokens=result['train']['input_ids'].shape[1],
                              fusion_tokens=4, insertion_position=pos)
    return result


@lru_cache(maxsize=4)
def vocabulary(path):
    return json.loads(Path(path).read_text())['vocab']


def training_identity(cfg, plan):
    value = identity(cfg, plan)
    value['calibration_sha256'] = a.native.sha(a.path(cfg['bridge']['calibration_file']))
    return value


def build_optimizer(model, settings):
    optimizer = original_optimizer(model, settings)
    slot = model.fusion_in.slot_embedding
    target = next(g for g in optimizer.param_groups if g['name'] == 'frontend_no_decay')
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params'] if p is not slot]
    target['params'].append(slot)
    return optimizer


def load_model(cfg, proc, device, *, require_calibration=True):
    thinker, info = a.native.load_model(cfg, device)
    model = CoreNativeAtsuko(thinker, cfg['bridge']['spectral'], slot_id(proc, cfg), cfg['bridge']['auxiliary'])
    signature = calibration_signature(model, cfg, a.native.sha(a.path(cfg['train_manifest'])), 31327)
    if require_calibration:
        artifact = json.loads(a.path(cfg['bridge']['calibration_file']).read_text())
        apply_calibration(model, artifact, signature)
    info['calibration_signature'] = signature
    info['initial_embedding_rms'] = model.initial_embedding_rms
    return model.to(device), info


def verify(cfg, directory):
    plan = json.loads((directory / 'plan.json').read_text())
    audit = json.loads((directory / 'audit.json').read_text())
    if not audit['passed'] or audit['identity'] != identity(cfg, plan):
        raise ValueError('B3 audit no longer matches config/source/plan')
    return plan


def state(model):
    values = a.trainable(model)
    values.update({n: b.detach().cpu().clone() for n,b in model.named_buffers() if n.startswith('frontend.')})
    return values


def save_checkpoint(path, model, optimizer, step, run_identity):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(format='native-omni-atsuko-core-v1', model=state(model), optimizer=optimizer.state_dict(),
                    step=step, identity=run_identity, rng=a.rng_state()), tmp)
    tmp.replace(path)
    a.save(path.with_suffix('.receipt.json'), dict(step=step, sha256=a.native.sha(path), identity=run_identity))


def restore_checkpoint(path, model, optimizer, run_identity):
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if receipt['identity'] != run_identity or receipt['sha256'] != a.native.sha(path):
        raise ValueError('B3 resume identity/hash mismatch')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if (cp['format'] != 'native-omni-atsuko-core-v1' or cp['identity'] != run_identity
            or type(cp.get('step')) is not int or cp['step'] != receipt.get('step')
            or cp['step'] < 0 or set(cp['model']) != set(state(model))):
        raise ValueError('Wrong B checkpoint payload')
    destinations = {n:p for n,p in model.named_parameters() if p.requires_grad}
    destinations.update({n:b for n,b in model.named_buffers() if n.startswith('frontend.')})
    with torch.no_grad():
        for name, value in cp['model'].items():
            dest = destinations[name]
            if dest.shape != value.shape or dest.dtype != value.dtype:
                raise ValueError('Checkpoint shape/dtype mismatch: ' + name)
            dest.copy_(value)
            if not torch.equal(dest.detach().cpu(), value):
                raise ValueError('Checkpoint restoration mismatch: ' + name)
    optimizer.load_state_dict(cp['optimizer'])
    a.restore_rng(cp['rng'])
    return cp['step']


def gradients(model):
    totals = dict(lora_A=0., lora_B=0., frontend=0., fusion_in=0.)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise ValueError('Frozen parameter received a gradient: ' + name)
            continue
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError('Disconnected or nonfinite gradient: ' + name)
        key = 'lora_A' if '.lora_A.' in name else 'lora_B' if '.lora_B.' in name else name.split('.')[0]
        totals[key] += float(p.grad.float().square().sum())
    if any(totals[k] <= 0 for k in ('lora_B', 'frontend', 'fusion_in')):
        raise ValueError('Missing gradient through added fusion branch or LoRA')
    return {k: math.sqrt(v) for k,v in totals.items()}


def update(model, optimizer, items, settings, step, check_gradients=False):
    factor = learning_rate(settings, step) / settings['lr']
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * factor
    optimizer.zero_grad(set_to_none=True)
    counts = [int((it['train']['labels'][:, 1:] != -100).sum()) for it in items]
    loss = 0.
    aux_values = {k:0. for k in ('total', 'fusion_bce', 'unimodal_bce', 'bimodal_bce', 'complementarity_kl')}
    coefficient = model.auxiliary_settings['weight']
    for item, count in zip(items, counts):
        batch = a.native.move(item['train'], next(model.parameters()).device)
        result = a.native.forward(model, batch)
        aux = result.auxiliary_losses
        objective = result.loss * (count / sum(counts)) + coefficient * aux['total'] / len(items)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite B3 objective')
        objective.backward()
        loss += float(result.loss.detach()) * count / sum(counts)
        for key in aux_values:
            aux_values[key] += float(aux[key].detach()) / len(items)
        del result, batch, aux, objective
    stats = gradients(model) if check_gradients else None
    norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                        settings['grad_clip'], error_if_nonfinite=True))
    optimizer.step()
    return dict(loss=loss, auxiliary=aux_values, total_loss=loss+coefficient*aux_values['total'],
                lr=learning_rate(settings, step), lr_frontend=settings['lr_frontend']*factor,
                supervised_tokens=sum(counts), gradient_norm_before_clip=norm,
                **({'gradient_norms': stats} if stats is not None else {}))


@torch.no_grad()
def calibrate(cfg, audit_dir, out, device):
    """Full training-set calibration only; no targets, optimizer or parameter updates."""
    verify(cfg, audit_dir)
    rows, proc, metadata = a.load_inputs(cfg)
    target = a.path(cfg['bridge']['calibration_file'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            raise FileExistsError('Refusing to overwrite B3 calibration')
        a.seed_all(cfg['seed'])
        model, info = load_model(cfg, proc, device, require_calibration=False)
        signature = info['calibration_signature']
        model.eval()
        progress = out / 'calibration-progress.json'
        energies = [torch.zeros(s.length, dtype=torch.float64) for s in model.frontend.spectra]
        processed, previous_seconds = 0, 0.
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved.get('signature') != signature:
                raise ValueError('Stale B3 calibration progress')
            processed = saved['processed']
            if type(processed) is not int or not 0 <= processed <= len(rows):
                raise ValueError('Invalid calibration coverage')
            restored = [torch.tensor(e, dtype=torch.float64) for e in saved['energy_sum']]
            if len(restored) != 3 or any(v.shape != e.shape or not torch.isfinite(v).all() or (v < 0).any()
                                         for e,v in zip(energies,restored)):
                raise ValueError('Invalid saved calibration energy')
            energies = restored
            previous_seconds = saved['elapsed_seconds']
        started = time.monotonic()
        for index in range(processed, len(rows)):
            row = {k:rows[index][k] for k in ('id','text','video','audio')}
            batch, _ = native_evaluation.prompt_only(row, cfg, proc, metadata[row['id']])
            batch['bypass_subtitle_ids'] = proc.tokenizer([row['text'] or ' '], padding=False,
                truncation=False, return_tensors='pt')['input_ids']
            batch = a.native.move(batch, device)
            sequences, _, _ = model.encode_sequences(batch)
            projected = model.frontend.project_sequences(sequences)
            increments = []
            for spectrum, values in zip(model.frontend.spectra, projected):
                coeffs = torch.einsum('kt,btd->bkd', spectrum.U.T.to(values), values)
                increments.append(coeffs.double().square().mean(-1).sum(0).cpu())
            if any(not torch.isfinite(e).all() for e in increments):
                raise ValueError('Nonfinite calibration spectrum: ' + row['id'])
            for total, value in zip(energies, increments):
                total.add_(value)
            processed = index + 1
            if processed == 1 or processed % 100 == 0 or processed == len(rows):
                a.save(progress, dict(signature=signature, processed=processed,
                    energy_sum=[e.tolist() for e in energies],
                    elapsed_seconds=previous_seconds+time.monotonic()-started))
                print(f'B3 calibration: {processed}/{len(rows)}', flush=True)
        energy = [(e/len(rows)).tolist() for e in energies]
        set_calibration(model.frontend, energy, len(rows))
        artifact = dict(format='native-atsuko-core-bands-v1', signature=signature, complete=True,
            processed=len(rows), energy=energy, band_edges=[s.band_edges.tolist() for s in model.frontend.spectra])
        apply_calibration(model, artifact, signature)
        a.save(target, artifact)
        a.save(out/'calibration.json', dict(complete=True, samples=len(rows), optimizer_steps=0,
            calibration=str(target), sha256=a.native.sha(target)))
