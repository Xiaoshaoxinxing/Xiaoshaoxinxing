"""Single-device AdamW training, token-correct accumulation, resumable adapters."""
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .protocol import sha256, config_without_data_locations
from .model import SequenceTooLong
from .objectives import combine_losses
from .console import panel, training_progress


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch, device):
    return dict(batch, omni_inputs={k: v.to(device) for k, v in batch['omni_inputs'].items()})


def build_optimizer(model, settings):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if parameter.ndim < 2 or any(x in name for x in ('bias', 'ln', 'bn')) else decay
        target.append(parameter)
    return torch.optim.AdamW([
        dict(params=decay, weight_decay=settings['weight_decay']),
        dict(params=no_decay, weight_decay=0.0)], lr=settings['lr'], betas=(0.9, 0.999))


def training_row(dataset, index, model, *, allow_replacement=True):
    # AffectGPT's combined-length failure retries up to ten samples. Sampling
    # uses the saved Python RNG, so resume reproduces these replacement choices.
    for _ in range(10):
        row = dataset[index]
        try:
            model.validate_training_text(row['question'], row['answer'])
            return row
        except SequenceTooLong:
            if not allow_replacement:
                raise SequenceTooLong(f'{row["id"]}: full-dataset epochs forbid sample replacement; '
                                      'resolve the text length limit before continuing') from None
            print(f"【样本替换】{row['id']} 的文本超过长度限制，正在重新抽样。", flush=True)
            index = random.randint(0, len(dataset) - 1)
    raise SequenceTooLong('Failed to find a training sample within the token limit after 10 attempts')


class SampleStream:
    """A shuffled, infinite sequence with an explicit restorable cursor."""
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError('Cannot sample an empty dataset')
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.position = [], 0

    def take(self, count):
        result = []
        while len(result) < count:
            if self.position == len(self.order):
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
                self.position = 0
            n = min(count - len(result), len(self.order) - self.position)
            result.extend(self.order[self.position:self.position+n])
            self.position += n
        return result

    def state_dict(self):
        return dict(size=self.size, order=self.order, position=self.position, generator=self.generator.get_state())

    def load_state_dict(self, state):
        if state['size'] != self.size:
            raise ValueError('Training stream dataset size changed')
        self.order, self.position = state['order'], state['position']
        self.generator.set_state(state['generator'])


def epoch_settings(settings, dataset_size):
    """Resolve a data epoch without changing the config stored in checkpoints."""
    mode = settings.get('epoch_mode', 'fixed_steps')
    if mode not in ('fixed_steps', 'full_dataset'):
        raise ValueError('Unknown training epoch_mode')
    if dataset_size < 1:
        raise ValueError('Cannot train on an empty dataset')
    if mode == 'full_dataset':
        effective_batch = settings['batch_size'] * settings['gradient_accumulation']
        return dict(settings, steps_per_epoch=math.ceil(dataset_size / effective_batch))
    return dict(settings)


def step_sample_counts(settings, dataset_size, step):
    """Keep the last partial update inside this epoch; never wrap to the next."""
    batch = settings['batch_size']
    count = batch * settings['gradient_accumulation']
    if settings.get('epoch_mode') == 'full_dataset':
        count = min(count, dataset_size - (step % settings['steps_per_epoch']) * count)
    return [min(batch, count - offset) for offset in range(0, count, batch)]


def learning_rate(settings, step):
    if step < settings['warmup_steps']:
        return settings['warmup_lr'] + (settings['lr'] - settings['warmup_lr']) * step / max(settings['warmup_steps'], 1)
    total = settings['epochs'] * settings['steps_per_epoch']
    return settings['min_lr'] + (settings['lr'] - settings['min_lr']) * .5 * (1 + math.cos(math.pi * step / total))


def adapter_state(model):
    names = {name for name, param in model.named_parameters() if param.requires_grad}
    names.update(name for name, _ in model.named_buffers() if name.startswith('frontend.'))
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def load_adapter(model, state):
    expected = set(adapter_state(model))
    if set(state) != expected:
        raise ValueError(f'Adapter state mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}')
    model.load_state_dict(state, strict=False)


def save_checkpoint(path, model, optimizer, stream, step, config, manifest_hash):
    state = np.random.get_state()
    rng = dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
               torch=torch.get_rng_state())
    if next(model.parameters()).device.type == 'cuda':
        rng['cuda'] = torch.cuda.get_rng_state_all()
    payload = dict(format='atsuko-mer-omni-v2', model=adapter_state(model), optimizer=optimizer.state_dict(),
                   stream=stream.state_dict(), step=step, config=config, manifest_sha256=manifest_hash,
                   rng=rng, pretrained=model.components.load_report, calibration=model.calibration_info)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'atsuko-mer-omni-v2':
        raise ValueError('Requires an aligned v2 checkpoint; v1 omits Atsuko branches and cannot be resumed')
    return checkpoint


def train(model, dataset, collator, config, output, device, *, resume=None, max_steps=None,
          log_every=1, verbose_batches=False):
    settings = epoch_settings(config['train'], len(dataset))
    full_dataset = settings.get('epoch_mode') == 'full_dataset'
    loss_mode = config['mcm'].get('loss_mode', 'full')
    if log_every < 1:
        raise ValueError('log_every must be positive')
    if settings['num_workers'] != 0:
        raise ValueError('Exact-resume runner uses num_workers=0; parallel decoding is not implemented')
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError('Choose an empty run directory or explicitly resume')
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(config['train_manifest'])
    optimizer = build_optimizer(model, settings)
    stream = SampleStream(len(dataset), settings['seed'])
    start = 0
    if resume is not None:
        checkpoint = read_checkpoint(resume)
        if (config_without_data_locations(checkpoint['config']) != config_without_data_locations(config)
                or checkpoint['manifest_sha256'] != manifest_hash):
            raise ValueError('Resume requires identical experiment settings and training manifest')
        load_adapter(model, checkpoint['model'])
        model.calibration_info = checkpoint['calibration']
        optimizer.load_state_dict(checkpoint['optimizer'])
        stream.load_state_dict(checkpoint['stream'])
        start = checkpoint['step']
        if full_dataset:
            expected_position = (start % settings['steps_per_epoch']) * settings['batch_size'] * settings['gradient_accumulation']
            if start and expected_position == 0:
                expected_position = len(dataset)
            if (stream.position != expected_position or
                    sorted(stream.order) != list(range(len(dataset)))):
                raise ValueError('Checkpoint stream does not match full-dataset epoch coverage')
        log_path = output / 'train.jsonl'
        if log_path.exists():
            lines = [line for line in log_path.read_text().splitlines() if line.strip()]
            if lines and json.loads(lines[-1])['step'] != start:
                raise ValueError('Run log extends beyond or does not match resume checkpoint; use a new output directory')
        rng = checkpoint['rng']
        random.setstate(rng['python'])
        state = rng['numpy']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(rng['torch'])
        if 'cuda' in rng:
            torch.cuda.set_rng_state_all(rng['cuda'])
    if int(model.frontend.calibration_samples) < 1:
        raise RuntimeError('Calibrate spectral bands on training data before training')
    total = settings['epochs'] * settings['steps_per_epoch']
    stop = total if max_steps is None else min(total, max_steps)
    if stop <= start:
        raise ValueError('No steps requested after the checkpoint')
    active_epochs = math.ceil(stop / settings['steps_per_epoch'])
    (output / 'effective_config.json').write_text(json.dumps(config, indent=2))
    (output / 'run.json').write_text(json.dumps(dict(
        dataset='MER-Caption+', samples=len(dataset), manifest_sha256=manifest_hash,
        effective_batch=settings['batch_size'] * settings['gradient_accumulation'],
        world_size=1, optimizer_steps=stop, full_schedule_steps=total,
        epoch_mode=settings.get('epoch_mode', 'fixed_steps'),
        steps_per_epoch=settings['steps_per_epoch'],
        final_step_samples=(len(dataset) - (settings['steps_per_epoch'] - 1) *
                            settings['batch_size'] * settings['gradient_accumulation']) if full_dataset else None,
        sample_replacement_allowed=not full_dataset,
        loss_mode=loss_mode,
        checkpoint_selection='final scheduled checkpoint; no benchmark-based selection',
        pretrained=model.components.load_report, calibration=model.calibration_info), indent=2))
    print(panel('训练任务 · Atsuko + Qwen2.5-Omni-3B', [
        f'训练数据  {len(dataset):,} 条    本次停止轮次  {active_epochs}',
        f'更新步数  {start:,} → {stop:,}    每轮 {settings["steps_per_epoch"]:,} 步',
        f'有效批量  {settings["batch_size"]} 条 × 累积 {settings["gradient_accumulation"]} 次 '
        f'= {settings["batch_size"] * settings["gradient_accumulation"]} 条/步',
        f'保存频率  每 {settings["steps_per_epoch"]:,} 步及本次结束时保存',
        ('轮次定义  完整遍历训练集，尾批按实际样本数更新，不替换样本' if full_dataset else
         '轮次定义  固定优化器步数，不代表完整遍历训练集'),
        f'配置预算  {settings["epochs"]} 轮 / {total:,} 步（本次以停止步数为准）',
        '训练损失  仅主路径语言 CE' if loss_mode == 'ce_only' else '训练损失  主任务及 Atsuko 辅助目标',
        f'输出目录  {output}',
    ]) + '\n', flush=True)
    print('【训练开始】正在处理首批样本，完成参数更新后显示进度。', flush=True)
    model.train()
    started = time.perf_counter()
    for step in range(start, stop):
        step_started = time.perf_counter()
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(settings, step)
        optimizer.zero_grad(set_to_none=True)
        sample_counts = step_sample_counts(settings, len(dataset), step)
        index_groups, groups = [], []
        for count in sample_counts:
            indices = stream.take(count)
            index_groups.append(indices)
            groups.append([training_row(dataset, index, model, allow_replacement=not full_dataset)
                           for index in indices])
        counts = [sum(len(model._ids(r['answer'] + '###')) + 1 for r in rows) for rows in groups]
        metrics = dict(nll=0.0, unimodal_nll=0.0, auxiliary_nll=0.0, router_kl=0.0,
                       distillation=0.0, modality_entropy=0.0, band_entropy=0.0)
        if loss_mode == 'ce_only':
            metrics = dict(nll=0.0)
        for micro_step, (rows, count) in enumerate(zip(groups, counts), 1):
            if verbose_batches:
                print(f'【批次 {micro_step}/{len(groups)} · 第 {step+1:,} 步】'
                      f'解码与预处理：{", ".join(row["id"] for row in rows)}；答案词元 {count}', flush=True)
            batch = move_batch(collator(rows), device)
            if verbose_batches:
                print(f'【批次 {micro_step}/{len(groups)} · 第 {step+1:,} 步】前向计算与反向传播', flush=True)
            result = model(**batch)
            if not torch.isfinite(result['loss']):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            token_weight = count / sum(counts)
            sample_weight = len(rows) / sum(len(group) for group in groups)
            loss = combine_losses(result, config['mcm'], token_weight=token_weight, sample_weight=sample_weight)
            loss.backward()
            for key in metrics:
                weight = token_weight if key == 'nll' else sample_weight
                metrics[key] += float(result[key].detach()) * weight
        metrics['loss'] = combine_losses(metrics, config['mcm'])
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError(f'Non-finite gradient at step {step}')
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        step_seconds = time.perf_counter() - step_started
        eta_seconds = elapsed / (step+1-start) * (stop-step-1)
        log = dict(step=step+1, lr=optimizer.param_groups[0]['lr'], loss_mode=loss_mode, **metrics,
                   epoch=step // settings['steps_per_epoch'] + 1, total_steps=total,
                   stop_step=stop, active_epochs=active_epochs,
                   step_seconds=step_seconds, elapsed_seconds=elapsed, eta_seconds=eta_seconds)
        if full_dataset:
            log.update(epoch_samples_seen=stream.position, epoch_samples_total=len(dataset),
                       samples_this_step=sum(sample_counts),
                       sample_indices=[index for indices in index_groups for index in indices],
                       sample_ids=[row['id'] for rows in groups for row in rows])
        if device.type == 'cuda':
            log.update(gpu_allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                       gpu_reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
                       gpu_peak_gib=torch.cuda.max_memory_allocated(device) / 2**30)
        with (output / 'train.jsonl').open('a') as handle:
            handle.write(json.dumps(log) + '\n')
        if step == start or (step+1) % log_every == 0 or step+1 == stop:
            print(training_progress(log) + '\n', flush=True)
        if (step+1) % settings['steps_per_epoch'] == 0 or step+1 == stop:
            checkpoint_path = output / f'checkpoint_step_{step+1:07d}.pt'
            print(f'【保存检查点】正在写入 {checkpoint_path}', flush=True)
            save_checkpoint(checkpoint_path, model, optimizer, stream, step+1, config, manifest_hash)
            print(f'【保存完成】第 {step+1:,} 步，文件大小 {checkpoint_path.stat().st_size / 2**30:.2f} GiB。', flush=True)
            if full_dataset:
                seen = stream.order[:stream.position]
                complete = len(seen) == len(dataset) and len(set(seen)) == len(dataset)
                if (step+1) % settings['steps_per_epoch'] == 0 and not complete:
                    raise RuntimeError('Epoch ended without full training-set coverage')
                coverage = dict(epoch=log['epoch'], complete=complete, optimizer_step=step+1,
                                expected_samples=len(dataset), seen_samples=len(seen),
                                unique_samples=len(set(seen)), sample_indices=seen,
                                manifest_sha256=manifest_hash, checkpoint=checkpoint_path.name,
                                sample_replacement_allowed=False)
                (output / f'coverage_epoch_{log["epoch"]:04d}.json').write_text(json.dumps(coverage, indent=2))
    return output / f'checkpoint_step_{stop:07d}.pt'
