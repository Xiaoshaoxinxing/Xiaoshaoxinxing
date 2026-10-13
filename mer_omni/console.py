"""Readable Chinese terminal cards; plain UTF-8 also works through tee."""
import logging
import shutil
import unicodedata
import warnings


def display_width(text):
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
               for char in text)


def _wrap(text, width):
    line, used = '', 0
    for char in text:
        size = display_width(char)
        if used + size > width:
            yield line
            line, used = '', 0
        line += char
        used += size
    yield line


def panel(title, lines, *, width=None):
    width = width or min(88, max(24, shutil.get_terminal_size((80, 24)).columns - 2))
    inner = width - 4
    title = next(_wrap(title, width - 6))
    heading = '╭─ ' + title + ' '
    result = [heading + '─' * (width - display_width(heading) - 1) + '╮']
    for line in lines:
        for part in _wrap(str(line), inner):
            result.append('│ ' + part + ' ' * (inner - display_width(part)) + ' │')
    return '\n'.join([*result, '╰' + '─' * (width - 2) + '╯'])


def duration(seconds):
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f'{days}天 {hours:02d}小时 {minutes:02d}分'
    if hours:
        return f'{hours}小时 {minutes:02d}分 {seconds:02d}秒'
    return f'{minutes}分 {seconds:02d}秒'


def training_progress(log):
    step, stop = log['step'], log['stop_step']
    fraction = min(1.0, max(0.0, step / stop))
    filled = int(fraction * 24)
    bar = '█' * filled + '░' * (24 - filled)
    lines = [
        f'轮次 {log["epoch"]}/{log["active_epochs"]}    更新步 {step:,} / {stop:,}',
        f'进度  {bar}  {100*fraction:6.2f}%',
        '',
        f'总损失  {log["loss"]:.4f}       学习率  {log["lr"]:.3e}',
        ('训练目标  仅主路径语言 CE' if log.get('loss_mode') == 'ce_only' else
         f'主任务  {log["nll"]:.4f}    单模态  {log["unimodal_nll"]:.4f}    双模态  {log["auxiliary_nll"]:.4f}'),
        '',
        f'单步耗时  {log["step_seconds"]:.1f} 秒    已用时间  {duration(log["elapsed_seconds"])}',
        f'预计剩余  {duration(log["eta_seconds"])}',
    ]
    if 'epoch_samples_seen' in log:
        lines.insert(2, f'本轮样本  {log["epoch_samples_seen"]:,} / {log["epoch_samples_total"]:,}'
                     f'    本步 {log["samples_this_step"]} 条')
    if 'gpu_allocated_gib' in log:
        lines += ['', f'显存（GiB） 已用 {log["gpu_allocated_gib"]:.2f}  '
                  f'预留 {log["gpu_reserved_gib"]:.2f}  峰值 {log["gpu_peak_gib"]:.2f}']
    return panel('训练进度', lines)


class _MediaNoticeOnce(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = False

    def filter(self, record):
        if record.getMessage() != 'Unused or unrecognized kwargs: return_tensors.':
            return True
        if self.seen:
            return False
        self.seen = True
        return True


_notices_configured = False


def configure_console_notices():
    """Show two known repetitive compatibility notices once; keep other warnings."""
    global _notices_configured
    if _notices_configured:
        return
    _notices_configured = True
    logging.getLogger('transformers.image_utils').addFilter(_MediaNoticeOnce())
    original = warnings.showwarning
    seen = set()

    def showwarning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if (issubclass(category, FutureWarning) and
                text.startswith('`torch.cpu.amp.autocast(args...)` is deprecated.')):
            if text in seen:
                return
            seen.add(text)
        original(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = showwarning
