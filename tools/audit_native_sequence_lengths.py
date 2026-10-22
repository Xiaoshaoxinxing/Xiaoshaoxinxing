#!/usr/bin/env python3
"""Read-only length audit for choosing fixed Omni feature lengths on training data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer
from mer_omni.data import read_manifest, verify_manifest_receipt
from mer_omni.protocol import sha256


def video_length(row, frame_limits):
    """Verify all frames, using the same ffprobe check as decode_video's fallback."""
    path = row['video']
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_frames,nb_read_frames,duration:format=duration',
             '-of', 'json', str(path)], capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f'Video frame verification failed: {path}: {exc}') from exc
    if probe.returncode or probe.stderr.strip():
        raise ValueError(f'Video frame verification failed: {path}: '
                         f'{probe.stderr.decode(errors="replace")[-500:]}')
    try:
        payload = json.loads(probe.stdout)
        stream = payload['streams'][0]
        decoded = int(stream['nb_read_frames'])
        native = stream.get('nb_frames')
        container = None if native in (None, 'N/A') else int(native)
        if decoded < 1 or (container is not None and container < 0):
            raise ValueError('Invalid frame count')
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError(f'Cannot determine video frame counts: {path}') from exc
    duration = None
    duration_source = None
    for source, metadata in [('stream', stream), ('format', payload.get('format', {}))]:
        try:
            value = float(metadata.get('duration'))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            duration, duration_source = value, source
            break
    if duration is None:
        raise ValueError(f'Cannot determine video duration: {path}')
    return dict(video_nb_frames=container, video_decoded_frames=decoded,
                video_frame_count_mismatch=None if container is None else container != decoded,
                video_duration_seconds=duration, video_duration_source=duration_source,
                video_positions_by_max_frames={str(limit): min(limit, decoded)//2
                                               for limit in frame_limits})


def distribution(values, *, integer=True):
    values = np.asarray(values)
    cast = int if integer else float
    return dict(minimum=cast(values.min()), maximum=cast(values.max()),
                quantiles=dict(zip(['p50', 'p90', 'p95', 'p99'],
                                   np.quantile(values, [.5, .9, .95, .99]).tolist())))


def overflow_counts(values, candidates):
    values = np.asarray(values)
    return {str(length): int((values > length).sum()) for length in candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/mercaptionplus.train.jsonl')
    parser.add_argument('--tokenizer', default='models/Qwen2.5-Omni-3B')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-frames', type=int, default=16,
                        help='Video sampling cap (>=2); also report caps 8 and 16')
    parser.add_argument('--limit', type=int,
                        help='Audit only the first N training rows (default: all)')
    options = parser.parse_args()
    if options.max_frames < 2:
        parser.error('--max-frames must be at least 2')
    if options.limit is not None and options.limit < 1:
        parser.error('--limit must be positive')
    output = ROOT/Path(options.output).expanduser()
    if output.exists():
        raise FileExistsError(output)
    manifest = ROOT/Path(options.manifest).expanduser()
    verify_manifest_receipt(manifest)
    rows = read_manifest(manifest)
    if not rows or any(r['split'] != 'train' or r['dataset'] != 'MER-Caption+' for r in rows):
        raise ValueError('Choose sequence lengths only from MER-Caption+ training inputs')
    manifest_samples = len(rows)
    if options.limit is not None:
        rows = rows[:options.limit]
    tokenizer = AutoTokenizer.from_pretrained(ROOT/Path(options.tokenizer).expanduser(), local_files_only=True)
    text_lengths = []
    for start in range(0, len(rows), 512):
        texts = [r['text'] or ' ' for r in rows[start:start+512]]
        text_lengths.extend(map(len, tokenizer(texts, add_special_tokens=True)['input_ids']))
    def audio_length(row):
        info = sf.info(row['audio'])
        mel = math.ceil(info.frames/info.samplerate*16000/160)
        return dict(id=row['id'], duration=info.frames/info.samplerate,
                    estimated_audio_positions=((mel+1)//2)//2)
    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(pool.map(audio_length, rows))
    for record, count in zip(records, text_lengths):
        record['text_positions'] = count
    frame_limits = sorted({8, 16, options.max_frames})
    with ThreadPoolExecutor(max_workers=16) as pool:
        videos = pool.map(lambda row: video_length(row, frame_limits), rows)
        for record, video in zip(records, videos):
            record.update(video)
            record['video_positions'] = video['video_positions_by_max_frames'][str(options.max_frames)]
    summary = {}
    for name, key in [('text','text_positions'), ('audio','estimated_audio_positions')]:
        values = np.array([r[key] for r in records])
        summary[name] = dict(minimum=int(values.min()), maximum=int(values.max()),
            quantiles=dict(zip(['p50','p90','p95','p99'], np.quantile(values,[.5,.9,.95,.99]).tolist())),
            over_50=int((values>50).sum()), over_50_fraction=float((values>50).mean()),
            fraction_positions_discarded_at_50=float(np.maximum(values-50,0).sum()/values.sum()))
    summary['video'] = distribution([r['video_positions'] for r in records])
    summary['video_by_max_frames'] = {
        str(limit): distribution([r['video_positions_by_max_frames'][str(limit)] for r in records])
        for limit in frame_limits}
    summary['video_duration_seconds'] = distribution(
        [r['video_duration_seconds'] for r in records], integer=False)
    overflows = dict(text=overflow_counts(text_lengths, [96]),
                     audio=overflow_counts([r['estimated_audio_positions'] for r in records],
                                           [400, 512, 704]),
                     video_by_max_frames={str(limit): overflow_counts(
                         [r['video_positions_by_max_frames'][str(limit)] for r in records], [50])
                         for limit in frame_limits})
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(samples=len(rows), manifest_sha256=sha256(manifest), summary=summary, records=records,
        notes=['Text lengths use the local original English-input tokenizer.',
               'Audio lengths are estimates from original WAV headers and native Omni downsampling; verify against encoder outputs.',
               'No media, labels, transcripts, splits or model weights are changed.'])
    report.update(manifest_samples=manifest_samples, limit=options.limit,
                  sampling='first_n' if options.limit is not None else 'all',
                  overflow_counts=overflows,
                  video_audit=dict(method='ffprobe_count_frames', workers=16, temporal_patch_size=2,
                                   max_frames=options.max_frames, compared_max_frames=frame_limits,
                                   frame_count_mismatches=sum(r['video_frame_count_mismatch'] is True
                                                              for r in records),
                                   missing_container_frame_counts=sum(r['video_nb_frames'] is None
                                                                      for r in records),
                                   single_frame_videos=sum(r['video_decoded_frames'] == 1 for r in records)))
    report['notes'].extend([
        'Video counts use verified decoded frames, even when nb_frames differs; unavailable nb_frames/mismatch are null.',
        'Video positions use min(max_frames, decoded_frames)//2; spatial patches are not counted.',
        'For one-frame clips this requested formula gives 0; decode_video duplicates the frame and produces 1 temporal position.',
        'Overflow counts use positions > L, not >= L. --limit selects the first N rows without shuffling.'])
    with output.open('x') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(summary, overflow_counts=overflows), indent=2), flush=True)


if __name__ == '__main__':
    main()
