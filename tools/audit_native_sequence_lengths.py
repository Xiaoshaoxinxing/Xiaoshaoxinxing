#!/usr/bin/env python3
"""Read-only length audit for choosing fixed Omni feature lengths on training data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer
from mer_omni.data import read_manifest, verify_manifest_receipt
from mer_omni.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifests/mercaptionplus.train.jsonl')
    parser.add_argument('--tokenizer', default='models/Qwen2.5-Omni-3B')
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    output = ROOT/Path(options.output).expanduser()
    if output.exists():
        raise FileExistsError(output)
    manifest = ROOT/Path(options.manifest).expanduser()
    verify_manifest_receipt(manifest)
    rows = read_manifest(manifest)
    if not rows or any(r['split'] != 'train' or r['dataset'] != 'MER-Caption+' for r in rows):
        raise ValueError('Choose sequence lengths only from MER-Caption+ training inputs')
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
    summary = {}
    for name, key in [('text','text_positions'), ('audio','estimated_audio_positions')]:
        values = np.array([r[key] for r in records])
        summary[name] = dict(minimum=int(values.min()), maximum=int(values.max()),
            quantiles=dict(zip(['p50','p90','p95','p99'], np.quantile(values,[.5,.9,.95,.99]).tolist())),
            over_50=int((values>50).sum()), over_50_fraction=float((values>50).mean()),
            fraction_positions_discarded_at_50=float(np.maximum(values-50,0).sum()/values.sum()))
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(samples=len(rows), manifest_sha256=sha256(manifest), summary=summary, records=records,
        notes=['Text lengths use the local original English-input tokenizer.',
               'Audio lengths are estimates from original WAV headers and native Omni downsampling; verify against encoder outputs.',
               'No media, labels, transcripts, splits or model weights are changed.'])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
