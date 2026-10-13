"""Strict local manifests; hybrid caption/OV-label instruction targets."""
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .media import NativeMediaCollator
from .protocol import BENCHMARKS, DESCRIPTION_QUESTION, OV_QUESTION, check_coverage, sha256
from .reference.affectgpt import string_to_list


def read_csv_map(path, column, *, allow_empty=False, required_ids=None):
    frame = pd.read_csv(path, keep_default_na=False)
    if not {'name', column} <= set(frame.columns):
        raise ValueError(f'Missing name/{column} columns: {path}')
    frame['name'] = frame['name'].astype(str)
    if required_ids is not None:
        # Official transcript tables can contain unused utterances (including
        # duplicated IEMOCAP FXX IDs). Validate only the selected label split.
        required_ids = set(required_ids)
        frame = frame[frame['name'].isin(required_ids)]
        missing = required_ids - set(frame['name'])
        if missing:
            raise ValueError(f'Missing required transcript IDs: {sorted(missing)[:5]}: {path}')
    if frame['name'].duplicated().any() or (frame['name'] == '').any():
        raise ValueError(f'Duplicate/empty IDs: {path}')
    result = dict(zip(frame['name'].astype(str), frame[column].astype(str)))
    if not allow_empty and any(not value.strip() for value in result.values()):
        raise ValueError(f'Empty {column}: {path}')
    return result


def build_train_rows(root):
    # Keep configured symlinks in stored paths instead of binding to their target machine.
    root = Path(root).expanduser().absolute()
    captions = read_csv_map(root / 'track3_train_mercaptionplus.csv', 'reason')
    labels = read_csv_map(root / 'track2_train_mercaptionplus.csv', 'openset', allow_empty=True)
    subtitles = read_csv_map(root / 'subtitle_chieng.csv', 'english', allow_empty=True)
    if set(captions) != set(labels) or not set(labels) <= set(subtitles):
        raise ValueError('MER-Caption+ annotation ID sets do not match')
    return [dict(id=name, dataset='MER-Caption+', split='train', text=subtitles[name],
                 description=captions[name], openset=string_to_list(label) or ['neutral'],
                 video=str(root / 'video' / (name + '.mp4')),
                 audio=str(root / 'audio' / (name + '.wav')))
            for name, label in labels.items()]


def build_benchmark_rows(dataset_root, dataset):
    spec = BENCHMARKS[dataset]
    root = Path(dataset_root).expanduser().absolute() / spec['directory']
    subtitle_path = root / spec.get('transcript', 'transcription-engchi-polish.csv')
    subtitle_column = spec.get('text_column', 'english')
    if dataset == 'OVMERDPlus':
        subtitles = read_csv_map(subtitle_path, subtitle_column, allow_empty=True)
        labels = read_csv_map(root / spec['label'], 'openset', allow_empty=True)
        if set(labels) != set(subtitles):
            raise ValueError('OV-MERD+ subtitle/label IDs differ')
        labels = {name: string_to_list(labels[name]) or ['neutral'] for name in subtitles}
    else:
        # These are trusted local AffectGPT annotations, not arbitrary uploaded pickle files.
        with np.load(root / spec['label'], allow_pickle=True) as archive:
            corpus = archive[spec['corpus']].item()
        if dataset == 'IEMOCAPFour':
            corpus = {name: value for name, value in corpus.items() if name.startswith('Ses05')}
        labels = {name: row['val' if spec['task'] == 'sentiment' else 'emo'] for name, row in corpus.items()}
        indexed = {'MELD': ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise'],
                   'IEMOCAPFour': ['happy', 'sad', 'neutral', 'anger']}
        if dataset in indexed:
            labels = {name: indexed[dataset][int(label)] for name, label in labels.items()}
        if spec['task'] == 'sentiment':
            labels = {name: float(label) for name, label in labels.items()}
        subtitles = read_csv_map(subtitle_path, subtitle_column, allow_empty=True,
                                 required_ids=labels)
    rows = [dict(id=name, dataset=dataset, split='test', text=subtitles[name], label=label,
                 video=str(root / spec['video'] / (name + '.' + spec.get('video_ext', 'mp4'))),
                 audio=str(root / spec['audio'] / (name + '.wav'))) for name, label in labels.items()]
    check_coverage(rows, dataset)
    return rows


def validate_rows(rows, *, require_media=True):
    seen = set()
    if not rows:
        raise ValueError('Empty manifest')
    for row in rows:
        if not {'id', 'dataset', 'split', 'text', 'video', 'audio'} <= row.keys():
            raise ValueError('Incomplete MER manifest row')
        key = (row['dataset'], row['id'])
        if not isinstance(row['id'], str) or not row['id'] or key in seen:
            raise ValueError('Empty or duplicate sample ID')
        seen.add(key)
        if not isinstance(row['text'], str):
            raise ValueError('Transcript must be a string (empty is allowed)')
        if any(k in row for k in ('start', 'end', 'start_time', 'end_time')):
            raise ValueError('Provide presegmented clips')
        if row['split'] == 'train':
            if row['dataset'] != 'MER-Caption+' or not row.get('description', '').strip():
                raise ValueError('Training requires MER-Caption+ descriptions')
            if not isinstance(row.get('openset'), list) or not row['openset'] or any(
                    not isinstance(x, str) or not x.strip() for x in row['openset']):
                raise ValueError('Training requires nonempty string emotion labels')
        elif row['split'] == 'test':
            if row['dataset'] not in BENCHMARKS or 'label' not in row:
                raise ValueError('Unknown benchmark or missing reference label')
            if BENCHMARKS[row['dataset']]['task'] == 'sentiment' and not np.isfinite(float(row['label'])):
                raise ValueError('Non-finite sentiment reference')
        else:
            raise ValueError('This protocol uses train and held-out test only')
        for field in ('video', 'audio'):
            if require_media and not Path(row[field]).is_file():
                raise FileNotFoundError(f'{row["id"]}: missing {field}: {row[field]}')
    return rows


def read_manifest(path, *, require_media=True):
    path = Path(path).expanduser().resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        for key in ('video', 'audio'):
            if key in row and not Path(row[key]).is_absolute():
                row[key] = str((path.parent / row[key]).resolve())
    return validate_rows(rows, require_media=require_media)


def save_manifest(rows, path, sources, *, require_media=True):
    validate_rows(rows, require_media=require_media)
    path = Path(path).expanduser().resolve()
    if path.exists() or path.with_suffix('.receipt.json').exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    stored_rows = []
    for row in rows:
        stored = dict(row)
        for key in ('video', 'audio'):
            stored[key] = os.path.relpath(Path(row[key]).expanduser().absolute(), path.parent)
        stored_rows.append(stored)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in stored_rows))
    sources = [Path(p).expanduser().absolute() for p in sources]
    receipt = dict(samples=len(rows), manifest_sha256=sha256(path), media_checked=require_media,
                   path_base='manifest_dir',
                   sources={os.path.relpath(p, path.parent): sha256(p) for p in sources})
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2))


def verify_manifest_receipt(path):
    path = Path(path).expanduser().resolve()
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if receipt.get('path_base') not in (None, 'manifest_dir'):
        raise ValueError('Unknown annotation receipt path base')
    if receipt['manifest_sha256'] != sha256(path):
        raise ValueError(f'Manifest changed after preparation: {path}')
    for source, expected in receipt['sources'].items():
        if receipt.get('path_base') == 'manifest_dir':
            source = path.parent / source
        if sha256(source) != expected:
            raise ValueError(f'Annotation source changed: {source}')
    return receipt


def read_checkpoint_training_rows(checkpoint, config):
    """Use this machine's manifest location, retaining the checkpoint's data identity."""
    path = config['train_manifest']
    verify_manifest_receipt(path)
    if sha256(path) != checkpoint['manifest_sha256']:
        raise ValueError('Checkpoint training manifest changed')
    return read_manifest(path, require_media=False)


class MERDataset(Dataset):
    def __init__(self, rows, label_type='hybrid', training=True):
        self.rows, self.label_type, self.training = rows, label_type, training
        if any(row['split'] != ('train' if training else 'test') for row in rows):
            raise ValueError('Train/test split mismatch')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        target = random.choice(('description', 'ovlabel')) if self.label_type == 'hybrid' else self.label_type
        if not self.training:
            target = 'ovlabel'
        row['question'] = DESCRIPTION_QUESTION if target == 'description' else OV_QUESTION
        if self.training:
            row['answer'] = (row['description'] if target == 'description' else
                             "The character's emotional state is " + ', '.join(row['openset']) + '.')
        # Eval labels never enter the collator/model.
        return {key: row[key] for key in ('id', 'text', 'video', 'audio', 'question', 'answer') if key in row}


class MERCollator:
    def __init__(self, options, *, processor=None):
        self.native = NativeMediaCollator(options, processor=processor)

    def __call__(self, rows):
        inputs = self.native(rows)
        return dict(omni_inputs=inputs, ids=[r['id'] for r in rows], texts=[r['text'] for r in rows],
                    questions=[r['question'] for r in rows],
                    answers=[r['answer'] for r in rows] if 'answer' in rows[0] else None)
