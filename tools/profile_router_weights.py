#!/usr/bin/env python3
"""Read-only router diagnostics on the first 256 checkpoint-bound training rows.

Print one JSON summary to stdout; loading/progress/collapse warnings go to stderr.
No generation, optimization, calibration, test media, or output files are used.
"""
import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np

SAMPLES = 256
MODALITIES = ('text', 'video', 'audio')
BANDS = ('low', 'mid', 'high')


def distribution(values):
    """Population statistics and empirical quantiles, without rounding."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite one-dimensional distribution')
    return dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                min=float(values.min()), max=float(values.max()),
                quantiles=dict(zip(('p05', 'p50', 'p90', 'p95', 'p99'),
                                   np.quantile(values, (.05, .5, .9, .95, .99)).tolist())))


def summarize_weights(weights, labels):
    """Summarize [samples, 3] probabilities and per-sample entropy/max weight."""
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.ndim != 2 or weights.shape[0] < 1 or weights.shape[1] != len(labels)
            or not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any()
            or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Router output must contain finite normalized probability vectors')
    log_weights = np.zeros_like(weights)
    np.log(weights, out=log_weights, where=weights > 0)
    entropy = -(weights * log_weights).sum(axis=1)
    return dict(order=list(labels), mean=weights.mean(axis=0).tolist(),
                std=weights.std(axis=0, ddof=0).tolist(), min=weights.min(axis=0).tolist(),
                max=weights.max(axis=0).tolist(), entropy=distribution(entropy),
                max_weight=distribution(weights.max(axis=1)))


def collapse_warnings(modality_summary):
    warnings = []
    for name, mean in zip(modality_summary['order'], modality_summary['mean']):
        if mean > 0.8:
            warnings.append(f'Modality collapse: mean {name} weight {mean:.6f} > 0.8')
    median = modality_summary['max_weight']['quantiles']['p50']
    if median > 0.95:
        warnings.append(f'Modality collapse: median per-sample maximum weight {median:.6f} > 0.95')
    return warnings


def training_rows(checkpoint, manifest):
    from mer_omni.data import read_manifest, verify_manifest_receipt
    from mer_omni.protocol import sha256

    # Reject other splits before opening media or receipt annotation sources.
    rows = read_manifest(manifest, require_media=False)
    if any(row['dataset'] != 'MER-Caption+' or row['split'] != 'train' for row in rows):
        raise ValueError('Router profiling requires MER-Caption+ training rows only')
    if sha256(manifest) != checkpoint['manifest_sha256']:
        raise ValueError('Training manifest differs from the one bound to the checkpoint')
    verify_manifest_receipt(manifest)
    if len(rows) < SAMPLES:
        raise ValueError(f'Training manifest must contain at least {SAMPLES} rows')
    selected = rows[:SAMPLES]
    for row in selected:
        for field in ('video', 'audio'):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(f'{row["id"]}: missing {field}: {row[field]}')
    return selected, len(rows)


def profile(checkpoint_path, manifest_path, device_name):
    import torch
    from tools.mer_omni import build_model, resolve_config
    from mer_omni.training import read_checkpoint, load_adapter, move_batch
    from mer_omni.protocol import sha256

    checkpoint_path = (ROOT / checkpoint_path.expanduser()).resolve()
    checkpoint = read_checkpoint(checkpoint_path)
    config = resolve_config(copy.deepcopy(checkpoint['config']))
    manifest = (ROOT / (manifest_path or Path(config['train_manifest'])).expanduser()).resolve()
    rows, manifest_samples = training_rows(checkpoint, manifest)
    checkpoint_hash, manifest_hash = sha256(checkpoint_path), sha256(manifest)
    step = checkpoint['step']
    device = torch.device(device_name)
    model, collator = build_model(config, device)
    # Restore trained parameters AND frontend buffers, including calibrated bands.
    load_adapter(model, checkpoint['model'])
    del checkpoint
    model.eval()
    modality_weights, band_weights = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            # Do not use MERDataset's hybrid label sampling or read test manifests.
            item = {key: row[key] for key in ('id', 'text', 'video', 'audio')}
            item['question'] = ''
            inputs = move_batch(collator([item]), device)['omni_inputs']
            features = model.frontend(model.components.encode(inputs))
            modality = features['modality_weights'].detach().cpu().double().numpy()
            bands = features['band_weights'].detach().cpu().double().numpy()
            if modality.shape != (1, 3) or bands.shape != (1, 3, 3):
                raise ValueError(f'{row["id"]}: unexpected router output shapes '
                                 f'{modality.shape}, {bands.shape}')
            modality_weights.append(modality[0].copy())
            band_weights.append(bands[0].copy())
            if index == 1 or index % 32 == 0:
                print(f'Router profiling: {index}/{SAMPLES}; '
                      f'{time.monotonic() - started:.1f}s', file=sys.stderr, flush=True)
    modality_summary = summarize_weights(modality_weights, MODALITIES)
    bands = np.asarray(band_weights)
    band_summary = {name: summarize_weights(bands[:, i, :], BANDS)
                    for i, name in enumerate(MODALITIES)}
    warnings = collapse_warnings(modality_summary)
    for message in warnings:
        print(f'WARNING: {message}', file=sys.stderr, flush=True)
    return dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                checkpoint_step=step, manifest=str(manifest), manifest_sha256=manifest_hash,
                manifest_samples=manifest_samples, samples=len(rows),
                selection=dict(method='first_n_in_manifest_order', first_id=rows[0]['id'],
                               last_id=rows[-1]['id']),
                execution=dict(device=str(device), model_mode='eval', inference_mode=True,
                               max_frames=config['omni']['max_frames'],
                               input_mode=config['spectral'].get('input_mode', 'omni_resample'),
                               router_input=config['spectral']['router_input'],
                               router_temperature=config['spectral']['temperature']),
                statistics=dict(std_ddof=0, entropy_unit='nats', entropy_max=math.log(3)),
                modality_weights=modality_summary, band_routers=band_summary,
                collapse=dict(detected=bool(warnings), mean_weight_threshold=0.8,
                              median_max_weight_threshold=0.95, comparison='strictly_greater',
                              warnings=warnings),
                notes=['Entropy is computed per sample, not from the mean weights.',
                       'Band routers are reported separately in text/video/audio order.',
                       'Eval mode disables dropout and training-time band masking.',
                       'This is a diagnostic of the first 256 training rows, not the full dataset.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        help='Training manifest; defaults to checkpoint config and must match its hash')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    # Keep stdout machine-readable even if the existing loader prints notices.
    with contextlib.redirect_stdout(sys.stderr):
        result = profile(args.checkpoint, args.manifest, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
