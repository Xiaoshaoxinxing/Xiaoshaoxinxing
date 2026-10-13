"""Fixed equal-energy spectral bands, estimated only on the training manifest."""
import hashlib
import json
import fcntl
import os
from pathlib import Path
import time

import torch

from .layers.spectral import GraphSpectralBandDecomposer, _equal_energy_edges

EDGE_RULE = 'atsuko_original_searchsorted_v1'


class TrainingSetSpectralDecomposer(GraphSpectralBandDecomposer):
    """Reuse Atsuko's basis/reconstruction; never estimate edges from a test batch."""
    def __init__(self, length, num_bands=3):
        super().__init__(length, num_bands=num_bands, partition='equal_energy')
        self.register_buffer('band_edges', torch.full((num_bands - 1,), -1, dtype=torch.long))

    def _get_edges(self, x_hat):
        edges = self.band_edges.tolist()
        if (len(edges) != self.num_bands - 1 or not edges or
                not 0 < edges[0] <= edges[-1] < self.length or
                any(left > right for left, right in zip(edges, edges[1:]))):
            raise RuntimeError('Spectral bands are uncalibrated; run tools/mer_omni.py calibrate on the training set')
        return edges


def equal_energy_edges(energy, num_bands=3):
    energy = torch.as_tensor(energy, dtype=torch.float64, device='cpu')
    if energy.ndim != 1 or len(energy) < num_bands or not torch.isfinite(energy).all() or (energy < 0).any():
        raise ValueError('Invalid training spectral energy')
    if energy.sum() <= 0:
        raise ValueError('Cannot calibrate an all-zero spectrum')
    # Preserve Atsuko's actual searchsorted/clamping behavior, including its
    # boundary-index convention. Do not introduce an extra +1 or empty-band fix.
    return _equal_energy_edges(energy, num_bands)


def band_energy_fractions(energy, edges):
    ends = [0, *edges, len(energy)]
    total = sum(energy)
    return [sum(energy[start:end]) / total for start, end in zip(ends, ends[1:])]


def projection_fingerprint(frontend):
    digest = hashlib.sha256()
    for name, value in frontend.state_dict().items():
        if name.startswith(('adapters.', 'shared_projection.')):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_signature(config):
    values = {key: config[key] for key in ('omni', 'spectral')}
    values['initialization_seed'] = config['train']['seed']
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def set_calibration(frontend, energies, samples):
    if len(energies) != 3 or samples < 1:
        raise ValueError('Calibration requires all three modalities and training samples')
    for spectral, energy in zip(frontend.spectra, energies):
        if len(energy) != spectral.length:
            raise ValueError('Calibration frequency grid differs from the model')
        edges = equal_energy_edges(energy, spectral.num_bands)
        spectral.band_edges.copy_(torch.tensor(edges, device=spectral.band_edges.device))
    frontend.calibration_samples.fill_(samples)


@torch.no_grad()
def _calibrate(model, dataset, collator, config, device, manifest_hash, output):
    """One complete, deterministic pass; labels do not enter the energy estimate."""
    if not dataset.training or any(row['split'] != 'train' or row['dataset'] != 'MER-Caption+'
                                   for row in dataset.rows):
        raise ValueError('Spectral calibration is training-only')
    path = Path(output)
    if path.exists():
        raise FileExistsError('Refusing to overwrite spectral calibration')
    was_training = model.training
    model.eval()
    energies = [torch.zeros(s.length, dtype=torch.float64) for s in model.frontend.spectra]
    signature = dict(manifest_sha256=manifest_hash, samples=len(dataset), model_signature=model_signature(config),
                     projection_sha256=projection_fingerprint(model.frontend))
    progress_path = path.with_suffix('.progress.json')
    processed, previous_seconds = 0, 0.0
    started = time.monotonic()
    def save_progress(status, error=None):
        elapsed = previous_seconds + time.monotonic() - started
        progress = dict(format='atsuko-training-bands-progress-v1', **signature,
                        processed=processed, energy_sum=[e.tolist() for e in energies],
                        elapsed_seconds=elapsed, status=status, pid=os.getpid(),
                        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), error=error)
        temporary = progress_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(progress, indent=2) + '\n')
        temporary.replace(progress_path)
    try:
        if progress_path.exists():
            saved = json.loads(progress_path.read_text())
            if saved.get('format') != 'atsuko-training-bands-progress-v1' or any(
                    saved.get(key) != value for key, value in signature.items()):
                raise ValueError('Stale calibration progress; manifest, model or initialization changed')
            processed = saved['processed']
            if not isinstance(processed, int) or not 0 <= processed <= len(dataset):
                raise ValueError('Invalid calibration progress count')
            restored = [torch.tensor(e, dtype=torch.float64) for e in saved['energy_sum']]
            if len(restored) != 3 or any(a.shape != b.shape or not torch.isfinite(b).all() or (b < 0).any()
                                      for a, b in zip(energies, restored)):
                raise ValueError('Invalid calibration energy checkpoint')
            energies = restored
            previous_seconds = float(saved['elapsed_seconds'])
            print(f'Resuming spectral calibration: {processed}/{len(dataset)}', flush=True)
        save_progress('running')
        for index in range(processed, len(dataset)):
            # Avoid hybrid-label sampling: calibration depends only on encoder inputs.
            row = {k: dataset.rows[index][k] for k in ('id', 'text', 'video', 'audio')}
            row['question'] = ''
            batch = collator([row])
            inputs = {k: v.to(device) for k, v in batch['omni_inputs'].items()}
            sequences = model.frontend.project_sequences(model.components.encode(inputs))
            increments = []
            for spectral, values in zip(model.frontend.spectra, sequences):
                coeffs = torch.einsum('kt,btd->bkd', spectral.U.T.to(values), values)
                increments.append(coeffs.double().square().mean(-1).sum(0).cpu())
            if any(not torch.isfinite(delta).all() for delta in increments):
                raise ValueError(f'Non-finite encoder spectrum at training sample {row["id"]}')
            for total, delta in zip(energies, increments):
                total.add_(delta)
            processed = index + 1
            if index == 0 or (index + 1) % 100 == 0:
                elapsed = previous_seconds + time.monotonic() - started
                speed = processed / max(elapsed, 1e-6)
                remaining = (len(dataset) - processed) / max(speed, 1e-6)
                print(f'Spectral calibration: {processed}/{len(dataset)}; {speed:.2f} samples/s; '
                      f'estimated remaining {remaining/60:.1f} min', flush=True)
                save_progress('running')
        averaged = [(energy / len(dataset)).tolist() for energy in energies]
        set_calibration(model.frontend, averaged, len(dataset))
        edges = [s.band_edges.tolist() for s in model.frontend.spectra]
        artifact = dict(format='atsuko-training-bands-v1', **signature,
                        energy=averaged, band_edges=edges, edge_rule=EDGE_RULE,
                        band_energy_fractions=[band_energy_fractions(e, b) for e, b in zip(averaged, edges)])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(artifact, indent=2) + '\n')
        temporary.replace(path)
        model.calibration_info = artifact
        save_progress('completed')
        return artifact
    except Exception as exc:
        # Preserve previously valid statistics, but never overwrite stale state.
        if not isinstance(exc, ValueError) or not str(exc).startswith(('Stale calibration', 'Invalid calibration')):
            save_progress('failed', f'{type(exc).__name__}: {exc}')
        raise
    finally:
        model.train(was_training)


def calibrate(model, dataset, collator, config, device, manifest_hash, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('A calibration for this output is already running') from exc
        return _calibrate(model, dataset, collator, config, device, manifest_hash, path)


def load_calibration(model, path, config, manifest_hash, samples):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing training-set spectral bands: {path}. Run tools/mer_omni.py calibrate first.')
    artifact = json.loads(path.read_text())
    expected = dict(format='atsuko-training-bands-v1', manifest_sha256=manifest_hash,
                    samples=samples, model_signature=model_signature(config),
                    projection_sha256=projection_fingerprint(model.frontend), edge_rule=EDGE_RULE)
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError('Stale calibration: training data, model, initialization or frequency grid changed')
    set_calibration(model.frontend, artifact['energy'], samples)
    if artifact['band_edges'] != [s.band_edges.tolist() for s in model.frontend.spectra]:
        raise ValueError('Calibration edges do not match the saved training energy')
    model.calibration_info = artifact
    return artifact
