"""Training-only vocabulary and fixed-band calibration identity for B3."""
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import torch

from .calibration import EDGE_RULE, set_calibration


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def train_vocabulary(rows, manifest_sha256, classes=64):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Vocabulary requires unique training samples')
    if any(r.get('dataset') != 'MER-Caption+' or r.get('split') != 'train' for r in rows):
        raise ValueError('Auxiliary vocabulary is training-only')
    labels = [{v.strip().lower() for v in r['openset']} for r in rows]
    if any(not values or '' in values for values in labels):
        raise ValueError('Empty training emotion label')
    counts = Counter(v for values in labels for v in values)
    vocab = sorted(counts, key=lambda v: (-counts[v], v))[:classes]
    if len(vocab) != classes:
        raise ValueError('Not enough training emotion classes')
    covered = sum(bool(set(vocab) & values) for values in labels)
    return dict(format='native-atsuko-core-vocab-v1', manifest_sha256=manifest_sha256,
                samples=len(rows), vocab=vocab, counts=[counts[v] for v in vocab],
                samples_with_known_label=covered, all_zero_targets=len(rows)-covered,
                normalization='strip_lower; unique labels per sample; count desc then lexical tie break')


def auxiliary_target(labels, vocabulary):
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError('Duplicate auxiliary classes')
    values = {v.strip().lower() for v in labels}
    return torch.tensor([[float(v in values) for v in vocabulary]], dtype=torch.float32)


def calibration_signature(model, cfg, manifest_sha256, samples):
    h = hashlib.sha256()
    # Includes new input normalization and actual spectral bases, not only projections.
    for name, value in sorted(model.frontend.state_dict().items()):
        if name.startswith(('input_norms.', 'adapters.', 'shared_projection.')):
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    for spectrum in model.frontend.spectra:
        h.update(spectrum.U.detach().cpu().contiguous().numpy().tobytes())
    root = Path(__file__).resolve().parents[1]
    sources = [root/'mer_omni/core_protocol.py', root/'mer_omni/native_atsuko_core.py',
               root/'mer_omni/native_atsuko.py', root/'mer_omni/native_atsuko_norm.py',
               root/'mer_omni/model.py', root/'mer_omni/calibration.py', root/'mer_omni/media.py',
               root/'mer_omni/sft/native_inputs.py', root/'mer_omni/sft/atsuko.py',
               root/'mer_omni/sft/omni.py']
    sources += sorted((root/'mer_omni/layers').glob('*.py'))
    return dict(manifest_sha256=manifest_sha256, samples=samples, initialization_sha256=h.hexdigest(),
                runtime_versions={name:version(name) for name in ('torch','transformers','peft')},
                config_sha256=digest({k:cfg[k] for k in ('seed', 'model_source', 'media', 'bridge')}),
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                edge_rule=EDGE_RULE)


def apply_calibration(model, artifact, signature):
    if artifact.get('format') != 'native-atsuko-core-bands-v1' or artifact.get('signature') != signature or artifact.get('complete') is not True:
        raise ValueError('Missing, stale, or incompatible B3 training calibration')
    if artifact.get('processed') != signature['samples']:
        raise ValueError('Calibration does not cover the complete training manifest')
    # set_calibration validates energy grids. Also reject invalid values before applying.
    energy = artifact['energy']
    if len(energy) != 3 or any(not torch.isfinite(torch.tensor(v)).all() or min(v) < 0 or sum(v) <= 0 for v in energy):
        raise ValueError('Invalid calibration energy')
    set_calibration(model.frontend, energy, signature['samples'])
    if artifact['band_edges'] != [s.band_edges.tolist() for s in model.frontend.spectra]:
        raise ValueError('Saved calibration edges disagree with saved energy')
    model.calibration_info = artifact
