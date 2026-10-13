"""AffectGPT's published nine-dataset MER-UniBench protocol."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION_QUESTION = "Please infer the person's emotional state and provide your reasoning process."
OV_QUESTION = "Please recognize all possible emotional states of the character."

# Local AffectGPT dataset classes and paper Table 11. Counts are before metric filtering.
BENCHMARKS = {
    'MER2023': dict(task='basic', count=411, directory='mer2023-dataset-process',
                    label='label-6way.npz', corpus='test1_corpus', video='video', video_ext='avi', audio='audio'),
    'MER2024': dict(task='basic', count=1169, directory='mer2024-dataset-process',
                    label='label-6way.npz', corpus='test1_corpus', video='video', audio='audio',
                    transcript='transcription_merge.csv', text_column='chinese'),
    'MELD': dict(task='basic', count=2610, directory='meld-process',
                 label='label.npz', corpus='test_corpus', video='subvideo', audio='subaudio'),
    'IEMOCAPFour': dict(task='basic', count=1241, directory='iemocap-process',
                        label='label_4way.npz', corpus='whole_corpus', video='subvideo-tgt', video_ext='avi', audio='subaudio'),
    'CMUMOSI': dict(task='sentiment', count=686, directory='cmumosi-process',
                    label='label.npz', corpus='test_corpus', video='subvideo', audio='subaudio'),
    'CMUMOSEI': dict(task='sentiment', count=4659, directory='cmumosei-process',
                     label='label.npz', corpus='test_corpus', video='subvideo_new', audio='subaudio'),
    'SIMS': dict(task='sentiment', count=457, directory='sims-process',
                 label='label.npz', corpus='test_corpus', video='video', audio='audio'),
    'SIMSv2': dict(task='sentiment', count=1034, directory='simsv2-process',
                   label='label.npz', corpus='test_corpus', video='video_new', audio='audio'),
    'OVMERDPlus': dict(task='fine_grained', count=532, directory='ovmerdplus-process',
                       label='ovlabel.csv', video='video', audio='audio',
                       transcript='subtitle_eng.csv', text_column='sentence'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def spectral_sequence_lengths(spectral):
    """Resolve fixed text/video/audio lengths; old configs keep one shared grid."""
    lengths = spectral.get('sequence_lengths', [spectral['sequence_length']] * 3)
    if (not isinstance(lengths, (list, tuple)) or len(lengths) != 3 or
            any(not isinstance(n, int) or isinstance(n, bool) or n < 3 for n in lengths)):
        raise ValueError('Expected three positive spectral sequence lengths: text, video, audio')
    kernels = spectral['conv_kernel_sizes']
    if (len(kernels) != 3 or any(not isinstance(k, int) or isinstance(k, bool) or
                               k < 1 or n-k+1 < 3 for n, k in zip(lengths, kernels))):
        raise ValueError('Each convolution must leave at least three frequency bins')
    return list(lengths)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['training_dataset'] != 'MER-Caption+':
        raise ValueError('This protocol explicitly uses MER-Caption+, not MER-Caption')
    cfg['omni']['generative'] = True
    if cfg['omni']['repo_id'] != 'Qwen/Qwen2.5-Omni-3B':
        raise ValueError('This experiment requires Qwen2.5-Omni-3B')
    if cfg['omni']['dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('Use float32 or bfloat16')
    epoch_mode = cfg['train'].get('epoch_mode', 'fixed_steps')
    if epoch_mode not in ('fixed_steps', 'full_dataset'):
        raise ValueError('Unknown training epoch_mode')
    keys = ['batch_size', 'gradient_accumulation', 'epochs']
    if epoch_mode == 'fixed_steps':
        keys.append('steps_per_epoch')
    for key in keys:
        if cfg['train'][key] < 1:
            raise ValueError(f'{key} must be positive')
    if cfg['train']['label_type'] not in ('hybrid', 'description', 'ovlabel'):
        raise ValueError('Unknown instruction target')
    if cfg['spectral']['partition'] != 'equal_energy':
        raise ValueError('Atsuko alignment requires fixed training-set equal-energy partitioning')
    if cfg['spectral']['dim'] % cfg['spectral']['heads'] or cfg['spectral']['sequence_length'] < 3:
        raise ValueError('Invalid spectral dimensions')
    spectral = cfg['spectral']
    spectral_sequence_lengths(spectral)
    if spectral.get('input_mode', 'omni_resample') not in ('omni_resample', 'prealigned', 'zero_pad'):
        raise ValueError('Unknown spectral input_mode')
    if spectral.get('sequence_overflow', 'error') not in ('error', 'truncate_right'):
        raise ValueError('Unknown spectral sequence_overflow')
    if spectral.get('input_mode') != 'zero_pad' and 'sequence_overflow' in spectral:
        raise ValueError('sequence_overflow applies only to zero_pad inputs')
    if spectral['router_input'] not in ('pre_attn', 'post_attn'):
        raise ValueError('Unknown Atsuko modality router pathway')
    dropouts = spectral['attn_dropout'] + [spectral[k] for k in
                ('relu_dropout', 'res_dropout', 'embed_dropout', 'text_dropout', 'output_dropout', 'band_mask_rate')]
    if len(spectral['attn_dropout']) != 3 or any(not 0 <= p < 1 for p in dropouts):
        raise ValueError('Invalid per-modality dropout/masking probabilities')
    for key in ('unimodal_weight', 'auxiliary_weight', 'router_weight', 'distillation_weight',
                'modality_entropy_weight', 'band_entropy_weight'):
        if cfg['mcm'][key] < 0:
            raise ValueError('Loss weights must be nonnegative')
        if cfg['mcm'].get('loss_mode') == 'ce_only' and cfg['mcm'][key] != 0:
            raise ValueError('ce_only requires all auxiliary loss weights to be zero')
    if cfg['mcm'].get('loss_mode', 'full') not in ('full', 'ce_only'):
        raise ValueError('Unknown training loss_mode')
    if cfg['mcm'].get('loss_mode') == 'ce_only' and cfg['mcm']['enabled']:
        raise ValueError('ce_only requires mcm.enabled=false')
    if cfg['train']['max_length'] < 2 or cfg['omni']['lora_r'] < 1:
        raise ValueError('Invalid generation head rank or training length')
    if cfg['mcm']['temperature'] <= 0 or cfg['spectral']['temperature'] <= 0:
        raise ValueError('Routing temperatures must be positive')
    if cfg['generation']['temperature'] <= 0 or not 0 < cfg['generation']['top_p'] <= 1:
        raise ValueError('Invalid generation sampling settings')
    top_k = cfg['generation'].get('top_k', 0)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('Generation top_k must be a nonnegative integer')
    if cfg['omni']['audio_sample_rate'] != 16000 or cfg['omni']['max_frames'] < 2 or cfg['omni']['max_frames'] % 2:
        raise ValueError('Native Omni requires 16 kHz audio and an even frame budget >= 2')
    accepted = cfg.get('accepted_release_overlaps', {})
    if not isinstance(accepted, dict):
        raise ValueError('Accepted release overlaps must map benchmark names to sample IDs')
    for dataset, names in accepted.items():
        if (dataset not in BENCHMARKS or not isinstance(names, list) or
                any(not isinstance(name, str) or not name for name in names) or
                len(names) != len(set(names))):
            raise ValueError('Invalid accepted release overlap IDs')
    return cfg


def check_coverage(rows, dataset, *, strict=True):
    if dataset not in BENCHMARKS:
        raise ValueError(f'Not part of MER-UniBench: {dataset}')
    names = [row['id'] for row in rows]
    if not names or len(set(names)) != len(names):
        raise ValueError('Empty or duplicate benchmark IDs')
    if any(row['dataset'] != dataset or row['split'] != 'test' for row in rows):
        raise ValueError('Evaluation requires the official test split of one dataset')
    if strict and len(names) != BENCHMARKS[dataset]['count']:
        raise ValueError(f'{dataset}: expected {BENCHMARKS[dataset]["count"]} samples, got {len(names)}')


def check_prediction_ids(rows, predictions):
    expected = {row['id'] for row in rows}
    actual = set(predictions)
    if expected != actual:
        raise ValueError(f'Prediction coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')


def check_no_overlap(train_rows, eval_rows, *, accepted_overlaps=None):
    """Reject undeclared overlap; return explicitly accepted original-release IDs."""
    # The user retained the original releases after auditing their shared IDs.
    # Exceptions are scoped to a benchmark and matching MER-Caption+ training ID;
    # a differently named sample aliasing a training file is never exempted.
    names = {r['id'] for r in train_rows}
    paths = {str(Path(r['video']).resolve()) for r in train_rows}
    release_names = {r['id'] for r in train_rows
                     if r.get('dataset') == 'MER-Caption+' and r.get('split') == 'train'}
    accepted = accepted_overlaps or {}
    overlap = [r for r in eval_rows if r['id'] in names or str(Path(r['video']).resolve()) in paths]
    unexpected = [r['id'] for r in overlap
                  if r.get('split') != 'test' or r['id'] not in release_names or
                  r['id'] not in accepted.get(r.get('dataset'), ())]
    if unexpected:
        raise ValueError(f'Training/evaluation overlap: {unexpected[:10]}')
    return [r['id'] for r in overlap]
