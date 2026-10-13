"""Qwen2.5-Omni front-end encoders and Thinker, loaded from local weights only.

The text front end is Omni's token embedding, not a fictitious text encoder.
Audio/video use their native encoders, including the original input projections.
Frozen native encoders may run without autograd; all trainable Atsuko adapters
are downstream of them. Thinker stays differentiable with respect to its inputs.
"""
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OMNI_REPO = 'Qwen/Qwen2.5-Omni-3B'


def prepare_thinker_config(config):
    # Transformers 4.52 validates the built-in tensor-parallel plan even for a
    # single-device model, but its plan registry is None on torch 2.4. Clear
    # unused dispatch metadata on THIS config only; no library/global patch,
    # tensor-parallel execution, weight change, or PyTorch upgrade is involved.
    from packaging.version import Version
    if Version(torch.__version__.split('+')[0]) < Version('2.5'):
        config.text_config.base_model_tp_plan = {}
    return config




def local_source(options):
    path = Path(options['source']).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Local Qwen2.5-Omni-3B weights are missing: {path}. No download is started automatically.')
    config = json.loads((path / 'config.json').read_text())
    thinker = config.get('thinker_config', {})
    text = thinker.get('text_config', {})
    if (config.get('model_type') != 'qwen2_5_omni' or text.get('hidden_size') != 2048
            or text.get('num_hidden_layers') != 36):
        raise ValueError(f'Expected Qwen2.5-Omni-3B architecture: {path}')
    return path


def resample_sequence(sequence, length):
    """Resample a valid [time, channels] sequence; padding never enters the spectrum."""
    if sequence.ndim != 2 or sequence.shape[0] < 1:
        raise ValueError('An encoder returned an empty or malformed sequence')
    sequence = sequence.float()
    if sequence.shape[0] == length:
        return sequence
    return F.interpolate(sequence.T.unsqueeze(0), size=length, mode='linear',
                         align_corners=False)[0].T


class OmniComponents(nn.Module):
    def __init__(self, thinker, options):
        super().__init__()
        self.thinker = thinker
        self.options = options
        self.hidden_size = thinker.config.text_config.hidden_size
        self.load_report = {}
        thinker.requires_grad_(False)
        if options['backbone_lora']:
            from peft import LoraConfig, get_peft_model
            targets = ['v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj']
            thinker.model = get_peft_model(thinker.model, LoraConfig(
                r=options['lora_r'], lora_alpha=options['lora_alpha'],
                lora_dropout=options['lora_dropout'], target_modules=targets, bias='none'))
            # Trainable LoRA matrices use FP32, the frozen backbone uses its load dtype.
            for parameter in thinker.model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        if options['gradient_checkpointing']:
            thinker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
        thinker.model.config.use_cache = False
        self.train(self.training)

    @classmethod
    def from_pretrained(cls, options):
        source = local_source(options)
        try:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniThinkerConfig
        except ImportError as exc:
            raise RuntimeError('Qwen Omni needs the requirements/train.txt environment.') from exc
        dtype = torch.bfloat16 if options['dtype'] == 'bfloat16' else torch.float32
        config = prepare_thinker_config(Qwen2_5OmniThinkerConfig.from_pretrained(
            str(source), local_files_only=True))
        thinker, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(source), config=config, local_files_only=True, torch_dtype=dtype,
            use_safetensors=True, low_cpu_mem_usage=True,
            attn_implementation=options['attn_implementation'], output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise RuntimeError(f'Incomplete Omni Thinker weights: {info}')
        unexpected = [key for key in info.get('unexpected_keys', [])
                      if not key.startswith(('talker.', 'token2wav.'))]
        if unexpected:
            raise RuntimeError(f'Unexpected Thinker parameters: {unexpected[:20]}')
        model = cls(thinker, options)
        model.load_report = {'source': str(source), 'requested_repo': options['repo_id'],
                             'requested_revision': options['revision'], 'loading_info': info,
                             'local_config_architecture_verified': True,
                             'note': 'Revision identity requires hash-verified downloaded files; config alone only verifies architecture.'}
        return model

    def train(self, mode=True):
        super().train(mode)
        self.thinker.visual.eval()
        self.thinker.audio_tower.eval()
        self.thinker.get_input_embeddings().eval()
        if not self.options['backbone_lora']:
            self.thinker.model.eval()
        return self

    def encode(self, inputs):
        required = {'input_ids', 'text_attention_mask', 'input_features',
                    'feature_attention_mask', 'pixel_values_videos', 'video_grid_thw'}
        if required - inputs.keys():
            raise ValueError(f'Missing native Omni inputs: {sorted(required - inputs.keys())}')
        ids, text_mask = inputs['input_ids'], inputs['text_attention_mask'].bool()
        grids = inputs['video_grid_thw']
        batch = ids.shape[0]
        if text_mask.shape != ids.shape or grids.shape != (batch, 3):
            raise ValueError('Expected one text, one audio and one video per sample')
        audio_mask = inputs['feature_attention_mask']
        if audio_mask.shape[0] != batch or inputs['input_features'].shape[0] != batch:
            raise ValueError('Audio batch does not match text/video batch')
        with torch.no_grad():
            text = self.thinker.get_input_embeddings()(ids)
            audio_dtype = next(self.thinker.audio_tower.parameters()).dtype
            audio = self.thinker.get_audio_features(
                inputs['input_features'].to(audio_dtype), feature_attention_mask=audio_mask)
            vision = self.thinker.get_video_features(inputs['pixel_values_videos'], grids)
        _, audio_lengths = self.thinker.audio_tower._get_feat_extract_output_lengths(audio_mask.sum(-1))
        audio_lengths = [int(n) for n in audio_lengths.tolist()]
        merge = self.thinker.config.vision_config.spatial_merge_size
        video_lengths = [int(t * h * w // (merge * merge)) for t, h, w in grids.tolist()]
        if sum(video_lengths) != len(vision) or sum(audio_lengths) != len(audio):
            raise ValueError('Native encoder output lengths disagree with media metadata')
        text_sequences = [value[mask] for value, mask in zip(text, text_mask)]
        audio_sequences = list(audio.split(audio_lengths))
        # Aggregate spatial patches within each time step, never across time.
        video_sequences = []
        for value, (time_steps, height, width) in zip(vision.split(video_lengths), grids.tolist()):
            if height % merge or width % merge:
                raise ValueError('Video grid is incompatible with the spatial merger')
            video_sequences.append(value.reshape(time_steps, -1, self.hidden_size).mean(dim=1))
        return text_sequences, video_sequences, audio_sequences
