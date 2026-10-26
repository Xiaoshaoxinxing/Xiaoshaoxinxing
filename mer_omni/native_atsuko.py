"""Experiment B: native Omni plus one original Atsuko fusion token, CE only.

The native audio/video tensors feed both paths. The original Thinker remains
responsible for positions, decoder forward, loss, and cached generation.
"""
import contextlib
import torch
from torch import nn
from .model import SpectralFrontEnd


def insert_fusion_slot(batch, token_id, im_end_id):
    """Insert one masked user token; removing it exactly recovers experiment A."""
    ids = batch['input_ids']
    if ids.shape[0] != 1 or (ids == token_id).any():
        raise ValueError('Expected batch1 and no reserved fusion placeholder in original text')
    endings = (ids[0] == im_end_id).nonzero().flatten()
    # system end, user end, and optionally the supervised assistant end.
    if len(endings) not in (2, 3):
        raise ValueError('Unexpected native conversation boundaries')
    position = int(endings[1])
    result = dict(batch)
    for key, value in (('input_ids', token_id), ('attention_mask', 1), ('labels', -100)):
        if key in batch:
            tensor = batch[key]
            result[key] = torch.cat((tensor[:, :position], tensor.new_full((1, 1), value), tensor[:, position:]), dim=1)
    return result, position


class NativeAtsuko(nn.Module):
    def __init__(self, thinker, spectral, token_id):
        super().__init__()
        self.thinker = thinker
        self.fusion_token_id = token_id
        # Preserve the loaded LoRA initialization and subsequent global RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.frontend = SpectralFrontEnd(thinker.config.text_config.hidden_size, spectral)
            self.fusion_in = nn.Linear(spectral['dim'], thinker.config.text_config.hidden_size)
        self.calibration_info = None

    @property
    def audio_tower(self):
        return self.thinker.audio_tower

    @property
    def visual(self):
        return self.thinker.visual

    @property
    def rope_deltas(self):
        return self.thinker.rope_deltas

    @rope_deltas.setter
    def rope_deltas(self, value):
        self.thinker.rope_deltas = value

    def train(self, mode=True):
        super().train(mode)
        self.audio_tower.eval()
        self.visual.eval()
        return self

    def prepare_embeddings(self, batch, *, with_fusion=True):
        ids = batch['input_ids']
        if ids.shape[0] != 1:
            raise ValueError('Experiment B uses microbatch1')
        thinker = self.thinker
        with torch.no_grad():
            embeddings = thinker.get_input_embeddings()(ids)
            audio = thinker.get_audio_features(batch['input_features'], feature_attention_mask=batch['feature_attention_mask'])
            video = thinker.get_video_features(batch['pixel_values_videos'], batch['video_grid_thw'])
            text = thinker.get_input_embeddings()(batch['bypass_subtitle_ids'])[0]
        for token, features in ((thinker.config.audio_token_id, audio), (thinker.config.video_token_id, video)):
            mask = ids == token
            if int(mask.sum()) != len(features):
                raise ValueError('Native media count mismatch')
            embeddings = embeddings.masked_scatter(mask.unsqueeze(-1).expand_as(embeddings), features.to(embeddings))
        if with_fusion:
            fusion_mask = ids == self.fusion_token_id
            if int(fusion_mask.sum()) != 1:
                raise ValueError('Expected exactly one fusion placeholder')
            time, height, width = batch['video_grid_thw'][0].tolist()
            # Spatial pooling applies ONLY to the Atsuko branch; native video stays complete.
            temporal_video = video.reshape(time, -1, video.shape[-1]).mean(dim=1)
            features = self.frontend(([text], [temporal_video], [audio]))
            fused = self.fusion_in(features['fusion'])
            embeddings = embeddings.masked_scatter(fusion_mask.unsqueeze(-1).expand_as(embeddings), fused.to(embeddings))
        elif (ids == self.fusion_token_id).any():
            raise ValueError('Disabled bridge requires the exact A input without a placeholder')
        return embeddings

    @staticmethod
    def native_kwargs(batch):
        return {k: v for k, v in batch.items()
                if k not in ('bypass_subtitle_ids', 'input_features', 'pixel_values_videos')}

    def forward(self, **batch):
        embeds = self.prepare_embeddings(batch)
        return self.thinker(**self.native_kwargs(batch), inputs_embeds=embeds)

    @torch.no_grad()
    def generate(self, **kwargs):
        keys = ('input_ids', 'attention_mask', 'input_features', 'feature_attention_mask',
                'pixel_values_videos', 'video_grid_thw', 'video_second_per_grid', 'bypass_subtitle_ids')
        batch = {k: kwargs[k] for k in keys}
        embeds = self.prepare_embeddings(batch)
        self.thinker.rope_deltas = None
        seen = []
        # Keep HF's normal input_ids/cache/3-D RoPE handling. Replace only its
        # prefill embeddings; no decoder hooks or hooks during backward.
        def prefill(module, args, values):
            ids = values.get('input_ids')
            if ids is not None and ids.shape[1] != 1:
                if seen or not torch.equal(ids, batch['input_ids']):
                    raise ValueError('Unexpected generation prefill/batch expansion')
                values['inputs_embeds'] = embeds
                seen.append(True)
            values['input_features'] = None
            values['pixel_values_videos'] = None
            return args, values
        handle = self.thinker.register_forward_pre_hook(prefill, with_kwargs=True)
        try:
            options = {k: v for k, v in kwargs.items() if k != 'bypass_subtitle_ids'}
            result = self.thinker.generate(**options)
        finally:
            handle.remove()
        if len(seen) != 1:
            raise ValueError('Native generation did not consume the prepared prefill')
        return result
