"""B3: normalized inputs, four summary tokens, and core complementarity supervision.

Reuses this project's frozen native-media path and fixed-band SpectralFrontEnd.
No resampling, EMA, entropy loss, distillation, or output router scaling.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .model import SpectralFrontEnd
from .native_atsuko import NativeAtsuko
from .native_atsuko_norm import embedding_mean_row_rms


def insert_affect_slots(batch, token_id, im_end_id):
    ids = batch['input_ids']
    if ids.shape[0] != 1 or (ids == token_id).any():
        raise ValueError('Expected microbatch1 without reserved affect placeholders')
    endings = (ids[0] == im_end_id).nonzero().flatten()
    if len(endings) not in (2, 3):
        raise ValueError('Unexpected native conversation boundaries')
    position = int(endings[1])
    result = dict(batch)
    for key, value in (('input_ids', token_id), ('attention_mask', 1), ('labels', -100)):
        if key in batch:
            x = batch[key]
            result[key] = torch.cat((x[:, :position], x.new_full((1, 4), value), x[:, position:]), 1)
    return result, position


class NormalizedSpectralFrontEnd(SpectralFrontEnd):
    def __init__(self, hidden, cfg):
        if cfg['router_input'] != 'post_attn' or cfg['input_mode'] != 'zero_pad':
            raise ValueError('B3 requires post-attention routing and native zero padding')
        super().__init__(hidden, cfg)
        self.input_norms = nn.ModuleList([nn.LayerNorm(hidden, eps=1e-5) for _ in range(3)])

    def prepare_sequences(self, sequences):
        if len(sequences) != 3:
            raise ValueError('Expected text/video/audio')
        # Normalize only real encoder positions; zero padding is applied afterwards.
        normalized = [[norm(value.float()) for value in rows]
                      for norm, rows in zip(self.input_norms, sequences)]
        return super().prepare_sequences(normalized)


class ResidualEmotionHead(nn.Module):
    def __init__(self, dim, classes, dropout):
        super().__init__()
        self.first, self.second = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, classes)
        self.dropout = dropout

    def forward(self, x):
        projected = x + self.second(F.dropout(F.relu(self.first(x)), self.dropout, self.training))
        return projected, self.classifier(projected)


class FourTokenHead(nn.Module):
    def __init__(self, dim, hidden, classes, dropout, embed_rms):
        super().__init__()
        self.unimodal = nn.ModuleList([ResidualEmotionHead(dim, classes, dropout) for _ in range(3)])
        # Without text, without video, without audio: VA / LA / LV.
        self.bimodal = nn.ModuleList([ResidualEmotionHead(dim, classes, dropout) for _ in range(3)])
        self.fused = ResidualEmotionHead(dim, classes, dropout)
        self.projector = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.slot_embedding = nn.Parameter(torch.zeros(4, hidden))
        self.output_norm = nn.LayerNorm(hidden, eps=1e-5)
        with torch.no_grad():
            self.output_norm.weight.fill_(embed_rms)
            self.output_norm.bias.zero_()

    def forward(self, features):
        h = features['modalities']  # text, video, audio
        projections, logits = [], {}
        for i, (name, head) in enumerate(zip(('text', 'video', 'audio'), self.unimodal)):
            value, logits[name] = head(h[:, i])
            projections.append(value)
        for name, (left, right), head in zip(('va', 'la', 'lv'), ((1, 2), (0, 2), (0, 1)), self.bimodal):
            _, logits[name] = head(h[:, left] + h[:, right])
        fused, logits['fusion'] = self.fused(features['fusion'])
        z = torch.stack([fused, *projections], dim=1)
        tokens = self.output_norm(self.projector(z) + self.slot_embedding)
        return tokens, dict(logits=logits, weights=features['modality_weights'],
                            band_weights=features['band_weights'])


def core_auxiliary_loss(aux, target, settings):
    logits = aux['logits']
    if target.shape != logits['fusion'].shape or not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('Auxiliary targets must be binary [batch, classes]')
    losses = {k: F.binary_cross_entropy_with_logits(v.float(), target.float(), reduction='none').mean(-1)
              for k, v in logits.items()}
    importance = torch.stack([losses[k] - losses['fusion'] for k in ('va', 'la', 'lv')], -1).detach()
    complementarity_target = F.softmax(importance / settings['temperature'], -1)
    complementarity = F.kl_div(aux['weights'].float().clamp_min(1e-8).log(),
                              complementarity_target, reduction='batchmean')
    fused = losses['fusion'].mean()
    uni = torch.stack([losses[k] for k in ('text', 'video', 'audio')]).mean()
    bi = torch.stack([losses[k] for k in ('va', 'la', 'lv')]).mean()
    total = fused + uni + settings['bimodal_weight'] * bi + settings['complementarity_weight'] * complementarity
    return dict(total=total, fusion_bce=fused, unimodal_bce=uni, bimodal_bce=bi,
                complementarity_kl=complementarity, complementarity_target=complementarity_target)


class CoreNativeAtsuko(NativeAtsuko):
    def __init__(self, thinker, spectral, token_id, auxiliary):
        # Reuse NativeAtsuko forward/generation protocol, but instantiate no unused B heads.
        nn.Module.__init__(self)
        self.thinker, self.fusion_token_id = thinker, token_id
        self.auxiliary_settings = dict(auxiliary)
        self.calibration_info = None
        rms = embedding_mean_row_rms(thinker.get_input_embeddings().weight)
        with torch.random.fork_rng(devices=[]):
            self.frontend = NormalizedSpectralFrontEnd(thinker.config.text_config.hidden_size, spectral)
            self.fusion_in = FourTokenHead(spectral['dim'], thinker.config.text_config.hidden_size,
                auxiliary['classes'], spectral['output_dropout'], rms)
        self.initial_embedding_rms = rms

    @torch.no_grad()
    def encode_sequences(self, batch):
        thinker = self.thinker
        audio = thinker.get_audio_features(batch['input_features'], feature_attention_mask=batch['feature_attention_mask'])
        video = thinker.get_video_features(batch['pixel_values_videos'], batch['video_grid_thw'])
        text = thinker.get_input_embeddings()(batch['bypass_subtitle_ids'])[0]
        time = int(batch['video_grid_thw'][0, 0])
        temporal = video.reshape(time, -1, video.shape[-1]).mean(1)
        return ([text], [temporal], [audio]), audio, video

    def _prepare(self, batch, with_fusion=True):
        ids = batch['input_ids']
        if ids.shape[0] != 1:
            raise ValueError('B3 uses microbatch1')
        with torch.no_grad():
            embeds = self.thinker.get_input_embeddings()(ids)
        sequences, audio, video = self.encode_sequences(batch)
        for token, features in ((self.thinker.config.audio_token_id, audio), (self.thinker.config.video_token_id, video)):
            mask = ids == token
            if int(mask.sum()) != len(features):
                raise ValueError('Native media count mismatch')
            embeds = embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), features.to(embeds))
        mask = ids == self.fusion_token_id
        if not with_fusion:
            if mask.any():
                raise ValueError('Disabled branch requires exact A input')
            return embeds, None
        locations = mask[0].nonzero().flatten()
        if len(locations) != 4 or not torch.equal(locations, torch.arange(int(locations[0]), int(locations[0])+4, device=ids.device)):
            raise ValueError('Expected four consecutive affect slots')
        tokens, aux = self.fusion_in(self.frontend(sequences))
        return embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), tokens.to(embeds)), aux

    def prepare_embeddings(self, batch, *, with_fusion=True):
        return self._prepare(batch, with_fusion)[0]

    @staticmethod
    def native_kwargs(batch):
        return {k: v for k, v in NativeAtsuko.native_kwargs(batch).items() if k != 'aux_target'}

    def forward(self, **batch):
        embeds, aux = self._prepare(batch)
        output = self.thinker(**self.native_kwargs(batch), inputs_embeds=embeds)
        if 'aux_target' in batch:
            output.auxiliary_losses = core_auxiliary_loss(aux, batch['aux_target'], self.auxiliary_settings)
        # Keep native .loss as answer CE, so logs never conflate CE and auxiliary loss.
        return output
