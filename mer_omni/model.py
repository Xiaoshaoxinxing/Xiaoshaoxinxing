"""Omni encoders/fusion with Atsuko routing, auxiliary branches and full objectives."""
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .encoders.omni import OmniComponents, resample_sequence
from .layers.spectral import BandRouter, apply_band_masking
from .layers.router import router
from .layers.transformer import TransformerEncoder
from .calibration import TrainingSetSpectralDecomposer
from .protocol import spectral_sequence_lengths
from .objectives import entropy_balance, feature_distillation, complementarity, combine_losses


class SequenceTooLong(ValueError):
    """Training resamples this case using AffectGPT's bounded retry policy."""


def generation_processors(settings):
    """Use the same HF temperature -> top-k -> top-p order as AffectGPT."""
    from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
    processors = LogitsProcessorList([TemperatureLogitsWarper(settings['temperature'])])
    if settings.get('top_k', 0) > 0:
        processors.append(TopKLogitsWarper(settings['top_k']))
    processors.append(TopPLogitsWarper(settings['top_p']))
    return processors


class ResidualMLP(nn.Module):
    """Atsuko's proj2(dropout(relu(proj1(x)))) + x."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.proj1, self.proj2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        return x + self.proj2(F.dropout(F.relu(self.proj1(x)), self.dropout, self.training))


class AuxiliaryPredictionHead(nn.Module):
    """Independent Atsuko residual branch, adapted to a language prediction head.

    W_vocab = frozen Omni LM head + independent B @ A for each branch. This
    preserves pretrained vocabulary knowledge without six copies of a dense head.
    The autoregressive Thinker backbone is shared, as part of the text-output
    adaptation; neither the residual MLP nor the trainable output head is shared.
    """
    def __init__(self, dim, hidden, vocabulary, dropout, rank, alpha):
        super().__init__()
        self.refine = ResidualMLP(dim, dropout)
        self.to_llm = nn.Linear(dim, hidden)
        self.output_a = nn.Linear(hidden, rank, bias=False)
        self.output_b = nn.Linear(rank, vocabulary, bias=False)
        nn.init.zeros_(self.output_b.weight)
        self.scale = alpha / rank

    def logits(self, hidden, base_head):
        dtype = self.output_a.weight.dtype
        delta = self.output_b(self.output_a(hidden.to(dtype))) * self.scale
        return base_head(hidden).float() + delta.float()


class SpectralFrontEnd(nn.Module):
    def __init__(self, hidden, cfg):
        super().__init__()
        self.cfg = cfg
        # Uploaded Atsuko consumes already constructed fixed-length features.
        # The existing Omni checkpoints instead use an explicit resampling adapter.
        self.input_mode = cfg.get('input_mode', 'omni_resample')
        if self.input_mode not in {'omni_resample', 'prealigned', 'zero_pad'}:
            raise ValueError(f'Unknown frontend input_mode: {self.input_mode}')
        self.sequence_lengths = spectral_sequence_lengths(cfg)
        self.sequence_overflow = cfg.get('sequence_overflow', 'error')
        if self.sequence_overflow not in {'error', 'truncate_right'}:
            raise ValueError(f'Unknown sequence_overflow: {self.sequence_overflow}')
        self.input_dims = [hidden] * 3 if isinstance(hidden, int) else list(hidden)
        if len(self.input_dims) != 3 or any(d < 1 for d in self.input_dims):
            raise ValueError('Expected positive text/video/audio input dimensions')
        dim = cfg['dim']
        if cfg['partition'] != 'equal_energy':
            raise ValueError('The aligned model requires fixed training-set equal-energy bands')
        self.adapters = nn.ModuleList([nn.Conv1d(d, dim, kernel_size=k, bias=False)
                                       for d, k in zip(self.input_dims, cfg['conv_kernel_sizes'])])
        self.shared_projection = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.spectra = nn.ModuleList([TrainingSetSpectralDecomposer(
            length=n - k + 1) for n, k in zip(self.sequence_lengths, cfg['conv_kernel_sizes'])])
        self.register_buffer('calibration_samples', torch.tensor(0, dtype=torch.long))
        self.band_routers = nn.ModuleList([BandRouter(dim, 3, cfg['temperature']) for _ in range(3)])
        self.attentions = nn.ModuleList([TransformerEncoder(
            embed_dim=dim, num_heads=cfg['heads'], layers=cfg['layers'],
            attn_dropout=dropout, relu_dropout=cfg['relu_dropout'],
            res_dropout=cfg['res_dropout'], embed_dropout=cfg['embed_dropout'],
            attn_mask=cfg['attn_mask']) for dropout in cfg['attn_dropout']])
        # The original positional encoder uses an uninitialized one-element
        # buffer only as a dtype/device marker. Give new MER checkpoints stable bytes.
        for name, buffer in self.attentions.named_buffers():
            if name.endswith('_float_tensor'):
                buffer.zero_()
        self.modality_router = router(3 * dim, 3, cfg['temperature'])

    def prepare_sequences(self, sequences):
        """Return text/video/audio batches without inventing temporal alignment.

        In prealigned mode callers supply the common time grid, including any
        padding. Shape validation cannot establish the origin of that alignment.
        zero_pad keeps native values and right-pads each modality independently;
        longer inputs require an explicit truncate_right policy, otherwise fail.
        Neither zero padding nor equal sequence length implies word alignment.
        Missing mode preserves the input adapter used by existing checkpoints.
        """
        if len(sequences) != 3:
            raise ValueError('Expected text, video and audio sequences, in that order')
        prepared, batch_size = [], None
        for name, values, channels, length in zip(('text', 'video', 'audio'), sequences,
                                                  self.input_dims, self.sequence_lengths):
            if len(values) == 0 or (batch_size is not None and len(values) != batch_size):
                raise ValueError('Modality batch sizes must agree and be nonempty')
            batch_size = len(values)
            items = []
            for value in values:
                if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != channels:
                    raise ValueError(f'{name}: expected nonempty [time, {channels}] features')
                if self.input_mode == 'prealigned':
                    if value.shape[0] != length:
                        raise ValueError(f'{name}: prealigned mode requires {length} supplied positions; '
                                         'provide aligned features with their original padding')
                    items.append(value.float())
                elif self.input_mode == 'zero_pad':
                    if len(value) > length and self.sequence_overflow == 'error':
                        raise ValueError(f'{name}: {len(value)} native positions exceed fixed length {length}; '
                                         'increase sequence_lengths or explicitly select truncate_right')
                    kept = value[:length].float()
                    items.append(F.pad(kept, (0, 0, 0, length-len(kept))))
                else:
                    items.append(resample_sequence(value, length))
            prepared.append(torch.stack(items))
        return prepared

    def project_sequences(self, sequences):
        result = []
        for index, (values, projection) in enumerate(zip(self.prepare_sequences(sequences), self.adapters)):
            seq = values.transpose(1, 2)
            if index == 0:
                seq = F.dropout(seq, p=self.cfg['text_dropout'], training=self.training)
            result.append(self.shared_projection(projection(seq)).transpose(1, 2))
        return result

    def forward(self, sequences):
        bands = [torch.stack(spectral.split_bands(seq), dim=1) for seq, spectral in
                 zip(self.project_sequences(sequences), self.spectra)]
        band_weights = [band_router(values.mean(dim=2)) for values, band_router in zip(bands, self.band_routers)]
        # Preserve Atsuko's RNG order: all band masks precede all attention dropouts.
        if self.training:
            band_weights = [apply_band_masking(w, self.cfg['band_mask_rate']) for w in band_weights]
        sequences = [(values * w[:, :, None, None]).sum(1) for values, w in zip(bands, band_weights)]
        pooled = [seq.mean(1) for seq in sequences]
        attended = []
        for seq, attention in zip(sequences, self.attentions):
            result = attention(seq.transpose(0, 1))
            attended.append((result[0] if isinstance(result, tuple) else result)[-1])
        attended, pooled = torch.stack(attended, dim=1), torch.stack(pooled, dim=1)
        decision = pooled if self.cfg['router_input'] == 'pre_attn' else attended
        weights = self.modality_router(decision.flatten(1))
        return dict(modalities=attended, modality_weights=weights,
                    band_weights=torch.stack(band_weights, dim=1),
                    fusion=(attended * weights[:, :, None]).sum(1))


class MEROmni(nn.Module):
    def __init__(self, config, tokenizer, *, components=None):
        super().__init__()
        self.config, self.tokenizer = config, tokenizer
        options = dict(config['omni'], generative=True)
        self.components = components if components is not None else OmniComponents.from_pretrained(options)
        if not hasattr(self.components.thinker, 'lm_head'):
            raise ValueError('Generative MER requires the pretrained vocabulary head')
        self.frontend = SpectralFrontEnd(self.components.hidden_size, config['spectral'])
        dim, hidden = config['spectral']['dim'], self.components.hidden_size
        self.fusion_in, self.fusion_out = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        def auxiliary_head():
            return AuxiliaryPredictionHead(dim, hidden, self.components.thinker.lm_head.out_features,
                config['spectral']['output_dropout'], config['omni']['lora_r'], config['omni']['lora_alpha'])
        self.unimodal_heads = nn.ModuleList([auxiliary_head() for _ in range(3)])
        self.bimodal_heads = nn.ModuleList([auxiliary_head() for _ in range(3)])
        self.ce_only = config['mcm'].get('loss_mode') == 'ce_only'
        if self.ce_only:
            # Keep construction/RNG identical to the full-loss experiment, but
            # exclude unused auxiliary heads and the distillation adapter from
            # optimization and trainable-only checkpoints. Forward skips them.
            self.unimodal_heads.requires_grad_(False)
            self.bimodal_heads.requires_grad_(False)
            self.fusion_out.requires_grad_(False)
        self.calibration_info = None

    def _amp(self):
        weight = self.components.thinker.get_input_embeddings().weight
        return torch.autocast(device_type=weight.device.type, dtype=torch.bfloat16) if weight.dtype == torch.bfloat16 else nullcontext()

    def _ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _embed(self, ids):
        embedding = self.components.thinker.get_input_embeddings()
        return embedding(torch.tensor(ids, device=embedding.weight.device, dtype=torch.long))

    def _prompt_segments(self, token, question):
        # Only the routed feature enters the main fusion layer. Raw subtitle and
        # unweighted audio/vision tokens would bypass Atsuko's modality routing.
        return ['###Human: The multimodal information is: <Multi>', token, '</Multi>. '
                'Now, please answer my question based on all the provided information. '
                + question + ' ###Assistant: ']

    def validate_training_text(self, question, answer):
        prompt_length = sum(len(self._ids(x)) if isinstance(x, str) else 1
                            for x in self._prompt_segments(None, question))
        target_length = len(self._ids(answer + '###'))
        # AffectGPT's limit is checked before the collator appends BOS/EOS.
        if prompt_length + target_length > self.config['train']['max_length']:
            raise SequenceTooLong('Combined prompt and answer exceed the AffectGPT training max_length')
        return target_length + 1

    def _assemble(self, tokens, questions, answers=None):
        embeddings, labels = [], []
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError('Tokenizer must define eos_token_id')
        for i, question in enumerate(questions):
            segments = self._prompt_segments(tokens[i:i+1], question)
            pieces = [self._embed([bos])] if bos is not None else []
            pieces += [self._embed(self._ids(x)) if isinstance(x, str) else x for x in segments]
            prompt = torch.cat(pieces)
            lab = torch.full((len(prompt),), -100, device=prompt.device, dtype=torch.long)
            if answers is not None:
                self.validate_training_text(question, answers[i])
                target = self._ids(answers[i] + '###') + [eos]
                prompt = torch.cat((prompt, self._embed(target)))
                lab = torch.cat((lab, torch.tensor(target, device=lab.device)))
            embeddings.append(prompt)
            labels.append(lab)
        mask = pad_sequence([torch.ones(len(x), device=x.device, dtype=torch.long) for x in embeddings], batch_first=True)
        return pad_sequence(embeddings, batch_first=True), mask, pad_sequence(labels, batch_first=True, padding_value=-100)

    def _nll(self, embeddings, mask, labels, *, head=None, return_feature=False):
        thinker = self.components.thinker
        positions = (mask.cumsum(-1) - 1).clamp_min(0)[None].expand(3, -1, -1)
        with self._amp():
            hidden = thinker.model(inputs_embeds=embeddings.to(thinker.get_input_embeddings().weight.dtype),
                                   attention_mask=mask, position_ids=positions, use_cache=False,
                                   return_dict=True).last_hidden_state
            losses = []
            for h, target in zip(hidden, labels):
                target = target[1:]
                valid = target != -100
                if not valid.any():
                    raise ValueError('No supervised answer tokens')
                # Do not allocate vocabulary logits for masked prompt/padding tokens.
                answer_hidden = h[:-1][valid]
                logits = thinker.lm_head(answer_hidden) if head is None else head.logits(answer_hidden, thinker.lm_head)
                losses.append(F.cross_entropy(logits.float(), target[valid]))
        losses = torch.stack(losses)
        if return_feature:
            # Last prompt position is answer-free by causality. Do not pool
            # teacher-forced answer states into the distilled fusion feature.
            positions = (labels != -100).long().argmax(-1) - 1
            context = hidden[torch.arange(len(hidden), device=hidden.device), positions]
            feature = self.fusion_out(context.to(self.fusion_out.weight.dtype))
            return losses, feature
        return losses

    def branch_features(self, modalities):
        refined = torch.stack([head.refine(modalities[:, index])
                               for index, head in enumerate(self.unimodal_heads)], dim=1)
        # Atsuko Eq. O_{not m}: unweighted SUM of the remaining H, independent heads.
        bimodal = [head.refine(modalities[:, [j for j in range(3) if j != index]].sum(1))
                   for index, head in enumerate(self.bimodal_heads)]
        return refined, bimodal

    def forward(self, *, omni_inputs, texts, questions, answers, **unused):
        if answers is None:
            raise ValueError('Training requires assistant answers')
        features = self.frontend(self.components.encode(omni_inputs))
        weights = features['modality_weights']
        assembled = self._assemble(self.fusion_in(features['fusion']), questions, answers)
        if self.ce_only:
            full = self._nll(*assembled)
            counts = (assembled[2] != -100).sum(-1)
            main_nll = (full * counts).sum() / counts.sum()
            return dict(loss=main_nll, nll=main_nll)
        full, student = self._nll(*assembled, return_feature=True)
        counts = (assembled[2] != -100).sum(-1)
        main_nll = (full * counts).sum() / counts.sum()
        mcm = self.config['mcm']
        refined, bimodal = self.branch_features(features['modalities'])
        unimodal = [self._nll(*self._assemble(head.to_llm(refined[:, index]), questions, answers), head=head)
                    for index, head in enumerate(self.unimodal_heads)]
        auxiliary, divergence = full.new_zeros(()), full.new_zeros(())
        if mcm['enabled']:
            omitted = []
            for value, head in zip(bimodal, self.bimodal_heads):
                omitted.append(self._nll(*self._assemble(head.to_llm(value), questions, answers), head=head))
            partial_losses = torch.stack(omitted, dim=1)
            auxiliary = partial_losses.mean()
            divergence = complementarity(full, partial_losses, weights, mcm['temperature'])
        terms = dict(nll=main_nll, unimodal_nll=torch.stack(unimodal, dim=1).mean(),
                     auxiliary_nll=auxiliary, router_kl=divergence,
                     distillation=feature_distillation(student, refined, weights),
                     modality_entropy=entropy_balance(weights),
                     band_entropy=entropy_balance(features['band_weights']))
        return dict(loss=combine_losses(terms, mcm), **terms)

    @torch.no_grad()
    def generate(self, *, omni_inputs, texts, questions, **unused):
        self.eval()
        features = self.frontend(self.components.encode(omni_inputs))
        embeddings, masks, _ = self._assemble(self.fusion_in(features['fusion']), questions)
        results = []
        settings = self.config['generation']
        processors = generation_processors(settings) if settings['do_sample'] else None
        eos_ids = set(settings.get('eos_token_ids', [self.tokenizer.eos_token_id]))
        eos_ids.add(self.tokenizer.eos_token_id)
        thinker = self.components.thinker
        for embedding, mask in zip(embeddings, masks):
            length = int(mask.sum())
            remaining = min(settings['max_new_tokens'], settings['max_length'] - length)
            if remaining < 1:
                raise ValueError('Generation context has no room for an answer')
            x = embedding[:length][None].to(thinker.get_input_embeddings().weight.dtype)
            cache, generated = None, []
            with self._amp():
                for step in range(remaining):
                    start = 0 if cache is None else length + step - 1
                    positions = torch.arange(start, start + x.shape[1], device=x.device)[None, None].expand(3, 1, -1)
                    output = thinker.model(inputs_embeds=x, position_ids=positions, past_key_values=cache,
                                           attention_mask=torch.ones(1, length + step, device=x.device, dtype=torch.long),
                                           use_cache=True, return_dict=True)
                    cache = output.past_key_values
                    logits = thinker.lm_head(output.last_hidden_state[:, -1]).float()[0]
                    if settings['do_sample']:
                        scores = processors(None, logits[None])
                        token = torch.multinomial(scores.softmax(-1), 1).item()
                    else:
                        token = logits.argmax().item()
                    if token in eos_ids:
                        break
                    generated.append(token)
                    decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                    if '###' in decoded:
                        break
                    x = self._embed([token])[None]
            answer = self.tokenizer.decode(generated, skip_special_tokens=True).split('###')[0]
            results.append(answer.split('Assistant:')[-1].strip())
        return results
