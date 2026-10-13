"""Atsuko objectives (paper Eqs. 8–14), independent of the output task."""
import torch.nn.functional as F


def entropy_balance(probs):
    # Same N * mean(sum(p log p)) as Atsuko/trains/utils/functions.py.
    probs = probs.clamp_min(1e-9)
    return probs.shape[-1] * (probs * probs.log()).sum(-1).mean()


def feature_distillation(student, refined_modalities, weights):
    teacher = (refined_modalities * weights[:, :, None]).sum(1).detach()
    return (student.softmax(-1) - teacher.softmax(-1)).square().mean()


def complementarity(full, partial, weights, temperature):
    importance = ((partial - full[:, None]).detach() / temperature).softmax(-1)
    return F.kl_div(weights.clamp_min(1e-8).log(), importance, reduction='batchmean')


def combine_losses(terms, settings, *, token_weight=1.0, sample_weight=1.0):
    """Main language CE is token-mean; Atsuko auxiliary/regularizers are sample-mean.

    Separate denominators make gradient accumulation agree with a physical batch.
    """
    if settings.get('loss_mode') == 'ce_only':
        return token_weight * terms['nll']
    auxiliary = (settings['unimodal_weight'] * terms['unimodal_nll']
                 + settings['auxiliary_weight'] * terms['auxiliary_nll']
                 + settings['router_weight'] * terms['router_kl']
                 + settings['distillation_weight'] * terms['distillation']
                 + settings['modality_entropy_weight'] * terms['modality_entropy']
                 + settings['band_entropy_weight'] * terms['band_entropy'])
    return token_weight * terms['nll'] + sample_weight * auxiliary
