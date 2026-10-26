"""B2: B's single fusion token with output LayerNorm and embedding-scale init."""
from collections import OrderedDict

import torch
from torch import nn

from .native_atsuko import NativeAtsuko


@torch.no_grad()
def embedding_mean_row_rms(weight, chunk_rows=4096):
    """Mean per-token RMS of the frozen embedding table, without a full FP32 copy."""
    total = torch.zeros((), device=weight.device, dtype=torch.float64)
    for block in weight.split(chunk_rows, dim=0):
        total += block.float().square().mean(dim=-1).sqrt().double().sum()
    rms = float(total / weight.shape[0])
    if not 0 < rms < float('inf'):
        raise ValueError('Invalid frozen embedding RMS')
    return rms


class NormalizedNativeAtsuko(NativeAtsuko):
    def __init__(self, thinker, spectral, token_id):
        # Construct exactly B's frontend/projection first; no changed random draws.
        super().__init__(thinker, spectral, token_id)
        hidden = thinker.config.text_config.hidden_size
        rms = embedding_mean_row_rms(thinker.get_input_embeddings().weight)
        norm = nn.LayerNorm(hidden, eps=1e-5)
        with torch.no_grad():
            norm.weight.fill_(rms)
            norm.bias.zero_()
        self.fusion_in = nn.Sequential(OrderedDict([
            ('projection', self.fusion_in), ('out_norm', norm)]))
        # A descriptive scalar, not a new model parameter or calibration statistic.
        self.initial_embedding_rms = rms
