"""Tiny offline fixtures; never used by the application."""
import torch
from mer_omni.encoders.omni import prepare_thinker_config
from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
HAS_OMNI = True


def tiny_thinker():
    config = Qwen2_5OmniThinkerConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}),
        audio_config=dict(num_mel_bins=128, d_model=32, output_dim=32, encoder_layers=1,
                          encoder_attention_heads=4, encoder_ffn_dim=64, n_window=4,
                          max_source_positions=32),
        vision_config=dict(hidden_size=32, out_hidden_size=32, depth=1, intermediate_size=64,
                           num_heads=4, fullatt_block_indexes=[0], window_size=56),
        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = 'sdpa'
    return Qwen2_5OmniThinkerForConditionalGeneration(prepare_thinker_config(config))

def native_batch():
    return {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5], [0, 0, 6, 7, 8]]),
        'text_attention_mask': torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        'input_features': torch.randn(2, 128, 12),
        'feature_attention_mask': torch.tensor([[1] * 8 + [0] * 4, [1] * 12]),
        'pixel_values_videos': torch.randn(48, 3 * 2 * 14 * 14),
        'video_grid_thw': torch.tensor([[2, 4, 4], [1, 4, 4]]),
    }
