"""B input isolation, gradient connectivity and native media preservation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from mer_omni.native_atsuko import NativeAtsuko, insert_fusion_slot
from mer_omni.calibration import set_calibration

ROOT = Path(__file__).resolve().parents[1]


class Thinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8), audio_token_id=3, video_token_id=4)
        self.embedding = nn.Embedding(30,8)
        self.embedding.requires_grad_(False)
        self.audio_tower, self.visual = nn.Identity(), nn.Identity()
        self.calls = [0,0]

    def get_input_embeddings(self):
        return self.embedding

    def get_audio_features(self, values, **kwargs):
        self.calls[0] += 1
        return values[0]

    def get_video_features(self, values, grid):
        self.calls[1] += 1
        return values


class NativeAtsukoTests(unittest.TestCase):
    def fixture(self):
        cfg = json.loads((ROOT/'tests/fixtures/config/mer_omni_av_tokens_frames16.json').read_text())['spectral']
        cfg.update(dim=8, heads=2, layers=1, sequence_lengths=[8,8,8], conv_kernel_sizes=[1,1,1])
        model = NativeAtsuko(Thinker(), cfg, 10)
        set_calibration(model.frontend, [torch.ones(8)]*3, 31327)
        model.eval()
        batch = dict(input_ids=torch.tensor([[1,3,3,4,4,10,2]]), attention_mask=torch.ones(1,7,dtype=torch.long),
            input_features=torch.randn(1,2,8), pixel_values_videos=torch.randn(2,8),
            feature_attention_mask=torch.ones(1,8), video_grid_thw=torch.tensor([[1,2,4]]),
            bypass_subtitle_ids=torch.tensor([[7,8]]))
        return model,batch

    def test_slot_is_masked_and_removal_recovers_exact_A(self):
        batch = dict(input_ids=torch.tensor([[1,9,5,6,9,7,8,9]]), attention_mask=torch.ones(1,8,dtype=torch.long),
                     labels=torch.tensor([[-100,-100,-100,-100,-100,-100,8,9]]))
        changed, pos = insert_fusion_slot(batch,10,9)
        self.assertEqual(pos,4)
        self.assertEqual(int(changed['labels'][0,pos]),-100)
        for k in batch:
            torch.testing.assert_close(torch.cat((changed[k][:,:pos],changed[k][:,pos+1:]),1),batch[k],rtol=0,atol=0)
        torch.testing.assert_close(changed['labels'][changed['labels']!=-100],batch['labels'][batch['labels']!=-100])

    def test_reserved_token_collision_rejected(self):
        with self.assertRaisesRegex(ValueError,'reserved'):
            insert_fusion_slot(dict(input_ids=torch.tensor([[10,9,9]])),10,9)

    def test_preserves_all_native_media_and_encodes_each_once(self):
        model,batch=self.fixture()
        result=model.prepare_embeddings(batch)
        torch.testing.assert_close(result[0,1:3],batch['input_features'][0],rtol=0,atol=0)
        torch.testing.assert_close(result[0,3:5],batch['pixel_values_videos'],rtol=0,atol=0)
        for position in (0,6):
            torch.testing.assert_close(result[:,position],model.thinker.embedding(batch['input_ids'][:,position]),rtol=0,atol=0)
        self.assertEqual(model.thinker.calls,[1,1])

    def test_fusion_gradient_reaches_frontend_without_encoder_gradient(self):
        model,batch=self.fixture()
        result=model.prepare_embeddings(batch)
        result[:,5].square().mean().backward()
        self.assertGreater(float(model.fusion_in.weight.grad.abs().sum()),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.frontend.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in model.thinker.parameters()))

    def test_original_inputs_not_modified_in_place(self):
        model,batch=self.fixture()
        saved={k:v.clone() for k,v in batch.items()}
        model.prepare_embeddings(batch)
        for k in batch:
            torch.testing.assert_close(batch[k],saved[k],rtol=0,atol=0)

    def test_bad_media_count_fails_before_scatter(self):
        model,batch=self.fixture()
        batch['input_features']=torch.randn(1,1,8)
        with self.assertRaisesRegex(ValueError,'media count'):
            model.prepare_embeddings(batch)

    def test_frontend_initialization_does_not_advance_global_rng(self):
        model,_=self.fixture()
        rng=torch.get_rng_state().clone()
        NativeAtsuko(model.thinker, model.frontend.cfg,10)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':
    unittest.main()
