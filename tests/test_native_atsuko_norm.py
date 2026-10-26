"""B2 output scale, B initialization isolation, and native-media regression checks."""
import copy
import unittest

import torch

from test_native_omni_atsuko import NativeAtsukoTests
from mer_omni.native_atsuko import NativeAtsuko
from mer_omni.native_atsuko_norm import NormalizedNativeAtsuko, embedding_mean_row_rms
from mer_omni.calibration import set_calibration
from mer_omni.training import build_optimizer


class NormalizedNativeAtsukoTests(NativeAtsukoTests):
    def fixture(self):
        base, batch = super().fixture()
        model = NormalizedNativeAtsuko(base.thinker, base.frontend.cfg, 10)
        set_calibration(model.frontend, [torch.ones(8)] * 3, 31327)
        model.eval()
        return model, batch

    def test_fusion_gradient_reaches_frontend_without_encoder_gradient(self):
        model, batch = self.fixture()
        result = model.prepare_embeddings(batch)
        # Nonuniform upstream gradient avoids the nearly constant squared norm of LN.
        (result[:, 5] * torch.arange(8, dtype=result.dtype)).sum().backward()
        for module in (model.frontend, model.fusion_in.projection, model.fusion_in.out_norm):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()
                                   if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in model.thinker.parameters()))

    def test_new_interface_preserves_B_initial_parameters_and_rng(self):
        model, _ = self.fixture()
        rng = torch.get_rng_state().clone()
        baseline = NativeAtsuko(model.thinker, model.frontend.cfg, 10)
        normalized = NormalizedNativeAtsuko(model.thinker, model.frontend.cfg, 10)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, param in baseline.frontend.named_parameters():
            torch.testing.assert_close(param, dict(normalized.frontend.named_parameters())[name], rtol=0, atol=0)
        for name, param in baseline.fusion_in.named_parameters():
            torch.testing.assert_close(param, dict(normalized.fusion_in.projection.named_parameters())[name], rtol=0, atol=0)
        extra = sum(p.numel() for p in normalized.parameters()) - sum(p.numel() for p in baseline.parameters())
        self.assertEqual(extra, 2 * model.thinker.config.text_config.hidden_size)

    def test_only_fusion_embedding_changes_and_scale_matches_initial_target(self):
        model, batch = self.fixture()
        raw = copy.deepcopy(model)
        raw.fusion_in = raw.fusion_in.projection
        expected = raw.prepare_embeddings(batch)
        actual = model.prepare_embeddings(batch)
        mask = batch['input_ids'] != model.fusion_token_id
        torch.testing.assert_close(actual[mask], expected[mask], rtol=0, atol=0)
        fused = actual[~mask]
        self.assertLess(abs(float(fused.mean())), 1e-6)
        self.assertAlmostEqual(float(fused.square().mean().sqrt()), model.initial_embedding_rms, delta=1e-3)

    def test_chunked_embedding_scale_equals_direct_computation(self):
        weight = torch.randn(31, 8)
        expected = float(weight.square().mean(-1).sqrt().double().mean())
        self.assertAlmostEqual(embedding_mean_row_rms(weight, chunk_rows=7), expected, places=10)

    def test_norm_uses_frontend_lr_without_weight_decay(self):
        model, _ = self.fixture()
        optimizer = build_optimizer(model, dict(lr=2e-5, lr_frontend=1e-4, weight_decay=.05))
        for param in model.fusion_in.out_norm.parameters():
            groups = [g for g in optimizer.param_groups if any(p is param for p in g['params'])]
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]['base_lr'], 1e-4)
            self.assertEqual(groups[0]['weight_decay'], 0.)


if __name__ == '__main__':
    unittest.main()
