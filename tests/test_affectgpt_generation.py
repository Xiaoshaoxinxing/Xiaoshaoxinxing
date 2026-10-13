"""Regression checks for the inherited AffectGPT Qwen decoding defaults."""
import unittest

import numpy as np
import torch

from mer_omni.model import generation_processors


class AffectGPTSamplingTests(unittest.TestCase):
    def test_top_k_is_applied_before_nucleus_filtering(self):
        logits = np.linspace(-1, 1, 64, dtype=np.float32)
        settings = dict(temperature=0.7, top_k=20, top_p=0.9)
        # Independent probability calculation in descending score order.
        candidates = np.argsort(-logits)[:20]
        scores = logits[candidates].astype(np.float64) / 0.7
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        retained = int(np.searchsorted(probabilities.cumsum(), 0.9)) + 1
        expected = np.zeros(64)
        expected[candidates[:retained]] = probabilities[:retained] / probabilities[:retained].sum()
        actual = generation_processors(settings)(None, torch.tensor(logits)[None]).softmax(-1)[0]
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7, rtol=1e-6)
        self.assertTrue(torch.all(actual[:44] == 0))

    def test_top_k_keeps_tied_boundary_logits_as_transformers_does(self):
        scores = torch.tensor([[0.0, 1.0, 1.0, 2.0]])
        filtered = generation_processors(dict(temperature=1.0, top_k=2, top_p=1.0))(None, scores)
        self.assertEqual(torch.isfinite(filtered).tolist(), [[False, True, True, True]])


if __name__ == '__main__':
    unittest.main()
