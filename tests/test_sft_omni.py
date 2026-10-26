"""Regression checks for shared native Qwen input helpers; no model load."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from mer_omni.sft import omni as native


class NativeSFTDiagnosticTests(unittest.TestCase):
    def test_only_answer_and_eos_supervised(self):
        prompt = torch.tensor([[1, 80, 80, 90, 90, 2]])
        full = torch.tensor([[1, 80, 80, 90, 90, 2, 11, 12, 99, 10]])
        labels, start, stop = native.assistant_labels(full, prompt, 99)
        self.assertEqual((start, stop), (6, 9))
        self.assertEqual(labels.tolist(), [[-100]*6 + [11, 12, 99, -100]])

    def test_mismatched_multimodal_prefix_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exact inference prompt prefix'):
            native.assistant_labels(torch.tensor([[1, 80, 80, 2, 11, 99]]),
                                    torch.tensor([[1, 80, 2]]), 99)

    def test_missing_duplicate_or_empty_end_rejected(self):
        for full in ([1, 2, 11], [1, 2, 11, 99, 99], [1, 2, 99]):
            with self.subTest(full=full), self.assertRaises(ValueError):
                native.assistant_labels(torch.tensor([full]), torch.tensor([[1, 2]]), 99)

    def test_fixed_selection_and_exact_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'clip').touch()
            rows = [dict(id=f'sample{i}', dataset='MER-Caption+', split='train',
                         openset=['happy', 'hopeful'], text='original subtitle',
                         audio='clip', video='clip') for i in range(50)]
            manifest = root/'train.jsonl'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            cfg = dict(train_manifest=str(manifest), sample_seed=42, sample_count=32, question='question')
            first, second = native.select_rows(cfg), native.select_rows(cfg)
            self.assertEqual(first, second)
            self.assertEqual(len({r['id'] for r in first}), 32)
            self.assertTrue(all(r['answer'] == "The character's emotional state is happy, hopeful." for r in first))
            self.assertTrue(all(r['text'] == 'original subtitle' for r in first))
            for r in rows:
                r['split'] = 'test'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'training examples'):
                native.select_rows(cfg)


if __name__ == '__main__':
    unittest.main()
