"""Inference truncation preserves calibrated shapes, valid inputs and training policy."""
import copy
import unittest

import torch

from mer_omni.calibration import set_calibration
from mer_omni.inference_inputs import InferenceInputAudit
from mer_omni.model import SpectralFrontEnd
from test_mer_omni import config


class InferenceInputPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = copy.deepcopy(config()['spectral'])
        self.cfg.update(input_mode='zero_pad', sequence_lengths=[6, 7, 8],
                        conv_kernel_sizes=[1, 1, 1], sequence_overflow='error')
        self.frontend = SpectralFrontEnd(32, self.cfg).eval()
        set_calibration(self.frontend, [[1.0]*n for n in (6, 7, 8)], 10)

    def test_long_input_matches_explicit_prefix_and_restores_strict_mode(self):
        values = [[torch.randn(n, 32)] for n in (9, 4, 12)]
        cropped = [[x[0][:limit]] for x, limit in zip(values, (6, 7, 8))]
        before = {name: x.clone() for name, x in self.frontend.state_dict().items()}
        with torch.no_grad():
            expected = self.frontend(cropped)
            with InferenceInputAudit(self.frontend, 'truncate_right') as audit:
                audit.begin_sample('long-example')
                actual = self.frontend(values)
                report = audit.report()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(report['affected'], [dict(id='long-example', native_lengths=[9,4,12],
            retained_lengths=[6,4,8], discarded_positions=[3,0,4])])
        self.assertEqual(report['processed_samples'], 1)
        self.assertEqual(self.cfg['sequence_overflow'], 'error')
        self.assertEqual(self.frontend.sequence_overflow, 'error')
        for name, value in self.frontend.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'exceed fixed length'):
            self.frontend(values)

    def test_valid_inputs_are_unchanged_and_not_marked_truncated(self):
        values = [[torch.randn(n,32)] for n in (5,4,7)]
        with torch.no_grad():
            expected = self.frontend(values)
            with InferenceInputAudit(self.frontend, 'truncate_right') as audit:
                audit.begin_sample('valid-example')
                actual = self.frontend(values)
                report = audit.report()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(report['affected_samples'], 0)

    def test_default_still_rejects_overflow_and_removes_hook_after_failure(self):
        values = [[torch.randn(n,32)] for n in (9,4,7)]
        with self.assertRaisesRegex(ValueError, 'exceed fixed length'):
            with InferenceInputAudit(self.frontend) as audit:
                audit.begin_sample('invalid-example')
                self.frontend(values)
        self.assertFalse(self.frontend._forward_pre_hooks)
        self.assertEqual(self.frontend.sequence_overflow, 'error')


if __name__ == '__main__':
    unittest.main()
