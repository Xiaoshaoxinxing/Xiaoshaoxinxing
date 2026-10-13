"""Exercise complete data passes, partial accumulation and exact resumption."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from mer_omni.model import SequenceTooLong
from mer_omni.objectives import combine_losses
from mer_omni.protocol import ROOT, load_config
from mer_omni.training import (adapter_state, epoch_settings, read_checkpoint,
                               seed_all, train, training_row)


class LossModel(torch.nn.Module):
    """Small differentiable objective with unequal answer lengths and sample losses."""
    def __init__(self, cfg):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
        self.frontend = torch.nn.Module()
        self.frontend.register_buffer('calibration_samples', torch.tensor(1))
        self.components = SimpleNamespace(load_report={})
        self.calibration_info = {}
        self.cfg = cfg

    def _ids(self, text):
        return text.split()

    def validate_training_text(self, question, answer):
        if answer == 'too_long':
            raise SequenceTooLong('fixture length limit')

    def forward(self, *, rows, omni_inputs):
        x = torch.tensor([int(r['id']) + 1 for r in rows], dtype=torch.float64)
        tokens = torch.tensor([len(self._ids(r['answer'] + '###')) + 1 for r in rows])
        values = (self.weight * x - 1).square()
        terms = dict(nll=(values * tokens).sum() / tokens.sum(),
                     unimodal_nll=values.mean(), auxiliary_nll=values.mean() * .3,
                     router_kl=values.mean() * .1, distillation=values.mean() * .2,
                     modality_entropy=values.mean() * -.1, band_entropy=values.mean() * -.2)
        return dict(terms, loss=combine_losses(terms, self.cfg['mcm']))


def collate(rows):
    return dict(rows=rows, omni_inputs={})


class FullDatasetEpochTests(unittest.TestCase):
    def setup_run(self, root, size=7):
        cfg = load_config(ROOT / 'config/mer_omni_atsuko_padding.json')
        cfg['train'].update(batch_size=2, gradient_accumulation=2,
                            warmup_steps=0, lr=.01, min_lr=.01)
        manifest = root / 'manifest.jsonl'
        manifest.write_text('synthetic fixture\n')
        cfg['train_manifest'] = str(manifest)
        rows = [dict(id=str(i), question='emotion?', answer='word ' * (i % 3 + 1)) for i in range(size)]
        return cfg, rows

    def run_train(self, model, rows, cfg, path, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(model, rows, collate, cfg, path, torch.device('cpu'), **kwargs)

    def test_full_pass_and_partial_microbatch_match_physical_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, rows = self.setup_run(root)
            model = LossModel(cfg)
            physical_cfg = copy.deepcopy(cfg)
            physical_cfg['train'].update(batch_size=4, gradient_accumulation=1)
            physical = LossModel(physical_cfg)
            checkpoint = self.run_train(model, rows, cfg, root / 'accumulated')
            self.run_train(physical, rows, physical_cfg, root / 'physical')
            torch.testing.assert_close(model.weight, physical.weight, rtol=1e-12, atol=1e-12)
            log = [json.loads(s) for s in (root / 'accumulated/train.jsonl').read_text().splitlines()]
            self.assertEqual([r['samples_this_step'] for r in log], [4, 3])
            self.assertCountEqual([i for r in log for i in r['sample_ids']], [str(i) for i in range(7)])
            self.assertEqual(read_checkpoint(checkpoint)['stream']['position'], 7)
            coverage = json.loads((root / 'accumulated/coverage_epoch_0001.json').read_text())
            self.assertTrue(coverage['complete'])
            self.assertEqual(coverage['unique_samples'], 7)

    def test_resume_before_single_sample_tail_and_epoch_boundary(self):
        for stop in (1, 2):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, rows = self.setup_run(root, size=5)
                cfg['train']['epochs'] = 2
                seed_all(19)
                full = LossModel(cfg)
                self.run_train(full, rows, cfg, root / 'full')
                seed_all(19)
                split = LossModel(cfg)
                checkpoint = self.run_train(split, rows, cfg, root / 'split', max_steps=stop)
                first_coverage = json.loads((root / 'split/coverage_epoch_0001.json').read_text())
                self.assertEqual(first_coverage['complete'], stop == 2)
                resumed = LossModel(cfg)
                self.run_train(resumed, rows, cfg, root / 'split', resume=checkpoint)
                for name, value in adapter_state(full).items():
                    torch.testing.assert_close(value, adapter_state(resumed)[name], rtol=0, atol=0)
                logs = [json.loads(s) for s in (root / 'split/train.jsonl').read_text().splitlines()]
                self.assertEqual([r['samples_this_step'] for r in logs], [4, 1, 4, 1])
                for epoch in (1, 2):
                    seen = [i for r in logs if r['epoch'] == epoch for i in r['sample_ids']]
                    self.assertCountEqual(seen, [str(i) for i in range(5)])

    def test_full_pass_forbids_silent_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, rows = self.setup_run(Path(tmp))
            rows[0]['answer'] = 'too_long'
            with self.assertRaisesRegex(SequenceTooLong, 'forbid sample replacement'):
                training_row(rows, 0, LossModel(cfg), allow_replacement=False)

    def test_dataset_smaller_than_batch_and_exact_division(self):
        for size in (1, 4):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, rows = self.setup_run(root, size)
                checkpoint = self.run_train(LossModel(cfg), rows, cfg, root / 'run')
                saved = read_checkpoint(checkpoint)
                self.assertEqual(saved['step'], 1)
                self.assertEqual(saved['stream']['position'], size)
                self.assertCountEqual(saved['stream']['order'], range(size))

    def test_real_config_budget_and_legacy_budget(self):
        new = load_config(ROOT / 'config/mer_omni_atsuko_padding.json')
        settings = epoch_settings(new['train'], 31327)
        self.assertEqual(settings['epochs'], 1)
        self.assertEqual(settings['steps_per_epoch'], 10443)
        old = load_config(ROOT / 'tests/fixtures/legacy_full_loss.json')
        self.assertEqual(epoch_settings(old['train'], 31327), old['train'])


if __name__ == '__main__':
    unittest.main()
