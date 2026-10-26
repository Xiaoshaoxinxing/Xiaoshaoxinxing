"""Main language CE only: numerical loss, gradient path and actual trainer resume."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from mer_omni.calibration import model_signature
from mer_omni.data import MERDataset
from mer_omni.objectives import combine_losses
from mer_omni.protocol import ROOT, load_config
from mer_omni.training import (adapter_state, build_optimizer, read_checkpoint,
                               seed_all, train)
from test_mer_omni import config, make_model, batch
from test_atsuko_alignment import split_batch


def ce_config():
    cfg = config()
    cfg['mcm'] = load_config(ROOT / 'tests/fixtures/config/mer_omni_atsuko_padding.json')['mcm']
    return cfg


class CEOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_one_language_forward_matches_direct_shifted_ce(self):
        seed_all(42)
        model = make_model(ce_config()).eval()
        b = batch()
        with ExitStack() as stack:
            for name in ('branch_features',):
                stack.enter_context(patch.object(model, name, side_effect=AssertionError('auxiliary branch executed')))
            stack.enter_context(patch.object(model.fusion_out, 'forward', side_effect=AssertionError('distillation executed')))
            for name in ('feature_distillation', 'entropy_balance', 'complementarity'):
                stack.enter_context(patch('mer_omni.model.' + name, side_effect=AssertionError('auxiliary loss executed')))
            language = stack.enter_context(patch.object(model.components.thinker.model, 'forward',
                                                        wraps=model.components.thinker.model.forward))
            result = model(**b)
            self.assertEqual(language.call_count, 1)
        self.assertEqual(set(result), {'loss', 'nll'})
        self.assertIs(result['loss'], result['nll'])
        features = model.frontend(model.components.encode(b['omni_inputs']))
        x, mask, labels = model._assemble(model.fusion_in(features['fusion']), b['questions'], b['answers'])
        pos = (mask.cumsum(-1) - 1).clamp_min(0)[None].expand(3, -1, -1)
        h = model.components.thinker.model(inputs_embeds=x, attention_mask=mask,
            position_ids=pos, use_cache=False).last_hidden_state
        logits = model.components.thinker.lm_head(h[:, :-1]).float()
        expected = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(result['loss'], expected)
        result['loss'].backward()
        for parameter in [model.fusion_in.weight, model.frontend.modality_router.l2.weight,
                          model.frontend.band_routers[0].fc2.weight]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if 'lora_' in n))
        optimized = {id(p) for group in build_optimizer(model, model.config['train']).param_groups
                     for p in group['params']}
        for module in [model.unimodal_heads, model.bimodal_heads, model.fusion_out]:
            self.assertTrue(all(not p.requires_grad and p.grad is None and id(p) not in optimized
                                for p in module.parameters()))

    def test_token_weighted_accumulation_matches_full_batch(self):
        seed_all(4)
        cfg = ce_config()
        model = make_model(cfg).eval()
        b = batch()
        whole = model(**b)['loss']
        whole.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        counts = [len(model._ids(a + '###')) + 1 for a in b['answers']]
        accumulated = 0.
        for index in range(2):
            result = model(**split_batch(b, index))
            loss = combine_losses(result, cfg['mcm'], token_weight=counts[index]/sum(counts))
            accumulated += float(loss.detach())
            loss.backward()
        self.assertAlmostEqual(float(whole), accumulated, places=5)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-4, msg=name)

    def test_main_initialization_and_calibration_signature_unchanged(self):
        old, new = config(), ce_config()
        self.assertEqual(model_signature(old), model_signature(new))
        seed_all(42); full = make_model(old)
        seed_all(42); ce = make_model(new)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, ce.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(any(n.startswith(('unimodal_heads.', 'bimodal_heads.', 'fusion_out.'))
                             for n in adapter_state(ce)))

    def test_actual_ce_trainer_covers_tail_and_resumes_exactly(self):
        cfg = ce_config()
        cfg['train'].update(epoch_mode='full_dataset', epochs=1, batch_size=1, gradient_accumulation=2)
        cfg['spectral'].update(band_mask_rate=.15, embed_dropout=.1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest'; manifest.write_text('synthetic only\n')
            cfg['train_manifest'] = str(manifest)
            rows = [dict(id=str(i), split='train', text='hi', video='x', audio='y',
                         description='happy indeed', openset=['happy']) for i in range(5)]
            dataset, fixed = MERDataset(rows, 'hybrid'), batch()
            def collate(values):
                value = split_batch(fixed, int(values[0]['id']) % 2)
                value.update(texts=[r['text'] for r in values], questions=[r['question'] for r in values],
                             answers=[r['answer'] for r in values])
                return value
            seed_all(11); full = make_model(cfg); partial = copy.deepcopy(full)
            with redirect_stdout(io.StringIO()):
                seed_all(15)
                train(full, dataset, collate, cfg, root/'full', torch.device('cpu'))
                seed_all(15)
                checkpoint = train(partial, dataset, collate, cfg, root/'split', torch.device('cpu'), max_steps=1)
                resumed = copy.deepcopy(partial)
                final = train(resumed, dataset, collate, cfg, root/'split', torch.device('cpu'), resume=checkpoint)
            for name, value in adapter_state(full).items():
                torch.testing.assert_close(value, adapter_state(resumed)[name], rtol=0, atol=0)
            logs = [json.loads(s) for s in (root/'split/train.jsonl').read_text().splitlines()]
            self.assertEqual([r['samples_this_step'] for r in logs], [2, 2, 1])
            self.assertCountEqual([i for r in logs for i in r['sample_ids']], [str(i) for i in range(5)])
            self.assertTrue(all(r['loss'] == r['nll'] and 'unimodal_nll' not in r for r in logs))
            self.assertEqual(read_checkpoint(final)['config']['mcm']['loss_mode'], 'ce_only')

    def test_invalid_ce_settings_are_rejected(self):
        cfg = load_config(ROOT/'tests/fixtures/config/mer_omni_atsuko_padding.json')
        for key, value in [('loss_mode', 'typo'), ('enabled', True), ('distillation_weight', .1)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                invalid = copy.deepcopy(cfg)
                invalid['mcm'][key] = value
                path = Path(tmp)/'config.json'; path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
