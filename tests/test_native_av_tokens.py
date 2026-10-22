"""Native AV bypass: ragged pooling, gradient isolation, old-checkpoint compatibility."""
import copy
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mer_omni.calibration import model_signature, set_calibration
from mer_omni.encoders.omni import OmniComponents
from mer_omni.protocol import ROOT, av_token_source, load_config, sha256
from mer_omni.training import adapter_state, load_adapter, seed_all
from fixtures import tiny_thinker
from test_mer_omni import TinyTokenizer, make_model, batch
from test_av_tokens import av_config


def native_config():
    cfg = av_config()
    cfg['prompt']['av_token_source'] = 'native_mean'
    return cfg


class NativeAVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_weights_rng_calibration_and_fusion_are_identical(self):
        seed_all(42)
        control = make_model(av_config()).eval()
        rng = torch.get_rng_state().clone()
        seed_all(42)
        model = make_model(native_config()).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertEqual(set(model.state_dict())-set(control.state_dict()),
                         {'audio_native_in.weight', 'video_native_in.weight'})
        self.assertFalse(torch.equal(model.audio_native_in.weight, model.video_native_in.weight))
        self.assertEqual(model_signature(model.config), model_signature(control.config))
        raw = model.components.encode(batch()['omni_inputs'])
        torch.testing.assert_close(model.frontend(raw)['fusion'], control.frontend(raw)['fusion'], rtol=0, atol=0)
        added = sum(p.numel() for p in model.parameters())-sum(p.numel() for p in control.parameters())
        self.assertEqual(added, 2*model.components.hidden_size*model.config['spectral']['dim'])

    def test_ragged_means_preserve_full_native_scope_and_ignore_frontend_vectors(self):
        model = make_model(native_config()).eval()
        h, d = model.components.hidden_size, model.config['spectral']['dim']
        # Lengths deliberately differ between samples and exceed the frontend grid.
        raw = [[torch.randn(n, h) for n in ns] for ns in [(2,3),(1,8),(13,2)]]
        features = dict(fusion=torch.randn(2,d), modalities=torch.randn(2,3,d))
        actual = model._project_prompt_features(features, native_sequences=raw)
        for name,m in [('audio',2),('video',1)]:
            expected = torch.cat([getattr(model,name+'_in')(getattr(model,name+'_native_in')(v.mean(0)[None])) for v in raw[m]])
            torch.testing.assert_close(actual[name+'_tokens'], expected)
        # No dependency on the frontend's modality vectors or pooling denominator.
        other = model._project_prompt_features(dict(features,modalities=features['modalities']*0), native_sequences=raw)
        for key in actual:
            torch.testing.assert_close(other[key], actual[key], rtol=0, atol=0)
        # Alter positions beyond the fusion grid: native mean must still respond.
        changed = [[v.clone() for v in values] for values in raw]
        changed[2][0][-1] += 10
        alternative = model._project_prompt_features(features,native_sequences=changed)
        self.assertFalse(torch.equal(actual['audio_tokens'][0],alternative['audio_tokens'][0]))
        torch.testing.assert_close(actual['audio_tokens'][1],alternative['audio_tokens'][1], rtol=0, atol=0)
        for key in ['tokens','video_tokens']:
            torch.testing.assert_close(actual[key],alternative[key],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'unpadded native'):
            model._project_prompt_features(features)
        changed[1][0] = torch.empty(0,h)
        with self.assertRaisesRegex(ValueError,'nonempty native'):
            model._project_prompt_features(features,native_sequences=changed)

    def test_direct_path_gradient_bypasses_frontend_and_encoder(self):
        model = make_model(native_config()).eval()
        raw = model.components.encode(batch()['omni_inputs'])
        features = model.frontend(raw)
        model._project_prompt_features(features,native_sequences=raw)['audio_tokens'].square().mean().backward()
        for name in ['audio_native_in','audio_in']:
            self.assertGreater(float(getattr(model,name).weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.frontend.parameters()))
        self.assertTrue(all(p.grad is None for p in model.components.parameters()))
        self.assertIsNone(model.video_native_in.weight.grad)

    def test_real_tiny_forward_backward_generation_and_prefix(self):
        model = make_model(native_config()).eval()
        data = batch()
        original = model.components.thinker.model.forward
        with patch.object(model.components.thinker.model,'forward',wraps=original) as calls:
            loss = model(**data)['loss']
            train_inputs = calls.call_args.kwargs['inputs_embeds'].detach().clone()
        loss.backward()
        for name in ['audio_native_in','video_native_in','audio_in','video_in','fusion_in']:
            grad = getattr(model,name).weight.grad
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.abs().sum()),0,name)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.frontend.parameters() if p.grad is not None),0)
        model.config['generation']['max_new_tokens'] = 1
        with patch.object(model.components.thinker.model,'forward',wraps=original) as calls:
            model.generate(**data)
        self.assertEqual(len(calls.call_args_list),2)
        for i,call in enumerate(calls.call_args_list):
            value = call.kwargs['inputs_embeds'][0]
            torch.testing.assert_close(value,train_inputs[i,:len(value)],rtol=0,atol=0)

    def test_checkpoint_roundtrip_and_source_mismatch_rejected_before_inference(self):
        seed_all(42)
        first = make_model(native_config()).eval()
        seed_all(42)
        second = make_model(native_config()).eval()
        with torch.no_grad():
            first.audio_native_in.weight.add_(.01)
        saved = adapter_state(first)
        self.assertIn('audio_native_in.weight',saved)
        load_adapter(second,saved)
        data = batch()
        torch.testing.assert_close(first(**data)['loss'],second(**data)['loss'],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'Adapter state mismatch'):
            load_adapter(make_model(av_config()),saved)
        from tools.mer_omni import run
        args = SimpleNamespace(config=ROOT/'config/mer_omni_native_av_evaluation.json',command='infer',checkpoint='unused')
        old_cfg = load_config(ROOT/'config/mer_omni_av_tokens_ablation.json')
        with patch('mer_omni.training.read_checkpoint',return_value={'config':old_cfg}):
            with self.assertRaisesRegex(ValueError,'token source differs'):
                run(args)

    def test_invalid_native_source_configuration_is_rejected(self):
        self.assertEqual(av_token_source(av_config()),'frontend')
        cfg = native_config()
        cfg['prompt']['include_av_tokens'] = False
        with self.assertRaisesRegex(ValueError,'requires prompt.include_av_tokens'):
            av_token_source(cfg)
        cfg = native_config()
        cfg['prompt']['av_token_source'] = 'typo'
        with self.assertRaisesRegex(ValueError,'frontend or native_mean'):
            av_token_source(cfg)
        cfg = native_config()
        cfg['mcm']['loss_mode'] = 'full'
        with self.assertRaisesRegex(ValueError,'ce_only'):
            make_model(cfg)

    def test_default_av_model_matches_frozen_source_initialization_loss_and_generation(self):
        path = ROOT/'tests/fixtures/model_before_native_av.py'
        receipt = json.loads(path.with_suffix('.provenance.json').read_text())
        self.assertEqual(sha256(path),receipt['sha256'])
        spec = importlib.util.spec_from_file_location('mer_omni.before_native_av',path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cfg = av_config()
        seed_all(42)
        control = old.MEROmni(cfg,TinyTokenizer(),components=OmniComponents(tiny_thinker(),cfg['omni']))
        set_calibration(control.frontend,[[1.]*s.length for s in control.frontend.spectra],samples=2)
        control.eval()
        rng = torch.get_rng_state().clone()
        seed_all(42)
        current = make_model(copy.deepcopy(cfg)).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(control.state_dict()),set(current.state_dict()))
        for name,value in control.state_dict().items():
            torch.testing.assert_close(value,current.state_dict()[name],rtol=0,atol=0)
        data = batch()
        torch.testing.assert_close(control(**data)['loss'],current(**data)['loss'],rtol=0,atol=0)
        self.assertEqual(control.generate(**data),current.generate(**data))


if __name__ == '__main__':
    unittest.main()
