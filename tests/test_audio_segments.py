"""Ordered audio segment means, ragged prompts and exact one-token compatibility."""
import copy
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mer_omni.calibration import model_signature, set_calibration
from mer_omni.encoders.omni import OmniComponents
from mer_omni.model import SequenceTooLong
from mer_omni.protocol import ROOT, audio_token_count, load_config, sha256
from mer_omni.training import adapter_state, load_adapter, seed_all
from fixtures import tiny_thinker
from test_mer_omni import TinyTokenizer, make_model, batch
from test_native_av_tokens import native_config


def segmented_config():
    cfg = native_config()
    cfg['prompt']['audio_token_count'] = 4
    return cfg


class AudioSegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_four_tokens_add_no_parameters_and_preserve_rng_fusion_video(self):
        seed_all(42)
        control = make_model(native_config()).eval()
        rng = torch.get_rng_state().clone()
        seed_all(42)
        model = make_model(segmented_config()).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(control.state_dict()), set(model.state_dict()))
        self.assertEqual(sum(p.numel() for p in control.parameters()), sum(p.numel() for p in model.parameters()))
        for key, value in control.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(model_signature(model.config), model_signature(control.config))
        raw = model.components.encode(batch()['omni_inputs'])
        first = control._project_prompt_features(control.frontend(raw), native_sequences=raw)
        changed = model._project_prompt_features(model.frontend(raw), native_sequences=raw)
        for key in ('tokens', 'video_tokens'):
            torch.testing.assert_close(first[key], changed[key], rtol=0, atol=0)

    def test_partition_covers_every_position_once_in_order_and_preserves_full_scope(self):
        model = make_model(segmented_config()).eval()
        h, d = model.components.hidden_size, model.config['spectral']['dim']
        lengths = [1, 2, 3, 4, 5, 9]
        raw = [[torch.randn(2, h) for _ in lengths], [torch.randn(4, h) for _ in lengths],
               [(torch.arange(n, dtype=torch.float32)[:, None]+torch.ones(1, h)).requires_grad_() for n in lengths]]
        features = dict(fusion=torch.randn(len(lengths), d), modalities=torch.randn(len(lengths), 3, d))
        result = model._project_prompt_features(features, native_sequences=raw)
        self.assertEqual([len(v) for v in result['audio_tokens']], [1, 2, 3, 4, 4, 4])
        # Explicit expected native means for T=5 and T=9 (not another tensor_split).
        for sample, means in [(4, [1.5, 3., 4., 5.]), (5, [2., 4.5, 6.5, 8.5])]:
            expected = model.audio_in(model.audio_native_in(torch.tensor(means)[:, None].expand(-1, h)))
            torch.testing.assert_close(result['audio_tokens'][sample], expected, rtol=0, atol=0)
        changed = [[v.detach().clone() for v in seq] for seq in raw]
        changed[2][-1][-1] += 10
        other = model._project_prompt_features(features, native_sequences=changed)
        torch.testing.assert_close(other['audio_tokens'][-1][:3], result['audio_tokens'][-1][:3], rtol=0, atol=0)
        self.assertFalse(torch.equal(other['audio_tokens'][-1][-1], result['audio_tokens'][-1][-1]))
        for key in ('tokens', 'video_tokens'):
            torch.testing.assert_close(other[key], result[key], rtol=0, atol=0)
        sum(v.square().mean() for v in result['audio_tokens']).backward()
        for value in raw[2]:
            self.assertTrue((value.grad.abs().sum(-1) > 0).all())
        for module in (model.audio_native_in, model.audio_in):
            self.assertGreater(float(module.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.frontend.parameters()))
        self.assertIsNone(model.video_native_in.weight.grad)

    def test_ragged_prompt_uses_exact_tokens_and_masks_all_evidence(self):
        model = make_model(segmented_config()).eval()
        h = model.components.hidden_size
        fusion, video = torch.randn(2, h), torch.randn(2, h)
        audio = [torch.randn(4, h), torch.randn(1, h)]
        data = batch()
        infer, masks, _ = model._assemble(fusion, data['questions'], texts=data['texts'], audio_tokens=audio, video_tokens=video)
        train, _, labels = model._assemble(fusion, data['questions'], data['answers'], texts=data['texts'], audio_tokens=audio, video_tokens=video)
        for i in range(2):
            pieces = model._prompt_segments(fusion[i:i+1], data['questions'][i], data['texts'][i],
                                           audio_token=audio[i], video_token=video[i:i+1])
            expected = torch.cat([model._embed([model.tokenizer.bos_token_id])] +
                                 [model._embed(model._ids(p)) if isinstance(p, str) else p for p in pieces])
            n = int(masks[i].sum())
            torch.testing.assert_close(infer[i, :n], expected, rtol=0, atol=0)
            torch.testing.assert_close(train[i, :n], expected, rtol=0, atol=0)
            self.assertTrue((labels[i, :n] == -100).all())
            self.assertEqual(labels[i][labels[i] != -100].tolist(), model._ids(data['answers'][i]+'###')+[model.tokenizer.eos_token_id])
        for invalid in (torch.randn(2, h), [audio[0]], [torch.empty(0, h), audio[1]],
                        [torch.randn(5, h), audio[1]], [torch.randn(4, h+1), audio[1]]):
            with self.subTest(invalid=str(type(invalid))), self.assertRaisesRegex(ValueError, 'nonempty projected'):
                model._assemble(fusion, data['questions'], texts=data['texts'], audio_tokens=invalid, video_tokens=video)

    def test_text_limit_counts_extra_tokens_and_allows_short_actual_sequence(self):
        model = make_model(segmented_config()).eval()
        question, answer, text = 'emotion?', 'happy', 'original subtitle'
        legacy_length = sum(len(model._ids(p)) if isinstance(p, str) else 1
                            for p in model._prompt_segments(None, question, text)) + len(model._ids(answer+'###'))
        model.config['train']['max_length'] = legacy_length+3
        model.validate_training_text(question, answer, text=text)
        model.config['train']['max_length'] -= 1
        with self.assertRaises(SequenceTooLong):
            model.validate_training_text(question, answer, text=text)
        model.config['train']['max_length'] = legacy_length
        model.validate_training_text(question, answer, text=text, actual_audio_tokens=1)
        h = model.components.hidden_size
        model._assemble(torch.randn(1, h), [question], [answer], texts=[text],
                        audio_tokens=[torch.randn(1, h)], video_tokens=torch.randn(1, h))
        with self.assertRaises(SequenceTooLong):
            model._assemble(torch.randn(1, h), [question], [answer], texts=[text],
                            audio_tokens=[torch.randn(4, h)], video_tokens=torch.randn(1, h))

    def test_real_decoder_backward_and_generation_prefix_for_ragged_audio(self):
        model = make_model(segmented_config()).eval()
        data = batch()
        raw = model.components.encode(data['omni_inputs'])
        h = model.components.hidden_size
        raw = [raw[0], raw[1], [torch.randn(9, h), torch.randn(1, h)]]
        original = model.components.thinker.model.forward
        with patch.object(model.components, 'encode', return_value=raw), \
                patch.object(model.components.thinker.model, 'forward', wraps=original) as calls:
            loss = model(**data)['loss']
            training = calls.call_args.kwargs['inputs_embeds'].detach().clone()
        loss.backward()
        for name in ('audio_native_in', 'audio_in', 'video_native_in', 'video_in', 'fusion_in'):
            grad = getattr(model, name).weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.frontend.parameters() if p.grad is not None), 0)
        model.config['generation']['max_new_tokens'] = 1
        with patch.object(model.components, 'encode', return_value=raw), \
                patch.object(model.components.thinker.model, 'forward', wraps=original) as calls:
            model.generate(**data)
        self.assertEqual(len(calls.call_args_list), 2)
        for i, call in enumerate(calls.call_args_list):
            value = call.kwargs['inputs_embeds'][0]
            torch.testing.assert_close(value, training[i, :len(value)], rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_semantic_mismatch_guard(self):
        seed_all(42)
        first = make_model(segmented_config()).eval()
        seed_all(42)
        second = make_model(segmented_config()).eval()
        with torch.no_grad():
            first.audio_native_in.weight.add_(.01)
        load_adapter(second, adapter_state(first))
        data = batch()
        torch.testing.assert_close(first(**data)['loss'], second(**data)['loss'], rtol=0, atol=0)
        from tools.mer_omni import run
        for cfg_file, saved_file in [('mer_omni_audio_segments_evaluation.json', 'mer_omni_native_av_ablation.json'),
                                     ('mer_omni_native_av_evaluation.json', 'mer_omni_audio_segments_ablation.json')]:
            args = SimpleNamespace(config=ROOT/'config'/cfg_file, command='infer', checkpoint='unused')
            with patch('mer_omni.training.read_checkpoint', return_value={'config': load_config(ROOT/'config'/saved_file)}):
                with self.assertRaisesRegex(ValueError, 'audio token count differs'):
                    run(args)

    def test_invalid_count_and_source_combinations_rejected(self):
        self.assertEqual(audio_token_count(native_config()), 1)
        self.assertEqual(audio_token_count(segmented_config()), 4)
        for value in (True, False, 0, -1, 2, 5, '4', 4.0):
            cfg = native_config()
            cfg['prompt']['audio_token_count'] = value
            with self.assertRaisesRegex(ValueError, 'must be 1 or 4'):
                audio_token_count(cfg)
        cfg = segmented_config()
        cfg['prompt']['av_token_source'] = 'frontend'
        with self.assertRaisesRegex(ValueError, 'require.*native_mean'):
            make_model(cfg)

    def test_one_token_exactly_matches_frozen_native_model(self):
        path = ROOT/'tests/fixtures/model_before_audio_segments.py'
        self.assertEqual(sha256(path), json.loads(path.with_suffix('.provenance.json').read_text())['sha256'])
        spec = importlib.util.spec_from_file_location('mer_omni.before_audio_segments', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for explicit in (False, True):
            cfg = native_config()
            if explicit:
                cfg['prompt']['audio_token_count'] = 1
            seed_all(42)
            control = old.MEROmni(cfg, TinyTokenizer(), components=OmniComponents(tiny_thinker(), cfg['omni']))
            set_calibration(control.frontend, [[1.]*s.length for s in control.frontend.spectra], samples=2)
            control.eval()
            rng = torch.get_rng_state().clone()
            seed_all(42)
            current = make_model(copy.deepcopy(cfg)).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(set(control.state_dict()), set(current.state_dict()))
            for key, value in control.state_dict().items():
                torch.testing.assert_close(value, current.state_dict()[key], rtol=0, atol=0)
            data = batch()
            torch.testing.assert_close(control(**data)['loss'], current(**data)['loss'], rtol=0, atol=0)
            self.assertEqual(control.generate(**data), current.generate(**data))


if __name__ == '__main__':
    unittest.main()
