"""Independent AV prompt inputs: isolation, gradients, checkpoint and legacy compatibility."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mer_omni.calibration import model_signature, set_calibration
from mer_omni.encoders.omni import OmniComponents
from mer_omni.model import SequenceTooLong
from mer_omni.protocol import ROOT, includes_av_tokens, load_config
from mer_omni.training import adapter_state, load_adapter, seed_all
from fixtures import tiny_thinker
from test_mer_omni import TinyTokenizer, make_model, batch
from test_subtitle_prompt import subtitle_config


def av_config():
    cfg = subtitle_config()
    cfg['prompt']['include_av_tokens'] = True
    return cfg


class AVPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_new_heads_preserve_shared_initialization_rng_and_calibration(self):
        seed_all(42)
        control = make_model(subtitle_config())
        rng = torch.get_rng_state().clone()
        seed_all(42)
        changed = make_model(av_config())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, changed.state_dict()[name]), name)
        self.assertEqual(set(changed.state_dict()) - set(control.state_dict()),
                         {'audio_in.weight', 'audio_in.bias', 'video_in.weight', 'video_in.bias'})
        self.assertEqual(model_signature(control.config), model_signature(changed.config))
        self.assertFalse(torch.equal(changed.audio_in.weight, changed.video_in.weight))

    def test_prompt_has_real_av_embeddings_in_order_and_masks_all_evidence(self):
        model = make_model(av_config()).eval()
        hidden = model.components.hidden_size
        fusion, audio, video = [torch.randn(2, hidden) for _ in range(3)]
        texts = ['I did not enjoy this. 我不喜欢。', '']
        questions, answers = ['emotion?', 'emotion?'], ['sad', 'neutral']
        infer, mask, _ = model._assemble(fusion, questions, texts=texts, audio_tokens=audio, video_tokens=video)
        train, _, labels = model._assemble(fusion, questions, answers, texts=texts,
                                           audio_tokens=audio, video_tokens=video)
        for i in range(2):
            parts = model._prompt_segments(fusion[i:i+1], questions[i], texts[i],
                                           audio_token=audio[i:i+1], video_token=video[i:i+1])
            expected = [model._embed([model.tokenizer.bos_token_id])]
            expected += [model._embed(model._ids(p)) if isinstance(p, str) else p for p in parts]
            expected = torch.cat(expected)
            length = int(mask[i].sum())
            torch.testing.assert_close(infer[i, :length], expected, rtol=0, atol=0)
            torch.testing.assert_close(train[i, :length], expected, rtol=0, atol=0)
            self.assertTrue((labels[i, :length] == -100).all())
            self.assertEqual(labels[i][labels[i] != -100].tolist(),
                             model._ids(answers[i] + '###') + [model.tokenizer.eos_token_id])
        with self.assertRaisesRegex(ValueError, 'one projected token'):
            model._assemble(fusion, questions, texts=texts)
        with self.assertRaisesRegex(ValueError, 'one projected token'):
            model._assemble(fusion, questions, texts=texts, audio_tokens=audio[:1], video_tokens=video)
        length = sum(len(model._ids(p)) if isinstance(p, str) else 1
                     for p in model._prompt_segments(None, questions[0], texts[0]))
        model.config['train']['max_length'] = length + len(model._ids(answers[0] + '###'))
        model.validate_training_text(questions[0], answers[0], text=texts[0])
        model.config['train']['max_length'] -= 1
        with self.assertRaises(SequenceTooLong):
            model.validate_training_text(questions[0], answers[0], text=texts[0])

    def test_direct_tokens_use_correct_ungated_modality_and_have_separate_paths(self):
        model = make_model(av_config()).eval()
        features = dict(fusion=torch.randn(2, model.config['spectral']['dim']),
                        modalities=torch.randn(2, 3, model.config['spectral']['dim']))
        original = model._project_prompt_features(features)
        changed = dict(features, modalities=features['modalities'].clone())
        changed['modalities'][:, 2] += 2
        alternative = model._project_prompt_features(changed)
        torch.testing.assert_close(original['tokens'], alternative['tokens'], rtol=0, atol=0)
        torch.testing.assert_close(original['video_tokens'], alternative['video_tokens'], rtol=0, atol=0)
        self.assertFalse(torch.equal(original['audio_tokens'], alternative['audio_tokens']))
        torch.testing.assert_close(original['audio_tokens'], model.audio_in(features['modalities'][:, 2]))
        torch.testing.assert_close(original['video_tokens'], model.video_in(features['modalities'][:, 1]))
        # Freeze fusion and subtitles: changes in each direct AV embedding affect CE.
        b = batch()
        baseline = model._nll(*model._assemble(questions=b['questions'], answers=b['answers'],
                                               texts=b['texts'], **original))
        for key in ['audio_tokens', 'video_tokens']:
            altered = dict(original, **{key: original[key] + torch.randn_like(original[key])})
            actual = model._nll(*model._assemble(questions=b['questions'], answers=b['answers'],
                                                texts=b['texts'], **altered))
            self.assertFalse(torch.equal(baseline, actual), key)

    def test_forward_generation_prefixes_match_and_new_heads_receive_gradients(self):
        model = make_model(av_config()).eval()
        b = batch()
        original = model.components.thinker.model.forward
        with patch.object(model.components.thinker.model, 'forward', wraps=original) as observed:
            loss = model(**b)['loss']
            training_inputs = observed.call_args.kwargs['inputs_embeds'].detach().clone()
        loss.backward()
        for name in ['audio_in', 'video_in', 'fusion_in']:
            gradient = getattr(model, name).weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0, name)
        model.config['generation']['max_new_tokens'] = 1
        with patch.object(model.components.thinker.model, 'forward', wraps=original) as observed:
            model.generate(**b)
        self.assertEqual(len(observed.call_args_list), 2)
        for i, call in enumerate(observed.call_args_list):
            actual = call.kwargs['inputs_embeds'][0]
            torch.testing.assert_close(actual, training_inputs[i, :len(actual)], rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_cross_architecture_rejection(self):
        seed_all(42)
        first = make_model(av_config()).eval()
        seed_all(42)
        second = make_model(av_config()).eval()
        saved = adapter_state(first)
        self.assertTrue({'audio_in.weight', 'video_in.weight'} <= set(saved))
        load_adapter(second, saved)
        data = batch()
        torch.testing.assert_close(first(**data)['loss'], second(**data)['loss'], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'Adapter state mismatch'):
            load_adapter(make_model(subtitle_config()), saved)
        from fixture_cli import run
        control = load_config(ROOT / 'tests/fixtures/config/mer_omni_subtitle_ablation.json')
        args = SimpleNamespace(config=ROOT / 'tests/fixtures/config/mer_omni_av_tokens_evaluation.json',
                               command='infer', checkpoint='unused')
        with patch('mer_omni.training.read_checkpoint', return_value={'config': control}):
            with self.assertRaisesRegex(ValueError, 'audio/video token prompt differs'):
                run(args)

    def test_invalid_flags_and_auxiliary_loss_are_rejected(self):
        self.assertFalse(includes_av_tokens(subtitle_config()))
        cfg = av_config()
        cfg['prompt']['include_av_tokens'] = 'true'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            includes_av_tokens(cfg)
        cfg = av_config()
        cfg['prompt']['include_subtitle'] = False
        cfg['mcm']['loss_mode'] = 'full'
        with self.assertRaisesRegex(ValueError, 'ce_only'):
            includes_av_tokens(cfg)

    def test_legacy_model_output_matches_archived_source_exactly(self):
        path = ROOT / 'tests/fixtures/model_before_av_tokens.py'
        spec = importlib.util.spec_from_file_location('mer_omni.archived_before_av', path)
        archived = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archived)
        for enabled in [False, True]:
            cfg = subtitle_config(enabled)
            seed_all(42)
            old = archived.MEROmni(cfg, TinyTokenizer(),
                                   components=OmniComponents(tiny_thinker(), cfg['omni']))
            set_calibration(old.frontend, [[1.] * s.length for s in old.frontend.spectra], samples=2)
            seed_all(42)
            new = make_model(copy.deepcopy(cfg))
            old.eval(); new.eval()
            for name, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
            data = batch()
            torch.testing.assert_close(old(**data)['loss'], new(**data)['loss'], rtol=0, atol=0)
            self.assertEqual(old.generate(**data), new.generate(**data))


if __name__ == '__main__':
    unittest.main()
