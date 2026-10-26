"""The subtitle ablation preserves initialization and exposes text in both paths."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mer_omni.calibration import model_signature
from mer_omni.model import SequenceTooLong
from mer_omni.protocol import ROOT, includes_raw_subtitle, load_config
from mer_omni.training import seed_all, training_row
from test_mer_omni import config, make_model, batch


def subtitle_config(enabled=True):
    cfg = config()
    cfg['mcm'].update(loss_mode='ce_only', enabled=False)
    for key in cfg['mcm']:
        if key.endswith('_weight'):
            cfg['mcm'][key] = 0.0
    if enabled:
        cfg['prompt'] = {'include_subtitle': True}
    return cfg


class SubtitlePromptTests(unittest.TestCase):
    def test_only_prompt_changes_initialization_and_rng_identical(self):
        baseline, changed = subtitle_config(False), subtitle_config()
        seed_all(42)
        first = make_model(baseline)
        first_rng = torch.get_rng_state().clone()
        seed_all(42)
        second = make_model(changed)
        self.assertEqual(model_signature(baseline), model_signature(changed))
        self.assertTrue(torch.equal(first_rng, torch.get_rng_state()))
        for key, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[key]), key)
        expected = ['###Human: The multimodal information is: <Multi>', None,
                    '</Multi>. Now, please answer my question based on all the provided information. '
                    'question ###Assistant: ']
        self.assertEqual(first._prompt_segments(None, 'question', 'arbitrary subtitle'), expected)

    def test_subtitle_is_full_prompt_evidence_and_never_an_answer_target(self):
        model = make_model(subtitle_config()).eval()
        token = torch.randn(2, model.components.hidden_size)
        texts = ['我不喜欢。 I did not enjoy this movie!', '']
        questions, answers = ['emotion?', 'emotion?'], ['negative', 'neutral']
        inference, masks, _ = model._assemble(token, questions, texts=texts)
        training, _, labels = model._assemble(token, questions, answers, texts=texts)
        for i, text in enumerate(texts):
            length = int(masks[i].sum())
            self.assertTrue(torch.equal(inference[i, :length], training[i, :length]))
            self.assertTrue((labels[i, :length] == -100).all())
            expected_ids = model._ids(answers[i] + '###') + [model.tokenizer.eos_token_id]
            self.assertEqual(labels[i][labels[i] != -100].tolist(), expected_ids)
            suffix = ('</Multi>. The subtitle of this video is: <Subtitle>' + text +
                      '</Subtitle>. Now, please answer my question based on all the provided information. '
                      + questions[i] + ' ###Assistant: ')
            expected = model._embed(model._ids(suffix))
            self.assertTrue(torch.equal(inference[i, length-len(expected):length], expected))
        with self.assertRaisesRegex(ValueError, 'one text'):
            model._assemble(token, questions)

    def test_length_guard_counts_subtitle_before_sample_selection(self):
        model = make_model(subtitle_config())
        question, answer = 'emotion?', 'negative'
        text = 'complete original subtitle'
        length = sum(len(model._ids(s)) if isinstance(s, str) else 1
                     for s in model._prompt_segments(None, question, text))
        model.config['train']['max_length'] = length + len(model._ids(answer + '###'))
        model.validate_training_text(question, answer, text=text)
        row = dict(id='long', question=question, answer=answer, text=text + ' extra')
        with self.assertRaisesRegex(SequenceTooLong, 'forbid sample replacement'):
            training_row([row], 0, model, allow_replacement=False)

    def test_forward_and_generation_use_the_same_subtitle_prefix(self):
        model = make_model(subtitle_config()).eval()
        data = batch()
        original_forward = model.components.thinker.model.forward
        with patch.object(model.components.thinker.model, 'forward', wraps=original_forward) as observed:
            result = model(**data)
            training_inputs = observed.call_args.kwargs['inputs_embeds'].detach().clone()
        result['loss'].backward()
        self.assertIsNotNone(model.fusion_in.weight.grad)
        self.assertTrue(torch.isfinite(model.fusion_in.weight.grad).all())
        model.zero_grad(set_to_none=True)
        model.config['generation']['max_new_tokens'] = 1
        with patch.object(model.components.thinker.model, 'forward', wraps=original_forward) as observed:
            output = model.generate(**data)
            calls = observed.call_args_list
        self.assertEqual(len(output), 2)
        self.assertEqual(len(calls), 2)
        for i, call in enumerate(calls):
            inference = call.kwargs['inputs_embeds'][0]
            self.assertTrue(torch.equal(inference, training_inputs[i, :len(inference)]))
        altered = dict(data, texts=['different words and meaning', 'also changed'])
        self.assertNotEqual(float(model(**data)['loss']), float(model(**altered)['loss']))

    def test_subtitle_switch_cannot_leak_into_auxiliary_omission_branches(self):
        cfg = config()
        cfg['prompt'] = {'include_subtitle': True}
        with self.assertRaisesRegex(ValueError, 'ce_only'):
            includes_raw_subtitle(cfg)
        cfg = subtitle_config()
        cfg['prompt']['include_subtitle'] = 'false'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            includes_raw_subtitle(cfg)

    def test_inference_rejects_checkpoint_with_different_subtitle_policy(self):
        from fixture_cli import run
        baseline = load_config(ROOT / 'tests/fixtures/config/mer_omni_atsuko_padding.json')
        args = SimpleNamespace(config=ROOT / 'tests/fixtures/config/mer_omni_subtitle_evaluation.json',
                               command='infer', checkpoint='unused')
        with patch('mer_omni.training.read_checkpoint', return_value={'config': baseline}):
            with self.assertRaisesRegex(ValueError, 'subtitle prompt differs'):
                run(args)


if __name__ == '__main__':
    unittest.main()
