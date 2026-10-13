"""Offline end-to-end tests: real tiny Omni modules, synthetic inputs only."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mer_omni.data import MERDataset, build_train_rows, validate_rows, build_benchmark_rows
from mer_omni.evaluation import score_dataset, aggregate_scores
from mer_omni.model import MEROmni
from mer_omni.protocol import ROOT, BENCHMARKS, check_no_overlap, check_prediction_ids, sha256
from mer_omni.training import SampleStream, adapter_state, load_adapter, train, read_checkpoint, seed_all
from mer_omni.encoders.omni import OmniComponents
from mer_omni.calibration import set_calibration
from fixtures import tiny_thinker, native_batch, HAS_OMNI


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False):
        return [3 + sum(word.encode()) % 59 for word in text.split()]
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(x) for x in ids if x > 2)


def config():
    cfg = json.loads((ROOT / 'tests/fixtures/legacy_full_loss.json').read_text())
    cfg['omni'].update(dtype='float32', lora_r=2, lora_alpha=4, lora_dropout=0.0)
    cfg['spectral'].update(dim=16, heads=2, layers=1, sequence_length=6, conv_kernel_sizes=[1, 1, 1],
        attn_dropout=[0.0, 0.0, 0.0], relu_dropout=0.0, res_dropout=0.0, embed_dropout=0.0,
        text_dropout=0.0, output_dropout=0.0, band_mask_rate=0)
    cfg['train'].update(epochs=1, steps_per_epoch=2, batch_size=2, gradient_accumulation=2,
                         warmup_steps=1, warmup_lr=1e-3, lr=1e-3, min_lr=1e-3)
    cfg['generation'].update(do_sample=False, max_new_tokens=3)
    return cfg


def make_model(cfg):
    model = MEROmni(cfg, TinyTokenizer(), components=OmniComponents(tiny_thinker(), cfg['omni']))
    # Synthetic training calibration, used exclusively in tests.
    set_calibration(model.frontend, [[1.0] * s.length for s in model.frontend.spectra], samples=2)
    return model


def batch(inputs=None):
    return dict(omni_inputs=native_batch() if inputs is None else inputs,
                texts=['first transcript', 'second transcript'], questions=['what emotion', 'what emotion'],
                answers=['happy', 'very sad indeed'])


class ProtocolTests(unittest.TestCase):
    def test_no_overlap_duplicates_or_partial_benchmark_mean(self):
        rows = [dict(id='x', dataset='MER-Caption+', split='train', text='', description='happy',
                     openset=['happy'], video='synthetic/x.mp4', audio='synthetic/x.wav')]
        validate_rows(rows, require_media=False)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_rows(rows * 2, require_media=False)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_no_overlap(rows, [dict(id='x', video='synthetic/different/file.mp4')])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            check_prediction_ids(rows, {})
        with self.assertRaisesRegex(ValueError, 'nine'):
            aggregate_scores([])

    def test_hybrid_targets_and_evaluation_label_isolation(self):
        row = dict(id='x', split='train', text='', video='v', audio='a', description='looks happy', openset=['happy'])
        description = MERDataset([row], 'description')[0]
        labels = MERDataset([row], 'ovlabel')[0]
        self.assertEqual(description['answer'], 'looks happy')
        self.assertEqual(labels['answer'], "The character's emotional state is happy.")
        self.assertNotEqual(description['question'], labels['question'])
        evaluation = MERDataset([dict(row, split='test', label='SECRET')], training=False)[0]
        self.assertNotIn('answer', evaluation)
        self.assertNotIn('label', evaluation)
        self.assertNotIn('description', evaluation)

    def test_original_release_overlap_acceptance_is_scoped(self):
        train = [dict(id='shared', dataset='MER-Caption+', split='train', video='synthetic/train/shared.mp4'),
                 dict(id='unexpected', dataset='MER-Caption+', split='train', video='synthetic/train/new.mp4')]
        row = dict(id='shared', dataset='OVMERDPlus', split='test', video='synthetic/eval/shared.mp4')
        policy = {'OVMERDPlus': ['shared']}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_no_overlap(train, [row])
        self.assertEqual(check_no_overlap(train, [row], accepted_overlaps=policy), ['shared'])
        for bad in (dict(row, dataset='MER2024'), dict(row, id='unexpected'),
                    dict(row, split='train'), dict(row, id='alias', video=train[0]['video'])):
            with self.subTest(row=bad), self.assertRaisesRegex(ValueError, 'overlap'):
                check_no_overlap(train, [bad], accepted_overlaps=policy)
        # A path-only collision cannot become an accepted shared ID.
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_no_overlap(train, [dict(row, id='alias', video=train[0]['video'])],
                             accepted_overlaps={'OVMERDPlus': ['alias']})
        fresh = dict(row, id='fresh', video='synthetic/eval/fresh.mp4')
        self.assertEqual(check_no_overlap(train, [fresh], accepted_overlaps=policy), [])

    def test_overlap_policy_preserves_calibration_signature(self):
        from mer_omni.calibration import model_signature
        cfg = config()
        signature = model_signature(cfg)
        cfg.pop('accepted_release_overlaps', None)
        self.assertEqual(signature, model_signature(cfg))

    def test_sentiment_neutral_rule_and_weighted_f1(self):
        rows = [dict(id=str(i), dataset='CMUMOSI', split='test', label=label) for i, label in enumerate([-1, 0, 1])]
        score = score_dataset(rows, dict(sentiment={'0':'neutral', '1':'positive', '2':'positive'}), strict=False)
        self.assertEqual(score['primary'], 1.0)
        self.assertEqual(score['evaluated_samples'], 2)
        with self.assertRaisesRegex(ValueError, 'expected'):
            score_dataset(rows, dict(sentiment={}))

    def test_wheel_metrics_known_perfect_and_all_miss_cases(self):
        rows = [dict(id='x', dataset='MER2023', split='test', label='happy'),
                dict(id='y', dataset='MER2023', split='test', label='sad')]
        perfect = score_dataset(rows, dict(openset={'x':'[happy]', 'y':'[sad]'}), strict=False)
        self.assertEqual(perfect['hit_rate'], 1)
        fine = [dict(r, dataset='OVMERDPlus', label=[r['label']]) for r in rows]
        miss = score_dataset(fine, dict(openset={'x':'[]', 'y':'[]'}), strict=False)
        self.assertEqual(miss['wheel_f1'], 0)
        self.assertEqual(miss['precision'], 0)

    def test_saved_reference_integrity_and_optional_upstream_comparison(self):
        root = ROOT / 'mer_omni/reference'
        receipt = json.loads((root / 'provenance.json').read_text())
        self.assertEqual(sha256(root / 'affectgpt.py'), receipt['vendored_sha256'])
        for path, info in receipt['files'].items():
            if path.startswith('emotion_wheel/'):
                self.assertEqual(sha256(root / path), info['sha256'])
        source = os.environ.get('AFFECTGPT_REFERENCE_ROOT')
        if not source:
            return  # The checkout integrity checks above always run.
        original = Path(source).expanduser()
        self.assertTrue(original.is_dir(), str(original))
        copied = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse((root / 'affectgpt.py').read_text()).body
                  if isinstance(n, ast.FunctionDef)}
        for path, info in receipt['files'].items():
            if 'functions' not in info:
                continue
            for node in ast.parse((original / path).read_text()).body:
                if isinstance(node, ast.FunctionDef) and node.name in info['functions']:
                    self.assertEqual(copied[node.name], ast.dump(node, include_attributes=False), node.name)

    def test_benchmark_mosi_test_split_count_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / BENCHMARKS['CMUMOSI']['directory']
            root.mkdir()
            corpus = {str(i): dict(val=float(i % 7 - 3)) for i in range(686)}
            np.savez(root / 'label.npz', test_corpus=corpus, train_corpus={'NEVER_TRAIN': dict(val=1)})
            (root / 'transcription-engchi-polish.csv').write_text('name,english\n'+''.join(f'{i},hello\n' for i in range(686)))
            rows = build_benchmark_rows(tmp, 'CMUMOSI')
            self.assertEqual(len(rows), 686)
            self.assertEqual(rows[0]['label'], -3)
            self.assertNotIn('NEVER_TRAIN', [r['id'] for r in rows])

    def test_sample_stream_resumes_across_shuffle_boundary(self):
        stream = SampleStream(3, 42)
        stream.take(2)
        state = stream.state_dict()
        expected = stream.take(9)
        other = SampleStream(3, 11)
        other.load_state_dict(state)
        self.assertEqual(other.take(9), expected)

    def test_iemocap_transcripts_follow_official_test_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / BENCHMARKS['IEMOCAPFour']['directory']
            root.mkdir()
            corpus = {f'Ses05_test_{i}': dict(emo=i % 4) for i in range(1241)}
            corpus['Ses01_not_test'] = dict(emo=0)
            np.savez(root / 'label_4way.npz', whole_corpus=corpus)
            path = root / 'transcription-engchi-polish.csv'
            valid = 'name,english\n' + ''.join(f'Ses05_test_{i},hello\n' for i in range(1241))
            # Matches the official package's duplicate, unlabelled FXX rows.
            path.write_text(valid + 'Ses05M_impro03_FXX0,Awe.\nSes05M_impro03_FXX0,Awe.\n')
            rows = build_benchmark_rows(tmp, 'IEMOCAPFour')
            self.assertEqual(len(rows), 1241)
            self.assertEqual({r['id'] for r in rows}, set(corpus) - {'Ses01_not_test'})
            self.assertTrue(all(Path(r['video']).suffix == '.avi' for r in rows))
            path.write_text(valid + 'Ses05_test_0,conflicting subtitle\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                build_benchmark_rows(tmp, 'IEMOCAPFour')
            path.write_text('name,english\n' + ''.join(f'Ses05_test_{i},hello\n' for i in range(1,1241)))
            with self.assertRaisesRegex(ValueError, 'Missing required transcript'):
                build_benchmark_rows(tmp, 'IEMOCAPFour')


@unittest.skipUnless(HAS_OMNI, 'Needs isolated Omni dependencies')
class GenerativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_answer_loss_mcm_and_frozen_encoder_gradients(self):
        model = make_model(config()).train()
        result = model(**batch())
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertGreater(result['auxiliary_nll'], 0)
        result['loss'].backward()
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        for adapter in model.frontend.adapters:
            self.assertGreater(adapter.weight.grad.abs().sum(), 0)
        self.assertGreater(model.frontend.modality_router.l2.weight.grad.abs().sum(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in model.named_parameters() if 'lora_' in n))
        self.assertFalse(model.components.thinker.lm_head.weight.requires_grad)

    def test_prefix_and_padding_mask_and_causal_answer_shift(self):
        cfg = config(); cfg['mcm']['enabled'] = False
        model = make_model(cfg).eval()
        b = batch()
        features = model.frontend(model.components.encode(b['omni_inputs']))
        x, mask, labels = model._assemble(model.fusion_in(features['fusion']), b['questions'], b['answers'])
        for i in range(2):
            expected = model._ids(b['answers'][i] + '###') + [2]
            self.assertEqual(labels[i][labels[i] != -100].tolist(), expected)
            self.assertTrue((labels[i][mask[i] == 0] == -100).all())
        nll = model._nll(x, mask, labels)
        length = int(mask[0].sum())
        single = model._nll(x[:1, :length], mask[:1, :length], labels[:1, :length])
        torch.testing.assert_close(nll[:1], single, atol=1e-6, rtol=1e-5)
        # Directly verify shifted answer CE against the native model and lm_head.
        pos = (mask.cumsum(-1)-1).clamp_min(0)[None].expand(3, -1, -1)
        h = model.components.thinker.model(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                                            use_cache=False).last_hidden_state
        for i in range(2):
            logits = model.components.thinker.lm_head(h[i, :-1])
            expected = torch.nn.functional.cross_entropy(logits.float(), labels[i, 1:], ignore_index=-100)
            torch.testing.assert_close(nll[i], expected)

    def test_omitted_modalities_and_raw_subtitles_do_not_leak(self):
        model = make_model(config()).eval()
        b = batch()
        features = model.frontend(model.components.encode(b['omni_inputs']))
        for index in range(3):
            original = features['modalities']
            changed = original.clone(); changed[:, index] += 100
            _, first = model.branch_features(original)
            _, second = model.branch_features(changed)
            torch.testing.assert_close(first[index], second[index])
        first = model(**b)['loss']
        b['texts'] = ['SECRET1', 'SECRET2']
        torch.testing.assert_close(first, model(**b)['loss'])

    def test_generation_and_trainable_only_checkpoint_roundtrip(self):
        cfg = config()
        model = make_model(cfg).eval()
        b = batch(); b['answers'] = None
        generated = model.generate(**b)
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(isinstance(x, str) for x in generated))
        state = adapter_state(model)
        self.assertFalse(any('audio_tower' in k or 'lm_head' in k or '.visual.' in k for k in state))
        other = copy.deepcopy(model)
        with torch.no_grad():
            next(other.frontend.parameters()).add_(1)
        load_adapter(other, state)
        self.assertEqual(other.generate(**b), generated)
        state.pop(next(iter(state)))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            load_adapter(other, state)

    def test_bf16_generation_and_backward(self):
        cfg = config(); cfg['omni']['dtype'] = 'bfloat16'
        components = OmniComponents(tiny_thinker().to(torch.bfloat16), cfg['omni'])
        model = MEROmni(cfg, TinyTokenizer(), components=components).train()
        set_calibration(model.frontend, [[1.0] * s.length for s in model.frontend.spectra], 2)
        b = batch()
        output = model(**b)
        output['loss'].backward()
        self.assertTrue(torch.isfinite(output['loss']))
        self.assertTrue(torch.isfinite(model.fusion_in.weight.grad).all())
        b['answers'] = None
        self.assertEqual(len(model.generate(**b)), 2)

    def test_optimizer_accumulation_and_resume_match_uninterrupted(self):
        cfg = config()  # All auxiliary branches, regularizers and calibration buffers participate.
        cfg['spectral'].update(band_mask_rate=.15, output_dropout=.2, text_dropout=.2,
                               attn_dropout=[.1, .0, .1], embed_dropout=.1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'train.jsonl'; manifest.write_text('synthetic fixture\n')
            cfg['train_manifest'] = str(manifest)
            rows = [dict(id=str(i), split='train', text='hi', video='x', audio='y', description='happy indeed', openset=['happy'])
                    for i in range(5)]
            dataset = MERDataset(rows, 'hybrid')
            fixed = native_batch()
            def collate(values):
                return dict(omni_inputs=fixed, texts=[r['text'] for r in values],
                            questions=[r['question'] for r in values], answers=[r['answer'] for r in values])
            seed_all(11); full = make_model(cfg)
            initial = copy.deepcopy(full)
            seed_all(15)
            full_path = train(full, dataset, collate, cfg, root/'full', torch.device('cpu'))
            seed_all(15)
            one = train(initial, dataset, collate, cfg, root/'split', torch.device('cpu'), max_steps=1)
            # Keep identical frozen tiny weights; production reloads the pinned checkpoint.
            resumed = copy.deepcopy(initial)
            train(resumed, dataset, collate, cfg, root/'split', torch.device('cpu'), resume=one)
            self.assertEqual(read_checkpoint(full_path)['step'], 2)
            for name, value in adapter_state(full).items():
                torch.testing.assert_close(value, adapter_state(resumed)[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
