"""Research invariants and comparisons with user-uploaded Atsuko snapshots."""
import ast
import copy
import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from mer_omni.calibration import (calibrate, equal_energy_edges, load_calibration,
                                   projection_fingerprint, set_calibration)
from mer_omni.data import MERDataset
from mer_omni.layers.router import router
from mer_omni.layers.spectral import GraphSpectralBandDecomposer, BandRouter, apply_band_masking
from mer_omni.layers.transformer import TransformerEncoder
from mer_omni.model import MEROmni, SequenceTooLong, SpectralFrontEnd
from mer_omni.objectives import combine_losses, complementarity, entropy_balance, feature_distillation
from mer_omni.training import build_optimizer, adapter_state, load_adapter, seed_all, training_row
from test_mer_omni import config, make_model, batch

REFERENCE = Path(__file__).resolve().parents[1]/'mer_omni/reference/atsuko_original'


def reference_spectral():
    path = REFERENCE/'spectral_band.py'
    spec = importlib.util.spec_from_file_location('uploaded_atsuko_spectral_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_batch(value, index):
    source = value['omni_inputs']
    inputs = {k: v[index:index+1] for k, v in source.items() if k != 'pixel_values_videos'}
    sizes = source['video_grid_thw'].prod(-1).tolist()
    offset = sum(sizes[:index])
    inputs['pixel_values_videos'] = source['pixel_values_videos'][offset:offset+sizes[index]]
    return dict(omni_inputs=inputs, **{k: [v[index]] for k, v in value.items() if k != 'omni_inputs'})


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_component_bodies_are_unchanged(self):
        source = Path('/data/user02/Atsuko')
        current = Path(__file__).resolve().parents[1]/'mer_omni/layers'
        pairs = {'spectral.py':'trains/singleTask/model/spectral_band.py',
                 'router.py':'trains/singleTask/model/router.py',
                 **{name:f'trains/subNets/transformers_encoder/{name}' for name in
                    ('transformer.py', 'multihead_attention.py', 'position_embedding.py')}}
        def bodies(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name, original in pairs.items():
            expected = REFERENCE/'spectral_band.py' if name == 'spectral.py' else source/original
            if not expected.is_file():
                continue  # Non-uploaded dependency comparison is optional off-server.
            self.assertEqual(bodies(current/name), bodies(expected), name)

    def test_uploaded_reference_snapshots_match_provenance(self):
        from mer_omni.protocol import sha256
        receipt = json.loads((REFERENCE/'provenance.json').read_text())
        for name, info in receipt['files'].items():
            self.assertEqual(sha256(REFERENCE/name), info['sha256'])

    def test_default_model_settings_match_the_declared_atsuko_preset(self):
        source = REFERENCE/'config.json'
        if not source.is_file():
            self.skipTest('Reference server config is unavailable')
        reference = json.loads(source.read_text())['atsuko']
        original = dict(reference['commonParams'], **reference['datasetParams']['mosi'])
        current = json.loads((Path(__file__).resolve().parents[1]/'tests/fixtures/legacy_full_loss.json').read_text())
        spec = current['spectral']
        self.assertEqual([spec['dim'],spec['heads']], original['dst_feature_dim_nheads'])
        self.assertEqual(spec['layers'], original['nlevels'])
        self.assertEqual(spec['conv_kernel_sizes'], [original[f'conv1d_kernel_size_{m}'] for m in ('l','v','a')])
        self.assertEqual(spec['attn_dropout'], [original['attn_dropout'], original['attn_dropout_v'], original['attn_dropout_a']])
        for key in ('text_dropout','embed_dropout','res_dropout','relu_dropout','output_dropout',
                    'attn_mask','band_mask_rate','temperature'):
            self.assertEqual(spec[key], original[key], key)
        self.assertEqual(spec['router_input'],original.get('spec_router_input','pre_attn'))
        self.assertEqual(current['mcm']['modality_entropy_weight'], .1*original['loss_ety'])
        self.assertEqual(current['mcm']['band_entropy_weight'], original['band_entropy_weight'])

    def test_all_objectives_accumulate_like_a_physical_batch(self):
        seed_all(42)
        cfg = config()
        model = make_model(cfg).eval()
        b = batch()
        counts = [len(model._ids(a+'###'))+1 for a in b['answers']]
        whole = model(**b)
        whole['loss'].backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        accumulated = 0.0
        for index in range(2):
            result = model(**split_batch(b, index))
            loss = combine_losses(result, cfg['mcm'], token_weight=counts[index]/sum(counts), sample_weight=0.5)
            accumulated += float(loss.detach())
            loss.backward()
        self.assertAlmostEqual(float(whole['loss']), accumulated, places=5)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-4, msg=name)

    def test_loss_formulas_and_stop_gradients(self):
        seed_all(6)
        w = torch.softmax(torch.randn(2, 3), -1).requires_grad_()
        full = torch.tensor([1.0, 2.0], requires_grad=True)
        partial = torch.tensor([[2., 1., 4.], [1., 3., 2.]], requires_grad=True)
        importance = ((partial-full[:, None])/.1).softmax(-1).detach()
        expected = (importance * (importance.clamp_min(1e-30).log() - w.log())).sum(-1).mean()
        actual = complementarity(full, partial, w, .1)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(full.grad); self.assertIsNone(partial.grad)
        self.assertIsNotNone(w.grad)
        student = torch.randn(2, 8, requires_grad=True)
        refined = torch.randn(2, 3, 8, requires_grad=True)
        w.grad = None
        distill = feature_distillation(student, refined, w)
        teacher = (refined * w[:, :, None]).sum(1).detach()
        torch.testing.assert_close(distill, F.mse_loss(student.softmax(-1), teacher.softmax(-1)))
        distill.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(refined.grad); self.assertIsNone(w.grad)
        torch.testing.assert_close(entropy_balance(w), 3 * (w * w.log()).sum(-1).mean())

    def test_each_auxiliary_head_is_independent_and_supervised(self):
        model = make_model(config()).train()
        heads = [*model.unimodal_heads, *model.bimodal_heads]
        parameters = [set(id(p) for p in head.parameters()) for head in heads]
        self.assertEqual(sum(map(len, parameters)), len(set.union(*parameters)))
        output = model(**batch())
        output['loss'].backward()
        for head in heads:
            self.assertGreater(float(head.refine.proj1.weight.grad.abs().sum()), 0)
            self.assertGreater(float(head.output_b.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.fusion_out.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.frontend.shared_projection.weight.grad.abs().sum()), 0)

    def test_fusion_feature_cannot_read_teacher_forced_answers(self):
        model = make_model(config()).eval()
        b = batch()
        features = model.frontend(model.components.encode(b['omni_inputs']))
        token = model.fusion_in(features['fusion'])
        _, first = model._nll(*model._assemble(token, b['questions'], b['answers']), return_feature=True)
        _, changed = model._nll(*model._assemble(token, b['questions'], ['entirely different', 'target words here']),
                                return_feature=True)
        torch.testing.assert_close(first, changed)

    def test_optimizer_no_decay_group_matches_affectgpt(self):
        model = make_model(config())
        optimizer = build_optimizer(model, config()['train'])
        observed = {id(p): group['weight_decay'] for group in optimizer.param_groups for p in group['params']}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                no_decay = parameter.ndim < 2 or any(x in name for x in ('bias', 'ln', 'bn'))
                self.assertEqual(observed[id(parameter)], 0.0 if no_decay else .05, name)
            else:
                self.assertNotIn(id(parameter), observed)

    def test_combined_length_limit_and_bounded_retry(self):
        model = make_model(config())
        question, answer = 'what emotion', 'happy'
        segments = model._prompt_segments(None, question)
        prompt_length = sum(len(model._ids(x)) if isinstance(x, str) else 1 for x in segments)
        target_length = len(model._ids(answer+'###'))
        model.config['train']['max_length'] = prompt_length + target_length
        model.validate_training_text(question, answer)
        with self.assertRaises(SequenceTooLong):
            model.validate_training_text(question, answer+' more')
        rows = [dict(id='long', question=question, answer='word '*2000)]
        with self.assertRaisesRegex(SequenceTooLong, '10 attempts'):
            training_row(rows, 0, model)

    def test_equal_energy_edges_fixed_inference_and_checkpoint(self):
        self.assertEqual(equal_energy_edges([1]*6), [1, 3])
        self.assertEqual(equal_energy_edges([8, 1, 1, 1, 1, 1]), [1, 2])
        self.assertEqual(equal_energy_edges([0, 0, 0, 0, 0, 9]), [5, 5])
        model = make_model(config()).eval()
        spectral = model.frontend.spectra[0]
        value = torch.randn(2, 6, 16)
        together = spectral.split_bands(value)
        for index in range(2):
            alone = spectral.split_bands(value[index:index+1])
            for expected, actual in zip(together, alone):
                torch.testing.assert_close(expected[index:index+1], actual)
        torch.testing.assert_close(sum(together), value, atol=2e-6, rtol=1e-5)
        clone = copy.deepcopy(model)
        clone.frontend.spectra[0].band_edges.fill_(-1)
        with self.assertRaisesRegex(RuntimeError, 'uncalibrated'):
            clone.frontend.spectra[0].split_bands(value)
        load_adapter(clone, adapter_state(model))
        self.assertEqual(clone.frontend.spectra[0].band_edges.tolist(), [1, 3])

    def test_calibrated_edge_rule_matches_original_source_for_all_cases(self):
        source = REFERENCE/'spectral_band.py'
        if not source.exists():
            self.skipTest('Reference server source is unavailable')
        import typing
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_equal_energy_edges')
        namespace = dict(torch=torch, List=typing.List)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        reference = namespace['_equal_energy_edges']
        cases = [[1]*6, [8,1,1,1,1,1], [0,0,0,0,0,9]]
        generator = torch.Generator().manual_seed(5)
        cases.extend(torch.rand(46, generator=generator).square().tolist() for _ in range(100))
        for energy in cases:
            self.assertEqual(equal_energy_edges(energy), reference(torch.tensor(energy, dtype=torch.float64), 3))
        # The original helper can produce a zero-width band. Preserve the source
        # decomposition and reconstruct correctly instead of silently changing it.
        model = make_model(config())
        set_calibration(model.frontend, [[0,0,0,0,0,9]] * 3, 1)
        values = torch.randn(1,6,16)
        bands = model.frontend.spectra[0].split_bands(values)
        torch.testing.assert_close(bands[1], torch.zeros_like(values))
        torch.testing.assert_close(sum(bands), values, atol=2e-6, rtol=1e-5)

    def test_training_calibration_roundtrip_and_stale_source_rejection(self):
        cfg = config(); model = make_model(cfg)
        b = batch()
        rows = [dict(id=str(i), dataset='MER-Caption+', split='train', text='x',
                     video='x', audio='x', description='x', openset=['happy']) for i in range(2)]
        def collate(items):
            return split_batch(b, int(items[0]['id']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bands.json'
            report = calibrate(model, MERDataset(rows), collate, cfg, torch.device('cpu'), 'training-hash', path)
            self.assertEqual(report['samples'], 2)
            self.assertEqual(int(model.frontend.calibration_samples), 2)
            clone = copy.deepcopy(model)
            load_calibration(clone, path, cfg, 'training-hash', 2)
            with self.assertRaisesRegex(ValueError, 'Stale'):
                load_calibration(clone, path, cfg, 'different-training-hash', 2)
            with torch.no_grad():
                clone.frontend.shared_projection.weight.add_(.01)
            with self.assertRaisesRegex(ValueError, 'Stale'):
                load_calibration(clone, path, cfg, 'training-hash', 2)
            evaluation = MERDataset([dict(r, split='test') for r in rows], training=False)
            with self.assertRaisesRegex(ValueError, 'training-only'):
                calibrate(model, evaluation, collate, cfg, torch.device('cpu'), 'test-hash', Path(tmp)/'invalid.json')

    def test_interrupted_calibration_resumes_without_double_counting(self):
        cfg = config(); model = make_model(cfg)
        b = batch()
        rows = [dict(id=str(i), dataset='MER-Caption+', split='train', text='x', video='x', audio='x',
                     description='x', openset=['happy']) for i in range(2)]
        dataset = MERDataset(rows)
        def collate(items):
            return split_batch(b, int(items[0]['id']))
        def interrupted(items):
            if items[0]['id'] == '1':
                raise RuntimeError('simulated interruption')
            return collate(items)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resumed.json'
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                calibrate(model, dataset, interrupted, cfg, torch.device('cpu'), 'train', path)
            progress = json.loads(path.with_suffix('.progress.json').read_text())
            self.assertEqual(progress['processed'], 1)
            self.assertEqual(progress['status'], 'failed')
            resumed = calibrate(model, dataset, collate, cfg, torch.device('cpu'), 'train', path)
            fresh = calibrate(model, dataset, collate, cfg, torch.device('cpu'), 'train', Path(tmp)/'fresh.json')
            self.assertEqual(resumed['energy'], fresh['energy'])
            self.assertEqual(resumed['band_edges'], fresh['band_edges'])
            self.assertEqual(json.loads(path.with_suffix('.progress.json').read_text())['status'], 'completed')

    def test_frontend_and_residual_branches_against_actual_atsuko(self):
        source = REFERENCE/'atsuko_model.py'
        if not source.is_file():
            self.skipTest('Reference server source is unavailable')
        # Execute only the source class, not its package/imports or any trainer.
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Atsuko')
        spectral = reference_spectral()
        namespace = dict(torch=torch, nn=nn, F=F, TransformerEncoder=TransformerEncoder, router=router,
                         GraphSpectralBandDecomposer=spectral.GraphSpectralBandDecomposer,
                         BandRouter=spectral.BandRouter, apply_band_masking=spectral.apply_band_masking)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        for pathway in ('pre_attn', 'post_attn'):
            for training in (False, True):
                with self.subTest(pathway=pathway, training=training):
                    cfg = config()
                    cfg['spectral'].update(sequence_length=50, conv_kernel_sizes=[5, 5, 5], router_input=pathway,
                        input_mode='prealigned',
                        text_dropout=.3, attn_dropout=[.3, .0, .2], embed_dropout=.2, output_dropout=.5, band_mask_rate=.15)
                    new = make_model(cfg)
                    args = SimpleNamespace(use_bert=False, dst_feature_dim_nheads=[16, 2], dataset_name='mosi',
                        need_data_aligned=True, feature_dims=[32, 32, 32], nlevels=1,
                        attn_dropout=.3, attn_dropout_a=.2, attn_dropout_v=.0, relu_dropout=0.,
                        embed_dropout=.2, res_dropout=0., output_dropout=.5, text_dropout=.3,
                        attn_mask=True, fusion_method='sum', temperature=.1, conv1d_kernel_size_l=5,
                        conv1d_kernel_size_v=5, conv1d_kernel_size_a=5, enable_abcm=True,
                        enable_spectral_bands=True, band_mask_rate=.15, spec_router_input=pathway,
                        **{f'band_edges_{name}':s.band_edges.tolist() for name, s in zip(('l','v','a'),new.frontend.spectra)})
                    old = namespace['Atsuko'](args)
                    for index, name in enumerate(('l', 'v', 'a')):
                        getattr(old, f'proj_{name}').load_state_dict(new.frontend.adapters[index].state_dict())
                        getattr(old, f'band_router_{name}').load_state_dict(new.frontend.band_routers[index].state_dict())
                        getattr(old, f'self_attentions_{name}').load_state_dict(new.frontend.attentions[index].state_dict())
                        for number in (1, 2):
                            getattr(old, f'proj{number}_{name}').load_state_dict(
                                getattr(new.unimodal_heads[index].refine, f'proj{number}').state_dict())
                    old.encoder_c.load_state_dict(new.frontend.shared_projection.state_dict())
                    old.Router_spec.load_state_dict(new.frontend.modality_router.state_dict())
                    captured = {}
                    for index, name in enumerate(('va', 'la', 'lv')):
                        for number in (1, 2):
                            getattr(old, f'proj{number}_c_{name}').load_state_dict(
                                getattr(new.bimodal_heads[index].refine, f'proj{number}').state_dict())
                        getattr(old, f'out_layer_c_{name}').register_forward_pre_hook(
                            lambda module, inputs, name=name: captured.update({name:inputs[0].detach().clone()}))
                    raw = [torch.randn(2, 50, 32) for _ in range(3)]
                    # Original audio/video features can contain padding and internal zeros.
                    for values in raw[1:]:
                        values[0, 12:] = 0
                        values[1, 24:] = 0
                        values[:, 4] = 0
                    old.train(training); new.train(training)
                    seed_all(47)
                    expected = old(raw[0], raw[2], raw[1])
                    seed_all(47)
                    actual = new.frontend([list(x) for x in raw])
                    uni, bi = new.branch_features(actual['modalities'])
                    torch.testing.assert_close(actual['fusion'], expected['c_fea'])
                    torch.testing.assert_close(actual['modality_weights'], expected['channel_weight'])
                    for index, name in enumerate(('l','v','a')):
                        torch.testing.assert_close(actual['band_weights'][:,index], expected[f'band_weights_{name}'])
                        torch.testing.assert_close(uni[:,index], expected[f'{name}_proj'])
                    for value, name in zip(bi, ('va','la','lv')):
                        torch.testing.assert_close(value, captured[name])

    def test_prealigned_input_preserves_padding_and_rejects_length_guessing(self):
        cfg = config()['spectral']
        cfg.update(input_mode='prealigned', sequence_length=50, conv_kernel_sizes=[5]*3)
        frontend = SpectralFrontEnd([7, 5, 3], cfg)
        values = [torch.randn(2, 50, d) for d in [7, 5, 3]]
        for x in values:
            x[:, 9:] = 0
            x[:, 3] = 0
        prepared = frontend.prepare_sequences(values)
        for actual, original in zip(prepared, values):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        for length in (4, 49, 51):
            bad = [values[0], torch.randn(2, length, 5), values[2]]
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, 'prealigned mode requires'):
                frontend.project_sequences(bad)
        with self.assertRaisesRegex(ValueError, 'batch sizes'):
            frontend.prepare_sequences([values[0], values[1][:1], values[2]])

    def test_legacy_input_adapter_matches_checkpoint_behavior(self):
        from mer_omni.encoders.omni import resample_sequence
        cfg = config()['spectral']
        cfg.update(sequence_length=50, conv_kernel_sizes=[5]*3)
        cfg.pop('input_mode', None)
        frontend = SpectralFrontEnd(32, cfg).eval()
        values = [[torch.randn(n, 32), torch.randn(n+1, 32)] for n in [19, 4, 77]]
        expected = [frontend.shared_projection(proj(torch.stack(
            [resample_sequence(x, 50) for x in modality]).transpose(1, 2))).transpose(1, 2)
            for modality, proj in zip(values, frontend.adapters)]
        for actual, old in zip(frontend.project_sequences(values), expected):
            torch.testing.assert_close(actual, old, rtol=0, atol=0)

    def test_zero_pad_keeps_native_values_and_includes_padding_in_spectrum(self):
        cfg = config()['spectral']
        cfg.update(input_mode='zero_pad', sequence_length=8, conv_kernel_sizes=[1]*3,
                   sequence_lengths=[8, 9, 10], sequence_overflow='error')
        frontend = SpectralFrontEnd(1, cfg).eval()
        value = torch.tensor([[2.], [0.], [6.], [8.]], requires_grad=True)
        prepared = frontend.prepare_sequences([[value]]*3)
        for item, length in zip(prepared, [8, 9, 10]):
            self.assertEqual(item.shape, (1, length, 1))
            self.assertEqual(item[0, :, 0].tolist(), [2., 0., 6., 8.]+[0.]*(length-4))
        prepared[0].sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))
        # Padding remains on each full graph; no valid-length masking is added.
        spectral = reference_spectral().GraphSpectralBandDecomposer(8, band_edges=[2, 4])
        actual = spectral.split_bands(prepared[0].detach())
        torch.testing.assert_close(sum(actual), prepared[0].detach(), atol=2e-6, rtol=1e-5)
        self.assertEqual([s.length for s in frontend.spectra], [8, 9, 10])

    def test_zero_pad_overflow_is_explicit_and_retains_the_prefix(self):
        cfg = config()['spectral']
        cfg.update(input_mode='zero_pad', sequence_length=6, conv_kernel_sizes=[1]*3)
        frontend = SpectralFrontEnd(1, cfg)
        values = [[torch.arange(7.).reshape(-1, 1)]]*3
        with self.assertRaisesRegex(ValueError, '7 native positions exceed fixed length 6'):
            frontend.prepare_sequences(values)
        cfg['sequence_overflow'] = 'truncate_right'
        frontend = SpectralFrontEnd(1, cfg)
        for result in frontend.prepare_sequences(values):
            self.assertEqual(result[0, :, 0].tolist(), [0., 1., 2., 3., 4., 5.])

    def test_unequal_padded_modalities_forward_backward_and_old_calibration_rejection(self):
        cfg = config()
        model = make_model(cfg)
        b = batch()
        rows = [dict(id=str(i), dataset='MER-Caption+', split='train', text='x',
                     video='x', audio='x', description='x', openset=['happy']) for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy-bands.json'
            calibrate(model, MERDataset(rows), lambda items: split_batch(b, int(items[0]['id'])),
                      cfg, torch.device('cpu'), 'same-unchanged-manifest', path)
            padded_cfg = copy.deepcopy(cfg)
            padded_cfg['spectral'].update(input_mode='zero_pad', sequence_lengths=[8, 9, 10])
            padded = make_model(padded_cfg)
            with self.assertRaisesRegex(ValueError, 'Stale calibration'):
                load_calibration(padded, path, padded_cfg, 'same-unchanged-manifest', 2)
            result = padded(**b)
            self.assertTrue(torch.isfinite(result['loss']))
            result['loss'].backward()
            for projection in padded.frontend.adapters:
                self.assertIsNotNone(projection.weight.grad)
                self.assertTrue(torch.isfinite(projection.weight.grad).all())

    def test_separate_fixed_lengths_validate_each_convolution(self):
        from mer_omni.protocol import spectral_sequence_lengths
        cfg = config()['spectral']
        cfg.update(sequence_lengths=[8, 4, 10], conv_kernel_sizes=[1, 5, 1])
        with self.assertRaisesRegex(ValueError, 'at least three frequency bins'):
            spectral_sequence_lengths(cfg)
        cfg['sequence_lengths'] = [8, 9]
        with self.assertRaisesRegex(ValueError, 'three positive spectral sequence lengths'):
            spectral_sequence_lengths(cfg)


if __name__ == '__main__':
    unittest.main()
