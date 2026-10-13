"""Exercise relocation, including receipts and a real trainer checkpoint resume."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from mer_omni.data import (MERDataset, build_train_rows, read_checkpoint_training_rows,
                           read_manifest, save_manifest, verify_manifest_receipt)
from mer_omni.protocol import ROOT, load_config, sha256
from mer_omni.training import read_checkpoint, seed_all, train
from test_training_epochs import LossModel, collate
from tools import mer_omni as cli


class PortablePathTests(unittest.TestCase):
    def fixture(self, root):
        media = root / 'data/clips'
        media.mkdir(parents=True)
        (media / 'clip.mp4').write_bytes(b'synthetic video path fixture')
        (media / 'clip.wav').write_bytes(b'synthetic audio path fixture')
        source = root / 'data/labels.csv'
        source.write_text('name,description\n0,happy\n')
        rows = [dict(id=str(i), dataset='MER-Caption+', split='train', text='hello',
                     description='happy ' * (i + 1), openset=['happy'],
                     video=str(media / 'clip.mp4'), audio=str(media / 'clip.wav')) for i in range(5)]
        manifest = root / 'data/manifests/train.jsonl'
        save_manifest(rows, manifest, [source])
        return manifest, source, rows

    def test_manifest_and_receipt_survive_move_and_still_reject_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / 'original checkout', Path(tmp) / 'moved checkout'
            manifest, source, rows = self.fixture(old)
            raw = [json.loads(x) for x in manifest.read_text().splitlines()]
            self.assertTrue(all(not Path(r[k]).is_absolute() for r in raw for k in ('video', 'audio')))
            self.assertTrue(Path(rows[0]['video']).is_absolute())  # Caller-owned rows are untouched.
            receipt = verify_manifest_receipt(manifest)
            self.assertTrue(all(not Path(p).is_absolute() for p in receipt['sources']))
            original_hash = sha256(manifest)
            old.rename(new)
            moved = new / manifest.relative_to(old)
            self.assertFalse(old.exists())
            self.assertEqual(sha256(moved), original_hash)
            self.assertEqual(verify_manifest_receipt(moved), receipt)
            loaded = read_manifest(moved)
            self.assertTrue(all(Path(r['video']).is_relative_to(new) for r in loaded))
            (new / source.relative_to(old)).write_text('changed annotation\n')
            with self.assertRaisesRegex(ValueError, 'Annotation source changed'):
                verify_manifest_receipt(moved)

    def test_legacy_absolute_manifests_and_receipts_remain_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, source, rows = self.fixture(Path(tmp))
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            receipt = dict(samples=len(rows), manifest_sha256=sha256(manifest),
                           sources={str(source): sha256(source)}, media_checked=True)
            manifest.with_suffix('.receipt.json').write_text(json.dumps(receipt))
            self.assertEqual(verify_manifest_receipt(manifest), receipt)
            self.assertEqual(read_manifest(manifest), rows)

    def test_preparation_retains_the_configured_dataset_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            actual, checkout = parent / 'external dataset', parent / 'checkout'
            actual.mkdir(); checkout.mkdir()
            (actual / 'track3_train_mercaptionplus.csv').write_text('name,reason\nclip,happy\n')
            (actual / 'track2_train_mercaptionplus.csv').write_text('name,openset\nclip,happy\n')
            (actual / 'subtitle_chieng.csv').write_text('name,english\nclip,hello\n')
            link = checkout / 'dataset'
            link.symlink_to(actual, target_is_directory=True)
            rows = build_train_rows(link)
            self.assertEqual(rows[0]['video'], str(link / 'video/clip.mp4'))
            manifest = checkout / 'manifests/train.jsonl'
            save_manifest(rows, manifest, [link / 'subtitle_chieng.csv'], require_media=False)
            self.assertEqual(json.loads(manifest.read_text())['video'], '../dataset/video/clip.mp4')
            self.assertIn('../dataset/subtitle_chieng.csv', verify_manifest_receipt(manifest)['sources'])
            # A symlinked manifest directory must use the same physical base on write/read.
            external_manifests = parent / 'external manifests'
            external_manifests.mkdir()
            linked_manifests = checkout / 'linked manifests'
            linked_manifests.symlink_to(external_manifests, target_is_directory=True)
            path = linked_manifests / 'train.jsonl'
            save_manifest(rows, path, [link / 'subtitle_chieng.csv'], require_media=False)
            verify_manifest_receipt(path)
            self.assertEqual(read_manifest(path, require_media=False)[0]['video'], str(actual / 'video/clip.mp4'))

    def test_relocated_checkpoint_uses_current_manifest_and_resumes_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / 'before', Path(tmp) / 'after move'
            manifest, _, rows = self.fixture(old)
            cfg = load_config(ROOT / 'config/mer_omni_atsuko_padding.json')
            cfg.update(train_manifest=str(manifest), data_root=str(old / 'data'),
                       benchmark_manifest_dir=str(old / 'data/manifests/benchmark'),
                       calibration_file=str(old / 'data/calibration.json'))
            cfg['evaluator']['model'] = str(old / 'models/evaluator')
            cfg['train'].update(batch_size=1, gradient_accumulation=2, warmup_steps=0, lr=.01, min_lr=.01)
            full, partial = LossModel(cfg), LossModel(cfg)
            with redirect_stdout(io.StringIO()):
                seed_all(42)
                train(full, MERDataset(rows, 'description'), collate, cfg, old / 'full', torch.device('cpu'))
                seed_all(42)
                checkpoint = train(partial, MERDataset(rows, 'description'), collate, cfg,
                                   old / 'partial', torch.device('cpu'), max_steps=1)
            old.rename(new)
            moved_cfg = copy.deepcopy(cfg)
            for key in ('train_manifest', 'data_root', 'benchmark_manifest_dir', 'calibration_file'):
                moved_cfg[key] = str(new / Path(cfg[key]).relative_to(old))
            moved_cfg['evaluator']['model'] = str(new / 'models/evaluator')
            checkpoint = new / checkpoint.relative_to(old)
            saved = read_checkpoint(checkpoint)
            self.assertFalse(Path(saved['config']['train_manifest']).exists())
            loaded = read_checkpoint_training_rows(saved, moved_cfg)
            resumed = LossModel(moved_cfg)
            with redirect_stdout(io.StringIO()):
                train(resumed, MERDataset(loaded, 'description'), collate, moved_cfg,
                      new / 'partial', torch.device('cpu'), resume=checkpoint)
            torch.testing.assert_close(resumed.weight, full.weight, rtol=0, atol=0)

            # Path flexibility must not let a changed training setup or dataset through.
            wrong = copy.deepcopy(moved_cfg)
            wrong['train']['lr'] *= 2
            with self.assertRaisesRegex(ValueError, 'identical experiment settings'):
                train(LossModel(wrong), MERDataset(loaded, 'description'), collate, wrong,
                      new / 'rejected', torch.device('cpu'), resume=checkpoint)
            manifest = Path(moved_cfg['train_manifest'])
            manifest.write_text(manifest.read_text().replace('hello', 'changed text'))
            receipt_path = manifest.with_suffix('.receipt.json')
            receipt = json.loads(receipt_path.read_text())
            receipt['manifest_sha256'] = sha256(manifest)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'Checkpoint training manifest changed'):
                read_checkpoint_training_rows(saved, moved_cfg)
            with self.assertRaisesRegex(ValueError, 'identical experiment settings'):
                train(LossModel(moved_cfg), MERDataset(loaded, 'description'), collate, moved_cfg,
                      new / 'rejected data', torch.device('cpu'), resume=checkpoint)

    def test_config_resources_are_checkout_relative_and_support_home_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / 'checkout'
            cfg = load_config(ROOT / 'config/mer_omni_atsuko_padding.json')
            cfg['data_root'] = '~/mer-data'
            cfg['evaluator']['model'] = '~/evaluator'
            with patch.object(cli, 'ROOT', checkout):
                resolved = cli.resolve_config(cfg)
            self.assertEqual(Path(resolved['train_manifest']), checkout / 'data/manifests/mercaptionplus.train.jsonl')
            self.assertEqual(Path(resolved['data_root']), Path.home() / 'mer-data')
            self.assertEqual(Path(resolved['evaluator']['model']), Path.home() / 'evaluator')

    def test_cli_default_config_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / 'standalone code with spaces'
            shutil.copytree(ROOT / 'mer_omni', checkout / 'mer_omni',
                            ignore=shutil.ignore_patterns('__pycache__'))
            (checkout / 'tools').mkdir()
            (checkout / 'config').mkdir()
            shutil.copy2(ROOT / 'tools/mer_omni.py', checkout / 'tools/mer_omni.py')
            shutil.copy2(ROOT / 'config/mer_omni_atsuko_padding.json', checkout / 'config')
            result = subprocess.run([sys.executable, str(checkout / 'tools/mer_omni.py'), 'preflight'], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads(result.stdout)
            self.assertFalse(status['training']['ready'])
            self.assertFalse(status['model_config_present'])
            self.assertIn(str(checkout), status['training']['reason'])
            self.assertTrue(all(str(checkout) in v['reason'] for v in status['benchmarks'].values()))

    def test_launcher_supports_active_python_and_explicit_interpreter_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / 'checkout with spaces'
            checkout.mkdir()
            script = checkout / 'train.sh'
            shutil.copy2(ROOT / 'train.sh', script)
            calibration = checkout / 'data/calibration/mercaptionplus-atsuko-padding-bands.json'
            calibration.parent.mkdir(parents=True)
            calibration.write_text('{}')
            bindir = Path(tmp) / 'custom environment'
            bindir.mkdir()
            python = bindir / 'python'
            python.write_text('#!/bin/sh\nprintf "mock-python:%s\\n" "$PWD"\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'])
            env.pop('MER_OMNI_PYTHON', None)
            for explicit in (False, True):
                with self.subTest(explicit=explicit):
                    if explicit:
                        env['MER_OMNI_PYTHON'] = str(python)
                    result = subprocess.run(['bash', str(script)], cwd=tmp, env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('mock-python:' + str(checkout), result.stdout)
                    self.assertIn('tools/mer_omni.py', result.stdout)


if __name__ == '__main__':
    unittest.main()
