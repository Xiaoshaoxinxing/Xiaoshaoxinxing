"""Relocation must preserve receipts while still rejecting unrelated changes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mer_omni.peer_source_migration import (
    RECORD, source_identities_match, protocol_identities_match, training_configs_match,
)


class PeerSourceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.new_path = 'mer_omni/peer_lora_sft.py'
        source = self.root / self.new_path
        source.parent.mkdir()
        source.write_text('unchanged implementation\n')
        self.digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.old_path = 'previous-checkout/sft/train_sft.py'
        record = dict(schema_version=1, experiment='completed-run',
            training_source=dict(old_path=self.old_path, new_path=self.new_path),
            current_files={self.new_path: self.digest},
            changes=[dict(old_path=self.old_path, new_path=self.new_path,
                          old_sha256=self.digest, new_sha256=self.digest)])
        dest = self.root / RECORD
        dest.parent.mkdir(parents=True)
        dest.write_text(json.dumps(record))
        self.old = {self.old_path: self.digest, 'manifest.jsonl': 'fixed-data'}
        self.new = {self.new_path: self.digest, 'manifest.jsonl': 'fixed-data'}

    def test_relocated_source_matches_without_old_directory(self):
        self.assertFalse((self.root / self.old_path).exists())
        self.assertTrue(source_identities_match(self.old, self.new, root=self.root))
        self.assertEqual(self.old[self.old_path], self.digest)

    def test_changed_data_missing_file_or_wrong_source_cannot_match(self):
        for actual in [dict(self.new, **{'manifest.jsonl': 'different-data'}),
                       {self.new_path: self.digest},
                       dict(self.new, **{self.new_path: 'different-source'})]:
            with self.subTest(actual=actual):
                self.assertFalse(source_identities_match(self.old, actual, root=self.root))
        changed_old = dict(self.old, **{self.old_path: 'different-historical-source'})
        self.assertFalse(source_identities_match(changed_old, self.new, root=self.root))

    def test_disk_changes_invalidate_even_previously_computed_identity(self):
        (self.root / self.new_path).write_text('changed implementation\n')
        self.assertFalse(source_identities_match(self.old, self.new, root=self.root))

    def test_only_source_path_can_change_in_training_config(self):
        old = dict(experiment='completed-run', peer_source=self.old_path, lr=1e-4)
        new = dict(old, peer_source=self.new_path)
        self.assertTrue(training_configs_match(old, new, root=self.root))
        self.assertFalse(training_configs_match(old, dict(new, lr=2e-4), root=self.root))
        self.assertFalse(training_configs_match(old, dict(new, experiment='other-run'), root=self.root))

    def test_protocol_reference_and_extra_fields_remain_strict(self):
        old = dict(sources=self.old, reference_sha256='fixed-metrics')
        new = dict(sources=self.new, reference_sha256='fixed-metrics')
        self.assertTrue(protocol_identities_match(old, new, root=self.root))
        changed = copy.deepcopy(new)
        changed['reference_sha256'] = 'different-metrics'
        self.assertFalse(protocol_identities_match(old, changed, root=self.root))
        self.assertFalse(protocol_identities_match(old, dict(new, extra='unknown'), root=self.root))

    def test_retired_source_requires_the_exact_recorded_digest(self):
        path = self.root / RECORD
        record = json.loads(path.read_text())
        record['retired_sources'] = {'old_experiment.py': 'frozen-old-source'}
        path.write_text(json.dumps(record))
        expected = dict(self.old, **{'old_experiment.py': 'frozen-old-source'})
        self.assertTrue(source_identities_match(expected, self.new, root=self.root))
        expected['old_experiment.py'] = 'unexpected-old-source'
        self.assertFalse(source_identities_match(expected, self.new, root=self.root))
        self.assertFalse(source_identities_match(dict(self.old, unknown='extra'), self.new, root=self.root))

    def test_layout_config_migration_does_not_allow_hyperparameter_changes(self):
        path = self.root / RECORD
        record = json.loads(path.read_text())
        old = dict(experiment='completed-run', peer_source=self.old_path, lr=1e-4)
        new = dict(old, peer_source=self.new_path, source_bounds='prepared/bounds.json')
        record.update(training_config_before=old, training_config_after=new)
        path.write_text(json.dumps(record))
        self.assertTrue(training_configs_match(old, new, root=self.root))
        self.assertFalse(training_configs_match(old, dict(new, lr=2e-4), root=self.root))
        self.assertFalse(training_configs_match(dict(old, lr=2e-4), new, root=self.root))


if __name__ == '__main__':
    unittest.main()
