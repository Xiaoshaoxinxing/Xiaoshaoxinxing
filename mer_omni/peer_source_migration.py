"""Verify the recorded source relocation without rewriting historical receipts.

Live identities always contain actual paths and SHA-256 values. Only the exact
before/after pairs in the migration record may match an older sealed identity;
unrelated changes, changed weights, and changed training settings still fail.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'docs/results/sft-layout-migration-20261026.json'


def _record(root):
    record = json.loads((root / RECORD).read_text())
    if record['schema_version'] != 1:
        raise ValueError('Unsupported peer source migration')
    for name, expected in record['current_files'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Migrated source changed after verification: {name}')
    return record


def source_identities_match(expected, actual, *, root=ROOT):
    if expected == actual:
        return True
    try:
        record = _record(root)
        migrated = dict(expected)
        for name, digest in record.get('retired_sources', {}).items():
            if name in migrated:
                if migrated[name] != digest:
                    return False
                del migrated[name]
        for change in record['changes']:
            old, new = change['old_path'], change['new_path']
            if old not in expected:
                continue
            if expected[old] != change['old_sha256']:
                return False
            if old != new and new in migrated:
                return False
            del migrated[old]
            migrated[new] = change['new_sha256']
        return migrated == actual
    except (OSError, KeyError, TypeError, ValueError):
        return False


def protocol_identities_match(expected, actual, *, root=ROOT):
    return (set(expected) == set(actual)
            and {k: v for k, v in expected.items() if k != 'sources'}
            == {k: v for k, v in actual.items() if k != 'sources'}
            and source_identities_match(expected['sources'], actual['sources'], root=root))


def training_configs_match(expected, actual, *, root=ROOT):
    if expected == actual:
        return True
    try:
        record = _record(root)
        if 'training_config_before' in record:
            return (expected == record['training_config_before']
                    and actual == record['training_config_after'])
        relocation = record['training_source']
        if (expected['experiment'] != record['experiment']
                or expected['peer_source'] != relocation['old_path']):
            return False
        migrated = dict(expected, peer_source=relocation['new_path'])
        return migrated == actual
    except (OSError, KeyError, TypeError, ValueError):
        return False


def evaluation_configs_match(expected, actual, *, root=ROOT):
    if expected == actual:
        return True
    try:
        record = _record(root)
        return (expected == record['evaluation_config_before']
                and actual == record['evaluation_config_after'])
    except (OSError, KeyError, TypeError, ValueError):
        return False
