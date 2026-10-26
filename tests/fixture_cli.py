"""Frozen legacy CLI used only by shared-code regression tests."""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('legacy_test_cli', Path(__file__).parent/'fixtures/legacy_cli.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
run = cli.run
