"""The user-started runner displays child progress and never trains in check mode."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_av_tokens_ablation as runner


class AVRunnerTests(unittest.TestCase):
    def test_progress_reaches_terminal_and_log_and_failures_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = io.StringIO()
            with patch.object(runner, 'LOGS', root / 'logs'), patch.object(runner, 'REPORT', root), redirect_stdout(output):
                runner.run([sys.executable, '-u', '-c',
                            "print('更新步 10 / 10,443; loss=1.25; lr=1e-5; ETA=2小时', flush=True)"], 'training')
                self.assertIn('loss=1.25', output.getvalue())
                self.assertIn('10,443', output.getvalue())
                logfile = next((root / 'logs').glob('training.*.log'))
                self.assertIn('ETA=2小时', logfile.read_text())
                with self.assertRaisesRegex(RuntimeError, '退出码 7'):
                    runner.run([sys.executable, '-u', '-c', "print('fixture failure'); raise SystemExit(7)"], 'failed_fixture')
                self.assertIn('fixture failure', next((root / 'logs').glob('failed_fixture.*.log')).read_text())

    def test_check_mode_does_not_launch_or_create_training_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'plan.json').write_text('{}')
            (root / 'smoke.json').write_text(json.dumps(dict(passed=True, optimizer_updates=0, plan_sha256='verified')))
            args = SimpleNamespace(check=True, train_only=False, eval_only=False, smoke_only=False, resume=None)
            with patch.object(runner, 'REPORT', root), patch.object(runner, 'OUTPUT', root / 'run'), \
                    patch.object(runner, 'verify') as verify, patch.object(runner, 'sha', return_value='verified'), \
                    patch.object(runner, 'run') as run, redirect_stdout(io.StringIO()):
                runner.main(args)
                verify.assert_called_once()
                run.assert_not_called()
            self.assertFalse((root / 'run').exists())

    def test_smoke_only_cannot_start_training_even_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'plan.json').write_text('{}')
            args = SimpleNamespace(check=False, train_only=False, eval_only=False, smoke_only=True, resume=None)
            def complete_check(command, stage):
                self.assertEqual(stage, 'gpu_precheck')
                (root / 'smoke.json').write_text(json.dumps(dict(passed=True, optimizer_updates=0, plan_sha256='verified')))
            with patch.object(runner, 'REPORT', root), patch.object(runner, 'OUTPUT', root / 'run'), \
                    patch.object(runner, 'verify'), patch.object(runner, 'sha', return_value='verified'), \
                    patch.object(runner, 'run', side_effect=complete_check) as run, redirect_stdout(io.StringIO()):
                runner.main(args)
                self.assertEqual(run.call_count, 1)
            self.assertFalse((root / 'run').exists())
            self.assertFalse(json.loads((root / 'status.json').read_text())['formal_training_started'])


if __name__ == '__main__':
    unittest.main()
