import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import search_peer_lora_checkpoints as search


class CheckpointSearchTests(unittest.TestCase):
    def test_completed_training_supervisor_only_checks_current_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); out=root/'run'; out.mkdir()
            for name in ['completed.json','coverage.json','preflight.json']:
                (out/name).write_text('{}')
            with patch.object(search.train,'ROOT',root), \
                 patch.object(search.train,'check',return_value=({'experiment':'finished'},out)), \
                 patch.object(search.base,'check',return_value=({},{})), \
                 patch.object(search.train,'wait_resources',side_effect=AssertionError('Must not restart training')), \
                 patch.object(search.train.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                search.train.supervise()
            self.assertEqual(run.call_count,1)
            command=run.call_args.args[0]
            self.assertEqual(command[-2:],[str(root/'tools/search_peer_lora_checkpoints.py'),'pipeline'])
            self.assertNotIn('A',command)

    def test_fixed_sample_excludes_training_and_preserves_source_order(self):
        rows=[dict(id=f'id-{i:03d}') for i in range(80)]
        training={r['id'] for r in rows[:12]}
        first=search.sample_rows(rows,training,20,42)
        self.assertEqual(first,search.sample_rows(rows,training,20,42))
        self.assertEqual(len(first),20)
        self.assertFalse({r['id'] for r in first}&training)
        self.assertEqual(first,sorted(first,key=lambda r:r['id']))
        self.assertNotEqual(first,search.sample_rows(rows,training,20,43))
        with self.assertRaises(ValueError):search.sample_rows(rows,training,100,42)

    def scores(self):
        return [dict(dataset=name,samples=20,strict_protocol=False,primary=index/10,
            checkpoint_sha256='checkpoint',train_manifest_sha256='train',reference_sha256='reference',
            generation_sha256='generation',evaluator_sha256='evaluator')
            for index,name in enumerate(search.BENCHMARKS)]

    def test_screening_mean_requires_all_nine_and_consistent_weights(self):
        scores=self.scores()
        self.assertAlmostEqual(search.screen_mean(scores,20),.4)
        with self.assertRaises(AssertionError):search.screen_mean(scores[:-1],20)
        mixed=copy.deepcopy(scores);mixed[0]['checkpoint_sha256']='different'
        with self.assertRaises(AssertionError):search.screen_mean(mixed,20)
        partial=copy.deepcopy(scores);partial[0]['samples']=19
        with self.assertRaises(AssertionError):search.screen_mean(partial,20)
        with self.assertRaises(ValueError):search.aggregate_scores(scores)

    def test_highest_screening_mean_wins_and_ties_choose_earlier(self):
        ranked=search.choose([dict(checkpoint_step=62654,screening_mean=.8),
            dict(checkpoint_step=34000,screening_mean=.7),
            dict(checkpoint_step=58000,screening_mean=.8)])
        self.assertEqual([r['checkpoint_step'] for r in ranked],[58000,62654,34000])


if __name__=='__main__':unittest.main()
