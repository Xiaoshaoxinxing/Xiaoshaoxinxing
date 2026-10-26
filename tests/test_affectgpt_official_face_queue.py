import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_affectgpt_official_face_merunibench as task
from mer_omni.protocol import BENCHMARKS


class OfficialFaceQueueTests(unittest.TestCase):
    def test_screening_uses_all_nine_and_ties_choose_earlier(self):
        scores=[dict(dataset=name,samples=20,selection_only=True,per_checkpoint=[
            dict(epoch=epoch,primary=.8 if epoch in (35,50) else .7,checkpoint_sha256=str(epoch))
            for epoch in task.CFG['epochs']]) for name in task.CFG['datasets']]
        ranking=task.rank_screen(scores)
        self.assertEqual([s['epoch'] for s in ranking[:2]],[35,50])
        with self.assertRaises(AssertionError):task.rank_screen(scores[:-1])
        wrong=copy.deepcopy(scores);wrong[0]['per_checkpoint'][0]['checkpoint_sha256']='different'
        with self.assertRaises(AssertionError):task.rank_screen(wrong)

    def test_label_comparison_handles_open_vocabulary_lists(self):
        from mer_omni.reference.affectgpt import string_to_list
        self.assertEqual(task.normalized('worry, shocked, care',string_to_list),
                         task.normalized(['worry','shocked','care'],string_to_list))
        self.assertEqual(task.normalized('neutral',string_to_list),task.normalized(['neutral'],string_to_list))
        self.assertEqual(task.normalized(['sad','anger','worry']),task.normalized(['worry','angry','sadness']))
        self.assertNotEqual(task.normalized(['joy']),task.normalized(['sad']))
        self.assertEqual(task.normalized(-0.5),-0.5)

    def test_only_complete_nine_dataset_peer_result_releases_queue(self):
        summary=dict(selected_step=34000,dataset_mean=.75,datasets=[dict(
            dataset=name,samples=info['count'],strict_protocol=True,primary=.75,checkpoint_sha256='same')
            for name,info in BENCHMARKS.items()])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            self.assertFalse(task.dependency_ready(path))
            path.write_text(json.dumps(summary));self.assertTrue(task.dependency_ready(path))
            broken=copy.deepcopy(summary);broken['datasets'].pop()
            path.write_text(json.dumps(broken))
            with self.assertRaises(AssertionError):task.dependency_ready(path)
            broken=copy.deepcopy(summary);broken['datasets'][0]['samples']=20
            path.write_text(json.dumps(broken))
            with self.assertRaises(AssertionError):task.dependency_ready(path)
            broken=copy.deepcopy(summary);broken['datasets'][0]['checkpoint_sha256']='other'
            path.write_text(json.dumps(broken))
            with self.assertRaises(AssertionError):task.dependency_ready(path)


if __name__=='__main__':unittest.main()
