"""Protocol checks using real cached benchmark extraction, without GPU generation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import evaluate_peer_lora_epoch2 as evaluate


class PeerEvaluationTests(unittest.TestCase):
    def test_benchmark_labels_cannot_reach_prompt(self):
        seen={}
        class Peer:
            @staticmethod
            def conversation_for(row,cache,frames):
                seen.update(row)
                return row
        class Processor:
            @staticmethod
            def apply_chat_template(conversation,**kwargs):
                return {'input_ids':'sentinel'}
        row=dict(id='sample',text='subtitle',video='clip.mp4',audio='clip.wav',
                 label='secret-label',target='secret-target',question='answer from label')
        evaluate.peer_prompt(Peer,Processor(),row,{'max_frames':2},Path('cache'),'cpu')
        self.assertEqual(set(seen),{'id','text','video','audio','question'})
        self.assertEqual(seen['question'],evaluate.OV_QUESTION)

    def test_cached_official_full_score_and_overlap_subsets(self):
        manifest = ROOT / 'data/manifests/benchmark/OVMERDPlus.jsonl'
        if not manifest.exists() and not manifest.with_suffix('.receipt.json').exists():
            self.skipTest('Requires locally prepared OV-MERD+ annotations, excluded from the source repository')
        source=ROOT/'tests/fixtures/ovmerdplus-reference'
        rows=evaluate.benchmark('OVMERDPlus')
        receipt=json.loads((source/'predictions.receipt.json').read_text())
        cache=json.loads((source/'extraction.json').read_text())
        expected=json.loads((source/'score.json').read_text())
        cfg=json.loads(evaluate.CONFIG.read_text())
        # Historical cache stores an absolute path; new configs use a project-relative path.
        self.assertEqual((ROOT/cfg['evaluator']['model']).resolve(),Path(cache['evaluator']['model']).resolve())
        self.assertEqual({k:v for k,v in cfg['evaluator'].items() if k!='model'},
                         {k:v for k,v in cache['evaluator'].items() if k!='model'})
        cfg['evaluator']=cache['evaluator']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'OVMERDPlus';out.mkdir()
            (out/'extraction.json').write_text(json.dumps(cache))
            with patch.object(evaluate,'check',return_value=(cfg,{})), \
                 patch.object(evaluate,'layout',return_value=root), \
                 patch.object(evaluate,'verify_predictions',return_value=(rows,{},receipt)):
                evaluate.score('peer','OVMERDPlus')
                result=json.loads((out/'score.json').read_text())
                self.assertAlmostEqual(result['primary'],expected['primary'],places=12)
                subsets=json.loads((out/'overlap-subsets.json').read_text())
                self.assertEqual(subsets['overlap']['samples'],99)
                self.assertEqual(subsets['nonoverlap']['samples'],433)
                self.assertFalse(subsets['nonoverlap']['strict_protocol'])
                before=(out/'score.json').read_bytes()
                evaluate.score('peer','OVMERDPlus')
                self.assertEqual(before,(out/'score.json').read_bytes())
                result['primary']+=.01
                (out/'score.json').write_text(json.dumps(result))
                with self.assertRaises(ValueError):evaluate.score('peer','OVMERDPlus')

    def test_mean_requires_nine_complete_consistent_scores(self):
        scores=[dict(dataset=name,samples=spec['count'],strict_protocol=True,primary=.5,
                     checkpoint_sha256='checkpoint',train_manifest_sha256='train',
                     reference_sha256='reference',evaluator_sha256='evaluator',generation_sha256='generation')
                for name,spec in evaluate.BENCHMARKS.items()]
        self.assertEqual(evaluate.aggregate_scores(scores)['dataset_mean'],.5)
        with self.assertRaises(ValueError):evaluate.aggregate_scores(scores[:-1])
        mixed=copy.deepcopy(scores);mixed[0]['checkpoint_sha256']='other'
        with self.assertRaises(ValueError):evaluate.aggregate_scores(mixed)
        partial=copy.deepcopy(scores);partial[-1]['samples']=433
        with self.assertRaises(ValueError):evaluate.aggregate_scores(partial)

    def test_peer_greedy_budget_preserves_complete_native_input(self):
        settings=json.loads(evaluate.CONFIG.read_text())['generation']
        self.assertFalse(settings['do_sample'])
        self.assertEqual(settings['eos_token_id'],151645)
        self.assertEqual(evaluate.generation_budget(settings,500),256)
        self.assertEqual(evaluate.generation_budget(settings,2000),256)
        self.assertEqual(evaluate.generation_budget(settings,32512),256)
        with self.assertRaises(ValueError):evaluate.generation_budget(settings,32513)

    def test_long_benchmark_audio_extends_buffer_without_truncation(self):
        from types import SimpleNamespace
        seen={}
        peer=SimpleNamespace(conversation_for=lambda row,cache,frames:row)
        def process(conversation,**kwargs):
            seen.update(kwargs);return {}
        processor=SimpleNamespace(apply_chat_template=process,
                                  feature_extractor=SimpleNamespace(n_samples=4800000,hop_length=160))
        row=dict(id='long',text='subtitle',audio='long.wav',video='long.mp4')
        evaluate.peer_prompt(peer,processor,row,{'max_frames':2},Path('cache'),'cpu',audio_samples=4879360)
        self.assertEqual(seen['audio_kwargs'],dict(max_length=4879360,truncation=False))


if __name__=='__main__':unittest.main()
