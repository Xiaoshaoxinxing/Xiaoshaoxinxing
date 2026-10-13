"""Native media/load contracts retained for the standalone generative project."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import WhisperFeatureExtractor, Qwen2VLVideoProcessor
from safetensors.torch import save_file

from mer_omni.data import MERCollator
from mer_omni.encoders.omni import OmniComponents, local_source
from test_mer_omni import config, TinyTokenizer, make_model
from fixtures import tiny_thinker


class AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_runtime_modules_come_from_this_checkout(self):
        checkout = Path(__file__).resolve().parents[1]
        loaded = [Path(module.__file__).resolve() for name, module in list(sys.modules.items())
                  if (name == 'mer_omni' or name.startswith('mer_omni.'))
                  and getattr(module, '__file__', None)]
        self.assertTrue(loaded)
        self.assertTrue(all(path.is_relative_to(checkout) for path in loaded))

    def test_missing_model_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'No download'):
                local_source(dict(config()['omni'], source=tmp))

    def test_full_checkpoint_loading_keeps_lm_head_and_rejects_missing_weights(self):
        thinker = tiny_thinker()
        head = thinker.lm_head.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text(json.dumps(dict(model_type='qwen2_5_omni', thinker_config=thinker.config.to_dict())))
            weights = {'thinker.'+key: value for key, value in thinker.state_dict().items()}
            weights['talker.unused'] = torch.ones(1)
            weights['token2wav.unused'] = torch.ones(1)
            save_file(weights, str(root / 'model.safetensors'), metadata={'format':'pt'})
            with patch('mer_omni.encoders.omni.local_source', return_value=root):
                loaded = OmniComponents.from_pretrained(config()['omni'])
            torch.testing.assert_close(loaded.thinker.lm_head.weight, head, rtol=0, atol=0)
            del weights['thinker.audio_tower.conv1.weight']
            save_file(weights, str(root / 'model.safetensors'), metadata={'format':'pt'})
            with patch('mer_omni.encoders.omni.local_source', return_value=root):
                with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                    OmniComponents.from_pretrained(config()['omni'])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for synthetic media')
    def test_decoded_video_audio_to_generative_loss(self):
        class Tokenizer(TinyTokenizer):
            def __call__(self, texts, **kwargs):
                return dict(input_ids=torch.ones(len(texts), 4, dtype=torch.long),
                            attention_mask=torch.ones(len(texts), 4, dtype=torch.long))
        cfg = config()
        cfg['mcm']['enabled'] = False
        cfg['omni'].update(max_frames=4, min_pixels=56*56, max_pixels=56*56)
        processor = SimpleNamespace(tokenizer=Tokenizer(),
            feature_extractor=WhisperFeatureExtractor(feature_size=128),
            video_processor=Qwen2VLVideoProcessor(min_pixels=56*56, max_pixels=56*56))
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / 'clip.mp4'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                'color=c=blue:s=56x56:r=8:d=0.5', '-f', 'lavfi', '-i',
                'sine=frequency=440:sample_rate=16000:duration=0.5',
                '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(media)], check=True)
            batch = MERCollator(cfg['omni'], processor=processor)([dict(id='synthetic', text='hello',
                video=str(media), audio=str(media), question='what emotion', answer='happy')])
            self.assertEqual(batch['omni_inputs']['video_grid_thw'].tolist(), [[2,4,4]])
            model = make_model(cfg).eval()
            self.assertTrue(torch.isfinite(model(**batch)['loss']))


if __name__ == '__main__':
    unittest.main()
