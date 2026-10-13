"""Native video/audio decoding and Qwen Omni preprocessing; local assets only."""
import math
import json
import subprocess
import warnings
import numpy as np
import torch


def _video_frame(frame, max_pixels):
    import cv2
    h, w = frame.shape[:2]
    scale = min(1.0, math.sqrt(max_pixels / (h * w)))
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _decode_video_sequential(path, max_frames, max_pixels):
    """Recover inaccurate container counts/seeking using verified decoded frames."""
    import cv2
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)],
                           capture_output=True, timeout=120, check=False)
    if probe.returncode or probe.stderr.strip():
        raise ValueError(f'Video frame verification failed: {path}: {probe.stderr.decode(errors="replace")[-500:]}')
    try:
        count = int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError(f'Cannot determine decoded frame count: {path}') from exc
    if count < 1:
        raise ValueError(f'Video has no decoded frames: {path}')
    sampled_count = max(2, min(count, max_frames))
    sampled_count -= sampled_count % 2
    indices = np.linspace(0, count - 1, sampled_count).round().astype(int)
    wanted, frames = set(indices), {}
    capture = cv2.VideoCapture(str(path))
    try:
        for index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Sequential decode disagrees with ffprobe at frame {index}: {path}')
            if index in wanted:
                frames[index] = _video_frame(frame, max_pixels)
    finally:
        capture.release()
    warnings.warn(f'Video seek/count fallback: {path}; verified {count} decoded frames; '
                  f'sampled indices {indices.tolist()}', RuntimeWarning)
    return np.stack([frames[index] for index in indices])


def decode_video(path, max_frames, max_pixels):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise ValueError(f'Video has no frames: {path}')
        sampled_count = max(2, min(count, max_frames))
        if sampled_count % 2:
            sampled_count -= 1
        frames = []
        for index in np.linspace(0, count - 1, sampled_count).round().astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                capture.release()
                return _decode_video_sequential(path, max_frames, max_pixels)
            frames.append(_video_frame(frame, max_pixels))
        return np.stack(frames)
    finally:
        capture.release()

def decode_audio(path, sample_rate):
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
               '-vn', '-ac', '1', '-ar', str(sample_rate), '-f', 'f32le', 'pipe:1']
    result = subprocess.run(command, capture_output=True, timeout=120, check=False)
    if result.returncode:
        raise ValueError(f'Audio decode failed: {path}: {result.stderr.decode(errors="replace")[-500:]}')
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    return audio

class NativeMediaCollator:
    def __init__(self, options, *, processor=None):
        self.options = options
        self.processor = processor

    def __call__(self, rows):
        if self.processor is None:
            from .encoders.omni import local_source
            try:
                from transformers import Qwen2_5OmniProcessor
            except ImportError as exc:
                raise RuntimeError('Use the requirements/train.txt environment for raw media') from exc
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                str(local_source(self.options)), local_files_only=True)
        p, opts = self.processor, self.options
        text = p.tokenizer([row['text'] or ' ' for row in rows], padding=True,
                           truncation=False, return_tensors='pt')
        if text['input_ids'].shape[1] > opts['max_text_tokens']:
            raise ValueError('Text exceeds max_text_tokens; increase the explicit limit instead of silently truncating')
        videos = [decode_video(row['video'], opts['max_frames'], opts['max_pixels']) for row in rows]
        video = p.video_processor(videos=videos, return_tensors='pt',
                                  min_pixels=opts['min_pixels'], max_pixels=opts['max_pixels'])
        waveforms = [decode_audio(row.get('audio', row['video']), opts['audio_sample_rate']) for row in rows]
        audio = p.feature_extractor(waveforms, sampling_rate=opts['audio_sample_rate'],
                                    padding='longest', truncation=False,
                                    pad_to_multiple_of=p.feature_extractor.hop_length,
                                    return_attention_mask=True, return_tensors='pt')
        if audio['input_features'].shape[-1] != audio['attention_mask'].shape[-1]:
            raise ValueError('Audio processor produced inconsistent feature/mask lengths')
        return {
            'input_ids': text['input_ids'], 'text_attention_mask': text['attention_mask'],
            'pixel_values_videos': video['pixel_values_videos'], 'video_grid_thw': video['video_grid_thw'],
            'input_features': audio['input_features'], 'feature_attention_mask': audio['attention_mask'],
        }
