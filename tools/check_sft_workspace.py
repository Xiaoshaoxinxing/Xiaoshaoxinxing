#!/usr/bin/env python3
"""Read-only verification of the retained SFT and official face experiments."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))


def main():
    import peer_lora_epoch2 as train
    import evaluate_peer_lora_epoch2 as evaluation
    import search_peer_lora_checkpoints as search
    import run_affectgpt_official_face_merunibench as face

    cfg, run = train.check()
    retained = [31327, 34000, 62654]
    for step in retained:
        search.checkpoint_identity(step)
        checkpoint = run / f'checkpoint-{step:07d}'
        receipt = json.loads((checkpoint / 'receipt.json').read_text())
        for key, filename in [('state_sha256', 'state.pt'), ('rng_sha256', 'rng.pt')]:
            if train.sha(checkpoint / filename) != receipt[key]:
                raise ValueError(f'Checkpoint state changed: {step}/{filename}')
    selected = search.selection()
    _, plan, output = search.bind(selected['selected_step'], full=True)
    summary = json.loads((output / 'summary.json').read_text())
    recorded = {row['dataset']: row for row in summary['datasets']}
    scores = []
    for name in evaluation.BENCHMARKS:
        rows, _, receipt = evaluation.verify_predictions('peer', name)
        cache = json.loads((output / name / 'extraction.json').read_text())
        evaluation.validate_cache(cache, receipt, evaluation.check()[0])
        score = evaluation.score_dataset(rows, cache)
        if abs(score['primary'] - recorded[name]['primary']) > 1e-12:
            raise ValueError(f'Recorded score differs from cached official scoring: {name}')
        scores.append(score)
    if abs(sum(s['primary'] for s in scores) / 9 - summary['dataset_mean']) > 1e-12:
        raise ValueError('Nine-dataset mean mismatch')
    face.check()
    face_summary = json.loads((face.OUT / 'summary.json').read_text())
    face_selection = json.loads((face.OUT / 'selection.json').read_text())
    if (not face_summary['complete'] or face_summary['selected_epoch'] != face_selection['selected_epoch']
            or len(face_summary['datasets']) != 9):
        raise ValueError('Incomplete official face result')
    for row in face_summary['datasets']:
        score = json.loads((face.OUT / 'full/scores' / f"{row['dataset']}.json").read_text())
        if score != row:
            raise ValueError('Face score summary differs from per-dataset record')
    print(json.dumps(dict(passed=True, retained_checkpoints=retained,
        selected_step=selected['selected_step'], samples=sum(s['samples'] for s in scores),
        qwen_lora_mean=summary['dataset_mean'], affectgpt_face_mean=face_summary['dataset_mean'],
        face_epoch=face_summary['selected_epoch'], gpu_inference_started=False), indent=2))


if __name__ == '__main__':
    main()
