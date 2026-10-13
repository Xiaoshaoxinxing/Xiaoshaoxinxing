#!/usr/bin/env python3
"""Describe existing predictions without fitting or changing any model settings."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mer_omni.evaluation import mappings, ref
from mer_omni.data import read_manifest
from mer_omni.protocol import sha256


def profile(directory, dataset, manifest_dir=None):
    directory = Path(directory).expanduser()
    manifest_dir = Path(manifest_dir).expanduser() if manifest_dir is not None else ROOT / 'data/manifests/benchmark'
    path = manifest_dir / (dataset + '.jsonl')
    rows = read_manifest(path, require_media=False)
    predictions = [json.loads(x) for x in (directory / 'predictions.jsonl').read_text().splitlines()]
    cache = json.loads((directory / 'extraction.json').read_text())
    score = json.loads((directory / 'score.json').read_text())
    assert [x['id'] for x in rows] == [x['id'] for x in predictions]
    counts = Counter(x['prediction'] for x in predictions)
    result = dict(dataset=dataset, samples=len(rows), predictions_sha256=sha256(directory / 'predictions.jsonl'),
        extraction_sha256=sha256(directory / 'extraction.json'), unique_prediction_strings=len(counts),
        largest_exact_prediction_count=counts.most_common(1)[0][1],
        note='Post-hoc error description only; no threshold, prompt or checkpoint selection.')
    if dataset == 'CMUMOSEI':
        names = [r for r in rows if r['label'] != 0]
        cells = Counter(('positive' if r['label'] > 0 else 'negative',
                         'positive' if cache['sentiment'][r['id']] == 'positive' else 'negative') for r in names)
        confusion = {gt: {p: cells[gt, p] for p in ['positive', 'negative']} for gt in ['positive', 'negative']}
        result.update(evaluated_samples=len(names), confusion=confusion,
            raw_sentiment_counts=dict(Counter(cache['sentiment'].values())),
            class_recall={gt: confusion[gt][gt] / sum(confusion[gt].values()) for gt in confusion},
            truth_counts={gt: sum(confusion[gt].values()) for gt in confusion},
            predicted_counts={p: sum(confusion[gt][p] for gt in confusion) for p in confusion})
        accuracy = sum(cells[v, v] for v in confusion) / len(names)
        assert abs(accuracy - score['accuracy']) < 1e-12
        majority = max(result['truth_counts'].values()) / len(names)
        result['majority_constant_accuracy'] = majority
        result['majority_constant_weighted_f1'] = 2 * majority**2 / (1 + majority)
    else:
        fmt, raw = mappings()
        by_class = defaultdict(list)
        wheels = []
        for index in range(1, 6):
            metric = f'case3_wheel{index}_level1'
            wheel = ref.func_get_wheel_cluster(f'wheel{index}', 'level1')
            groups = defaultdict(list)
            hits = []
            def normalize(value):
                labels = [x.lower().strip() for x in ref.string_to_list(value)]
                return set(ref.func_map_label_to_synonym(labels, fmt, raw, wheel, metric))
            for row in rows:
                gt = normalize(row['label'])
                if not gt:
                    continue
                hit = len(gt & normalize(cache['openset'][row['id']]))
                hits.append(hit)
                groups[row['label']].append(hit)
            assert abs(sum(hits)/len(hits) - score['per_wheel'][index-1][0]) < 1e-12
            per_class = {label: dict(samples=len(values), hit_rate=sum(values)/len(values))
                         for label, values in groups.items()}
            for label, values in groups.items():
                by_class[label].append(sum(values)/len(values))
            wheels.append(dict(wheel=index, scored_samples=len(hits), classes=per_class))
        result.update(per_wheel=wheels, per_class_mean_over_applicable_wheels={
            label: sum(values)/len(values) for label, values in by_class.items()},
            truth_counts=dict(Counter(r['label'] for r in rows)))
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest-dir', type=Path, default=ROOT / 'data/manifests/benchmark')
    args = p.parse_args()
    args.evaluation, args.output = args.evaluation.expanduser(), args.output.expanduser()
    result = {d: profile(args.evaluation / d, d, args.manifest_dir) for d in ['MER2023', 'MER2024', 'CMUMOSEI']
              if (args.evaluation / d / 'score.json').exists()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({d: {k: v for k, v in obj.items() if k not in ['per_wheel']}
                      for d, obj in result.items()}, ensure_ascii=False, indent=2))
