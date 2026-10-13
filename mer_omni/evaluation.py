"""Offline official wheel metrics and a separate local Qwen extraction stage."""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .protocol import BENCHMARKS, check_coverage, check_prediction_ids, sha256
from .reference import affectgpt as ref


@lru_cache(maxsize=1)
def mappings():
    return ref.read_format2raws(), ref.read_candidate_synonym_merge()


def read_predictions(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=True) as data:
            predictions = data['name2reason'].item()
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate prediction IDs')
        predictions = {r['id']: r['prediction'] for r in rows}
    if not predictions or any(not isinstance(x, str) for x in predictions.values()):
        raise ValueError('Expected ID -> generated text predictions')
    return predictions


def reference_fingerprint():
    root = Path(ref.__file__).parent
    paths = [root / 'affectgpt.py', *sorted((root / 'emotion_wheel').glob('*'))]
    return hashlib.sha256(''.join(sha256(p) for p in paths).encode()).hexdigest()


class ContextCheckedLLM:
    """Reject oversized extraction prompts before vLLM can ignore a request."""
    def __init__(self, llm, tokenizer, max_model_len):
        self.llm, self.tokenizer, self.max_model_len = llm, tokenizer, max_model_len

    def generate(self, prompts, sampling_params):
        for prompt in prompts:
            required = len(self.tokenizer.encode(prompt)) + sampling_params.max_tokens
            if required > self.max_model_len:
                raise ValueError(f'Extraction requires {required} tokens, exceeds context capacity '
                                 f'{self.max_model_len}; refusing to truncate')
        return self.llm.generate(prompts, sampling_params)


def extract_predictions(predictions, settings, *, sentiment=False, gpu_memory_utilization=None,
                        max_model_len=None, max_num_seqs=None, single_process_store=False):
    """No API calls. Uses AffectGPT's local vLLM backend and unchanged prompts."""
    path = Path(settings['model']).resolve()
    if not (path / 'config.json').is_file() or not list(path.glob('*.safetensors')):
        raise FileNotFoundError(f'Local evaluator weights missing: {path}; no automatic download')
    if settings['backend'] != 'vllm':
        raise ValueError('Official extraction uses vllm; no implicit alternative evaluator')
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError('Extraction requires vllm in a separate compatible environment; scoring cached outputs does not.') from exc
    for package in ('vllm', 'transformers'):
        actual = importlib.metadata.version(package)
        if actual != settings[package + '_version']:
            raise RuntimeError(f'Evaluator requires {package}=={settings[package + "_version"]}, found {actual}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    engine_options = {}
    if gpu_memory_utilization is not None:
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError('gpu_memory_utilization must be in (0, 1]')
        engine_options['gpu_memory_utilization'] = gpu_memory_utilization
    for name, value in (('max_model_len', max_model_len), ('max_num_seqs', max_num_seqs)):
        if value is not None:
            if value < 1:
                raise ValueError(f'{name} must be positive')
            engine_options[name] = value
    if single_process_store:
        import torch.distributed as dist
        if dist.is_initialized():
            if dist.get_world_size() != 1 or dist.get_rank() != 0:
                raise ValueError('In-memory rendezvous requires one evaluator process')
        else:
            dist.init_process_group('nccl', store=dist.HashStore(), rank=0, world_size=1)
    llm = LLM(model=str(path), seed=settings['seed'], **engine_options)
    if max_model_len is not None:
        llm = ContextCheckedLLM(llm, tokenizer, max_model_len)
    sampling = SamplingParams(**{key: settings[key] for key in
                              ('temperature', 'top_p', 'repetition_penalty', 'max_tokens')},
                              seed=settings.get('sampling_seed'))
    names = list(predictions)
    emotions, sentiments = {}, {}
    for start in range(0, len(names), settings['batch_size']):
        batch = names[start:start + settings['batch_size']]
        output = ref.reason_to_openset_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling,
                                           batch_reasons=[predictions[n] for n in batch])
        emotions.update(zip(batch, output))
    if sentiment:
        for start in range(0, len(names), settings['batch_size']):
            batch = names[start:start + settings['batch_size']]
            output = ref.openset_to_sentiment_qwen(llm=llm, tokenizer=tokenizer, sampling_params=sampling,
                                                   batch_reasons=[emotions[n] for n in batch])
            sentiments.update(zip(batch, output))
    return dict(openset=emotions, sentiment=sentiments,
                evaluator=settings, engine_options=engine_options,
                distributed_store='HashStore' if single_process_store else 'vllm_default',
                reference_sha256=reference_fingerprint(),
                evaluator_config_sha256=sha256(path / 'config.json'))


def score_dataset(rows, extraction, *, strict=True):
    dataset = rows[0]['dataset']
    check_coverage(rows, dataset, strict=strict)
    task = BENCHMARKS[dataset]['task']
    predictions = extraction['sentiment' if task == 'sentiment' else 'openset']
    check_prediction_ids(rows, predictions)
    gt = {r['id']: r['label'] for r in rows}
    result = dict(dataset=dataset, task=task, samples=len(rows), strict_protocol=strict,
                  expected_samples=BENCHMARKS[dataset]['count'], reference_sha256=reference_fingerprint())
    if task == 'sentiment':
        # AffectGPT drops gt==0. Neutral or malformed predictions map to 0,
        # then >0 binarization treats them as non-positive. Preserve this rule.
        names = [name for name in gt if float(gt[name]) != 0]
        if not names:
            raise ValueError('No nonzero sentiment ground truth')
        actual = [float(gt[n]) > 0 for n in names]
        predicted = [predictions[n] == 'positive' for n in names]
        weighted_f1 = float(f1_score(actual, predicted, average='weighted', zero_division=0))
        result.update(primary=weighted_f1, weighted_f1=weighted_f1,
                      accuracy=float(accuracy_score(actual, predicted)), evaluated_samples=len(names),
                      invalid_sentiment_outputs=sum(x not in ('positive', 'negative', 'neutral') for x in predictions.values()))
    else:
        format_map, raw_map = mappings()
        scores, counts = [], []
        for index in range(1, 6):
            metric = f'case3_wheel{index}_level1'
            wheel = ref.func_get_wheel_cluster(f'wheel{index}', 'level1')
            count = sum(bool(ref.func_map_label_to_synonym(
                [x.lower().strip() for x in ref.string_to_list(value)], format_map, raw_map, wheel, metric))
                        for value in gt.values())
            if count == 0:
                raise ValueError(f'No mapped ground truth for wheel{index}')
            counts.append(count)
            if task == 'basic':
                hit, mscore = ref.calculate_openset_onehot_hitrate(
                    name2gt=gt, name2pred=predictions, metric=metric,
                    format_mapping=format_map, raw_mapping=raw_map, inter_print=False)
                scores.append([float(hit), float(mscore)])
            else:
                precision, recall = ref.calculate_openset_overlap_rate(
                    name2gt=gt, name2pred=predictions, metric=metric,
                    format_mapping=format_map, raw_mapping=raw_map, inter_print=False)
                # Reference computes 0/0 when all predictions miss; define F1=0
                # for this degenerate case and record the convention in docs.
                fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                scores.append([float(fscore), float(precision), float(recall)])
        mean = np.mean(scores, axis=0).tolist()
        result.update(primary=mean[0], per_wheel=scores, evaluated_samples_per_wheel=counts)
        if task == 'basic':
            result.update(hit_rate=mean[0], mscore=mean[1])
        else:
            result.update(wheel_f1=mean[0], precision=mean[1], recall=mean[2])
    return result


def aggregate_scores(scores):
    names = [s['dataset'] for s in scores]
    if len(set(names)) != len(names) or set(names) != set(BENCHMARKS):
        raise ValueError('The MER-UniBench mean requires all nine datasets exactly once')
    for score in scores:
        if not score.get('strict_protocol') or score['samples'] != BENCHMARKS[score['dataset']]['count']:
            raise ValueError('Partial/debug scores cannot enter the benchmark mean')
        if not np.isfinite(score['primary']):
            raise ValueError('Non-finite score')
    for field in ('checkpoint_sha256', 'train_manifest_sha256', 'reference_sha256', 'evaluator_sha256', 'generation_sha256'):
        if any(not s.get(field) for s in scores) or len({s[field] for s in scores}) != 1:
            raise ValueError(f'All nine evaluations must share {field}')
    return dict(benchmark='MER-UniBench', dataset_mean=float(np.mean([s['primary'] for s in scores])),
                datasets=scores, scale='0..1', checkpoint_sha256=scores[0]['checkpoint_sha256'])
