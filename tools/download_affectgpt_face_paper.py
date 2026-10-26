#!/usr/bin/env python3
"""Download the two official face checkpoints identified by released paper scores."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/AffectGPT-face'
REVISION = '9d82d878cc6f5a530af01c68319a27362c29dc07'
NAMES = ['checkpoint_000050_loss_0.527.pth', 'checkpoint_000045_loss_0.551.pth']
CHUNK = 16 * 1024 * 1024


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fetch_part(job):
    entry, index, start, end, part = job
    wanted = end - start + 1
    for attempt in range(12):
        size = part.stat().st_size if part.exists() else 0
        if size == wanted:
            return part
        if size > wanted:
            raise ValueError(f'Oversized part: {part}')
        offset = start + size
        url = (f'https://huggingface.co/MERChallenge/AffectGPT/resolve/{REVISION}/'
               f'{entry["path"]}?download=true&offset={offset}')
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-{end}',
                                                       'User-Agent': 'AffectGPT-reproduction'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                expected = f'bytes {offset}-{end}/{entry["size"]}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError(f'Unexpected range response: {response.status}, {response.headers.get("Content-Range")}')
                with part.open('ab') as output:
                    while block := response.read(min(1024 * 1024, wanted - output.tell())):
                        output.write(block)
                    output.flush()
            if part.stat().st_size != wanted:
                raise IOError('Truncated response')
            return part
        except Exception as error:
            print(f'{Path(entry["path"]).name} part {index} attempt {attempt + 1}: '
                  f'{type(error).__name__}: {error}', flush=True)
            if attempt == 11:
                raise
            time.sleep(min(attempt + 1, 5))
    raise RuntimeError('Unreachable')


def main():
    metadata = json.loads((DEST / 'official-files.json').read_text())
    entries = [next(e for e in metadata if Path(e['path']).name == name) for name in NAMES]
    jobs, plans = [], []
    for entry in entries:
        target = DEST / Path(entry['path']).name
        if target.exists():
            if digest(target) != entry['lfs']['oid']:
                raise ValueError(f'Existing checkpoint has incorrect hash: {target}')
            print(f'Already verified: {target.name}', flush=True)
            continue
        parts = DEST / (target.stem + '.parts')
        parts.mkdir(exist_ok=True)
        current = []
        for index, start in enumerate(range(0, entry['size'], CHUNK)):
            end = min(start + CHUNK, entry['size']) - 1
            job = (entry, index, start, end, parts / f'{index:04d}.part')
            current.append(job)
            jobs.append(job)
        plans.append((entry, target, current))
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for future in concurrent.futures.as_completed([executor.submit(fetch_part, job) for job in jobs]):
            future.result()
            completed += 1
            print(f'Parts complete: {completed}/{len(jobs)}', flush=True)
    verified = []
    for entry, target, current in plans:
        temporary = target.with_suffix('.assembling')
        with temporary.open('wb') as output:
            for _, _, start, end, part in current:
                assert part.stat().st_size == end - start + 1
                with part.open('rb') as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        assert temporary.stat().st_size == entry['size']
        actual = digest(temporary)
        if actual != entry['lfs']['oid']:
            raise ValueError(f'Checkpoint hash mismatch: {target.name}: {actual}')
        temporary.replace(target)
        verified.append(dict(path=str(target), sha256=actual, bytes=entry['size']))
        print(f'VERIFIED {target.name} {actual}', flush=True)
    (DEST / 'paper-download-receipt.json').write_text(json.dumps(
        dict(revision=REVISION, downloaded=verified, requested=NAMES), indent=2) + '\n')


if __name__ == '__main__':
    main()
