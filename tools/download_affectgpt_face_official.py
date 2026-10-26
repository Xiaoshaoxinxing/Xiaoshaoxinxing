#!/usr/bin/env python3
"""Complete the seven upstream paper-series face checkpoints, verifying LFS hashes."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shutil
import time

import download_affectgpt_face_paper as download


def complete_checkpoint(entry,target):
    for attempt in range(3):
        parts = download.DEST / (target.stem + '.parts')
        parts.mkdir(exist_ok=True)
        jobs = [(entry, i, start, min(start + download.CHUNK, entry['size']) - 1,
                 parts / f'{i:04d}.part')
                for i, start in enumerate(range(0, entry['size'], download.CHUNK))]
        for _, _, start, end, part in jobs:
            if part.exists() and part.stat().st_size > end - start + 1:
                part.rename(part.with_name(part.name + f'.invalid-{time.time_ns()}'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            pending = [pool.submit(download.fetch_part, job) for job in jobs]
            for done, result in enumerate(concurrent.futures.as_completed(pending), 1):
                result.result()
                print(target.name, done, '/', len(jobs), flush=True)
        temporary = target.with_suffix('.assembling')
        with temporary.open('wb') as output:
            for _, _, start, end, part in jobs:
                assert part.stat().st_size == end - start + 1
                with part.open('rb') as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        assert temporary.stat().st_size == entry['size']
        if download.digest(temporary) == entry['lfs']['oid']:
            temporary.replace(target)
            return
        suffix = f'.invalid-{time.time_ns()}'
        parts.rename(parts.with_name(parts.name+suffix))
        temporary.rename(temporary.with_name(temporary.name+suffix))
        print('Hash mismatch: preserved invalid download, retrying',target.name,attempt+1,flush=True)
    raise ValueError(f'Checkpoint hash verification failed after three downloads: {target}')


def main():
    lock = (download.DEST / 'official-seven-download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    entries = json.loads((download.DEST / 'official-files.json').read_text())
    verified = []
    for entry in entries:
        target = download.DEST / Path(entry['path']).name
        if not target.exists():
            complete_checkpoint(entry,target)
        assert target.stat().st_size == entry['size']
        assert download.digest(target) == entry['lfs']['oid']
        verified.append(dict(path=str(target), sha256=entry['lfs']['oid'], bytes=entry['size']))
        print('VERIFIED', target.name, flush=True)
    path = download.DEST / 'official-seven-download-receipt.json'
    path.write_text(json.dumps(dict(revision=download.REVISION, files=verified), indent=2) + '\n')


if __name__ == '__main__':
    main()
