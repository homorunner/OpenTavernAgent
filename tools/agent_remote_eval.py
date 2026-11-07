#!/usr/bin/env python3
"""Evaluate immutable depth checkpoints on borrowed hosts without blocking training."""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import time
import zipfile

from agent_depth_experiment import DEPTHS, STAGES, atomic_json
from agent_remote_job import SSH, cleanup, run_remote


def ssh(host, args, **kw):
    return subprocess.run([*SSH, host, shlex.join(args)], check=True, capture_output=True, **kw)


def complete(path):
    try:
        json.loads(path.read_text())
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def fetch(source, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix+'.download')
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    f'{source}:{remote}', str(tmp)], check=True, timeout=180)
    tmp.replace(local)


def publish(source, archive, relative, local):
    remote = str(Path(archive)/relative)
    tmp = remote+'.remote-upload'
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    str(local), f'{source}:{tmp}'], check=True, timeout=60)
    ssh(source, ['mv', tmp, remote], timeout=30)


def snapshot(source, archive):
    code = r'''
import json,pathlib,sys
archive=pathlib.Path(sys.argv[1])
m=json.loads((archive/'manifest.json').read_text())
work=pathlib.Path(m['work'])
checkpoints={}
for root in [work,archive]:
 for d in m['depths']:
  for p in (root/f'd{d}').glob('model.step*.npz'):
   if '.trainer.' in p.name: continue
   checkpoints[f'd{d}/{p.name}']=str(p)
results={str(p.relative_to(archive)):str(p) for p in archive.glob('d*/eval_step*.json')}
results.update({p.name:str(p) for p in archive.glob('arena_*.json')})
status=json.loads((archive/'status.json').read_text())
matched=json.loads((archive/'matched_compute.json').read_text()) if (archive/'matched_compute.json').exists() else None
print(json.dumps(dict(manifest=m,checkpoints=checkpoints,results=results,status=status,matched=matched)))
'''
    return json.loads(ssh(source, ['python3', '-B', '-c', code, archive], timeout=45).stdout)


def jobs_for(index):
    checkpoints = index['checkpoints']
    jobs = []
    for step in STAGES:
        for depth in DEPTHS:
            for variant in ['ema', 'raw']:
                name = f'd{depth}/model.step{step}' + ('.ema.npz' if variant == 'ema' else '.npz')
                if name not in checkpoints:
                    continue
                jobs.append(dict(id=f'eval_{step}_d{depth}_{variant}', kind='eval', step=step,
                    relative=f'd{depth}/eval_step{step}_{variant}.json', inputs={'model.npz': name},
                    args=['eval', '--model', 'model.npz', '--games', str(512 if step == 100 else 2048),
                          '--sims', '64', '--seed', '61000000', '--no-baselines', '--out', 'result.json']))
        for a, b in itertools.combinations(DEPTHS, 2):
            inputs = {'a.npz': f'd{a}/model.step{step}.ema.npz', 'b.npz': f'd{b}/model.step{step}.ema.npz'}
            if all(p in checkpoints for p in inputs.values()):
                jobs.append(arena_job(f'step{step}_d{a}_d{b}', inputs, step))
    if index.get('matched'):
        steps = index['matched']['steps']
        for a, b in itertools.combinations(DEPTHS, 2):
            inputs = {'a.npz': f'd{a}/model.step{steps[str(a)]}.ema.npz',
                      'b.npz': f'd{b}/model.step{steps[str(b)]}.ema.npz'}
            if all(p in checkpoints for p in inputs.values()):
                jobs.append(arena_job(f'matched_compute_d{a}_d{b}', inputs, STAGES[-1]+1))
    return jobs


def arena_job(label, inputs, step):
    return dict(id='arena_'+label, kind='arena', step=step, relative=f'arena_{label}.json', inputs=inputs,
                args=['arena', '--a', 'a.npz', '--b', 'b.npz', '--games', '512', '--sims', '64',
                      '--seed', '71000000', '--out', 'result.json'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='experiment')
    p.add_argument('--archive', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--hosts', type=Path, required=True)
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    a.root = a.root.resolve()
    a.root.mkdir(parents=True, exist_ok=True)
    hosts = json.loads(a.hosts.read_text())
    if not hosts or len({h['host'] for h in hosts}) != len(hosts):
        p.error('Hosts must be nonempty and unique')
    total = len(STAGES)*(len(DEPTHS)*2+6)+6
    binary = a.root/'tavernagent'
    first = snapshot(a.source, a.archive)
    digest = first['manifest']['binary_sha256']
    if not binary.exists():
        fetch(a.source, first['manifest']['binary'], binary)
    if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
        raise ValueError('Evaluation binary differs from training binary')
    binary.chmod(0o700)
    atomic_json(a.root/'manifest.json', dict(source=a.source, archive=a.archive,
        training_binary_sha256=digest, hosts=hosts, stages=STAGES, total_jobs=total))
    quarantined, failures, retry_at = {}, {}, {}

    def recover(skip=()):
        quarantined.clear()
        for path in (a.root/'jobs').glob('*/assignment.json'):
            record = json.loads(path.read_text())
            if path.parent.name in skip or record.get('cleanup_verified'):
                continue
            try:
                cleanup(record['host'], record['remote'], record['token'])
                record['cleanup_verified'] = True
                atomic_json(path, record)
            except Exception as exc:
                quarantined[record['host']] = str(exc)

    recover()
    active = {}
    last_scan = 0
    last_recovery = time.time()
    index = first
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        while True:
            now = time.time()
            for future in list(active):
                if not future.done():
                    continue
                job, host = active.pop(future)
                try:
                    response = future.result()
                    result = response['result']
                    result['_execution'] = dict(host=host['host'], workers=host['cpus'],
                        binary_sha256=digest, seconds=response['seconds'], cleanup_verified=True,
                        checkpoints=job['checkpoints'], completed=time.time())
                    output = a.root/'results'/job['relative']
                    output.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(output, result)
                    failures.pop(job['id'], None)
                    print(json.dumps(dict(job=job['id'], host=host['host'], state='complete',
                                          seconds=response['seconds'])), flush=True)
                except Exception as exc:
                    failures[job['id']] = str(exc)
                    retry_at[job['id']] = now+300
                    assignment = a.root/'jobs'/job['id']/'assignment.json'
                    if assignment.exists() and not json.loads(assignment.read_text()).get('cleanup_verified'):
                        quarantined[host['host']] = str(exc)
                    print(json.dumps(dict(job=job['id'], host=host['host'], error=str(exc))), flush=True)
            if now-last_scan >= a.poll_seconds:
                index = snapshot(a.source, a.archive)
                if index['manifest']['binary_sha256'] != digest:
                    raise ValueError('Training binary changed during evaluation')
                for relative, remote in index['results'].items():
                    local = a.root/'results'/relative
                    if not complete(local):
                        fetch(a.source, remote, local)
                last_scan = now
            planned = jobs_for(index)
            busy_hosts = {h['host'] for _, h in active.values()}
            busy_jobs = {j['id'] for j, _ in active.values()}
            if quarantined and now-last_recovery > 300:
                recover(busy_jobs)
                last_recovery = now
            ready = [j for j in planned if j['id'] not in busy_jobs
                     and not complete(a.root/'results'/j['relative']) and retry_at.get(j['id'], 0) <= now]
            for host in hosts:
                if not ready or host['host'] in busy_hosts or host['host'] in quarantined:
                    continue
                job = ready.pop(0)
                resolved = {}
                for alias, name in job['inputs'].items():
                    local = a.root/'checkpoints'/name
                    if not local.exists():
                        fetch(a.source, index['checkpoints'][name], local)
                        with zipfile.ZipFile(local) as z:
                            if z.testzip() is not None:
                                local.unlink()
                                raise ValueError('Corrupt checkpoint transfer')
                    resolved[alias] = str(local)
                job = dict(job, checkpoints=job['inputs'], inputs=resolved,
                           args=job['args']+['--workers', str(host['cpus'])])
                future = pool.submit(run_remote, job, host, binary, a.root/'jobs'/job['id'])
                active[future] = (job, host)
            # Publishing is retried independently; failed transfers never repeat a completed match.
            published = 0
            for job in planned:
                local = a.root/'results'/job['relative']
                if not complete(local):
                    continue
                if job['relative'] not in index['results']:
                    publish(a.source, a.archive, job['relative'], local)
                    index['results'][job['relative']] = str(Path(a.archive)/job['relative'])
                published += 1
            finished = published == total and not active
            state = dict(state='complete' if finished else 'running', updated=time.time(),
                completed=published, total=total, available=len(planned), failed=failures,
                quarantined=quarantined, active=[dict(id=j['id'], kind=j['kind'], host=h['host']) for j,h in active.values()])
            atomic_json(a.root/'status.json', state)
            publish(a.source, a.archive, 'remote_eval_status.json', a.root/'status.json')
            if finished or (a.once and not active and not ready):
                break
            time.sleep(min(a.poll_seconds, 5))


if __name__ == '__main__':
    main()
