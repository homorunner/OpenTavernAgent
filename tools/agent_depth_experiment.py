#!/usr/bin/env python3
"""Run the four fixed-width depth ablations on experiment's four NUMA nodes.

Uses only the standard library. Training/replay snapshots stay in a caller-owned
scratch directory; checkpoints, logs, evaluations and the latest resumable state
are archived separately. Run under a user systemd service, with KillMode=control-group.
"""
import argparse
import ast
import concurrent.futures
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

DEPTHS = (6, 8, 10, 12)
STAGES = (100, 500, 1000, 1500, 2000, 2500, 3000)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def copy_atomic(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def metadata(path, name='metadata'):
    with zipfile.ZipFile(path) as z:
        raw = z.read(name + '.npy')
    if raw[:6] != b'\x93NUMPY':
        raise ValueError(f'Invalid NPY in {path}')
    n = 2 if raw[6] == 1 else 4
    end = 8 + n + int.from_bytes(raw[8:8+n], 'little')
    header = ast.literal_eval(raw[8+n:end].decode('ascii'))
    if header['descr'] != '|u1':
        raise ValueError(f'Invalid metadata dtype in {path}')
    return json.loads(raw[end:])


def training_args(binary, work, depth, stage, seed, workers):
    args = [str(binary), 'selfplay', '--hidden', '256', '--embedding-dim', '32',
            '--blocks', str(depth), '--steps', str(stage), '--seed', str(seed),
            '--games', '128', '--sims', '256', '--search-prob', '0.33',
            '--epochs', '2', '--lr', '0.00003', '--lr-decay-steps', '0',
            '--batch', '512', '--max-samples', '80000',
            '--temperature', '0.5', '--temperature-warm', '0.5', '--split-turn', '4',
            '--lcb-stdevs', '0', '--workers', str(workers), '--probe-every', '100',
            '--probe-games', '64', '--save-every', '100', '--save-trainer-state',
            '--out', str(work / 'model.npz')]
    state = work / 'model.trainer.npz'
    if state.exists():
        m = metadata(state)
        if m['format'] != 8 or m['net']['hidden'] != 256 or m['net']['embedding_dim'] != 32 or m['net']['blocks'] != depth:
            raise ValueError(f'Unexpected trainer schema: {state}')
        args += ['--resume-state', str(state)]
    return args


def run_lanes(lanes, workers):
    """One serial lane per NUMA node; never overlap two jobs on the same CPUs."""
    def lane(slot):
        lo, hi = slot * 128, slot * 128 + 127
        env = dict(os.environ, RAYON_NUM_THREADS=str(workers))
        for cmd, log in lanes[slot]:
            log.parent.mkdir(parents=True, exist_ok=True)
            command = ['numactl', f'--physcpubind={lo}-{hi}', f'--membind={slot}', *cmd]
            with log.open('a') as f:
                f.write('\n' + json.dumps({'started': time.time(), 'command': command}) + '\n')
                f.flush()
                start = time.monotonic()
                subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
                f.write(json.dumps({'finished': time.time(), 'seconds': time.monotonic()-start}) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lane, range(4)))


def arena_jobs(binary, archive, selected, label, workers):
    lanes = [[] for _ in DEPTHS]
    for i, (a, b) in enumerate(itertools.combinations(DEPTHS, 2)):
        out = archive / f'arena_{label}_d{a}_d{b}.json'
        if out.exists():
            json.loads(out.read_text())
            continue
        cmd = [str(binary), 'arena', '--a', str(selected[a]), '--b', str(selected[b]),
               '--games', '512', '--sims', '64', '--workers', str(workers),
               '--seed', '71000000', '--out', str(out)]
        lanes[i % 4].append((cmd, out.with_suffix('.log')))
    return lanes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--seed', type=int, default=444444444)
    p.add_argument('--workers', type=int, default=128)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    a.binary, a.work, a.archive = (x.resolve() for x in (a.binary, a.work, a.archive))
    if a.work == a.archive or a.work in a.archive.parents or a.archive in a.work.parents:
        p.error('work and archive must be separate directories')
    if not 1 <= a.workers <= 128:
        p.error('workers must be in 1..128')
    manifest = dict(revision=a.revision, binary=str(a.binary),
                    binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                    depths=DEPTHS, stages=STAGES, seed=a.seed, workers=a.workers,
                    hidden=256, embedding_dim=32, ff_hidden=384,
                    lr=3e-5, epochs=2, batch=512, max_samples=80000,
                    replay_schedule='min(6, 3 + step // 200)',
                    work=str(a.work), archive=str(a.archive),
                    evaluation='EMA+raw vs bots at each stage; six paired EMA arenas every 500 rounds through 3000; final matched-compute arenas',
                    matched_compute='sum of collect_seconds+train_seconds; excludes pauses, evaluation and checkpoint I/O',
                    limitation='One training seed: screening only; replicate winners before concluding depth is better.')
    manifest = json.loads(json.dumps(manifest))
    if a.dry_run:
        print(json.dumps(dict(manifest=manifest, first_commands=[training_args(a.binary,a.work/f'd{d}',d,100,a.seed,a.workers) for d in DEPTHS]), indent=2))
        return
    a.work.mkdir(parents=True, exist_ok=True)
    a.archive.mkdir(parents=True, exist_ok=True)
    mp = a.archive / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Manifest differs; use a new experiment directory')
    atomic_json(mp, manifest)
    for stage in STAGES:
        marker = a.archive / f'stage{stage}.complete.json'
        if marker.exists():
            continue
        atomic_json(a.archive/'status.json', dict(phase='training', stage=stage, updated=time.time()))
        lanes = [[] for _ in DEPTHS]
        for slot, depth in enumerate(DEPTHS):
            work = a.work / f'd{depth}'
            work.mkdir(exist_ok=True)
            state = work / 'model.trainer.npz'
            backup = a.archive / f'd{depth}' / 'model.trainer.npz'
            if not state.exists() and backup.exists():
                copy_atomic(backup, state)
            done = metadata(state)['step'] if state.exists() else 0
            if done < stage:
                lanes[slot].append((training_args(a.binary,work,depth,stage,a.seed,a.workers),
                                    a.archive/f'd{depth}'/f'train_to_{stage}.log'))
        run_lanes(lanes, a.workers)
        for depth in DEPTHS:
            work, dst = a.work/f'd{depth}', a.archive/f'd{depth}'
            if metadata(work/'model.trainer.npz')['step'] != stage:
                raise ValueError(f'Incomplete or unexpected training step: {depth}')
            for f in work.glob('model.*'):
                if f.is_file() and (f.suffix == '.json' or (f.suffix == '.npz' and '.trainer.' not in f.name)):
                    copy_atomic(f, dst/f.name)
            copy_atomic(work/'model.trainer.npz', dst/'model.trainer.npz')
            # Immutable weight checkpoints remain; redundant replay snapshots do not.
            for f in work.glob('model.step*.trainer.npz'):
                f.unlink()
        atomic_json(a.archive/'status.json', dict(phase='evaluation', stage=stage, updated=time.time()))
        lanes = [[] for _ in DEPTHS]
        for slot, depth in enumerate(DEPTHS):
            for variant in ['ema', 'raw']:
                out = a.archive/f'd{depth}'/f'eval_step{stage}_{variant}.json'
                if out.exists():
                    json.loads(out.read_text())
                    continue
                suffix = '.ema.npz' if variant == 'ema' else '.npz'
                cmd = [str(a.binary), 'eval', '--model', str(a.archive/f'd{depth}'/f'model.step{stage}{suffix}'),
                       '--games', str(512 if stage == 100 else 2048), '--sims', '64',
                       '--workers', str(a.workers), '--seed', '61000000', '--no-baselines', '--out', str(out)]
                lanes[slot].append((cmd, out.with_suffix('.log')))
        run_lanes(lanes, a.workers)
        if stage >= 500:
            selected = {d:a.archive/f'd{d}'/f'model.step{stage}.ema.npz' for d in DEPTHS}
            run_lanes(arena_jobs(a.binary,a.archive,selected,f'step{stage}',a.workers),a.workers)
        atomic_json(marker, dict(completed=time.time()))
    curves = {}
    for depth in DEPTHS:
        rows = json.loads((a.archive/f'd{depth}'/'model.history.json').read_text())
        total = 0.0
        curve = {}
        for row in rows:
            total += row['collect_seconds'] + row['train_seconds']
            if (a.archive/f'd{depth}'/f'model.step{row["step"]}.ema.npz').exists():
                curve[row['step']] = total
        curves[depth] = curve
    budget = min(max(c.values()) for c in curves.values())
    steps = {d:max(s for s,t in c.items() if t <= budget) for d,c in curves.items()}
    atomic_json(a.archive/'matched_compute.json', dict(budget_seconds=budget,steps=steps,curves=curves))
    selected = {d:a.archive/f'd{d}'/f'model.step{s}.ema.npz' for d,s in steps.items()}
    run_lanes(arena_jobs(a.binary,a.archive,selected,'matched_compute',a.workers),a.workers)
    atomic_json(a.archive/'status.json',dict(phase='complete',updated=time.time()))


if __name__ == '__main__':
    main()
