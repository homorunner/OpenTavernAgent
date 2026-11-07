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
import threading
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


def run_job(slot, cmd, log, workers, abort=None):
    """A lane owns its NUMA node for the whole subprocess lifetime."""
    lo, hi = slot * 128, slot * 128 + 127
    env = dict(os.environ, RAYON_NUM_THREADS=str(workers))
    log.parent.mkdir(parents=True, exist_ok=True)
    command = ['numactl', f'--physcpubind={lo}-{hi}', f'--membind={slot}', *cmd]
    with log.open('a') as f:
        f.write('\n' + json.dumps({'started': time.time(), 'command': command}) + '\n')
        f.flush()
        start = time.monotonic()
        with subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT) as proc:
            while True:
                if abort is not None and abort.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise RuntimeError('Another experiment lane failed')
                try:
                    code = proc.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if code:
                raise subprocess.CalledProcessError(code, command)
        f.write(json.dumps({'finished': time.time(), 'seconds': time.monotonic()-start}) + '\n')


def run_lanes(lanes, workers):
    """Run a final batch after independent training has completed."""
    abort = threading.Event()
    def lane(slot):
        try:
            for cmd, log in lanes[slot]:
                run_job(slot, cmd, log, workers, abort)
        except BaseException:
            abort.set()
            raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(DEPTHS)) as pool:
        list(pool.map(lane, range(len(DEPTHS))))


def result_complete(path):
    try:
        json.loads(path.read_text())
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def arena_jobs(binary, archive, selected, label, workers):
    lanes = [[] for _ in DEPTHS]
    for i, (a, b) in enumerate(itertools.combinations(DEPTHS, 2)):
        out = archive / f'arena_{label}_d{a}_d{b}.json'
        if result_complete(out):
            continue
        cmd = [str(binary), 'arena', '--a', str(selected[a]), '--b', str(selected[b]),
               '--games', '512', '--sims', '64', '--workers', str(workers),
               '--seed', '71000000', '--out', str(out)]
        lanes[i % 4].append((cmd, out.with_suffix('.log')))
    return lanes


class ReadyArenas:
    """Claim each ready pair once; idle lanes can help without occupying training CPUs."""
    def __init__(self, args):
        self.condition = threading.Condition()
        self.abort = threading.Event()
        self.finished = set()
        self.running = 0
        self.pending = []
        for stage in STAGES:
            if stage < 500:
                continue
            selected = {d: args.archive/f'd{d}'/f'model.step{stage}.ema.npz' for d in DEPTHS}
            for lane in arena_jobs(args.binary, args.archive, selected, f'step{stage}', args.workers):
                for cmd, log in lane:
                    inputs = [Path(cmd[cmd.index(flag)+1]) for flag in ['--a', '--b']]
                    self.pending.append((inputs, cmd, log))

    def claim(self):
        with self.condition:
            for i, (inputs, cmd, log) in enumerate(self.pending):
                if all(p.exists() for p in inputs):
                    self.pending.pop(i)
                    self.running += 1
                    return cmd, log
        return None

    def complete(self):
        with self.condition:
            self.running -= 1
            self.condition.notify_all()


def archive_depth(work, dst):
    for f in work.glob('model.*'):
        if f.is_file() and (f.suffix == '.json' or (f.suffix == '.npz' and '.trainer.' not in f.name)):
            copy_atomic(f, dst/f.name)
    copy_atomic(work/'model.trainer.npz', dst/'model.trainer.npz')
    for f in work.glob('model.step*.trainer.npz'):
        f.unlink()


def run_independent(args):
    arenas = ReadyArenas(args)

    def lane(slot):
        depth = DEPTHS[slot]
        work, dst = args.work/f'd{depth}', args.archive/f'd{depth}'
        work.mkdir(parents=True, exist_ok=True)
        dst.mkdir(parents=True, exist_ok=True)

        def status(phase, **fields):
            atomic_json(dst/'status.json', dict(phase=phase, depth=depth, updated=time.time(), **fields))

        def drain_arenas():
            while not arenas.abort.is_set():
                job = arenas.claim()
                if job is None:
                    return
                cmd, log = job
                status('arena', job=log.stem)
                run_job(slot, cmd, log, args.workers, arenas.abort)
                if not result_complete(Path(cmd[cmd.index('--out')+1])):
                    raise ValueError(f'Missing or incomplete arena result: {log}')
                arenas.complete()

        try:
            state, backup = work/'model.trainer.npz', dst/'model.trainer.npz'
            if not state.exists() and backup.exists():
                copy_atomic(backup, state)
            for stage in STAGES:
                if arenas.abort.is_set():
                    raise RuntimeError('Another experiment lane failed')
                marker = dst/f'stage{stage}.complete.json'
                if marker.exists():
                    continue
                done = metadata(state)['step'] if state.exists() else 0
                if done < stage:
                    status('training', stage=stage)
                    run_job(slot, training_args(args.binary, work, depth, stage, args.seed, args.workers),
                            dst/f'train_to_{stage}.log', args.workers, arenas.abort)
                    done = metadata(state)['step']
                    if done != stage:
                        raise ValueError(f'Unexpected training step for d{depth}: {done}, expected {stage}')
                status('archiving', stage=stage)
                if done == stage:
                    archive_depth(work, dst)
                # A resumed lane may already be past this stage. Reuse its immutable weights.
                for suffix in ['.ema.npz', '.npz']:
                    checkpoint = dst/f'model.step{stage}{suffix}'
                    if not checkpoint.exists():
                        copy_atomic(work/checkpoint.name, checkpoint)
                with arenas.condition:
                    arenas.condition.notify_all()
                status('evaluation', stage=stage)
                for variant in ['ema', 'raw']:
                    out = dst/f'eval_step{stage}_{variant}.json'
                    if result_complete(out):
                        continue
                    suffix = '.ema.npz' if variant == 'ema' else '.npz'
                    cmd = [str(args.binary), 'eval', '--model', str(dst/f'model.step{stage}{suffix}'),
                           '--games', str(512 if stage == 100 else 2048), '--sims', '64',
                           '--workers', str(args.workers), '--seed', '61000000', '--no-baselines', '--out', str(out)]
                    run_job(slot, cmd, out.with_suffix('.log'), args.workers, arenas.abort)
                    if not result_complete(out):
                        raise ValueError(f'Missing or incomplete evaluation: {out}')
                atomic_json(marker, dict(completed=time.time()))
                drain_arenas()
            with arenas.condition:
                arenas.finished.add(depth)
                arenas.condition.notify_all()
            # Finished lanes help with ready arenas while other depths keep training.
            while not arenas.abort.is_set():
                drain_arenas()
                with arenas.condition:
                    if len(arenas.finished) == len(DEPTHS):
                        if not arenas.pending and arenas.running == 0:
                            break
                        if arenas.pending and arenas.running == 0:
                            # All checkpoints must exist after every lane has finished.
                            if not any(all(p.exists() for p in inputs) for inputs, _, _ in arenas.pending):
                                raise ValueError('Arena inputs missing after training completed')
                    # Recheck readiness under the lock to avoid sleeping through a publication.
                    if any(all(p.exists() for p in inputs) for inputs, _, _ in arenas.pending):
                        continue
                    status('waiting', stage=STAGES[-1])
                    arenas.condition.wait(timeout=10)
            if arenas.abort.is_set():
                raise RuntimeError('Another experiment lane failed')
            status('complete', stage=STAGES[-1])
        except BaseException as exc:
            arenas.abort.set()
            with arenas.condition:
                arenas.condition.notify_all()
            status('failed', error=str(exc))
            raise

    atomic_json(args.archive/'status.json', dict(phase='independent', target_steps=STAGES[-1], updated=time.time()))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(DEPTHS)) as pool:
            list(pool.map(lane, range(len(DEPTHS))))
    except BaseException as exc:
        atomic_json(args.archive/'status.json', dict(phase='failed', error=str(exc), updated=time.time()))
        raise


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
                    scheduling='independent NUMA lanes; ready arenas at lane boundaries or on finished lanes',
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
    run_independent(a)
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
    atomic_json(a.archive/'status.json', dict(phase='matched_compute', updated=time.time()))
    run_lanes(arena_jobs(a.binary,a.archive,selected,'matched_compute',a.workers),a.workers)
    atomic_json(a.archive/'status.json',dict(phase='complete',updated=time.time()))


if __name__ == '__main__':
    main()
