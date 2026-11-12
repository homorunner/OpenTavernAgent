#!/usr/bin/env python3
"""Continuously evaluate immutable PPO checkpoints in one connected eight-seat league.

Each --run is LABEL=DIRECTORY, containing model.iterN.npz and/or initial.npz.
Uses the Rust league command for CPU FP32 games and the v1 Plackett-Luce fitter.
Requires numpy for ratings; plots are optional. Existing results are resumed, never
replayed into the rating data twice. No training checkpoints are modified.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time
import zipfile


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def checkpoint_state(path):
    with zipfile.ZipFile(path) as archive:
        data = archive.read('trainer_meta.npy')
    if data[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError(f'unsupported checkpoint metadata: {path}')
    return json.loads(data[10 + int.from_bytes(data[8:10], 'little'):])['state']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def schedule(names, counts, games, seed, anchor):
    """Four least-played entrants plus four random opponents keep cohorts connected."""
    rng = random.Random(seed)
    counts = Counter(counts)
    tables = []
    for game in range(games):
        priority = sorted(names, key=lambda name: (counts[name], rng.random()))[:4]
        table = priority + rng.sample([n for n in names if n not in priority], 4)
        if game == 0 and anchor not in table:
            table[-1] = anchor
        rng.shuffle(table)
        tables.append(table)
        counts.update(table)
    return tables


def stop(signum, frame):
    raise KeyboardInterrupt(f'stopped by signal {signum}')


class League:
    def __init__(self, args):
        self.args = args
        self.root = args.out.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.sources = {}
        for spec in args.run:
            name, path = spec.split('=', 1)
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in self.sources:
                raise ValueError('run labels must be unique simple names')
            self.sources[name] = Path(path).resolve()
        self.references = {}
        for spec in args.reference:
            name, path = spec.split('=', 1)
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in self.sources or name in self.references:
                raise ValueError('reference labels must be unique and distinct from run labels')
            self.references[name] = Path(path).resolve()
        self.models = json.loads((self.root / 'models.json').read_text()) if (self.root / 'models.json').exists() else {}
        settings = {'runs': {n: str(p) for n, p in self.sources.items()},
                    'references': {n: str(p) for n, p in self.references.items()},
                    'binary_sha256': digest(args.binary), 'fitter_sha256': digest(Path(__file__).with_name('league_elo.py')),
                    'controller_sha256': digest(Path(__file__)),
                    'interval': args.interval, 'target_iteration': args.target_iteration,
                    'games_per_model': args.games_per_model, 'batch_games': args.batch_games,
                    'seed': args.seed, 'anchor': args.anchor, 'anchor_rating': args.anchor_rating,
                    'bootstrap': args.bootstrap, 'backend': 'cpu', 'inference': 'fp32', 'sample': False}
        manifest = self.root / 'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != settings:
                raise ValueError('league settings changed; use a new output directory')
        else:
            save(manifest, settings)
        self.child = None
        self.counts = Counter()
        self.completed = []
        self.excluded = []
        self.batch = 0
        self.last_fit_batch = (json.loads((self.root / 'ratings-meta.json').read_text()).get('batches', -1)
                               if (self.root / 'ratings-meta.json').exists() else -1)
        self.recover()

    def status(self, phase, **extra):
        save(self.root / 'status.json', {'state': phase, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'models': len(self.models), 'games': sum(self.counts.values()) // 8,
            'excluded_games': len(self.excluded),
            'appearances': dict(self.counts), 'minimum_games': self.args.games_per_model,
            'batch': self.batch, **extra})

    def discover(self):
        for label in [*self.sources, *self.references]:
            fixed = label in self.references
            directory = self.sources.get(label)
            candidates = [self.references[label]] if fixed else sorted(directory.glob('model.iter*.npz'))
            if not fixed and (directory / 'initial.npz').exists():
                candidates.append(directory / 'initial.npz')
            for path in candidates:
                if any(m['path'] == str(path) for m in self.models.values()):
                    continue
                match = re.fullmatch(r'model\.iter(\d+)\.npz', path.name)
                if not fixed and match and (int(match[1]) % self.args.interval or int(match[1]) > self.args.target_iteration):
                    continue
                state = checkpoint_state(path)
                iteration = state['iteration']
                if not fixed and (iteration == 0 or iteration % self.args.interval or iteration > self.args.target_iteration):
                    continue
                if match and int(match[1]) != iteration:
                    raise ValueError(f'checkpoint name/metadata mismatch: {path}')
                name = f'{label}_s{iteration}'
                fingerprint = digest(path)
                if name in self.models:
                    if self.models[name]['sha256'] != fingerprint:
                        raise ValueError(f'checkpoint identity changed: {name}')
                    continue
                self.models[name] = {'path': str(path), 'sha256': fingerprint, 'run': label,
                    'iteration': iteration, 'training_seed': state['options']['seed'],
                    'hidden': state['options']['hidden'], 'blocks': state['options']['blocks'],
                    'policy_rules': state['options'].get('policy_rules', {}),
                    'engine_options': {k: state['options'][k] for k in ['max_turns', 'budgets']}}
        save(self.root / 'models.json', self.models)

    def recover(self):
        self.counts = Counter()
        self.completed = []
        self.excluded = []
        self.batch = 0
        for directory in sorted(self.root.glob('batch-*')):
            output = directory / 'games.jsonl'
            if not output.exists():
                break
            plan = json.loads((directory / 'plan.json').read_text())
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            if len(rows) != len(plan['tables']):
                raise ValueError(f'incomplete published batch: {directory}')
            indices = set()
            for row in rows:
                i = row['game']
                if i in indices or not 0 <= i < len(plan['tables']) or row['models'] != plan['tables'][i] or row['seed'] != plan['seed'] + i:
                    raise ValueError(f'duplicate or inconsistent league game: {directory}')
                if sorted(row['placements']) != list(range(1, 9)):
                    raise ValueError(f'invalid league ranking: {directory}')
                indices.add(i)
                if row['turn_limit_end'] or row['truncated_combats']:
                    self.excluded.append({'batch': directory.name, **row})
                    continue
                self.counts.update(row['models'])
            self.completed.append(output)
            self.batch += 1
        save(self.root / 'excluded-games.json', self.excluded)

    def play_batch(self):
        args = self.args
        directory = self.root / f'batch-{self.batch:05d}'
        directory.mkdir(exist_ok=True)
        plan_path = directory / 'plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            seed = args.seed + self.batch * args.batch_games
            tables = schedule(sorted(self.models), self.counts, args.batch_games, seed, args.anchor)
            plan = {'seed': seed, 'tables': tables}
            save(plan_path, plan)
        save(directory / 'tables.json', plan['tables'])
        names = sorted({name for table in plan['tables'] for name in table})
        command = ([] if args.numa_node is None else ['numactl', f'--membind={args.numa_node}'])
        command += [str(args.binary.resolve()), 'league', '--tables', str(directory / 'tables.json'),
                    '--seed', str(plan['seed']), '--workers', str(args.workers), '--out', str(directory / 'games.jsonl')]
        for name in names:
            command += ['--model', f"{name}={self.models[name]['path']}"]
        save(directory / 'command.json', command)
        started = time.monotonic()
        with (directory / 'stdout.log').open('w') as out, (directory / 'stderr.log').open('w') as err:
            self.child = subprocess.Popen(command, stdout=out, stderr=err)
            while self.child.poll() is None:
                self.status('playing', child_pid=self.child.pid, batch_seconds=time.monotonic() - started)
                time.sleep(5)
            if self.child.returncode:
                raise RuntimeError(f'league batch failed: {directory}; see stderr.log')
            self.child = None
        save(directory / 'resources.json', {'wall_seconds': time.monotonic() - started})
        self.recover()

    def fit(self):
        if self.last_fit_batch == self.batch:
            return
        from league_elo import load, ratings_with_bootstrap
        self.status('fitting')
        combined = self.root / 'league.jsonl'
        temporary = combined.with_suffix('.jsonl.tmp')
        with temporary.open('wb') as output:
            for path in self.completed:
                rows = sorted((json.loads(line) for line in path.read_text().splitlines()), key=lambda row: row['game'])
                for row in rows:
                    if row['turn_limit_end'] or row['truncated_combats']:
                        continue
                    output.write((json.dumps(row) + '\n').encode())
        temporary.replace(combined)
        games = load(combined)
        names = sorted({name for models, _ in games for name in models})
        elo, se, counts = ratings_with_bootstrap(games, names, bootstrap=self.args.bootstrap,
            workers=min(4, self.args.workers), seed=0, anchor=self.args.anchor, anchor_rating=self.args.anchor_rating)
        ratings = {name: {'elo': float(e), 'se': float(s), 'ci95_normal': [float(e - 1.96*s), float(e + 1.96*s)],
            'games': int(n), 'run': self.models[name]['run'], 'iteration': self.models[name]['iteration']}
            for name, e, s, n in zip(names, elo, se, counts)}
        save(self.root / 'ratings.json', ratings)
        save(self.root / 'ratings-meta.json', {'games': len(games), 'excluded_games': len(self.excluded), 'models': len(names), 'batches': self.batch, 'anchor': self.args.anchor,
            'anchor_rating': self.args.anchor_rating, 'bootstrap': self.args.bootstrap,
            'note': 'Conditional full-game bootstrap, not training-seed uncertainty. Scores are relative within this connected pool and may shift when games/models are added.'})
        self.last_fit_batch = self.batch

    def run(self):
        try:
            while True:
                self.discover()
                if len(self.models) >= 8 and self.args.anchor in self.models:
                    if min(self.counts[name] for name in self.models) < self.args.games_per_model:
                        self.play_batch()
                        if self.batch % 4 == 0:
                            self.fit()
                        continue
                    if self.completed:
                        self.fit()
                    finished = all(any(m['run'] == run and m['iteration'] == self.args.target_iteration
                                       for m in self.models.values()) for run in self.sources)
                    if finished or not self.args.watch:
                        self.status('complete')
                        return
                elif not self.args.watch:
                    raise ValueError('need at least eight checkpoints and the requested anchor')
                self.status('waiting_for_checkpoints')
                time.sleep(self.args.poll_seconds)
        except BaseException as error:
            self.status('stopped' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
            raise
        finally:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--reference', action='append', default=[], metavar='LABEL=CHECKPOINT',
                        help='frozen reference, independent of training iteration ceiling')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--anchor-rating', type=float, default=1000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--target-iteration', type=int, default=10000)
    parser.add_argument('--games-per-model', type=int, default=256)
    parser.add_argument('--batch-games', type=int, default=64)
    parser.add_argument('--seed', type=int, default=60_000_000)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--cpu-list', help='optional comma-separated CPU IDs or ranges for this evaluator')
    parser.add_argument('--numa-node', type=int)
    args = parser.parse_args()
    if min(args.interval, args.target_iteration, args.games_per_model, args.batch_games, args.workers, args.poll_seconds) <= 0 or args.bootstrap < 2 or args.seed < 0:
        parser.error('counts must be positive, seed nonnegative, bootstrap >= 2')
    if args.cpu_list:
        cpus = set()
        for segment in args.cpu_list.split(','):
            lo, _, hi = segment.partition('-')
            cpus.update(range(int(lo), int(hi or lo) + 1))
        if not cpus or not cpus <= os.sched_getaffinity(0):
            parser.error('CPU list is outside the available affinity')
        os.sched_setaffinity(0, cpus)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    League(args).run()


if __name__ == '__main__':
    main()
