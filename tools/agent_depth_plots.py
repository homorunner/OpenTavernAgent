#!/usr/bin/env python3
"""Render shareable placement and Arena figures from completed evaluation JSON."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEPTHS = (6, 8, 10, 12)
COLORS = ('#2563eb', '#059669', '#d97706', '#9333ea')


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def save_figure(fig, path):
    tmp = path.with_name(path.stem + f'.{os.getpid()}.tmp.png')
    fig.savefig(tmp, dpi=160)
    tmp.replace(path)


def league_ratings(root, out):
    from league_elo import ratings_with_bootstrap
    paths = sorted(root.glob('league_batch*.json'))
    stamp = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    fingerprint = hashlib.sha256(json.dumps(stamp).encode()).hexdigest()
    cached = read(out/'elo_cache.json')
    if cached and cached.get('version') == 1 and cached.get('fingerprint') == fingerprint:
        return cached['ratings'], cached['games']
    games = []
    for path in paths:
        result = read(path)
        if result:
            games.extend((r['models'], r['placements']) for r in result['matches'])
    if not games:
        return {}, 0
    started = time.monotonic()
    names = sorted({name for models, _ in games for name in models})
    elo, se, counts = ratings_with_bootstrap(games, names)
    ratings = {name: dict(elo=float(e), se=float(s), games=int(n))
               for name, e, s, n in zip(names, elo, se, counts)}
    atomic_json(out/'elo_cache.json', dict(version=1, fingerprint=fingerprint,
                ratings=ratings, games=len(games), fit_seconds=time.monotonic()-started,
                updated=time.time()))
    return ratings, len(games)


def render(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(root.parent/'manifest.json') or {}
    runs = manifest.get('runs') or [dict(id=f'd{d}', label=f'{d} blocks') for d in DEPTHS]
    colors = {r['id']: c for r, c in zip(runs, COLORS)}
    labels = {r['id']: r.get('label', r['id']) for r in runs}
    baselines = {b['name']: b.get('label', b['name']) for b in manifest.get('baselines', [])}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    eval_paths = []
    for prefix, label, filename in [('', 'scripted bots', 'eval.png'),
                                    ('teacher_', 'teacher bots (no composition tables)', 'teacher_eval.png')]:
        fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
        n = 0
        latest = 100
        for run in runs:
            color = colors[run['id']]
            for variant, style in [('ema', '-'), ('raw', '--')]:
                rows = []
                for path in (root/run['id']).glob(f'{prefix}eval_step*_{variant}.json'):
                    result = read(path)
                    if result:
                        step = int(re.search(r'step(\d+)', path.name)[1])
                        summary = result['summary']['network']
                        rows.append((step, summary['mean'], summary['se']))
                rows.sort()
                if not rows:
                    continue
                n += len(rows)
                latest=max(latest,rows[-1][0])
                x, y, se = np.array(rows).T
                ax.errorbar(x, y, yerr=1.96*se, color=color, ls=style, marker='o' if variant=='ema' else '.',
                            capsize=3, alpha=1 if variant=='ema' else .45, label=f"{labels[run['id']]} / {variant.upper()}")
        ax.set(title=f'Evaluation against {label}', xlabel='Training round', ylabel='Mean placement (lower is better)')
        ax.grid(alpha=.2)
        ax.set_xlim(0,latest+100)
        if n:
            ax.legend(ncol=2, fontsize=9)
        else:
            ax.text(.5,.5,'Awaiting evaluation results',ha='center',transform=ax.transAxes)
        fig.suptitle('Error bars: 95% confidence intervals | 64 search simulations',fontsize=10)
        eval_path = out/filename
        save_figure(fig, eval_path)
        plt.close(fig)
        eval_paths.append(eval_path)

    ratings, game_count = league_ratings(root, out)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    if ratings:
        groups={}
        for name, row in ratings.items():
            value, error = row['elo'], row['se']
            if name not in baselines:
                run,step=name.rsplit('_s',1)
                groups.setdefault(run,[]).append((int(step),value,error))
        for run,points in sorted(groups.items()):
            points.sort();x,y,e=np.array(points).T
            run_id, variant=run.rsplit('_', 1)
            ax.errorbar(x,y,yerr=1.96*e,color=colors[run_id],
                ls='-' if variant=='ema' else ':',marker='o',capsize=3,
                alpha=1 if variant=='ema' else .45,label=f'{labels[run_id]} / {variant.upper()}')
        for name, color in zip(baselines, itertools.cycle(('#64748b', '#a16207', '#be185d', '#0e7490', '#475569', '#7c3aed'))):
            if name in ratings:
                row=ratings[name]
                ax.axhline(row['elo'], color=color, ls='--', label=baselines[name])
                ax.axhspan(row['elo']-1.96*row['se'], row['elo']+1.96*row['se'], color=color, alpha=.06)
        ax.legend(ncol=2,fontsize=9)
        ax.set_title(f'Eight-player league | {len(ratings)} models | {game_count} games')
        atomic_json(out/'elo.json', ratings)
    else:
        ax.text(.5,.5,'Awaiting eight-player league results',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Training round',ylabel='Elo (Plackett-Luce, pool mean = 0)')
    ax.grid(alpha=.2)
    fig.suptitle('All checkpoints share one pool | Random eight-player tables | Game-bootstrap 95% intervals',fontsize=10)
    league_path=out/'elo.png'
    save_figure(fig, league_path)
    plt.close(fig)
    atomic_json(out/'updated.json', dict(updated=time.time(), league_games=game_count))
    return [*eval_paths,league_path]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    for path in render(a.root,a.out):
        print(path)
