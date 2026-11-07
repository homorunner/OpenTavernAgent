#!/usr/bin/env python3
"""Render shareable placement and Arena figures from completed evaluation JSON."""
import argparse
import itertools
import json
from pathlib import Path
import re

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


def render(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    n = 0
    latest = 100
    for depth, color in zip(DEPTHS, COLORS):
        for variant, style in [('ema', '-'), ('raw', '--')]:
            rows = []
            for path in (root/f'd{depth}').glob(f'eval_step*_{variant}.json'):
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
                        capsize=3, alpha=1 if variant=='ema' else .45, label=f'{depth} blocks / {variant.upper()}')
    ax.set(title='Evaluation against scripted bots', xlabel='Training round', ylabel='Mean placement (lower is better)')
    ax.grid(alpha=.2)
    ax.set_xlim(0,latest+100)
    if n:
        ax.legend(ncol=2, fontsize=9)
    else:
        ax.text(.5,.5,'Awaiting evaluation results',ha='center',transform=ax.transAxes)
    fig.suptitle('Error bars: 95% confidence intervals | 64 search simulations',fontsize=10)
    eval_path = out/'eval.png'
    fig.savefig(eval_path, dpi=160)
    plt.close(fig)

    import importlib.util
    spec=importlib.util.spec_from_file_location("league_ratings",Path(__file__).with_name("league_elo.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plackett_luce=module.plackett_luce
    games=[]
    for path in sorted(root.glob('league_batch*.json')):
        result=read(path)
        if result:
            games.extend((r['models'],r['placements']) for r in result['matches'])
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    if games:
        names=sorted({name for models,_ in games for name in models})
        elo=plackett_luce(games,names)
        rng=np.random.default_rng(0)
        boots=[plackett_luce([games[i] for i in rng.integers(len(games),size=len(games))],names,iters=100) for _ in range(32)]
        se=np.std(boots,axis=0,ddof=1)
        groups={}
        counts={name:sum(models.count(name) for models,_ in games) for name in names}
        ratings={}
        for name,value,error in zip(names,elo,se):
            run,step=name.rsplit('_s',1)
            groups.setdefault(run,[]).append((int(step),value,error))
            ratings[name]=dict(elo=float(value),se=float(error),games=counts[name])
        for run,points in sorted(groups.items()):
            points.sort();x,y,e=np.array(points).T
            depth=int(run.split('_')[0][1:]); variant=run.split('_')[1]
            ax.errorbar(x,y,yerr=1.96*e,color=COLORS[DEPTHS.index(depth)],
                ls='-' if variant=='ema' else '--',marker='o',capsize=3,
                alpha=1 if variant=='ema' else .45,label=run)
        ax.legend(ncol=2,fontsize=9)
        ax.set_title(f'Eight-player league | {len(names)} models | {len(games)} games')
        (out/'elo.json').write_text(json.dumps(ratings,indent=2))
    else:
        ax.text(.5,.5,'Awaiting eight-player league results',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Training round',ylabel='Elo (Plackett-Luce, pool mean = 0)')
    ax.grid(alpha=.2)
    fig.suptitle('All checkpoints share one pool | Random eight-player tables | Game-bootstrap 95% intervals',fontsize=10)
    league_path=out/'elo.png'
    fig.savefig(league_path,dpi=160)
    plt.close(fig)
    return [eval_path,league_path]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    for path in render(a.root,a.out):
        print(path)
