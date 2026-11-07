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

    fig, (trend, heat) = plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.55,1]},layout='constrained')
    stages = {}
    for pair_index,(a,b) in enumerate(itertools.combinations(DEPTHS,2)):
        rows=[]
        for path in root.glob(f'arena_step*_d{a}_d{b}.json'):
            result=read(path)
            if result:
                step=int(re.search(r'step(\d+)',path.name)[1])
                rows.append((step,result['score'],*result['ci95']))
                stages.setdefault(step,{})[(a,b)]=result
        rows.sort()
        if rows:
            x,y,lo,hi=np.array(rows).T
            trend.errorbar(x,y*100,yerr=np.array([y-lo,hi-y])*100,marker='o',capsize=2,color=f'C{pair_index}',label=f'{a} vs {b}')
    trend.axhline(50,color='#64748b',ls=':',linewidth=1)
    trend.set(title='Paired Arena scores',xlabel='Training round',ylabel='First model score (%)',ylim=(0,100))
    trend.grid(alpha=.2)
    trend.set_xlim(0,max(stages,default=100)+100)
    if stages:
        trend.legend(ncol=2,fontsize=9)
        # Prefer the latest full comparison so cells refer to one common round.
        full=[s for s,pairs in stages.items() if len(pairs)==6]
        step=max(full or stages)
        pairs=stages[step]
        matrix=np.full((4,4),np.nan)
        np.fill_diagonal(matrix,50)
        for (a,b),r in pairs.items():
            i,j=DEPTHS.index(a),DEPTHS.index(b)
            matrix[i,j]=r['score']*100
            matrix[j,i]=(1-r['score'])*100
        cmap=plt.get_cmap('RdYlGn').copy()
        cmap.set_bad('#e5e7eb')
        heat.imshow(matrix,vmin=0,vmax=100,cmap=cmap)
        for i in range(4):
            for j in range(4):
                label='—' if i==j or np.isnan(matrix[i,j]) else f'{matrix[i,j]:.1f}%'
                heat.text(j,i,label,ha='center',va='center',fontsize=11)
        heat.set(title=f'Round {step} | {len(pairs)}/6 pairs',xticks=range(4),yticks=range(4),
                 xticklabels=DEPTHS,yticklabels=DEPTHS,xlabel='Opponent blocks',ylabel='Row model blocks')
    else:
        heat.axis('off')
        heat.text(.5,.5,'Awaiting paired Arena results',ha='center',transform=heat.transAxes)
    fig.suptitle('Arena: cross-team placement score, not game win rate | Pooled independent seeds with seat swaps | 95% CI',fontsize=10)
    arena_path=out/'arena.png'
    fig.savefig(arena_path,dpi=160)
    plt.close(fig)
    return [eval_path,arena_path]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    for path in render(a.root,a.out):
        print(path)
