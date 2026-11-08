"""Independent seed batches and pooled, game-clustered uncertainty estimates."""
import copy
import math
import statistics


def merge(base, extra, ident):
    base=copy.deepcopy(base)
    if ident in base.get('_batches',[]):
        return base
    if 'placements' in base:
        assert base.get('opponent', 'scripted') == extra.get('opponent', 'scripted'), 'Different evaluation opponents'
        assert not set(base['seeds']) & set(extra['seeds']), 'Repeated evaluation seeds'
        base['seeds']+=extra['seeds']
        for name,values in extra['placements'].items():
            base['placements'][name]+=values
            rows=base['placements'][name]; n=len(rows)
            base['summary'][name]=dict(n=n,mean=statistics.mean(rows),
                se=statistics.stdev(rows)/math.sqrt(n) if n>1 else 0,
                first_rate=sum(x==1 for x in rows)/n,top4_rate=sum(x<=4 for x in rows)/n)
    else:
        assert not {r['seed'] for r in base['matches']} & {r['seed'] for r in extra['matches']}, 'Repeated Arena seeds'
        base['matches']+=extra['matches']
        grouped={}
        for row in base['matches']:
            grouped.setdefault(row['seed'],[]).append(row['score'])
        assert all(len(v)==2 for v in grouped.values())
        scores=[statistics.mean(v) for v in grouped.values()]
        mean=statistics.mean(scores); se=statistics.stdev(scores)/math.sqrt(len(scores)) if len(scores)>1 else 0
        ci=[max(0,mean-1.96*se),min(1,mean+1.96*se)]
        def elo(p):
            p=min(1-1e-6,max(1e-6,p)); return 400*math.log10(p/(1-p))
        base.update(games=len(base['matches']),seed_clusters=len(scores),score=mean,se=se,ci95=ci,
                    elo_equivalent=elo(mean),elo_ci95=[elo(p) for p in ci])
    base.setdefault('_batches',[]).append(ident)
    return base


def extras(planned, root, busy, complete):
    latest={}
    for j in planned:
        if j['step']>3000: continue
        key=(j['kind'],j.get('opponent', 'scripted'),tuple((a,p.split('/model.step')[0],p.endswith('.ema.npz')) for a,p in j['inputs'].items()))
        if key not in latest or j['step']>latest[key]['step']: latest[key]=j
    candidates=[]
    import json
    for j in latest.values():
        p=root/'results'/j['relative']
        if not complete(p): continue
        base=json.loads(p.read_text())
        n=base['summary']['network']['n'] if j['kind']=='eval' else base['games']
        if n>=16384: continue
        batch=len(base.get('_batches',[]))+1
        ident=j['id']+f'_extra{batch}'
        if ident in busy: continue
        args=list(j['args'])
        args[args.index('--games')+1]='512'
        args[args.index('--seed')+1]=str((61000000 if j['kind']=='eval' else 71000000)+100000+batch*4096)
        candidates.append((n,dict(j,id=ident,args=args,aggregate=j['relative'],relative=f'batches/{ident}.json')))
    return [j for _,j in sorted(candidates,key=lambda x:(x[0],x[1]['id']))]
