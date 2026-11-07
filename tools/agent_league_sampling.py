"""Count-based, without-replacement league seating with a uniform probability floor."""
import random

UNIFORM_SHARE = 0.2
COUNT_PRIOR = 256
BATCH_GAMES = 128


def weighted_tables(counts, games, seed):
    """Increment counts in place so later tables/batches reserve these appearances."""
    if len(counts)<8 or any(n<0 for n in counts.values()):
        raise ValueError('Need at least eight models with nonnegative counts')
    rng=random.Random(seed)
    names=sorted(counts)
    tables=[]
    for _ in range(games):
        inverse=[1/(counts[name]+COUNT_PRIOR) for name in names]
        total=sum(inverse)
        weights=[UNIFORM_SHARE/len(names)+(1-UNIFORM_SHARE)*v/total for v in inverse]
        remaining=list(zip(names,weights))
        table=[]
        for _ in range(8):
            position=rng.choices(range(len(remaining)),weights=[v for _,v in remaining],k=1)[0]
            name,_=remaining.pop(position)
            table.append(name)
        rng.shuffle(table)
        for name in table:
            counts[name]+=1
        tables.append(table)
    return tables
