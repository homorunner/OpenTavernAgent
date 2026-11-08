#!/usr/bin/env python3
"""Append identity residual FFN blocks to a v10 checkpoint; keep every existing tensor.

New W_up is random, W_down and both biases are zero, LayerNorm gamma=1/beta=0.
Thus the residual branch is zero initially, while W_down can learn on the first update.
Only weights are transferred: selfplay --resume starts fresh Adam, EMA and replay state.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def expand(source, destination, blocks, seed=444444444):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Source and destination must differ')
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    meta = json.loads(arrays['meta'].tobytes())
    if meta['version'] != 10:
        raise ValueError('Only v10 pre-norm FFN checkpoints are supported')
    old = meta['blocks']
    if blocks < old:
        raise ValueError('Cannot remove existing blocks')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + '.tmp')
    if blocks == old:
        shutil.copyfile(source, tmp)
    else:
        h, m = arrays[f'W{old}_up'].shape
        for i in range(old + 1, blocks + 1):
            # Shared extra blocks start identically across the 8/10/12-block variants.
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            arrays[f'W{i}_up'] = (rng.standard_normal((h, m)) * np.sqrt(2 / h)).astype(np.float32)
            arrays[f'b{i}_up'] = np.zeros_like(arrays[f'b{old}_up'])
            arrays[f'b{i}_down'] = np.zeros_like(arrays[f'b{old}_down'])
            arrays[f'W{i}_down'] = np.zeros_like(arrays[f'W{old}_down'])
            arrays[f'g{i}'] = np.ones_like(arrays[f'g{old}'])
            arrays[f'k{i}'] = np.zeros_like(arrays[f'k{old}'])
        meta['blocks'] = blocks
        arrays['meta'] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        with tmp.open('wb') as f:
            np.savez(f, **arrays)
    tmp.replace(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--blocks', type=int, required=True)
    p.add_argument('--seed', type=int, default=444444444)
    a = p.parse_args()
    expand(a.model, a.out, a.blocks, a.seed)


if __name__ == '__main__':
    main()
