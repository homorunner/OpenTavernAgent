"""Latency acceptance test: evaluation + action prediction must stay under 0.5 ms.

python -m tavernagent.bench [--iters 2000] [--budget-ms 0.5]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import taverngym

from . import mcts
from .heroes import select_heroes
from .net import N_FEATURES, PolicyValueNet
from .rules import add_budget_args, game_config


def timed(fn, iters: int) -> np.ndarray:
    for _ in range(min(50, iters)):  # warm up BLAS
        fn()
    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1e3
    return samples


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--budget-ms", type=float, default=0.5)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    add_budget_args(ap)
    args = ap.parse_args(argv)

    net = PolicyValueNet(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # a real observation from a real game, so the input has the same sparsity
    game = taverngym.Game(
        seed=args.seed, config_json=game_config(args.action_cap, args.move_cap, args.freeze_cap)
    )
    select_heroes(game, args.seed)
    floats, card_bytes = game.observe(0)
    cards = np.frombuffer(card_bytes, dtype="<i4").copy()
    x = np.frombuffer(floats, dtype="<f4").copy()
    mask = np.frombuffer(game.legal_mask(0), dtype=np.uint8).astype(bool)
    if not mask.any():
        mask[:] = True

    report = {}
    report["forward_only"] = timed(lambda: net.forward(x, cards), args.iters)
    report["predict"] = timed(lambda: net.predict(x, cards, mask), args.iters)
    report["random_obs"] = timed(
        lambda: net.predict(rng.standard_normal(N_FEATURES, dtype=np.float32), cards, mask),
        args.iters,
    )

    print(
        f"net {net.n_params():,} params, hidden {net.hidden}, "
        f"inference over {args.iters} iterations"
    )
    print(f"{'case':<14} {'mean':>7} {'p50':>7} {'p99':>7} {'max':>7}   (ms)")
    for name, s in report.items():
        print(
            f"{name:<14} {s.mean():7.3f} {np.percentile(s, 50):7.3f} "
            f"{np.percentile(s, 99):7.3f} {s.max():7.3f}"
        )

    # correctness of the masked policy
    probs, value = net.predict(x, cards, mask)
    assert abs(probs.sum() - 1.0) < 1e-5, probs.sum()
    assert (probs[~mask] == 0).all(), "illegal actions got probability"
    assert -1.0 <= value <= 1.0
    print(f"\npolicy sums to 1 over {int(mask.sum())} legal actions, value {value:+.3f}")

    # a full decision, for context
    d = timed(lambda: mcts.search(game, 0, net, sims=args.sims), max(20, args.iters // 40))
    print(
        f"MCTS decision ({args.sims} sims): mean {d.mean():.2f} ms, "
        f"p99 {np.percentile(d, 99):.2f} ms"
    )

    worst = float(report["predict"].max())
    p99 = float(np.percentile(report["predict"], 99))
    ok = p99 < args.budget_ms
    print(
        f"\nbudget {args.budget_ms} ms: p99 {p99:.3f} ms -> "
        f"{'PASS' if ok else 'FAIL'} (worst single call {worst:.3f} ms)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
