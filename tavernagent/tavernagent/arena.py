"""Checkpoint self-play matches with seat swaps and game-clustered score intervals.

The reported Elo-equivalent is 400*log10(score/(1-score)), where score is the
fraction of cross-team placement comparisons won. It is not a fitted global Elo.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import taverngym

from .heroes import select_heroes
from .mcts import search
from .net import PolicyValueNet
from .rules import (
    ACTION_CAP,
    FREEZE_CAP,
    MOVE_CAP,
    add_budget_args,
    experiment_settings,
    game_config,
)
from .worker import worker_call

_NETS = None
_SIMS = None
_CAP = None
_HORIZON = None
_C_PUCT = None


def _init(a, b, sims, cap, horizon, c_puct, move_cap, freeze_cap):
    global _NETS, _SIMS, _CAP, _HORIZON, _C_PUCT
    _NETS = [PolicyValueNet.load(a), PolicyValueNet.load(b)]
    _SIMS, _CAP = sims, game_config(cap, move_cap, freeze_cap)
    _HORIZON = horizon
    _C_PUCT = c_puct


def _play_match(job):
    seed, swapped = job
    game = taverngym.Game(seed=seed, config_json=_CAP)
    select_heroes(game, seed)
    team_a = [s for s in range(8) if (s % 2) == swapped]
    while not game.is_over():
        acted = False
        for seat in range(8):
            if game.needs_action(seat):
                acted = True
                net = _NETS[0 if seat in team_a else 1]
                action = search(
                    game, seat, net, sims=_SIMS, horizon=_HORIZON, c_puct=_C_PUCT
                ).best_action
                game.step(seat, action)
        if not acted:
            raise RuntimeError(f"arena stalled: {seed}")
    places = list(game.placements())
    a = np.array([places[s] for s in team_a])
    b = np.array([places[s] for s in range(8) if s not in team_a])
    return {
        "seed": seed,
        "swapped": swapped,
        "placements": places,
        "score": float(np.mean(a[:, None] < b[None, :])),
    }


def _match(job):
    return worker_call(_play_match, f"arena seed={job[0]}, swapped={job[1]}", job)


def elo_equivalent(score):
    p = min(1 - 1e-6, max(1e-6, score))
    return 400 * math.log10(p / (1 - p))


def battle(
    a,
    b,
    games=512,
    seed=21_000_000,
    sims=8,
    workers=1,
    action_cap=ACTION_CAP,
    horizon="turn",
    c_puct=1.5,
    move_cap=MOVE_CAP,
    freeze_cap=FREEZE_CAP,
):
    if games < 2 or games % 2:
        raise ValueError("games must be positive and even for seat swaps")
    if not np.isfinite(c_puct) or c_puct < 0:
        raise ValueError("c_puct must be finite and nonnegative")
    jobs = [(s, swap) for s in range(seed, seed + games // 2) for swap in (0, 1)]
    initargs = (a, b, sims, action_cap, horizon, c_puct, move_cap, freeze_cap)
    _init(*initargs)
    if workers == 1:
        rows = list(map(_match, jobs))
    else:
        with mp.get_context("spawn").Pool(
            min(workers, games), initializer=_init, initargs=initargs
        ) as pool:
            rows = pool.map(_match, jobs)
    # Each seed's two seat-swapped games form one statistical cluster.
    scores = np.array([r["score"] for r in rows]).reshape(-1, 2).mean(axis=1)
    mean = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
    ci = [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)]
    return {
        "a": str(a),
        "b": str(b),
        "games": games,
        "seed_clusters": len(scores),
        "sims": sims,
        "horizon": horizon,
        "c_puct": c_puct,
        **experiment_settings(action_cap, move_cap, freeze_cap),
        "score": mean,
        "se": se,
        "ci95": ci,
        "elo_equivalent": elo_equivalent(mean),
        "elo_ci95": [elo_equivalent(p) for p in ci],
        "matches": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=21_000_000)
    parser.add_argument("--sims", type=int, default=8)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--horizon", choices=("turn", "game"), default="turn")
    parser.add_argument("--c-puct", type=float, default=1.5)
    add_budget_args(parser)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = battle(
        args.a,
        args.b,
        args.games,
        args.seed,
        args.sims,
        args.workers,
        horizon=args.horizon,
        c_puct=args.c_puct,
        action_cap=args.action_cap,
        move_cap=args.move_cap,
        freeze_cap=args.freeze_cap,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "matches"}, indent=2))


if __name__ == "__main__":
    main()
