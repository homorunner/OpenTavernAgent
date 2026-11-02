"""Measure full decision latency on identical saved recruit positions across budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import taverngym

from .heroes import select_heroes
from .mcts import observe, search
from .net import PolicyValueNet
from .rules import (
    ACTION_CAP,
    FREEZE_CAP,
    MOVE_CAP,
    add_budget_args,
    experiment_settings,
    game_config,
)


def positions(
    count=48, seed=40_000_000, action_cap=ACTION_CAP, move_cap=MOVE_CAP, freeze_cap=FREEZE_CAP
):
    pool = []
    for offset in range(32):
        game = taverngym.Game(
            seed=seed + offset, config_json=game_config(action_cap, move_cap, freeze_cap)
        )
        select_heroes(game, seed + offset)
        captured = {}
        while not game.is_over():
            acted = False
            for seat in range(8):
                if not game.needs_action(seat):
                    continue
                acted = True
                turn = game.turn
                if seat == 0 and turn in (1, 4, 8, 12, 16) and captured.get(turn, 0) < 3:
                    captured[turn] = captured.get(turn, 0) + 1
                    pool.append(game.copy())
                game.step(seat, game.bot_action(seat))
            if not acted:
                raise RuntimeError("benchmark position collection stalled")
        if len(pool) >= count * 2:
            break
    if len(pool) < count:
        raise RuntimeError("not enough benchmark positions")
    keep = np.random.default_rng(seed).choice(len(pool), count, replace=False)
    return [pool[i] for i in keep]


def summarize(values):
    a = np.asarray(values)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def benchmark(
    models,
    budgets=(1, 8, 16, 32, 64),
    count=48,
    repeats=3,
    seed=40_000_000,
    horizon="turn",
    c_puct=1.5,
    action_cap=ACTION_CAP,
    move_cap=MOVE_CAP,
    freeze_cap=FREEZE_CAP,
):
    if not np.isfinite(c_puct) or c_puct < 0:
        raise ValueError("c_puct must be finite and nonnegative")
    states = positions(count, seed, action_cap, move_cap, freeze_cap)
    description = []
    for game in states:
        x, _, mask = observe(game, 0)
        description.append(
            {
                "turn": game.turn,
                "legal_actions": int(mask.sum()),
                "board_count": round(float(x[33]) * 7),
                "hand_count": round(float(x[34]) * 10),
                "state_sha256": hashlib.sha256(game.state_json().encode()).hexdigest(),
            }
        )
    results = {}
    for name, path in models.items():
        net = PolicyValueNet.load(path)
        by_budget = {}
        for sims in budgets:
            search(states[0], 0, net, sims=sims, horizon=horizon, c_puct=c_puct)
            latency = []
            actions = []
            for state in states:
                for _ in range(repeats):
                    start = time.perf_counter()
                    result = search(state, 0, net, sims=sims, horizon=horizon, c_puct=c_puct)
                    latency.append((time.perf_counter() - start) * 1000)
                actions.append(result.best_action)
            by_budget[str(sims)] = {
                "latency_ms": summarize(latency),
                "actions": actions,
                "raw_ms": latency,
            }
        results[name] = {"path": str(path), "parameters": net.n_params(), "budgets": by_budget}
    bot_latency = []
    for state in states:
        for _ in range(repeats):
            copy = state.copy()
            start = time.perf_counter()
            copy.bot_action(0)
            bot_latency.append((time.perf_counter() - start) * 1000)
    return {
        "seed": seed,
        "horizon": horizon,
        "c_puct": c_puct,
        **experiment_settings(action_cap, move_cap, freeze_cap),
        "positions": description,
        "repeats": repeats,
        "models": results,
        "scripted_bot_action_ms": summarize(bot_latency),
        "note": "Wall clock on identical positions; game-copy cost included in MCTS, excluded for bot action.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name=path")
    parser.add_argument("--budgets", default="1,8,16,32,64")
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=40_000_000)
    parser.add_argument("--horizon", choices=("turn", "game"), default="turn")
    parser.add_argument("--c-puct", type=float, default=1.5)
    add_budget_args(parser)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    models = dict(spec.split("=", 1) for spec in args.model)
    result = benchmark(
        models,
        [int(s) for s in args.budgets.split(",")],
        args.positions,
        args.repeats,
        args.seed,
        args.horizon,
        args.c_puct,
        args.action_cap,
        args.move_cap,
        args.freeze_cap,
    )
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    for name, stats in result["models"].items():
        print(name, {s: round(d["latency_ms"]["median"], 3) for s, d in stats["budgets"].items()})


if __name__ == "__main__":
    main()
