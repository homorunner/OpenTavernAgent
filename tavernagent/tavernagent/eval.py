"""Evaluate a checkpoint against the baselines that matter, on paired seeds.

    python -m tavernagent.eval --model models/demo.npz --games 40 --sims 48
    python -m tavernagent.eval --model models/demo.npz --games 512 --sims 8 --workers 32 --out results.json

Plays seat 0 with the checkpoints and the engine's scripted bot over the same seeds,
and reports mean placement, standard error and paired differences.
Paired seeds matter: an 8-player placement has a standard deviation of ~2.2, so a
5-game sample can be off by whole placements.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np
import taverngym

from . import mcts
from .heroes import select_heroes
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


def play(policy, seed: int, action_cap: int, move_cap=MOVE_CAP, freeze_cap=FREEZE_CAP) -> int:
    config = game_config(action_cap, move_cap, freeze_cap)
    game = taverngym.Game(seed=seed, config_json=config)
    select_heroes(game, seed)
    while not game.is_over():
        if game.needs_action(0):
            game.step(0, policy(game))
            continue
        moved = False
        for p in range(1, game.num_players):
            if game.needs_action(p):
                game.step(p, game.bot_action(p))
                moved = True
        if not moved:
            raise RuntimeError(f"evaluation stalled: seed={seed}, turn={game.turn}")
    return next(iter(game.placements()))


def bot_policy(game):
    return game.bot_action(0)


def net_policy(net, sims, horizon="turn", c_puct=1.5):
    def policy(game):
        return mcts.search(game, 0, net, sims=sims, horizon=horizon, c_puct=c_puct).best_action

    return policy


def summarize(places):
    places = np.asarray(places, dtype=np.float64)
    sd = float(places.std(ddof=1)) if len(places) > 1 else 0.0
    return {
        "n": len(places),
        "mean": float(places.mean()),
        "se": sd / np.sqrt(len(places)),
        "first_rate": float(np.mean(places == 1)),
        "top4_rate": float(np.mean(places <= 4)),
    }


def report(name, places):
    stats = summarize(places)
    print(
        f"{name:<14} n={stats['n']:<3} mean {stats['mean']:.3f}  se {stats['se']:.3f}  "
        f"top4 {stats['top4_rate']:.1%}",
        flush=True,
    )
    return stats["mean"]


_EVAL_POLICIES = None
_EVAL_CAP = None


def _init_eval(models, sims, horizon, action_cap, baselines, c_puct, move_cap, freeze_cap):
    global _EVAL_POLICIES, _EVAL_CAP
    _EVAL_POLICIES = {
        name: net_policy(PolicyValueNet.load(path), sims, horizon, c_puct)
        for name, path in models.items()
    }
    if baselines:
        _EVAL_POLICIES["bot"] = bot_policy
    _EVAL_CAP = (action_cap, move_cap, freeze_cap)


def _eval_seed(seed):
    return {
        name: worker_call(play, f"evaluation seed={seed}, policy={name}", policy, seed, *_EVAL_CAP)
        for name, policy in _EVAL_POLICIES.items()
    }


def evaluate(
    models,
    games=512,
    seed=10_000_000,
    sims=8,
    horizon="turn",
    action_cap=ACTION_CAP,
    workers=1,
    baselines=True,
    c_puct=1.5,
    move_cap=MOVE_CAP,
    freeze_cap=FREEZE_CAP,
):
    if not np.isfinite(c_puct) or c_puct < 0:
        raise ValueError("c_puct must be finite and nonnegative")
    seeds = list(range(seed, seed + games))
    game_config(action_cap, move_cap, freeze_cap)
    initargs = (models, sims, horizon, action_cap, baselines, c_puct, move_cap, freeze_cap)
    # Validate checkpoints before spawning to avoid Pool initializer restart loops.
    _init_eval(*initargs)
    if workers == 1:
        rows = list(map(_eval_seed, seeds))
    else:
        with mp.get_context("spawn").Pool(
            min(workers, games), initializer=_init_eval, initargs=initargs
        ) as pool:
            rows = pool.map(_eval_seed, seeds)
    placements = {name: [r[name] for r in rows] for name in rows[0]}
    result = {
        "seeds": seeds,
        "sims": sims,
        "horizon": horizon,
        "c_puct": c_puct,
        "models": {name: str(path) for name, path in models.items()},
        **experiment_settings(action_cap, move_cap, freeze_cap),
        "placements": placements,
        "summary": {name: summarize(p) for name, p in placements.items()},
        "paired_vs_bot": {},
    }
    if "bot" in placements:
        bot = np.array(placements["bot"])
        for name, places in placements.items():
            if name == "bot":
                continue
            delta = np.asarray(places) - bot
            se = float(delta.std(ddof=1) / np.sqrt(games)) if games > 1 else 0.0
            mean = float(delta.mean())
            result["paired_vs_bot"][name] = {
                "mean_delta": mean,
                "se": se,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                "significantly_better": mean + 1.96 * se < 0,
            }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/demo.npz")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    add_budget_args(ap)
    ap.add_argument("--horizon", choices=("turn", "game"), default="turn")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="save per-seed placements and paired statistics")
    ap.add_argument("--compare-model", action="append", default=[], help="name=path")
    args = ap.parse_args(argv)

    if min(args.games, args.sims, args.workers) < 1:
        ap.error("games, sims and workers must be positive")
    models = {"network": args.model}
    for spec in args.compare_model:
        name, path = spec.split("=", 1)
        models[name] = path
    result = evaluate(
        models,
        args.games,
        args.seed,
        args.sims,
        args.horizon,
        args.action_cap,
        args.workers,
        not args.no_baselines,
        args.c_puct,
        args.move_cap,
        args.freeze_cap,
    )
    for name, places in result["placements"].items():
        report(name, places)
    for name, stats in result["paired_vs_bot"].items():
        print(
            f"{name} - bot: {stats['mean_delta']:+.3f} placements, "
            f"95% CI {stats['ci95']}, significantly better: {stats['significantly_better']}",
            flush=True,
        )
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
