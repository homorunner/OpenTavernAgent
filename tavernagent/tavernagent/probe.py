"""What the agent does on its first decision of turn 1 and of turn 2.

The first two turns are the cheapest possible sanity check for the whole pipeline:
turn 1 should buy a minion or a spell (never roll the Tavern away), turn 2 should
upgrade, and the failure mode to look for is "spend the gold on rerolls".

    python -m tavernagent.probe --model models/sp.npz --games 32 --sims 8
"""

from __future__ import annotations

import argparse
import collections
import json

import taverngym

from . import mcts
from .heroes import select_heroes
from .net import PolicyValueNet
from .rules import ACTION_CAP, FREEZE_CAP, MOVE_CAP, add_budget_args, game_config

FAMILY = {
    "EndTurn": "EndTurn",
    "Refresh": "Refresh",
    "Freeze": "Freeze",
    "Upgrade": "Upgrade",
    "Buy": "Buy",
    "Play": "Play",
    "Sell": "Sell",
    "Choose": "Choose",
    "HeroPower": "HeroPower",
    "DarkGift": "DarkGift",
    "Move": "Move",
    "Activate": "Activate",
    "Magnetize": "Magnetize",
}


def family(name: str) -> str:
    head = name.split(" ")[0].split("(")[0]
    return FAMILY.get(head, head)


def behaviour(
    net,
    games: int = 16,
    sims: int = 8,
    action_cap: int = ACTION_CAP,
    seed: int = 9000,
    horizon: str = "turn",
    c_puct: float = 1.5,
    move_cap: int = MOVE_CAP,
    freeze_cap: int = FREEZE_CAP,
):
    """Distribution of seat 0's first action in turn 1 and in turn 2, plus gold left."""
    config = game_config(action_cap, move_cap, freeze_cap)
    first = {1: collections.Counter(), 2: collections.Counter()}
    gold_left = {1: [], 2: []}
    for g in range(games):
        game = taverngym.Game(seed=seed + g, config_json=config)
        select_heroes(game, seed + g)
        seen: set[int] = set()
        while not game.is_over():
            if game.needs_action(0):
                result = mcts.search(game, 0, net, sims=sims, horizon=horizon, c_puct=c_puct)
                action = result.best_action
                turn = game.turn
                if turn in (1, 2) and turn not in seen:
                    seen.add(turn)
                    first[turn][family(taverngym.action_name(action))] += 1
                    gold_left[turn].append(_gold(game))
                game.step(0, action)
                continue
            moved = False
            for p in range(1, game.num_players):
                if game.needs_action(p):
                    game.step(p, game.bot_action(p))
                    moved = True
            if not moved:
                break
    return first, gold_left


def _gold(game) -> int:
    try:
        return int(json.loads(game.view(0))["me"]["gold"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return -1


def _fmt(counter, gold):
    total = sum(counter.values()) or 1
    parts = [f"{k} {100 * v / total:.0f}%" for k, v in counter.most_common()]
    avg_gold = sum(gold) / len(gold) if gold else 0
    return ", ".join(parts) + f"  (avg gold {avg_gold:.1f})"


def behaviour_to_string(
    net,
    games: int = 16,
    sims: int = 8,
    *,
    horizon="turn",
    action_cap=ACTION_CAP,
    c_puct=1.5,
    move_cap=MOVE_CAP,
    freeze_cap=FREEZE_CAP,
) -> str:
    first, gold = behaviour(
        net,
        games=games,
        sims=sims,
        horizon=horizon,
        action_cap=action_cap,
        c_puct=c_puct,
        move_cap=move_cap,
        freeze_cap=freeze_cap,
    )
    return f"T1 [{_fmt(first[1], gold[1])}]  T2 [{_fmt(first[2], gold[2])}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="checkpoint; omit to probe the initial net")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--horizon", choices=("turn", "game"), default="turn")
    add_budget_args(ap)
    ap.add_argument("--c-puct", type=float, default=1.5)
    args = ap.parse_args(argv)
    net = PolicyValueNet.load(args.model) if args.model else PolicyValueNet(seed=0)
    print(f"{args.model or 'initial net'}  ({args.games} games, {args.sims} sims)")
    first, gold = behaviour(
        net,
        games=args.games,
        sims=args.sims,
        horizon=args.horizon,
        action_cap=args.action_cap,
        c_puct=args.c_puct,
        move_cap=args.move_cap,
        freeze_cap=args.freeze_cap,
    )
    for turn in (1, 2):
        print(f"  turn {turn}: {_fmt(first[turn], gold[turn])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
