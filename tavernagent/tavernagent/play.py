"""Play games with a checkpoint; prints placements and an action histogram.

python -m tavernagent.play --model models/demo.npz --games 8 --sims 48
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import taverngym

from . import mcts
from .heroes import select_heroes
from .net import PolicyValueNet
from .rules import add_budget_args, game_config


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/demo.npz")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    add_budget_args(ap)
    ap.add_argument("--horizon", choices=("turn", "game"), default="turn")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args(argv)
    if not np.isfinite(args.c_puct) or args.c_puct < 0:
        ap.error("c-puct must be finite and nonnegative")

    net = PolicyValueNet.load(args.model)
    print(f"{args.model}: {net.n_params():,} params, hidden {net.hidden}")
    config = game_config(args.action_cap, args.move_cap, args.freeze_cap)

    places = []
    hist = collections.Counter()
    for g in range(args.games):
        game = taverngym.Game(seed=args.seed + g, config_json=config)
        select_heroes(game, args.seed + g)
        while not game.is_over():
            if not game.needs_action(0):
                moved = False
                for p in range(1, game.num_players):
                    if game.needs_action(p):
                        game.step(p, game.bot_action(p))
                        moved = True
                if not moved:
                    break
                continue
            result = mcts.search(
                game, 0, net, sims=args.sims, horizon=args.horizon, c_puct=args.c_puct
            )
            action = result.best_action
            hist[taverngym.action_name(action).split(" ")[0][:6]] += 1
            mcts.advance(game, 0, action)
        placement = next(iter(game.placements()))
        places.append(placement)
        print(f"game {g + 1}: placed {placement}  (turn {game.turn})")

    print(f"\nmean placement {np.mean(places):.2f}  placements {places}")
    print("actions:", dict(hist.most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
