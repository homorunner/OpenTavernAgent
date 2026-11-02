"""Whole-game, per-turn action traces and conservative loop diagnostics.

Diagnostics use only the controlled seat's visible preparation state. A repeated
state is a candidate loop, not proof of a useless action: budgets are deliberately
excluded from the signature, while resources, buffs and ability uses are retained.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import taverngym

from . import mcts
from .heroes import select_heroes
from .net import PolicyValueNet
from .probe import family
from .rules import add_budget_args, experiment_settings, game_config
from .selfplay import _sample


def material_state(me):
    """Keep gameplay state; omit logs, opponent histories and action budgets."""
    keys = (
        "board",
        "hand",
        "shop",
        "gold",
        "tier",
        "health",
        "armor",
        "pending",
        "effects",
        "hero_power",
        "hero_powers",
        "trinkets",
        "quests",
        "rewards",
        "scheduled",
        "secrets",
        "free_refreshes",
        "gold_next_turn",
        "dark_gift",
        "casting",
        "hand_queue",
        "chronum_balance",
        "deity",
        "buddy_button",
        "max_gold",
        "max_gold_bonus",
        "minion_price",
        "refresh_price",
        "upgrade_price",
    )
    return {k: me[k] for k in keys if k in me}


def state_key(state):
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compact_state(me):
    fields = ("eid", "id", "atk", "hp", "golden", "keywords", "counter", "frozen")
    return {
        **{k: me.get(k) for k in ("gold", "tier", "health", "armor", "upgrade_price")},
        **{
            zone: [{k: c[k] for k in fields if k in c} for c in me.get(zone, [])]
            for zone in ("board", "hand", "shop")
        },
        "pending": me.get("pending"),
    }


def summarize_turn(turn):
    steps = turn["actions"]
    counts = collections.Counter(a["family"] for a in steps)
    seen = {turn["initial_key"]}
    candidates = []
    noops = 0
    longest = {"Move": 0, "Freeze": 0}
    streak = 0
    previous = None
    for i, action in enumerate(steps):
        kind = action["family"]
        streak = streak + 1 if kind == previous else 1
        previous = kind
        if kind in longest:
            longest[kind] = max(longest[kind], streak)
        if kind in ("Move", "Freeze"):
            noops += action["before_key"] == action["after_key"]
            if action["after_key"] in seen:
                candidates.append(i)
        seen.add(action["after_key"])
    last = steps[-1] if steps else None
    ended = last is not None and last["family"] == "EndTurn"
    return {
        "turn": turn["turn"],
        "actions": len(steps),
        "families": dict(counts),
        "candidate_cycle_closures": len(candidates),
        "candidate_indices": candidates,
        "unchanged_move_freeze": int(noops),
        "longest_streak": longest,
        "gold_left": last["before"]["gold"] if ended else None,
        "end_board_size": len(last["before"]["board"]) if ended else None,
        "tier": last["before"]["tier"] if ended else None,
        "forced_end": bool(ended and last["remaining_before"][0] == 0),
        "ended_with_buy_available": bool(ended and last["buy_available"]),
        "move_cap_reached": any(a["remaining_after"][1] == 0 for a in steps),
        "freeze_cap_reached": any(a["remaining_after"][2] == 0 for a in steps),
    }


_NET = None
_OPTIONS = None


def _init(model, options):
    global _NET, _OPTIONS
    _NET = PolicyValueNet.load(model) if model else None
    _OPTIONS = options


def _game(seed):
    o = _OPTIONS
    game = taverngym.Game(
        seed=seed, config_json=game_config(o["action_cap"], o["move_cap"], o["freeze_cap"])
    )
    select_heroes(game, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x504C4159]))
    turns = {}
    # selfplay mode follows the actual all-eight-seat training schedule and RNG.
    while not game.is_over():
        moved = False
        seats = range(game.num_players) if o["mode"] == "selfplay" else (0,)
        for seat in seats:
            if not game.needs_action(seat):
                continue
            moved = True
            if _NET is None:
                action = game.bot_action(seat)
            else:
                result = mcts.search(
                    game,
                    seat,
                    _NET,
                    sims=o["sims"],
                    horizon="turn",
                    c_puct=o["c_puct"],
                    root_noise=o["noise"] if o["mode"] == "selfplay" else 0,
                    noise_alpha=o["noise_alpha"],
                    rng=rng if o["mode"] == "selfplay" else None,
                )
                action = (
                    _sample(result, o["temperature"], rng)
                    if o["mode"] == "selfplay" and o["temperature"] > 0
                    else result.best_action
                )
            if seat == 0:
                view = json.loads(game.view(0))
                before = view["me"]
                before_key = state_key(material_state(before))
                turn = turns.setdefault(
                    game.turn, {"turn": game.turn, "initial_key": before_key, "actions": []}
                )
                mask = np.frombuffer(game.legal_mask(0), dtype=np.uint8)
                record = {
                    "action": int(action),
                    "name": taverngym.action_name(action),
                    "family": family(taverngym.action_name(action)),
                    "before": compact_state(before),
                    "before_key": before_key,
                    "remaining_before": game.action_budgets(0),
                    "buy_available": bool(mask[8:16].any()),
                }
            game.step(seat, action)
            if seat == 0:
                after = json.loads(game.view(0))["me"]
                record.update(
                    after_key=state_key(material_state(after)),
                    after=compact_state(after),
                    remaining_after=game.action_budgets(0),
                )
                turn["actions"].append(record)
        if o["mode"] != "selfplay":
            # Match eval.play: controlled seat completes its turn before bots act.
            if game.needs_action(0):
                continue
            for seat in range(1, game.num_players):
                if game.needs_action(seat):
                    game.step(seat, game.bot_action(seat))
                    moved = True
        if not moved and not game.is_over():
            raise RuntimeError(f"pattern game stalled: seed={seed}")
    values = list(turns.values())
    return {
        "seed": seed,
        "placement": next(iter(game.placements())),
        "turns": values,
        "summaries": [summarize_turn(t) for t in values],
    }


def aggregate(rows):
    summaries = [t for g in rows for t in g["summaries"]]
    counts = collections.Counter()
    per_turn = {}
    for t in summaries:
        counts.update(t["families"])
        group = per_turn.setdefault(t["turn"], [])
        group.append(t)

    def stats(ts):
        ended = [t for t in ts if t["gold_left"] is not None]
        return {
            "n": len(ts),
            "mean_actions": float(np.mean([t["actions"] for t in ts])),
            "mean_gold_left": float(np.mean([t["gold_left"] for t in ended])) if ended else None,
            "mean_board_size": float(np.mean([t["end_board_size"] for t in ended]))
            if ended
            else None,
            "mean_tier": float(np.mean([t["tier"] for t in ended])) if ended else None,
            **{
                key: sum(t[key] for t in ts)
                for key in (
                    "candidate_cycle_closures",
                    "unchanged_move_freeze",
                    "forced_end",
                    "ended_with_buy_available",
                    "move_cap_reached",
                    "freeze_cap_reached",
                )
            },
            "turns_with_candidate_cycle": sum(t["candidate_cycle_closures"] > 0 for t in ts),
            "families": dict(
                sum((collections.Counter(t["families"]) for t in ts), collections.Counter())
            ),
        }

    return {
        "games": len(rows),
        "mean_placement": float(np.mean([g["placement"] for g in rows])),
        "families": dict(counts),
        "overall": stats(summaries),
        "per_turn": {str(t): stats(ts) for t, ts in sorted(per_turn.items())},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=54_000_000)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--c-puct", type=float, default=0.75)
    ap.add_argument("--mode", choices=("eval", "selfplay"), default="eval")
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--noise-alpha", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.5)
    add_budget_args(ap)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if min(args.games, args.sims, args.workers) < 1 or (args.mode == "selfplay" and not args.model):
        ap.error("positive counts and a model for selfplay are required")
    options = {k: v for k, v in vars(args).items() if k not in ("out", "model")}
    _init(args.model, options)  # fail before starting a process pool
    if args.workers == 1:
        rows = [_game(args.seed + i) for i in range(args.games)]
    else:
        with mp.get_context("spawn").Pool(
            min(args.workers, args.games), initializer=_init, initargs=(args.model, options)
        ) as pool:
            rows = pool.map(_game, range(args.seed, args.seed + args.games))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix(".traces.json.gz")
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        json.dump(rows, handle)
    report = {
        "model": args.model,
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
        if args.model
        else None,
        "options": options,
        **experiment_settings(args.action_cap, args.move_cap, args.freeze_cap),
        "summary": aggregate(rows),
        "traces": str(trace),
        "diagnostic_note": "Repeated material states are candidate loops; counters are excluded; inspect traces before judging utility.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]["overall"]), flush=True)


if __name__ == "__main__":
    main()
