"""Shared host budgets for current training, evaluation and interactive play."""

import json

ACTION_CAP = 100
MOVE_CAP = 10
FREEZE_CAP = 5


def game_rules(action_cap=ACTION_CAP, move_cap=MOVE_CAP, freeze_cap=FREEZE_CAP):
    for name, value in (
        ("action_cap", action_cap),
        ("move_cap", move_cap),
        ("freeze_cap", freeze_cap),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "turn_action_caps": [action_cap],
        "move_actions_per_turn": move_cap,
        "freeze_presses_per_turn": freeze_cap,
    }


def game_config(action_cap=ACTION_CAP, move_cap=MOVE_CAP, freeze_cap=FREEZE_CAP):
    return json.dumps(game_rules(action_cap, move_cap, freeze_cap))


def add_budget_args(parser):
    parser.add_argument("--action-cap", type=int, default=ACTION_CAP)
    parser.add_argument("--move-cap", type=int, default=MOVE_CAP)
    parser.add_argument("--freeze-cap", type=int, default=FREEZE_CAP)


def experiment_settings(action_cap=ACTION_CAP, move_cap=MOVE_CAP, freeze_cap=FREEZE_CAP):
    from .mcts import SEARCH_VERSION

    return {
        "search_version": SEARCH_VERSION,
        "action_cap": action_cap,
        "move_cap": move_cap,
        "freeze_cap": freeze_cap,
        "rules": game_rules(action_cap, move_cap, freeze_cap),
    }
