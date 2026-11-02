"""Behavior diagnostics must preserve the evaluated and self-play policies."""

import pytest
from tavernagent.eval import net_policy, play
from tavernagent.net import PolicyValueNet
from tavernagent.selfplay import play_one

from tavernagent import patterns


def test_cycle_diagnostics_distinguish_return_and_resources():
    def action(kind, before, after):
        return {
            "family": kind,
            "before_key": before,
            "after_key": after,
            "before": {"gold": 3, "board": [], "tier": 1},
            "remaining_before": [100, 10, 5],
            "remaining_after": [99, 9, 4],
            "buy_available": True,
        }

    summary = patterns.summarize_turn(
        {
            "turn": 2,
            "initial_key": "a",
            "actions": [
                action("Move", "a", "b"),
                action("Move", "b", "a"),
                action("Freeze", "a", "c"),
                action("Freeze", "c", "a"),
                action("Move", "a", "a"),
                action("Buy", "a", "d"),
                action("EndTurn", "d", "d"),
            ],
        }
    )
    assert summary["candidate_indices"] == [1, 3, 4]
    assert summary["unchanged_move_freeze"] == 1
    assert summary["longest_streak"] == {"Move": 2, "Freeze": 2}
    assert summary["gold_left"] == 3
    assert summary["ended_with_buy_available"]
    assert not summary["forced_end"]


def test_signature_retains_ability_uses_but_ignores_logs():
    base = {"gold": 3, "hero_power": {"uses_turn": 0}, "turn_events": ["A"]}
    k = patterns.state_key(patterns.material_state(base))
    assert patterns.state_key(patterns.material_state({**base, "turn_events": ["B"]})) == k
    assert (
        patterns.state_key(patterns.material_state({**base, "hero_power": {"uses_turn": 1}})) != k
    )


@pytest.mark.parametrize("mode", ["eval", "selfplay"])
def test_trace_matches_actual_game_policy(tmp_path, mode):
    net = PolicyValueNet(seed=7)
    model = tmp_path / "net.npz"
    net.save(model)
    o = {
        "action_cap": 100,
        "move_cap": 2,
        "freeze_cap": 2,
        "mode": mode,
        "sims": 2,
        "c_puct": 0.75,
        "noise": 0.2,
        "noise_alpha": 0.3,
        "temperature": 0.5,
    }
    patterns._init(str(model), o)
    trace = patterns._game(76)
    if mode == "eval":
        placement = play(net_policy(net, 2, c_puct=0.75), 76, 100, 2, 2)
        assert trace["placement"] == placement
    else:
        _, reward = play_one(
            net, 2, 76, 100, "turn", 0.5, noise=0.2, c_puct=0.75, move_cap=2, freeze_cap=2
        )
        assert 1 - (trace["placement"] - 1) * (2 / 7) == pytest.approx(reward)
    report = patterns.aggregate([trace])
    assert report["overall"]["n"] == len(trace["turns"])
    assert all(t["families"].get("Move", 0) <= 2 for t in trace["summaries"])
    assert all(t["families"].get("Freeze", 0) <= 2 for t in trace["summaries"])
