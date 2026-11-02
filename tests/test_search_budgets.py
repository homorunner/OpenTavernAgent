"""Search information boundary, chance sampling and finite host budgets."""

import json

import numpy as np
import pytest
import taverngym
from tavernagent.heroes import select_heroes
from tavernagent.net import BUDGET_FEATURE_START, N_INPUT_FEATURES, PolicyValueNet
from tavernagent.rules import game_config, game_rules

from tavernagent import mcts


def position(seed=19, **rules):
    game = taverngym.Game(seed, game_config(**rules))
    select_heroes(game, seed)
    # Give the controlled seat a plain recruit phase, without hero-specific menus.
    state = json.loads(game.state_json())
    state["players"][0].update(pending=[], skipping=False, gold=30)
    return taverngym.Game.from_json(json.dumps(state))


class SingleAction:
    def __init__(self, action):
        self.action = action

    def predict(self, x, cards, mask):
        prior = np.zeros(245, np.float32)
        action = self.action if mask[self.action] else int(np.flatnonzero(mask)[0])
        prior[action] = 1
        # Positive but outcome-dependent, so root exploration keeps the forced action.
        return prior, 0.1 + float(cards[30] % 10) / 20


def test_seeded_copy_changes_only_future_rng_and_keeps_exact_copy_available():
    game = position()
    state = json.loads(game.state_json())
    assert game.copy().state_json() == game.state_json()
    copy = game.copy(seed=12345)
    changed = json.loads(copy.state_json())
    assert changed.pop("rng") != state.pop("rng")
    assert changed == state
    assert copy.copy(seed=12345).state_json() == copy.state_json()
    # Different future streams actually produce different Refresh outcomes.
    shops = []
    for seed in range(8):
        sample = game.copy(seed=seed)
        sample.step(0, 1)
        shops.append(tuple(mcts.observe(sample, 0)[1]))
    assert len(set(shops)) > 1


@pytest.mark.parametrize("horizon", ["turn", "game"])
def test_search_is_independent_of_real_rng_and_does_not_mutate_game(horizon):
    game = position()
    other = game.copy(seed=987654321)
    before = game.state_json()
    a = mcts.search(game, 0, SingleAction(1), sims=24, horizon=horizon)
    b = mcts.search(other, 0, SingleAction(1), sims=24, horizon=horizon)
    np.testing.assert_array_equal(a.visits, b.visits)
    np.testing.assert_array_equal(a.root.cw, b.root.cw)
    assert game.state_json() == before
    assert mcts._key(mcts.observe(game, 0)) == mcts._key(mcts.observe(other, 0))
    # Executing the real next draw after search is identical to an untouched snapshot.
    reference = taverngym.Game.from_json(before)
    game.step(0, 1)
    reference.step(0, 1)
    assert game.state_json() == reference.state_json()


def test_repeated_refresh_visits_sample_multiple_outcomes_on_one_action_edge():
    game = position()
    result = mcts.search(game, 0, SingleAction(1), sims=64, rng=np.random.default_rng(123))
    index = int(np.flatnonzero(result.root.actions == 1)[0])
    outcomes = result.root.children[index]
    assert result.visits[1] == 64
    assert len(outcomes) > 8
    assert all(node.game.turn == game.turn for node in outcomes.values())
    assert all(np.isfinite(node.value) for node in outcomes.values())


def test_deterministic_actions_reuse_nodes_despite_new_simulation_rng():
    game = position()
    result = mcts.search(game, 0, SingleAction(2), sims=12, rng=np.random.default_rng(9))
    index = int(np.flatnonzero(result.root.actions == 2)[0])
    assert len(result.root.children[index]) == 1
    child = next(iter(result.root.children[index].values()))
    assert child.n > 0


def test_turn_search_never_advances_other_seats_or_end_turn(monkeypatch):
    game = position()
    # The controlled seat is the last seat: stepping EndTurn would resolve combat.
    for seat in range(1, game.num_players):
        while game.needs_action(seat):
            game.step(seat, game.bot_action(seat))
    before = game.state_json()

    def forbidden(*args, **kwargs):
        pytest.fail("turn search attempted a bot/combat rollout")

    monkeypatch.setattr(mcts, "advance", forbidden)
    end = mcts.search(game, 0, SingleAction(0), sims=16)
    assert end.visits[0] == 16
    assert all(not outcomes for outcomes in end.root.children)
    assert end.root.cw[end.root.actions == 0][0] == pytest.approx(16 * end.value)
    mcts.search(game, 0, SingleAction(1), sims=16)
    assert game.state_json() == before
    # Changing a current hidden board cannot affect an EndTurn estimate.
    state = json.loads(before)
    state["players"][1]["board"] = []
    changed = taverngym.Game.from_json(json.dumps(state))
    same = mcts.search(changed, 0, SingleAction(0), sims=16)
    np.testing.assert_array_equal(end.visits, same.visits)
    np.testing.assert_array_equal(end.root.cw, same.root.cw)


def test_remaining_budgets_are_network_inputs_without_opponent_history():
    game = position()
    assert game.action_budgets(0) == (100, 10, 5)
    x, cards, _ = mcts.observe(game, 0)
    np.testing.assert_array_equal(x[-3:], [1, 1, 1])
    assert all(i in range(BUDGET_FEATURE_START, len(x)) for i in range(len(x) - 3, len(x)))
    assert N_INPUT_FEATURES == 1715
    net = PolicyValueNet(seed=13)
    base = net.forward(x, cards)[0].copy()
    state = json.loads(game.state_json())
    state["players"][0].update(actions_this_turn=90, moves_this_turn=8, freezes_this_turn=5)
    changed = taverngym.Game.from_json(json.dumps(state))
    y, other_cards, _ = mcts.observe(changed, 0)
    np.testing.assert_allclose(y[-3:], [0.1, 0.2, 0])
    assert changed.action_budgets(0) == (10, 2, 0)
    assert not np.allclose(net.forward(y, other_cards)[0], base)


def test_default_limits_are_shared_and_invalid_limits_fail_early():
    assert game_rules() == {
        "turn_action_caps": [100],
        "move_actions_per_turn": 10,
        "freeze_presses_per_turn": 5,
    }
    for key in ("action_cap", "move_cap", "freeze_cap"):
        for value in (0, -1, 1.5, True, None):
            with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
                game_config(**{key: value})
    assert position(action_cap=1, move_cap=1, freeze_cap=1).action_budgets(0) == (1, 1, 1)
