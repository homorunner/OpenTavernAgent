"""Regression checks for identity-aware policy/value inputs and training gradients."""

import itertools
import json

import numpy as np
import pytest
import taverngym
from tavernagent.heroes import is_hero_selection, select_heroes
from tavernagent.net import (
    N_CARD_SLOTS,
    N_FEATURES,
    N_INPUT_FEATURES,
    N_OPP_BOARD_SLOTS,
    OPP_BOARD_FEATURE_START,
    PolicyValueNet,
)
from tavernagent.selfplay import Trainer

from tavernagent import mcts


def inputs():
    game = taverngym.Game(seed=17)
    select_heroes(game, 17)
    return game, mcts.observe(game, 0)


def test_card_identity_and_ignored_categories():
    _, (x, cards, mask) = inputs()
    net = PolicyValueNet(seed=2)
    assert net.n_params() <= 1_000_000
    catalog = json.loads(taverngym.catalog_json())
    for kind in (
        "minion",
        "tavern_spell",
        "hero",
        "trinket_lesser",
        "hero_power",
        "quest",
        "quest_reward",
    ):
        ids = [c["cid"] + 1 for c in catalog if c["kind"] == kind][:2]
        a = np.zeros_like(cards)
        b = a.copy()
        a[30], b[30] = ids
        pa, va = net.predict(x, a, mask)
        pb, vb = net.predict(x, b, mask)
        assert not np.allclose(pa, pb) or not np.isclose(va, vb)
    empty = np.zeros_like(cards)
    ignored = empty.copy()
    ignored[3] = next(c["cid"] + 1 for c in catalog if c["kind"] == "hero")
    ignored[14] = next(c["cid"] + 1 for c in catalog if c["kind"] == "quest_reward")
    np.testing.assert_array_equal(net.predict(x, empty, mask)[0], net.predict(x, ignored, mask)[0])


def test_embedding_gradient_accumulates_repeated_cards():
    net = PolicyValueNet(seed=3)
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.3, (2, N_FEATURES)).astype(np.float32)
    cards = np.zeros((2, N_CARD_SLOTS), np.int32)
    cid = int(np.flatnonzero(net.allowed_cards)[0])
    cards[:, 19:22] = cid
    dp = rng.normal(0, 0.01, (2, 245)).astype(np.float32)
    dv = np.array([[0.03], [-0.02]], np.float32)
    _, v = net.forward(x, cards)
    grads = net.backward(dp, dv * (1 - v[:, None] ** 2))
    assert np.all(grads["E"][0] == 0)
    for name, idx in [("E", (cid, j)) for j in range(3)] + [
        (
            "W0",
            (
                N_INPUT_FEATURES
                + int(np.flatnonzero(np.flatnonzero(net.card_slots) == 19)[0]) * net.embedding_dim,
                0,
            ),
        )
    ]:
        p = net.p[name]
        old = p[idx].copy()

        def loss():
            logits, value = net.forward(x, cards)
            return float(np.sum(logits.astype(np.float64) * dp) + np.sum(value[:, None] * dv))

        eps = 0.01
        p[idx] = old + eps
        hi = loss()
        p[idx] = old - eps
        lo = loss()
        p[idx] = old
        np.testing.assert_allclose(grads[name][idx], (hi - lo) / (2 * eps), atol=2e-4, rtol=0.05)


def test_optimizer_search_and_checkpoint(tmp_path):
    game, (x, cards, mask) = inputs()
    net = PolicyValueNet(seed=5)
    result = mcts.search(game, 0, net, sims=2)
    assert mask[result.best_action]
    assert mask[mcts.search(game, 0, net, sims=2, horizon="game").best_action]
    before = net.p["E"].copy()
    pl, vl = Trainer(net).step(
        x[None], cards[None], mask[None], result.policy[None], np.array([0.5], np.float32), lr=0.001
    )
    assert np.isfinite(pl + vl)
    assert np.any(before != net.p["E"])
    assert np.all(net.p["E"][0] == 0)
    path = tmp_path / "model.npz"
    net.save(path)
    restored = PolicyValueNet.load(path)
    np.testing.assert_array_equal(
        net.predict(x, cards, mask)[0], restored.predict(x, cards, mask)[0]
    )
    with np.load(path) as saved:
        bad = {k: saved[k] for k in saved.files}
    bad["__card_fingerprint"] = "wrong-card-table"
    np.savez(tmp_path / "wrong.npz", **bad)
    with pytest.raises(ValueError, match="schema"):
        PolicyValueNet.load(tmp_path / "wrong.npz")
    with np.load(path) as saved:
        assert int(saved["__version"]) == 2
        missing_format = {k: saved[k] for k in saved.files if k != "__version"}
    np.savez(tmp_path / "missing-format.npz", **missing_format)
    with pytest.raises(ValueError, match="expected model format v2"):
        PolicyValueNet.load(tmp_path / "missing-format.npz")
    for version in (1, 3, 4):
        with np.load(path) as saved:
            unsupported = {k: saved[k] for k in saved.files}
        unsupported["__version"] = version
        np.savez(tmp_path / "unsupported.npz", **unsupported)
        with pytest.raises(ValueError, match="expected model format v2"):
            PolicyValueNet.load(tmp_path / "unsupported.npz")
    for field in ("__activation", "__residual_scale", "__head_norm"):
        with np.load(path) as saved:
            incomplete = {k: saved[k] for k in saved.files if k != field}
        np.savez(tmp_path / "incomplete.npz", **incomplete)
        with pytest.raises(ValueError, match="incomplete model format v2"):
            PolicyValueNet.load(tmp_path / "incomplete.npz")


def test_opponent_history_is_excluded_but_public_info_and_heroes_remain():
    _, (x, cards, _) = inputs()
    net = PolicyValueNet(seed=11)
    x = np.array(x, copy=True)
    cards = cards.copy()
    base_logits, base_value = net.forward(x, cards)
    changed = x.copy()
    changed[OPP_BOARD_FEATURE_START:-3] = 123.0
    changed_cards = cards.copy()
    changed_cards[-N_OPP_BOARD_SLOTS:] = int(np.flatnonzero(net.allowed_cards)[0])
    logits, value = net.forward(changed, changed_cards)
    np.testing.assert_array_equal(logits, base_logits)
    np.testing.assert_array_equal(value, base_value)
    assert net.p["W0"].shape[0] == 1715 + 69 * 8
    # Public opponent summaries sit before the omitted board entities.
    changed = x.copy()
    changed[OPP_BOARD_FEATURE_START - 1] += 1.0
    assert not np.allclose(net.forward(changed, cards)[0], base_logits)
    heroes = [c["cid"] + 1 for c in json.loads(taverngym.catalog_json()) if c["kind"] == "hero"]
    changed_cards = cards.copy()
    changed_cards[-N_OPP_BOARD_SLOTS - 1] = heroes[0]
    first = net.forward(x, changed_cards)[0].copy()
    changed_cards[-N_OPP_BOARD_SLOTS - 1] = heroes[1]
    assert not np.allclose(net.forward(x, changed_cards)[0], first)


def test_phase_flags_are_not_network_inputs():
    _, (x, cards, _) = inputs()
    net = PolicyValueNet(seed=23)
    expected = net.forward(x, cards)
    changed = x.copy()
    changed[1:3] = 1.0 - changed[1:3]
    actual = net.forward(changed, cards)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_random_hero_selection_is_reproducible_and_bypasses_network():
    heroes = []
    for seed in range(12):
        a = taverngym.Game(seed=37)
        b = a.copy()
        assert is_hero_selection(a)
        with pytest.raises(ValueError, match="external policy"):
            mcts.search(a, 0, None)
        select_heroes(a, seed)
        select_heroes(b, seed)
        assert a.state_json() == b.state_json()
        assert not is_hero_selection(a)
        # Hero IDs stay in the model input after the external selection.
        heroes.append(tuple(mcts.observe(a, seat)[1][0] for seat in range(8)))
        before = a.state_json()
        select_heroes(a, seed)
        assert a.state_json() == before
    assert len(set(heroes)) > 1


def test_selfplay_never_records_hero_selection_samples():
    from tavernagent.selfplay import play_one

    samples, _ = play_one(
        PolicyValueNet(seed=1), sims=1, seed=71, action_cap=2, horizon="turn", temperature=0
    )
    assert samples
    assert all(s.x[1] == 0 and s.x[2] == 0 for s in samples)


def test_terminal_search_bypasses_network():
    game = taverngym.Game(seed=81, config_json=json.dumps({"max_turns": 1}))
    select_heroes(game, 81)
    while not game.is_over():
        for seat in range(game.num_players):
            if game.needs_action(seat):
                game.bot_play(seat)
    assert mcts.terminal_value(game, 0) is not None
    with pytest.raises(ValueError, match="terminal rewards"):
        mcts.search(game, 0, None)


def test_shop_freeze_is_per_item_and_global_freeze_is_ignored():
    game, _ = inputs()
    state = json.loads(game.state_json())
    shop = state["players"][0]["shop"]
    assert len(shop) >= 2
    for item in shop:
        item["frozen"] = False
    shop[0]["frozen"] = True
    first = taverngym.Game.from_json(json.dumps(state))
    x, cards, _ = mcts.observe(first, 0)
    layout = dict(taverngym.layout())
    shop_start = (
        layout["GLOBAL_F"]
        + layout["N_QUESTS"] * layout["QUEST_F"]
        + layout["PENDING_KINDS"]
        + layout["N_BOARD"] * layout["MINION_F"]
        + layout["N_HAND"] * layout["ITEM_F"]
    )
    freeze0 = shop_start + 6
    freeze1 = freeze0 + layout["ITEM_F"]
    assert x[freeze0] == 1 and x[freeze1] == 0
    net = PolicyValueNet(seed=27)
    logits, value = net.forward(x, cards)
    changed = x.copy()
    changed[15] = 1.0 - changed[15]
    actual_logits, actual_value = net.forward(changed, cards)
    np.testing.assert_array_equal(actual_logits, logits)
    np.testing.assert_array_equal(actual_value, value)
    # Both states have at least one frozen card, but different frozen slots.
    shop[0]["frozen"], shop[1]["frozen"] = False, True
    second = taverngym.Game.from_json(json.dumps(state))
    y, other_cards, _ = mcts.observe(second, 0)
    assert x[15] == y[15] == 1
    assert y[freeze0] == 0 and y[freeze1] == 1
    assert not np.allclose(net.forward(y, other_cards)[0], logits)


def test_quest_and_power_slots_share_trainable_embeddings():
    _, (x, cards, _) = inputs()
    net = PolicyValueNet(seed=31)
    catalog = json.loads(taverngym.catalog_json())
    slots = {
        "hero_power": [1, 2, 4],
        "quest": [5, 8, 11],
        "quest_reward": [6, 7, 9, 10, 12, 13, 30],
    }
    empty = np.zeros_like(cards)
    baseline = net.forward(x, empty)[0].copy()
    for kind, positions in slots.items():
        cid = next(c["cid"] + 1 for c in catalog if c["kind"] == kind)
        for slot in positions:
            assert net.card_slots[slot]
            changed = empty.copy()
            changed[slot] = cid
            logits, _value = net.forward(x, changed)
            assert not np.allclose(logits, baseline)
            grads = net.backward(np.ones_like(logits), np.zeros((1, 1), np.float32))
            assert np.any(grads["E"][cid] != 0)
            assert np.all(grads["E"][0] == 0)


def test_cosine_schedule_reaches_both_endpoints():
    from tavernagent.selfplay import cosine_lr

    rates = [cosine_lr(1e-3, 1e-4, i, 100) for i in range(100)]
    assert rates[0] == pytest.approx(1e-3)
    assert rates[-1] == pytest.approx(1e-4)
    assert all(a >= b for a, b in itertools.pairwise(rates))


def test_selfplay_seed_reproduces_search_noise_and_sampling():
    from tavernagent.selfplay import play_one

    net = PolicyValueNet(seed=9)
    a, za = play_one(net, 2, 301, 2, "turn", 0.25, 0.35)
    b, zb = play_one(net, 2, 301, 2, "turn", 0.25, 0.35)
    assert za == zb
    assert [s.action for s in a] == [s.action for s in b]
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa.pi, sb.pi)


def test_eliminated_seat_has_terminal_value_before_lobby_finishes():
    game, _ = inputs()
    state = json.loads(game.state_json())
    state["players"][0]["alive"] = False
    state["players"][0]["placement"] = 8
    game = taverngym.Game.from_json(json.dumps(state))
    assert not game.is_over()
    assert mcts.terminal_value(game, 0) == -1


def test_zero_prior_legal_actions_remain_available_for_root_noise():
    game, (_, _, mask) = inputs()

    class Saturated:
        def predict(self, x, cards, mask):
            prior = np.zeros(245, np.float32)
            prior[np.flatnonzero(mask)[0]] = 1.0
            return prior, 0.0

    result = mcts.search(
        game, 0, Saturated(), sims=1, root_noise=0.35, rng=np.random.default_rng(5)
    )
    np.testing.assert_array_equal(result.root.actions, np.flatnonzero(mask))
    assert np.all(result.root.prior[mask] > 0)


def test_replay_window_evicts_old_rounds_and_samples_without_replacement():
    from tavernagent.selfplay import ReplayWindow, cosine_lr

    rng = np.random.default_rng(7)
    replay = ReplayWindow(2)
    assert replay.add_and_sample([1, 2], 2, rng) == [1, 2]
    replay.add_and_sample([3, 4], 2, rng)
    result = replay.add_and_sample([5, 6], 4, rng)
    assert sorted(result) == [3, 4, 5, 6]
    assert cosine_lr(0.001, 0.0001, 199, 100) == pytest.approx(0.0001)


def test_training_diagnostics_are_finite_and_parameter_count_is_fixed():
    _game, (x, cards, mask) = inputs()
    net = PolicyValueNet(seed=4)
    trainer = Trainer(net)
    target = mask.astype(np.float32) / mask.sum()
    trainer.step(x[None], cards[None], mask[None], target[None], np.array([0.0]), 0.001)
    assert net.n_params() == 627510
    assert all(np.isfinite(v) for v in trainer.last_metrics.values())
    assert trainer.last_metrics["policy_entropy"] >= 0
    assert trainer.last_metrics["grad_norm"] >= 0


@pytest.mark.parametrize("horizon", ["turn", "game"])
def test_arena_identical_models_balance_exactly_under_seat_swaps(tmp_path, horizon):
    from tavernagent.arena import battle

    model = tmp_path / "same.npz"
    PolicyValueNet(seed=2).save(model)
    result = battle(
        model, model, games=2, sims=1, workers=1, action_cap=2, horizon=horizon, c_puct=0.75
    )
    assert result["horizon"] == horizon
    assert result["c_puct"] == 0.75
    assert result["score"] == 0.5
    assert result["elo_equivalent"] == 0


@pytest.mark.parametrize("activation", ["gelu", "silu", "relu"])
def test_parameter_free_activation_variants_round_trip_and_gradients(tmp_path, activation):
    _, (x, cards, mask) = inputs()
    net = PolicyValueNet(seed=8, activation=activation, residual_scale=0.5)
    assert net.n_params() == 627510
    logits, _value = net.forward(x, cards)
    dp = np.ones_like(logits, dtype=np.float32) * 0.01
    g = net.backward(dp, np.zeros((1, 1), np.float32))
    param = net.p["W1"]
    index = (2, 3)
    old = param[index].copy()
    eps = 0.001

    def loss():
        return float((net.forward(x, cards)[0] * dp).sum())

    param[index] = old + eps
    hi = loss()
    param[index] = old - eps
    lo = loss()
    param[index] = old
    np.testing.assert_allclose(g["W1"][index], (hi - lo) / (2 * eps), rtol=0.1, atol=0.0002)
    path = tmp_path / "variant.npz"
    net.save(path)
    restored = PolicyValueNet.load(path)
    assert restored.activation == activation and restored.residual_scale == 0.5
    np.testing.assert_array_equal(
        net.predict(x, cards, mask)[0], restored.predict(x, cards, mask)[0]
    )


@pytest.mark.parametrize("head_norm", [False, True])
def test_full_trainer_snapshot_restores_optimizer_rng_and_next_update(tmp_path, head_norm):
    from argparse import Namespace

    from tavernagent.selfplay import Stats, load_training_state, save_training_state

    _, (x, cards, mask) = inputs()
    net = PolicyValueNet(seed=12, head_norm=head_norm)
    trainer = Trainer(net, ema_decay=0.995, value_weight=0.25, weight_decay=0.02, clip=2.0)
    pi = (mask / mask.sum()).astype(np.float32)[None]
    data = (x[None], cards[None], mask[None], pi, np.array([0.5], np.float32), 0.0003)
    trainer.step(*data)
    rng = np.random.default_rng(10)
    replay_rng = np.random.default_rng(11)
    rng.random(5)
    replay_rng.random(3)
    path = tmp_path / "state.npz"
    save_training_state(path, trainer, 1, rng, replay_rng, [], Stats(), Namespace(replay_rounds=1))
    if head_norm:
        with pytest.raises(ValueError, match="schema"):
            load_training_state(
                path,
                Trainer(PolicyValueNet()),
                np.random.default_rng(1),
                np.random.default_rng(2),
            )
    restored = Trainer(PolicyValueNet(seed=99, head_norm=head_norm))
    rng2 = np.random.default_rng(1)
    replay_rng2 = np.random.default_rng(2)
    load_training_state(path, restored, rng2, replay_rng2)
    assert trainer.t == restored.t
    np.testing.assert_array_equal(rng.random(10), rng2.random(10))
    np.testing.assert_array_equal(replay_rng.random(10), replay_rng2.random(10))
    trainer.step(*data)
    restored.step(*data)
    for key in net.p:
        np.testing.assert_array_equal(net.p[key], restored.net.p[key])
        np.testing.assert_array_equal(trainer.ema[key], restored.ema[key])


@pytest.mark.parametrize("activation", ["gelu", "silu", "relu"])
def test_fixed_head_norm_preserves_parameters_and_backpropagates(tmp_path, activation):
    _, (x, cards, mask) = inputs()
    net = PolicyValueNet(seed=18, activation=activation, head_norm=True)
    original = PolicyValueNet(seed=18, activation=activation)
    assert net.n_params() == original.n_params() == 627510
    for key in net.p:
        np.testing.assert_array_equal(net.p[key], original.p[key])
    rng = np.random.default_rng(77)
    dp = rng.standard_normal((1, 245), dtype=np.float32) * 0.01
    dv = np.array([[0.03]], np.float32)
    _, value = net.forward(x, cards)
    grads = net.backward(dp, dv * (1 - value[:, None] ** 2))

    def loss():
        logits, value = net.forward(x, cards)
        return float((logits.astype(np.float64) * dp).sum() + (value[:, None] * dv).sum())

    for key in ("E", "W0", "W1", "Wp", "Wv"):
        parameter = net.p[key]
        index = np.unravel_index(np.abs(grads[key]).argmax(), parameter.shape)
        old = parameter[index].copy()
        parameter[index] = old + 0.001
        hi = loss()
        parameter[index] = old - 0.001
        lo = loss()
        parameter[index] = old
        np.testing.assert_allclose(grads[key][index], (hi - lo) / 0.002, rtol=0.03, atol=0.0001)
    path = tmp_path / "normalized.npz"
    net.save(path)
    restored = PolicyValueNet.load(path)
    assert restored.head_norm
    np.testing.assert_array_equal(
        net.predict(x, cards, mask)[0], restored.predict(x, cards, mask)[0]
    )


def test_weights_only_resume_preserves_architecture_in_full_state(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "source.npz"
    PolicyValueNet(activation="silu", residual_scale=0.5, head_norm=True).save(model)
    output = tmp_path / "trained.npz"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(root / "python"), str(root / "tavernagent")]),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    command = [sys.executable, "-m", "tavernagent.selfplay"]
    subprocess.run(
        command
        + [
            "--resume",
            str(model),
            "--steps",
            "1",
            "--games",
            "1",
            "--workers",
            "1",
            "--sims",
            "1",
            "--action-cap",
            "2",
            "--max-samples",
            "8",
            "--epochs",
            "1",
            "--probe-every",
            "0",
            "--save-trainer-state",
            "--out",
            str(output),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    options = json.loads(output.with_suffix(".config.json").read_text())
    assert options["activation"] == "silu"
    assert options["residual_scale"] == 0.5 and options["head_norm"]
    continued = tmp_path / "continued.npz"
    subprocess.run(
        command
        + [
            "--resume-state",
            str(tmp_path / "trained.trainer.npz"),
            "--steps",
            "2",
            "--out",
            str(continued),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    restored = PolicyValueNet.load(continued)
    assert restored.head_norm and restored.activation == "silu" and restored.residual_scale == 0.5
