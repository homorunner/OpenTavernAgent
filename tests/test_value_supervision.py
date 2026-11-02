"""A single value head can learn stopping positions without giving continuation states stop labels."""

import json
from argparse import Namespace

import numpy as np
import pytest
import taverngym
from tavernagent.heroes import select_heroes
from tavernagent.net import PolicyValueNet
from tavernagent.selfplay import (
    Sample,
    Stats,
    Trainer,
    load_training_state,
    save_training_state,
    train_epoch,
)

from tavernagent import mcts


def batch():
    game = taverngym.Game(seed=79)
    select_heroes(game, 79)
    x, cards, mask = mcts.observe(game, 0)
    pi = (mask / mask.sum()).astype(np.float32)
    return x, cards, mask, pi


def test_end_supervision_uses_only_selected_end_labels_and_adds_no_head():
    x, cards, mask, pi = batch()
    other = int(next(a for a in np.flatnonzero(mask) if a != 0))
    net = PolicyValueNet(seed=4)
    trainer = Trainer(net, weight_decay=0, value_samples="end")
    trainer.step(
        np.stack([x, x]),
        np.stack([cards, cards]),
        np.stack([mask, mask]),
        np.stack([pi, pi]),
        np.array([-1, 1], np.float32),
        0.001,
        actions=np.array([0, other]),
    )
    assert trainer.last_metrics["value_training_samples"] == 1
    assert net.p["bv"][0] < 0
    assert trainer.last_metrics["value_head_grad_norm"] > 0
    assert net.n_params() == 627510 and "We" not in net.p


def test_unknown_actions_do_not_become_end_value_labels_or_pg_labels():
    x, cards, mask, pi = batch()
    net = PolicyValueNet(seed=4)
    trainer = Trainer(net, weight_decay=0, value_samples="end")
    before = {k: net.p[k].copy() for k in ("Wv", "bv")}
    train_epoch(trainer, [Sample(x, cards, mask, pi, z=1, action=-1)], pg_weight=0.1)
    assert trainer.last_metrics["value_training_samples"] == 0
    assert trainer.last_metrics["value_head_grad_norm"] == 0
    assert trainer.epoch_metrics["value_training_samples"] == 0
    for k, expected in before.items():
        np.testing.assert_array_equal(expected, net.p[k])


def test_end_supervision_requires_real_action_metadata_and_legality():
    x, cards, mask, pi = batch()
    trainer = Trainer(PolicyValueNet(), value_samples="end")
    args = (x[None], cards[None], mask[None], pi[None], np.array([1], np.float32), 0.001)
    with pytest.raises(ValueError, match="requires selected actions"):
        trainer.step(*args)
    illegal = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError, match="legal"):
        trainer.step(*args, actions=np.array([illegal]))


def test_end_supervision_full_state_restores_mode_and_exact_next_update(tmp_path):
    x, cards, mask, pi = batch()
    trainer = Trainer(PolicyValueNet(seed=9), value_samples="end")
    args = (x[None], cards[None], mask[None], pi[None], np.array([-0.5], np.float32), 0.001)
    trainer.step(*args, actions=np.array([0]))
    path = tmp_path / "trainer.npz"
    save_training_state(
        path,
        trainer,
        1,
        np.random.default_rng(1),
        np.random.default_rng(2),
        [],
        Stats(games=1),
        Namespace(replay_rounds=1, value_samples="end"),
    )
    restored = Trainer(PolicyValueNet(seed=10))
    meta = load_training_state(path, restored, np.random.default_rng(), np.random.default_rng())
    assert meta["format"] == 2 and meta["search_version"] == mcts.SEARCH_VERSION
    assert restored.value_samples == "end"
    trainer.step(*args, actions=np.array([0]))
    restored.step(*args, actions=np.array([0]))
    for field in ("m", "v", "ema"):
        for key in trainer.net.p:
            np.testing.assert_array_equal(
                getattr(trainer, field)[key], getattr(restored, field)[key]
            )
    for key in trainer.net.p:
        np.testing.assert_array_equal(trainer.net.p[key], restored.net.p[key])


def test_incomplete_v2_snapshot_does_not_guess_value_supervision(tmp_path):
    path = tmp_path / "state.npz"
    trainer = Trainer(PolicyValueNet())
    save_training_state(
        path,
        trainer,
        0,
        np.random.default_rng(),
        np.random.default_rng(),
        [],
        Stats(),
        Namespace(replay_rounds=1),
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays["metadata"]))
    del meta["optimizer_hparams"]["value_samples"]
    arrays["metadata"] = json.dumps(meta)
    np.savez(path, **arrays)
    restored = Trainer(PolicyValueNet(), value_samples="end")
    with pytest.raises(ValueError, match="incomplete trainer format v2 optimizer settings"):
        load_training_state(path, restored, np.random.default_rng(), np.random.default_rng())


@pytest.mark.parametrize(
    "protocol,search_version",
    [
        (1, mcts.SEARCH_VERSION),
        (3, mcts.SEARCH_VERSION),
        (4, mcts.SEARCH_VERSION),
        (2, "copied-rng-v1"),
        (2, "sampled-prep-v3"),
        (2, None),
    ],
)
def test_trainer_only_accepts_v2_state_and_search(tmp_path, protocol, search_version):
    path = tmp_path / "end.npz"
    trainer = Trainer(PolicyValueNet(), value_samples="end")
    save_training_state(
        path,
        trainer,
        0,
        np.random.default_rng(),
        np.random.default_rng(),
        [],
        Stats(),
        Namespace(replay_rounds=1, value_samples="end"),
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays["metadata"]))
    assert meta["format"] == 2
    meta["format"] = protocol
    meta["search_version"] = search_version
    arrays["metadata"] = json.dumps(meta)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="schema"):
        load_training_state(path, trainer, np.random.default_rng(), np.random.default_rng())
