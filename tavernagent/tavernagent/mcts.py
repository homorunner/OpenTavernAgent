"""PUCT with sampled chance outcomes and an independent RNG per simulation.

The default ``turn`` horizon stops at End Turn without executing it: no opponents,
end-of-turn triggers or combat are simulated. The value network evaluates the
current recruit position. ``game`` is an explicit full-state experimental mode;
it advances other seats with the scripted bot and is not information-limited.

Action edges aggregate visits across distinct observable chance outcomes.
Exact Game.copy() remains available for replay; search always supplies a new seed.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .heroes import is_hero_selection
from .net import N_ACTIONS, PolicyValueNet

SEARCH_VERSION = "sampled-prep-v2"
MAX_PLIES = 2000
MAX_DEPTH = 100


class Node:
    __slots__ = ("actions", "children", "cn", "cw", "game", "n", "prior", "terminal", "value", "w")

    def __init__(self, game, terminal=None):
        self.game = game  # representative snapshot; never used to sample a transition
        self.prior = None
        self.actions = None
        self.children = None  # action -> {observable outcome: decision node}
        self.n = 0
        self.w = 0.0
        self.cn = None
        self.cw = None
        self.terminal = terminal
        self.value = 0.0

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


def placement_reward(placement: int) -> float:
    return 1.0 - (placement - 1) * (2.0 / 7.0)


def terminal_value(game, seat: int = 0):
    place = list(game.placements())[seat]
    return placement_reward(place) if place > 0 else None


def advance(game, seat: int, action: int) -> None:
    """Full-game mode only: apply an action and drive bots until we can act."""
    game.step(seat, action)
    for _ in range(MAX_PLIES):
        if terminal_value(game, seat) is not None or game.needs_action(seat):
            return
        moved = False
        for p in range(game.num_players):
            if p != seat and game.needs_action(p):
                game.step(p, game.bot_action(p))
                moved = True
        if not moved:
            return
    raise RuntimeError("search bot rollout exceeded its action guard")


def observe(game, seat: int):
    floats, cards = game.observe(seat)
    x = np.frombuffer(floats, dtype="<f4")
    mask = np.frombuffer(game.legal_mask(seat), dtype=np.uint8).astype(bool)
    return x, np.frombuffer(cards, dtype="<i4"), mask


def _key(observation):
    # Observation history distinguishes outcomes; future RNG is deliberately absent.
    x, cards, mask = observation
    return x.tobytes(), cards.tobytes(), mask.tobytes()


def _expand(node: Node, net: PolicyValueNet, observation) -> float:
    x, cards, mask = observation
    prior, value = net.predict(x, cards, mask)
    node.prior = prior
    node.actions = np.flatnonzero(mask).astype(np.int32)
    node.children = [{} for _ in node.actions]
    node.cn = np.zeros(len(node.actions), np.float32)
    node.cw = np.zeros(len(node.actions), np.float32)
    node.value = float(value)
    return node.value


def _backup(path, value):
    for parent, index in path:
        parent.n += 1
        parent.w += value
        parent.cn[index] += 1
        parent.cw[index] += value


def _simulation_seed(rng):
    return int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


def _search_rng(observation, seat):
    # Repeatable inference independent of the game's hidden RNG. Self-play instead
    # supplies its own seeded Generator, separate from the simulator's random stream.
    digest = hashlib.blake2b(digest_size=16)
    digest.update(int(seat).to_bytes(4, "little"))
    for part in _key(observation):
        digest.update(part)
    return np.random.default_rng(int.from_bytes(digest.digest(), "little"))


def search(
    game,
    seat: int,
    net: PolicyValueNet,
    sims: int = 64,
    c_puct: float = 1.5,
    chance_hook=None,
    horizon: str = "turn",
    max_depth: int = MAX_DEPTH,
    root_noise: float = 0.0,
    noise_alpha: float = 0.3,
    rng=None,
) -> SearchResult:
    """Search independent sampled trajectories without changing the real game.

    Optional ``chance_hook`` runs on each reseeded simulation root and must preserve
    the searching player's observation. Future RNG sampling needs no hook.
    """
    if game.is_over():
        raise ValueError("search requires an active game; terminal rewards bypass the network")
    if is_hero_selection(game):
        raise ValueError("select heroes with an external policy before searching")
    if not game.needs_action(seat) or terminal_value(game, seat) is not None:
        raise ValueError("search requires a seat that can act")
    if horizon not in ("turn", "game"):
        raise ValueError(f"unknown horizon {horizon!r}")
    if sims < 1 or max_depth < 1:
        raise ValueError("sims and max_depth must be positive")
    if not np.isfinite(c_puct) or c_puct < 0:
        raise ValueError("c_puct must be finite and nonnegative")
    if not 0 <= root_noise <= 1 or not np.isfinite(noise_alpha) or noise_alpha <= 0:
        raise ValueError("noise must be in [0, 1] and noise_alpha must be positive")
    observation = observe(game, seat)
    generator = _search_rng(observation, seat) if rng is None else rng
    root = Node(game.copy(seed=_simulation_seed(generator)))
    root_value = _expand(root, net, observation)
    _add_root_noise(root, root_noise, noise_alpha, generator)

    for _ in range(sims):
        simulation = game.copy(seed=_simulation_seed(generator))
        if chance_hook is not None:
            chance_hook(simulation)
            if _key(observe(simulation, seat)) != _key(observation):
                raise ValueError("chance_hook must preserve the root observation")
        node = root
        path = []
        for _depth in range(max_depth):
            if not len(node.actions):
                value = node.value
                break
            index = _select(node, c_puct)
            path.append((node, index))
            action = int(node.actions[index])
            if horizon == "turn" and action == 0:
                # Even just stepping EndTurn for the last seat could start combat.
                value = node.value
                break
            if horizon == "turn":
                simulation.step(seat, action)
            else:
                advance(simulation, seat, action)
            value = terminal_value(simulation, seat)
            if value is not None:
                break
            child_observation = observe(simulation, seat)
            outcome = _key(child_observation)
            child = node.children[index].get(outcome)
            if child is None:
                child = Node(simulation.copy())
                node.children[index][outcome] = child
                value = _expand(child, net, child_observation)
                break
            node = child
        else:
            value = node.value
        _backup(path, value)

    visits = np.zeros(N_ACTIONS, np.float32)
    visits[root.actions] = root.cn
    total = visits.sum()
    policy = visits / total if total > 0 else root.prior
    return SearchResult(policy, root_value, visits, root)


def _add_root_noise(root: Node, epsilon: float, alpha: float, rng) -> None:
    if not epsilon or not len(root.actions):
        return
    noise = rng.dirichlet([alpha] * len(root.actions))
    prior = root.prior.copy()
    prior[root.actions] = (1.0 - epsilon) * prior[root.actions] + epsilon * noise
    root.prior = prior


class SearchResult:
    __slots__ = ("policy", "root", "value", "visits")

    def __init__(self, policy, value, visits, root):
        self.policy = policy
        self.value = value
        self.visits = visits
        self.root = root

    @property
    def best_action(self) -> int:
        return int(np.argmax(self.visits)) if self.visits.any() else int(np.argmax(self.policy))


def _select(node: Node, c_puct: float) -> int:
    u = node.cw / np.maximum(node.cn, 1.0)
    u = u + c_puct * node.prior[node.actions] * math.sqrt(node.n + 1) / (1.0 + node.cn)
    return int(np.argmax(u))
