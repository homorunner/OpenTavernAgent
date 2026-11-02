"""TavernAgent: a small policy/value network plus PUCT search over TavernGym.

The network provides action priors and position values for Monte-Carlo tree search
and learns from all-seat self-play.

    from tavernagent import PolicyValueNet, search, select_heroes

    select_heroes(game, seed=0)
    net = PolicyValueNet()
    result = search(game, seat=0, net=net, sims=64)
    action = result.best_action
"""

from .heroes import select_heroes
from .mcts import SearchResult, placement_reward, search
from .net import N_ACTIONS, N_FEATURES, PolicyValueNet, masked_softmax

__all__ = [
    "N_ACTIONS",
    "N_FEATURES",
    "PolicyValueNet",
    "SearchResult",
    "masked_softmax",
    "placement_reward",
    "search",
    "select_heroes",
]
