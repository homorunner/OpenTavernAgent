"""Hero selection is an external policy, separate from recruit-phase learning."""

from __future__ import annotations

import numpy as np
import taverngym

_LAYOUT = dict(taverngym.layout())


def is_hero_selection(game) -> bool:
    features, _ = game.observe(0)
    return bool(np.frombuffer(features, dtype="<f4")[1])


def select_heroes(game, seed: int) -> None:
    """Uniformly choose an offered hero for every seat, reproducibly from seed.

    No network calls or training samples. Evaluation uses the same choices for all
    policies at a given seed. A live hero-ranking policy belongs at this boundary.
    Recruit-phase Discovers opened by hero effects remain for the normal policy.
    """
    if not is_hero_selection(game):
        return
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4845524F]))
    start = _LAYOUT["A_CHOOSE"]
    for seat in range(game.num_players):
        if game.needs_action(seat):
            mask = np.frombuffer(game.legal_mask(seat), dtype=np.uint8)
            choices = np.flatnonzero(mask[start : start + _LAYOUT["N_CHOICE"]]) + start
            if not len(choices):
                raise RuntimeError("hero selection has no legal offered hero")
            game.step(seat, int(rng.choice(choices)))
