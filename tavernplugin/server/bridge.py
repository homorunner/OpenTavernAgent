"""Bridge between a live-game snapshot and TavernGym.

The snapshot is produced by the in-game client (``tavernplugin/client``) from
Power.log.  This module turns it into TavernGym calls:

* ``combat``  -- build both boards in a scratch ``Game`` and run ``CombatPreview``
* ``align``   -- build the same boards and report where TavernGym and the game
                 disagree (goal 3 of the project: validate the simulator)
* ``advise``  -- rank candidate actions on the current board

TavernGym is imported lazily so the HTTP server still starts (and answers
``/health``) when the Rust extension has not been built yet.
"""

from __future__ import annotations

import json
import random
import time

KEYWORDS = {
    "taunt": "TAUNT",
    "divine_shield": "DIVINE_SHIELD",
    "poisonous": "POISONOUS",
    "windfury": "WINDFURY",
    "mega_windfury": "MEGA_WINDFURY",
    "reborn": "REBORN",
    "stealth": "STEALTH",
    "cleave": "CLEAVE",
    "rush": "RUSH",
}


class TavernGymUnavailable(RuntimeError):
    """The native extension is missing; /health reports ``taverngym: false``."""


def load_core():
    try:
        from taverngym import _core  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on the build
        raise TavernGymUnavailable(str(exc)) from exc
    return _core


_CARD_IDS = None
_CATALOG = None


def catalog():
    """The card table TavernGym ships, as id -> card dict."""
    global _CATALOG
    if _CATALOG is None:
        core = load_core()
        _CATALOG = {card["id"]: card for card in json.loads(core.catalog_json())}
    return _CATALOG


def card_ids():
    """The card ids TavernGym knows, built once from its catalog."""
    global _CARD_IDS
    if _CARD_IDS is None:
        _CARD_IDS = set(catalog())
    return _CARD_IDS


def attachable(card_id):
    """Should this attachment be replayed in the simulator?

    A minion's stats and keywords already reflect everything attached to it, so
    re-attaching a plain enchantment would double-count its bonuses. Only effects
    with behaviour the snapshot cannot express are worth rebuilding:

    * Dark Gifts (`BG36_MidGameEffect_*`) — they keep triggering ("whenever you
      play a card, +3 Health") and the golden merge unions them.
    * Magnetic minions — their Deathrattle/trigger has to keep firing.

    Everything else (`TB_BaconShopBadsongE`, `BG34_170` and other bookkeeping
    enchantments) is skipped.
    """
    if not card_id:
        return False
    if card_id.startswith("BG36_MidGameEffect_"):
        return True
    card = catalog().get(card_id) or {}
    if card.get("kind") != "minion":
        return False
    # the keyword line, not a card that merely mentions the word ("Battlecry: Get a
    # random Magnetic Volumizer" is the parent enchantment, not a magnetized minion)
    return (card.get("text") or "").lstrip().startswith("Magnetic")


def card_id_of(minion):
    """TavernGym card id for a logged minion, or None when the card is unknown."""
    cid = (minion or {}).get("card") or ""
    if not cid:
        return None
    return cid if cid in card_ids() else None


def _board(payload):
    return [m for m in (payload or {}).get("board", []) if m.get("atk") is not None]


def build_preview_game(me, them, seed=1):
    """A scratch ``Game`` holding the two boards, ready for ``CombatPreview``.

    Seat 0 is the local player, seat 1 the opponent (the ghost/opponent of the
    round).  Cards TavernGym does not know are skipped and reported back.
    """
    core = load_core()
    game = core.Game(int(seed))
    notes = []

    def dbg(**cmd):
        return json.loads(game.debug(json.dumps(cmd)))

    for seat, side in ((0, me), (1, them)):
        hero = (side or {}).get("hero")
        if hero:
            try:
                dbg(cmd="set_hero", p=seat, hero=hero)
            except Exception as exc:
                notes.append(f"seat {seat}: hero {hero} not set ({exc})")
        tier = (side or {}).get("tier")
        if tier:
            try:
                dbg(cmd="set_player", p=seat, tier=int(tier))
            except Exception as exc:
                notes.append(f"seat {seat}: tier not set ({exc})")

        for index, minion in enumerate(_board(side)):
            cid = card_id_of(minion)
            if cid is None:
                notes.append(f"seat {seat}: unknown card {minion.get('card')!r} at {index}")
                continue
            try:
                dbg(
                    cmd="add_minion",
                    p=seat,
                    zone="board",
                    card=cid,
                    pos=index,
                    golden=bool(minion.get("golden")),
                    atk=int(minion["atk"]),
                    hp=int(minion["hp"]),
                )
            except Exception as exc:
                notes.append(f"seat {seat}: add {cid} failed ({exc})")
                continue
            kws = [KEYWORDS[k] for k, v in (minion.get("keywords") or {}).items() if v and k in KEYWORDS]
            if kws:
                try:
                    dbg(cmd="edit_minion", p=seat, zone="board", index=index, keywords=kws)
                except Exception as exc:
                    notes.append(f"seat {seat}: keywords {kws} failed ({exc})")

            # Dark Gifts / magnetized cards. They are attached before the stats are
            # forced, because a gift's on_attach hook adds its own bonus and the
            # snapshot already reports the *resulting* numbers.
            wanted = [a for a in (minion.get("attached") or []) if attachable(a)]
            skipped = [a for a in (minion.get("attached") or []) if not attachable(a)]
            if skipped:
                notes.append(f"seat {seat} #{index}: skipped attachments {skipped}")
            for attached in wanted:
                try:
                    dbg(cmd="edit_minion", p=seat, zone="board", index=index, attach=attached)
                except Exception as exc:
                    notes.append(f"seat {seat}: attach {attached} failed ({exc})")
            if wanted:
                try:
                    dbg(
                        cmd="edit_minion",
                        p=seat,
                        zone="board",
                        index=index,
                        atk=int(minion["atk"]),
                        hp=int(minion["hp"]),
                        keywords=kws or None,
                    )
                except Exception as exc:
                    notes.append(f"seat {seat}: restat after attach failed ({exc})")
    return game, notes


def combat(payload, n=512):
    """Win / tie / loss probabilities for the two boards in ``payload``."""
    started = time.time()
    game, notes = build_preview_game(payload.get("me"), payload.get("them"), seed=payload.get("seed", 1))
    result = json.loads(
        game.debug(json.dumps({"cmd": "combat_preview", "a": 0, "b": 1, "n": int(n), "details": True}))
    )
    trials = result.get("trials") or []
    taken = [t["damage_taken"] for t in trials]
    dealt = [t["damage_dealt"] for t in trials]
    return {
        "ok": True,
        "n": int(result.get("n", n)),
        "win": round(float(result.get("win_a", 0.0)), 4),
        "tie": round(float(result.get("tie", 0.0)), 4),
        "loss": round(float(result.get("win_b", 0.0)), 4),
        "avg_damage_taken": round(sum(taken) / len(taken), 2) if taken else 0.0,
        "avg_damage_dealt": round(sum(dealt) / len(dealt), 2) if dealt else 0.0,
        "lethal_risk": round(sum(1 for t in trials if t.get("lethal_loss")) / len(trials), 4) if trials else 0.0,
        "elapsed_ms": int((time.time() - started) * 1000),
        "notes": notes,
    }


def align(payload):
    """Compare the logged boards with the boards TavernGym rebuilt from them."""
    game, notes = build_preview_game(payload.get("me"), payload.get("them"), seed=payload.get("seed", 1))
    view = json.loads(game.view(0))
    issues = []
    diffs = []

    def side_of(name):
        data = (payload or {}).get(name) or {}
        return data

    def taverngym_board(seat):
        if seat == 0:
            return (view.get("me") or {}).get("board") or []
        opponents = view.get("opponents") or []
        if isinstance(opponents, dict):
            opponents = list(opponents.values())
        if len(opponents) > seat - 1:
            return (opponents[seat - 1] or {}).get("board") or []
        return []

    for seat, name in ((0, "me"), (1, "them")):
        expected = _board(side_of(name))
        actual = taverngym_board(seat)
        if len(expected) != len(actual):
            diffs.append({"seat": seat, "field": "board_len", "game": len(expected), "taverngym": len(actual)})
        for i in range(min(len(expected), len(actual))):
            g, t = expected[i], actual[i]
            for field in ("atk", "hp"):
                tval = t.get(field)
                gval = g.get(field)
                if tval is not None and gval is not None and int(tval) != int(gval):
                    diffs.append({"seat": seat, "index": i, "field": field, "game": gval, "taverngym": tval})
            gcard = g.get("card")
            tcard = t.get("card") or t.get("card_id") or t.get("id")
            if gcard and tcard and gcard != tcard:
                diffs.append({"seat": seat, "index": i, "field": "card", "game": gcard, "taverngym": tcard})

    for name, data in (("me", side_of("me")), ("them", side_of("them"))):
        for minion in _board(data):
            if card_id_of(minion) is None:
                issues.append(f"{name}: card {minion.get('card')!r} is not in TavernGym's table")

    return {
        "ok": True,
        "aligned": not diffs and not issues,
        "diffs": diffs,
        "issues": issues + notes,
        "counts": {"me": len(_board(side_of("me"))), "them": len(_board(side_of("them")))},
    }


def _swap_candidates(board):
    """Boards differ by minion order; propose each value-preserving swap."""
    out = []
    if len(board) < 2:
        return out
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            out.append((i, j))
    return out


def advise(payload, n=128):
    """Rank a few candidate placements by simulated win rate.

    This is deliberately small: it only reorders the current warband.  Buying
    decisions need the Tavern contents, which Power.log does not expose in a
    form this prototype parses yet (see README, "Limitations").
    """
    me = payload.get("me") or {}
    them = payload.get("them") or {}
    board = _board(me)
    if not board:
        return {"ok": True, "recommendations": [], "note": "empty warband: nothing to arrange"}
    if not _board(them):
        return {"ok": True, "recommendations": [], "note": "no opponent board: wait for combat"}

    tests = [(None, None)] + _swap_candidates(board)[:12]
    results = []
    for i, j in tests:
        moved = list(board)
        if i is not None:
            moved[i], moved[j] = moved[j], moved[i]
        payload2 = dict(payload)
        payload2["me"] = dict(me, board=moved)
        stats = combat(payload2, n=n)
        label = "keep order" if i is None else f"swap {i + 1}<->{j + 1}"
        results.append(
            {
                "kind": "move",
                "text": label,
                "win": stats["win"],
                "avg_damage_taken": stats["avg_damage_taken"],
            }
        )
    results.sort(key=lambda r: (-r["win"], r["avg_damage_taken"]))
    best = results[0]["win"]
    for r in results:
        r["delta"] = round(r["win"] - best, 4)
    return {
        "ok": True,
        "recommendations": results[:5],
        "note": f"{len(tests)} placements x {n} simulated combats",
    }

