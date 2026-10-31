"""Summarise and diff the files written by the in-game F9 dump.

    python tavernplugin/tools/inspect_dumps.py [dumps_dir]

Each dump is a pair: `dump_<ts>.json` (raw game state) and `dump_<ts>.png`
(screenshot). The snapshot actually posted to the server is a third file,
`dump_<ts>.snapshot.json`, when one was captured at the same moment.
"""

import glob
import json
import os
import sys

DEFAULT_DIR = os.path.expandvars(r"%USERPROFILE%\Documents\TavernPlugin\dumps")


def main(dumps_dir=DEFAULT_DIR):
    files = sorted(f for f in glob.glob(os.path.join(dumps_dir, "dump_*.json"))
                   if not f.endswith(".snapshot.json"))
    if not files:
        raise SystemExit(f"no dumps in {dumps_dir}")

    print(f"{len(files)} dumps in {dumps_dir}\n")
    for path in files:
        d = json.load(open(path, encoding="utf-8"))
        s = d.get("summary") or {}
        me = d.get("me") or {}
        choice = d.get("choice") or {}
        png = os.path.splitext(path)[0] + ".png"
        print(f'{os.path.basename(path)}  turn={s.get("turn")} '
              f'combat={s.get("in_combat")}  entities={len(d.get("entities") or [])}'
              f'{"  [png]" if os.path.exists(png) else ""}')
        print(f'   me: board={len(me.get("board") or [])} hand={len(me.get("hand") or [])} '
              f'gold={me.get("gold")} tier={me.get("tavern_tier")} '
              f'hp={me.get("hero_health")} armor={me.get("hero_armor")}')
        for m in (me.get("board") or [])[:8]:
            print(f'      {m.get("card")} {m.get("atk")}/{m.get("hp")}')
        if choice.get("active"):
            options = ", ".join(o.get("card") or "?" for o in (choice.get("options") or []))
            print(f'   choice: type={choice.get("choice_type")} '
                  f'{choice.get("count_min")}..{choice.get("count_max")} [{options}]')


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
