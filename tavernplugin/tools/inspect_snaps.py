"""Print a table of the snapshots written by `tavernreplay snapshot`.

    python tavernplugin/tools/inspect_snaps.py tavernplugin/tools/replay/out/snaps.jsonl
"""

import json
import sys


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    print(f"snapshots {len(rows)}")
    for i, r in enumerate(rows):
        game = r.get("game", {})
        me, them = r.get("me", {}), r.get("them", {})
        print(
            f"{i:>3} {game.get('phase', '?'):<8} turn {game.get('turn', '-'):>3} "
            f"me {len(me.get('board', [])):>2} them {len(them.get('board', [])):>2} "
            f"shop {len(r.get('shop', [])):>2}  hero {me.get('hero', '') or '-':<10} "
            f"hp {me.get('health', '-')}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tavernplugin/tools/replay/out/snaps.jsonl")
