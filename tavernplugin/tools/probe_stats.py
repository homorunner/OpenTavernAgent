import json
import sys

sys.path.insert(0, "tavernplugin/server")
import bridge  # noqa: E402

core = bridge.load_core()
g = core.Game(1)


def dbg(**cmd):
    return json.loads(g.debug(json.dumps(cmd)))


dbg(cmd="set_hero", p=0, hero="BG32_HERO_001")
dbg(cmd="set_hero", p=1, hero="BG32_HERO_001")
dbg(cmd="add_minion", p=0, zone="board", card="BG36_201", atk=11, hp=10)
view = json.loads(g.view(0))
m = view["me"]["board"][0]
for key in ("atk", "hp", "base_atk", "base_hp", "max_hp", "tier", "golden",
            "pbonus_atk", "pbonus_hp", "breakdown", "keywords", "player_bonus"):
    if key in m:
        print(f"{key:14} {json.dumps(m[key], ensure_ascii=False)}")
print()
print("state.json player 0 keys:", list(json.loads(g.state_json())["players"][0].keys()))
print("player stats:", json.loads(g.state_json())["players"][0].get("stats"))
