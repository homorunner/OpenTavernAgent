# tavernplugin

An in-game companion for Hearthstone Battlegrounds, backed by TavernGym.

The module has two halves:

```
Hearthstone (Windows)                                    Python 3.10+
┌──────────────────────────────────────────┐            ┌──────────────────────────────┐
│ tavernplugin client  (UnityDoorstop DLL)  │   HTTP     │ tavernplugin server           │
│                                           │ 127.0.0.1  │                              │
│  Power.log ──► PowerLogParser ──► Snapshot│ ─────────► │ bridge.py ──► TavernGym       │
│       ▲                        │          │  :8765     │   /combat /align /advise      │
│       │                        ▼          │ ◄───────── │                              │
│  PowerLogTailer          OnGUI overlay    │            │ (a trained agent plugs in here)│
└──────────────────────────────────────────┘            └──────────────────────────────┘
```

The client reads the game's own state (`GameState` / `Entity`), with the Power.log
parser kept for offline replay. A parsing bug can only produce a wrong number, it
can never crash the game, and the same code runs offline against recorded logs
(`tools/replay`), so everything here is testable without opening the game.

The story behind this — what the simulator is, what went wrong at runtime, and how
the board and shop were finally aligned — is written up in
[docs/TAVERNPLUGIN_JOURNAL.md](../docs/TAVERNPLUGIN_JOURNAL.md).

## What it does

| Goal | Endpoint | Where the answer appears |
| :-- | :-- | :-- |
| 1. Win probability of the coming combat | `POST /combat` | `combat: win 62% tie 3% loss 35% dmg 6.0 (512 sims)` |
| 2. Recommended recruit-phase action | `POST /advise` | `advice: > swap 1<->3 58%` |
| 3. Validate TavernGym against the live game | `POST /align` | `aligned with TavernGym` / `align: 3 difference(s)` |

Goal 3 is the interesting one for the project itself: the client sends the state the
game says it has, `bridge.py` rebuilds the same two boards inside TavernGym with
debug commands, and the server reports every field where the simulator disagrees
with the client. Because Power.log is exact and deterministic, an action without
randomness must leave both states identical — any difference is a simulator bug or
an unimplemented card.

## Build

### 1. TavernGym (Rust → Python extension)

Requires a Rust toolchain matching [`rust-toolchain.toml`](../rust-toolchain.toml)
(rustup installs it on first use) and Python 3.10+.

```powershell
cd <repo root>
python -m pip install maturin      # once
python -m pip install .            # builds crates/taverngym-py and installs taverngym
python -c "import taverngym; print(taverngym.game_version())"
```

> Behind a proxy, cargo/rustup do **not** read the Windows proxy settings. Set
> `$env:HTTPS_PROXY` (and `$env:HTTP_PROXY`) before building, or the crates.io
> download stalls with no output.

### 2. The in-game plugin (C# → HSLibrary2.dll)

The plugin sources are compiled into the existing `hearthstone-helper-plugin`
project; nothing extra to install.

```powershell
cd hearthstone-helper-plugin
.\deploy.ps1            # builds HSLibrary2.dll and installs it (needs an elevated shell)
```

`deploy.ps1` picks the UnityDoorstop proxy that matches `Hearthstone.exe`, copies
the assembly to the path named in `doorstop_config.ini`, and verifies the hashes.
Start the game; `%USERPROFILE%\Documents\tavernplugin.log` and `doorstop-debug.log`
report progress.

The overlay is in the top-right corner and toggles with **F8**.

## Run

**The plugin starts the server itself.** On launch it runs

```
python <game folder>\tavernplugin\server\server.py --port 8765 --parent-pid <Hearthstone pid>
```

as a child process, waits for the port to accept connections, and reports the
result in the overlay (`server: auto-started (pid 1234)`). When the game exits —
or crashes — the server notices its parent is gone and shuts itself down, so no
python process is left behind. If a server is already listening on the port, the
plugin leaves it alone.

Settings live in `%USERPROFILE%\Documents\TavernPlugin\config.json` (written with
defaults on first run):

```json
{
  "auto_start": true,
  "port": 8765,
  "python": "python",
  "server_dir": "",
  "extra_args": []
}
```

`server_dir` empty means "find it": the plugin tries `<dll folder>\tavernplugin\server`,
then `<dll folder>\server`, then `Documents\TavernPlugin\server`. `python` can be a
full path if the interpreter is not on `PATH`. Server output goes to
`Documents\TavernPlugin\server.log`.

To run the bridge by hand instead (useful while developing the server):

```powershell
python tavernplugin\server\server.py            # http://127.0.0.1:8765
```

Then enter a Battlegrounds game. The client finds the newest
`<install>\Logs\Hearthstone_<timestamp>\Power.log` (or `Power_old.log`) by itself
and follows it. If no session directory exists, it tries to enable the `[Power]`
stanza in `%LOCALAPPDATA%\Blizzard\Hearthstone\log.config`; that needs a game
restart to take effect.

### Offline: replay a recorded log

This is how the parser and the server were developed and how you can check a change
without playing:

```powershell
# parse a real log into snapshots
dotnet run -c Release --project tavernplugin\tools\replay -- snapshot `
    "C:\Program Files (x86)\Hearthstone\Logs\Hearthstone_2026_10_31_17_07_27\Power_old.log" `
    --out tavernplugin\tools\replay\out\snaps.jsonl --every 4000

python tavernplugin\tools\inspect_snaps.py tavernplugin\tools\replay\out\snaps.jsonl

# post them to a running server (end-to-end check)
dotnet run -c Release --project tavernplugin\tools\replay -- post `
    tavernplugin\tools\replay\out\snaps.jsonl --limit 5

# exercise the auto-start path without launching the game:
# starts the server, probes /health, then stops it like the game does on quit
dotnet run -c Release --project tavernplugin\tools\replay -- server `
    --server-dir tavernplugin\server --seconds 3
```

`tavernreplay snapshot|summary|post` share the exact parser sources with the game
plugin (`client/Core/*.cs`), so an offline result is a real result.

## API

All responses are JSON with `"ok"` and `"elapsed_ms"`.

| Route | Body | Answer |
| :-- | :-- | :-- |
| `GET /health` | – | `{"ok":true,"taverngym":true,"game_version":"36.6.3.253932","cards":3983,...}` |
| `POST /combat` | snapshot, optional `"n"` (1..5000) | `{"win":…,"tie":…,"loss":…,"avg_damage_taken":…,"lethal_risk":…,"n":…}` |
| `POST /align` | snapshot | `{"aligned":bool,"diffs":[…],"issues":[…],"counts":{…}}` |
| `POST /advise` | snapshot, optional `"n"` | `{"recommendations":[{"kind":"move","text":"swap 1<->3","win":…}]}` |
| `POST /step` | snapshot + action | *(planned: apply one action and diff the result)* |

Snapshot (produced by `Snapshot.ToJson()` in
[client/Core/Snapshot.cs](client/Core/Snapshot.cs)):

```json
{
  "v": 1, "seq": 42, "ts": "2026-10-31 18:40:00.123",
  "log": {"file": "...\\Power.log", "line": 123456},
  "game": {"phase": "combat", "turn": 7, "local_player": 7, "local_player_confident": true},
  "me":   {"player": 7, "hero": "BG32_HERO_001", "health": 30, "armor": 11, "tier": 2,
           "board": [{"entity": 418, "card": "BG36_200", "name": "翩飞蝙蝠",
                      "atk": 3, "hp": 3, "tier": 1, "golden": false,
                      "keywords": {"taunt": true}}]},
  "them": {"player": 15, "board": [ … ]},
  "shop": [{"slot": 0, "card": "BG25_008", "atk": 2, "hp": 2, "cost": 3}]
}
```

## How the state is read

In a match the client reads the game directly
([client/Unity/GameStateSource.cs](client/Unity/GameStateSource.cs)), on the main
thread, four times a second:

| Signal | Used for |
| :-- | :-- |
| `Player.GetBattlefieldZone().GetCards()` | the warband, already in position order |
| `Player.GetHandZone().GetCards()` | the hand |
| `HAS_DRAG_TO_BUY` (2458) on a PLAY entity | a Tavern item (they live in the opposing zone) |
| `BACON_IN_COMBAT_PHASE` (1522) on the GameEntity | recruit vs combat |
| `ATK` / `HEALTH - DAMAGE` / `TECH_LEVEL` / `COST` | numbers |
| `GetCardId()` ending in `_G` | golden |
| `Entity.GetAttachments()` | Dark Gifts and magnetized cards |
| `GameState.GetFriendlyEntityChoices()` | a pending choice (Discover, hero pick, …) |

Outside a match the plugin falls back to
[client/Core/PowerLogParser.cs](client/Core/PowerLogParser.cs), which rebuilds the same
`Snapshot` from the game's log — that is what makes the offline `tavernreplay` possible.
## Limitations (read before trusting a number)

1. **`/advise` still ranks placements only.** The Tavern is readable (see
   [ALIGNMENT](docs/ALIGNMENT.md#7-商店已在真机确认)), but during the recruit phase the
   opposing zone *is* the Tavern, so there is no opponent warband to simulate against.
   Buy decisions need a reference board — last scouted, or supplied by the user.
2. **`them` changes meaning with the phase.** During recruit it is the Tavern; during
   combat it is the opponent's warband. `GameStateSource` keeps them apart (the shop
   items carry `HAS_DRAG_TO_BUY`), but anything reading a raw snapshot must check
   `game.phase` first.
3. **Hero fields are best effort.** A hero skin id (`TB_BaconShop_HERO_*_SKIN_*`) is not
   in TavernGym's card table, so `set_hero` fails and the hero is left unset rather
   than wrong.
4. **The log source is a fallback.** It drops combat copies without a `ZONE` change
   (bounded by `PowerLogParser.PruneLines`) and over-counts Tavern items; it exists for
   offline replay, not for live use.
5. **No hidden information.** The client only reads what the game already shows.
   No memory scanning, no packet sniffing, no reflection into the game assembly.
6. **Windows only**, because deployment and log discovery are Windows-specific. The
   parser and server are portable.

## Files

| Path | Role |
| :-- | :-- |
| `client/Core/PowerLog.cs` | find and tail the newest `Power.log` |
| `client/Core/PowerLogParser.cs` | entity table → `Snapshot` |
| `client/Core/Snapshot.cs` | wire format, `ToJson()`, overlay summary |
| `client/Core/TavernJson.cs` | dependency-free JSON writer/reader (C# 7.3) |
| `client/Core/BridgeClient.cs` | blocking HTTP calls to the server |
| `client/Core/ServerLauncher.cs` | starts/stops the Python bridge with the game |
| `client/Core/GameLog.cs` | the one logging primitive the game's runtime allows |
| `client/Core/NativeProcess.cs` | `CreateProcessW` / `TerminateProcess` (the BCL `Process` is unusable) |
| `client/Core/SelfCheck.cs` | runtime capability probe, run once at startup |
| `client/Core/Text.cs` | ordinal string helpers (the `StringComparison` overloads are stripped) |
| `client/Unity/GameStateSource.cs` | `GameState` -> `Snapshot`: board, hand, Tavern |
| `client/Unity/GameStateDump.cs` | F9 dump (state + pending choice) |
| `client/Unity/TavernPluginModule.cs` | MonoBehaviour: main-thread sampling, worker thread, overlay (F8/F9) |
| `server/server.py` | stdlib HTTP server on 127.0.0.1:8765 |
| `server/bridge.py` | snapshot → TavernGym (`CombatPreview`, alignment, advice) |
| `tools/replay/` | `tavernreplay` console: snapshot/summary/post offline |
| `tools/inspect_snaps.py` | print a table of snapshots |
| `tools/ilspec/` | decompile the game's own types (`ICSharpCode.Decompiler` from ILSpy) |
| `tools/probe_stats.py` | print TavernGym's stat model for one rebuilt minion |
| `docs/ALIGNMENT.md` | how the live board maps onto TavernGym, field by field |

## Where a trained agent goes

`server/bridge.py` is the only place that talks to TavernGym. A learned policy
would replace `advise()` (or add `POST /act`) and reuse the same snapshot, the same
`Game` rebuild and the same `BatchEnv`/`SoloEnv` interface TavernGym already
exposes; nothing in the client changes, because the client only renders text.

## Legal

For personal research on your own games. Automating a live multiplayer game breaks
Blizzard's terms of service; Battlegrounds solo games are the only supported target
and no hidden information is read. See the repository [NOTICE](../NOTICE).





