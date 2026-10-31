// Parses Hearthstone's Power.log into a Snapshot.
//
// Power.log is the game's own event stream: it is version independent, needs no
// reflection into Assembly-CSharp, and a bug here can never crash the game.
// Only what matters to TavernGym is tracked: the game phase, the turn, both
// warbands, hero health/armor/tier and the Tavern slots.
//
// The shapes understood here were verified against a real client log:
//
//   D 17:39:20.077 GameState.DebugPrintPower() - CREATE_GAME
//   D ... GameState.DebugPrintPower() -     GameEntity EntityID=19
//   D ... GameState.DebugPrintPower() -         tag=BACON_IN_COMBAT_PHASE value=1
//   D ... GameState.DebugPrintPower() - TAG_CHANGE Entity=[entityName=... id=418
//        zone=PLAY zonePos=1 cardId=BG36_200 player=15] tag=ATK value=3
//
// A minion is a PLAY entity with ZONE_POSITION >= 1 that carries ATK and HEALTH:
// the shop/UI actors also have ATK but sit at position 0.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace TavernPlugin
{
    public sealed class LogEntity
    {
        public int Id;
        public string Card = "";
        public string Name = "";
        public string Zone = "INVALID";
        public int ZonePos;
        public int Controller;
        // log line that last mentioned this entity; the Battlegrounds client drops
        // its combat copies without a ZONE change, so staleness is the only signal
        public long LastLine;
        public readonly Dictionary<string, string> Tags = new Dictionary<string, string>(StringComparer.Ordinal);

        public int IntTag(string name, int fallback = int.MinValue)
        {
            string v;
            if (!Tags.TryGetValue(name, out v)) return fallback;
            int parsed;
            return int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed) ? parsed : fallback;
        }

        public bool Flag(string name)
        {
            return IntTag(name, 0) > 0;
        }

        public void Set(string name, string value)
        {
            switch (name)
            {
                case "ZONE": Zone = value; break;
                case "ZONE_POSITION": int pos; ZonePos = int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out pos) ? pos : 0; break;
                case "CONTROLLER": int c; Controller = int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out c) ? c : 0; break;
            }
            Tags[name] = value;
        }
    }

    public sealed class PowerLogParser
    {
        private static readonly Regex PrefixRx = new Regex(
            @"\d\d:\d\d:\d\d\.\d+ (?:GameState|PowerTaskList)\.DebugPrintPower\(\) - ?", RegexOptions.Compiled);

        private static readonly Regex FullCreateRx = new Regex(
            @"^FULL_ENTITY - Creating ID=(?<id>\d+)(?: CardID=(?<card>\S*))?", RegexOptions.Compiled);

        // "SHOW_ENTITY - Updating Entity=6561 CardID=BG28_300" and its siblings name
        // the entity by id; the debug view uses the "[entityName=...]" form instead.
        private static readonly Regex EntityOpRx = new Regex(
            @"^(?:FULL_ENTITY|SHOW_ENTITY|CHANGE_ENTITY|HIDE_ENTITY) - (?:\w+ )?Entity=(?<id>\d+)(?: CardID=(?<card>\S+))?(?: tag=(?<tag>\w+) value=(?<val>\S+))?",
            RegexOptions.Compiled);

        private static readonly Regex EntityHeaderRx = new Regex(
            @"Entity=\[entityName=(?<name>.*?) id=(?<id>\d+) zone=(?<zone>\w+)(?: zonePos=(?<pos>\d+))? cardId=(?<card>\w*) player=(?<player>\d+)\]",
            RegexOptions.Compiled);

        private static readonly Regex TagRx = new Regex(@"tag=(?<tag>[A-Za-z_0-9]+) value=(?<val>\S+)", RegexOptions.Compiled);

        // most TAG_CHANGE lines name the entity by id only ("Entity=418"), the
        // descriptive "[entityName=... zone=...]" form appears in the debug view
        private static readonly Regex NumericEntityRx = new Regex(@"Entity=(?<id>\d+)", RegexOptions.Compiled);

        private readonly Dictionary<int, LogEntity> _entities = new Dictionary<int, LogEntity>();
        private readonly Dictionary<string, string> _gameTags = new Dictionary<string, string>(StringComparer.Ordinal);
        private LogEntity _open;
        private int _localPlayer = -1;
        private bool _localConfident;
        private long _line;
        private bool _sawGame;
        private int _gameCount;

        public string LogFile = "";
        public int Games => _gameCount;

        // An entity that has not been mentioned for this many log lines is no longer
        // on the board. The Battlegrounds client discards its combat copies without a
        // ZONE change, so this idle window is what bounds a warband; ~2 combats.
        public int PruneLines = 12000;

        public void Reset()
        {
            _entities.Clear();
            _gameTags.Clear();
            _open = null;
            _sawGame = false;
            _localConfident = false;
        }

        private LogEntity Touch(int id)
        {
            LogEntity e;
            if (!_entities.TryGetValue(id, out e))
            {
                e = new LogEntity { Id = id };
                _entities[id] = e;
            }
            e.LastLine = _line;
            return e;
        }

        private void Prune()
        {
            long cutoff = _line - PruneLines;
            if (cutoff <= 0) return;
            List<int> dead = null;
            foreach (KeyValuePair<int, LogEntity> pair in _entities)
            {
                if (pair.Value.LastLine >= cutoff) continue;
                if (pair.Value.Zone != "PLAY") continue;   // keep small, off-board records cheaply
                if (dead == null) dead = new List<int>();
                dead.Add(pair.Key);
            }
            if (dead == null) return;
            for (int i = 0; i < dead.Count; i++) _entities.Remove(dead[i]);
        }

        // Feed one raw log line (with or without the game's prefix).
        public void Feed(string rawLine)
        {
            _line++;
            if (rawLine == null) return;
            string s = rawLine;
            int at = Text.IndexOf(s, "DebugPrintPower()");
            if (at < 0) return;                      // not a power line
            int dash = Text.IndexOf(s, " - ", at);
            if (dash < 0) return;
            s = s.Substring(dash + 3).Trim();
            if (s.Length == 0) return;

            if (Text.StartsWith(s, "CREATE_GAME"))
            {
                Reset();
                _sawGame = true;
                _gameCount++;
                return;
            }
            if (!_sawGame) return;

            Match m;
            if (Text.StartsWith(s, "FULL_ENTITY - Creating"))
            {
                m = FullCreateRx.Match(s);
                if (!m.Success) return;
                _open = Touch(int.Parse(m.Groups["id"].Value, CultureInfo.InvariantCulture));
                if (m.Groups["card"].Success) _open.Card = m.Groups["card"].Value;
                return;
            }
            if (Text.StartsWith(s, "FULL_ENTITY - Updating")
                || Text.StartsWith(s, "SHOW_ENTITY - Updating")
                || Text.StartsWith(s, "CHANGE_ENTITY - Updating")
                || Text.StartsWith(s, "HIDE_ENTITY"))
            {
                Match h = EntityHeaderRx.Match(s);
                if (h.Success)
                {
                    ApplyHeader(h);
                    _open = Touch(int.Parse(h.Groups["id"].Value, CultureInfo.InvariantCulture));
                    return;
                }
                // the common form: "SHOW_ENTITY - Updating Entity=6561 CardID=BG28_300"
                m = EntityOpRx.Match(s);
                if (!m.Success) return;
                _open = Touch(int.Parse(m.Groups["id"].Value, CultureInfo.InvariantCulture));
                if (m.Groups["card"].Success) _open.Card = m.Groups["card"].Value;
                if (m.Groups["tag"].Success) _open.Set(m.Groups["tag"].Value, m.Groups["val"].Value);
                if (Text.StartsWith(s, "HIDE_ENTITY")) _open = null;
                return;
            }
            if (Text.StartsWith(s, "tag="))
            {
                if (_open == null) return;
                m = TagRx.Match(s);
                if (m.Success) _open.Set(m.Groups["tag"].Value, m.Groups["val"].Value);
                return;
            }
            if (Text.StartsWith(s, "TAG_CHANGE"))
            {
                _open = null;   // a tag line after this belongs to no entity block
                m = TagRx.Match(s);
                if (!m.Success) return;
                string tag = m.Groups["tag"].Value;
                string val = m.Groups["val"].Value;
                if (s.Contains("Entity=GameEntity"))
                {
                    _gameTags[tag] = val;
                    return;
                }
                Match h = EntityHeaderRx.Match(s);
                int id;
                if (h.Success)
                {
                    ApplyHeader(h);
                    id = int.Parse(h.Groups["id"].Value, CultureInfo.InvariantCulture);
                }
                else
                {
                    Match n = NumericEntityRx.Match(s);
                    if (!n.Success) return;
                    id = int.Parse(n.Groups["id"].Value, CultureInfo.InvariantCulture);
                }
                Touch(id).Set(tag, val);
            }
        }

        private void ApplyHeader(Match h)
        {
            int id = int.Parse(h.Groups["id"].Value, CultureInfo.InvariantCulture);
            LogEntity e = Touch(id);
            e.Name = h.Groups["name"].Value;
            e.Zone = h.Groups["zone"].Value;
            string pos = h.Groups["pos"].Value;
            e.ZonePos = pos.Length > 0 ? int.Parse(pos, CultureInfo.InvariantCulture) : 0;
            e.Controller = int.Parse(h.Groups["player"].Value, CultureInfo.InvariantCulture);
            string card = h.Groups["card"].Value;
            if (card.Length > 0) e.Card = card;
            // The local player is the one holding cards; remember it for the phases
            // where the hand is empty (combat).
            if (e.Zone == "HAND" && e.Controller > 0)
            {
                _localPlayer = e.Controller;
                _localConfident = true;
            }
        }

        private string GameTag(string name, string fallback = "")
        {
            string v;
            return _gameTags.TryGetValue(name, out v) ? v : fallback;
        }

        private int GameTagInt(string name, int fallback = -1)
        {
            string v = GameTag(name);
            int parsed;
            return int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out parsed) ? parsed : fallback;
        }

        public Snapshot Build()
        {
            Prune();
            var snap = new Snapshot
            {
                Timestamp = Snapshot.Now(),
                LogFile = LogFile,
                LogLine = _line,
                Phase = "unknown",
                Turn = GameTagInt("TURN"),
                LocalPlayer = _localPlayer,
                LocalPlayerConfident = _localConfident,
            };

            int inCombat = GameTagInt("BACON_IN_COMBAT_PHASE");
            bool bacongame = _gameTags.ContainsKey("BACON_BARTENDER_CARD_ID")
                             || _gameTags.ContainsKey("BACON_IN_COMBAT_PHASE")
                             || _gameTags.ContainsKey("BACON_TRINKETS_ACTIVE");

            // heroes, one per player id
            var heroes = new Dictionary<int, LogEntity>();
            var boards = new Dictionary<int, List<LogEntity>>();
            var shops = new List<LogEntity>();

            foreach (LogEntity e in _entities.Values)
            {
                if (e.Zone == "PLAY" && e.Tags.ContainsKey("PLAYER_ID"))
                {
                    int pid = e.IntTag("PLAYER_ID");
                    if (e.Card.Length > 0 && !heroes.ContainsKey(pid)) heroes[pid] = e;
                }
            }

            foreach (LogEntity e in _entities.Values)
            {
                if (e.Zone != "PLAY") continue;
                bool hasStats = e.Tags.ContainsKey("ATK") && e.Tags.ContainsKey("HEALTH");
                if (!hasStats) continue;
                if (e.Tags.ContainsKey("PLAYER_ID")) continue;   // the hero itself

                // The Tavern: parked in the opposing player's PLAY zone during the
                // recruit phase, every item marked with HAS_DRAG_TO_BUY (2458).
                if (e.Flag("HAS_DRAG_TO_BUY"))
                {
                    shops.Add(e);
                    continue;
                }

                if (e.ZonePos < 1) continue;

                int ctrl = e.IntTag("CONTROLLER", e.Controller);
                if (ctrl <= 0) continue;
                List<LogEntity> list;
                if (!boards.TryGetValue(ctrl, out list))
                {
                    list = new List<LogEntity>();
                    boards[ctrl] = list;
                }
                list.Add(e);
            }

            // The local player: known from the hand, else the seat that owns a board
            // while no other seat does (recruit phase), else unknown.
            if (_localPlayer < 0)
            {
                foreach (int pid in heroes.Keys)
                {
                    _localPlayer = pid;
                    break;
                }
            }

            int opponent = -1;
            foreach (int pid in boards.Keys)
            {
                if (pid != _localPlayer) { opponent = pid; break; }
            }

            if (bacongame)
            {
                if (inCombat == 1) snap.Phase = "combat";
                else if (heroes.Count > 0) snap.Phase = "recruit";
            }

            FillSide(snap.Me, _localPlayer, heroes, boards);
            if (opponent > 0) FillSide(snap.Them, opponent, heroes, boards);

            shops.Sort(delegate (LogEntity a, LogEntity b) { return a.ZonePos.CompareTo(b.ZonePos); });
            for (int i = 0; i < shops.Count; i++)
            {
                LogEntity e = shops[i];
                var slot = new ShopSlot
                {
                    Slot = e.ZonePos > 0 ? e.ZonePos - 1 : i,
                    Card = e.Card,
                    Name = e.Name,
                    Atk = e.IntTag("ATK"),
                    Hp = e.IntTag("HEALTH"),
                    Cost = e.IntTag("COST"),
                    Tier = e.IntTag("TECH_LEVEL", -1),
                    Spell = e.IntTag("CARDTYPE", 0) == 42,
                };
                snap.Shop.Add(slot);
            }

            return snap;
        }

        private static void FillSide(SideInfo side, int pid, Dictionary<int, LogEntity> heroes, Dictionary<int, List<LogEntity>> boards)
        {
            side.PlayerId = pid;
            LogEntity hero;
            if (pid > 0 && heroes.TryGetValue(pid, out hero))
            {
                side.Hero = hero.Card;
                side.Health = hero.IntTag("HEALTH", -1);
                side.Armor = hero.IntTag("ARMOR", -1);
                side.Tier = hero.IntTag("PLAYER_TECH_LEVEL", hero.IntTag("TECH_LEVEL", -1));
            }
            List<LogEntity> board;
            if (pid > 0 && boards.TryGetValue(pid, out board))
            {
                board.Sort(delegate (LogEntity a, LogEntity b) { return a.ZonePos.CompareTo(b.ZonePos); });
                foreach (LogEntity e in board) side.Board.Add(ToMinion(e));
            }
        }

        private static MinionInfo ToMinion(LogEntity e)
        {
            var m = new MinionInfo
            {
                EntityId = e.Id,
                Card = e.Card,
                Name = e.Name,
                Atk = e.IntTag("ATK"),
                Hp = e.IntTag("HEALTH"),
                Tier = e.IntTag("TECH_LEVEL", -1),
                Golden = Text.EndsWith(e.Card, "_G"),
                Taunt = e.Flag("TAUNT"),
                DivineShield = e.Flag("DIVINE_SHIELD"),
                Poisonous = e.Flag("POISONOUS"),
                Windfury = e.Flag("WINDFURY"),
                MegaWindfury = e.Flag("MEGA_WINDFURY"),
                Reborn = e.Flag("REBORN"),
                Stealth = e.Flag("STEALTH"),
                Cleave = e.Flag("CLEAVE"),
                Rush = e.Flag("RUSH"),
            };
            return m;
        }
    }
}


