// The wire format exchanged with the bridge server. Keep in sync with
// tavernplugin/server/bridge.py and the tables in tavernplugin/README.md.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace TavernPlugin
{
    public sealed class MinionInfo
    {
        public int EntityId;
        public string Card = "";
        public string Name = "";
        public int Atk;
        public int Hp;
        public int Tier;
        public bool Golden;
        public bool Taunt;
        public bool DivineShield;
        public bool Poisonous;
        public bool Windfury;
        public bool MegaWindfury;
        public bool Reborn;
        public bool Stealth;
        public bool Cleave;
        public bool Rush;
        public bool Spell;
        public readonly List<string> Attached = new List<string>();

        public bool HasKeywords
        {
            get
            {
                return Taunt || DivineShield || Poisonous || Windfury || MegaWindfury || Reborn || Stealth || Cleave || Rush;
            }
        }

        public void WriteKeywords(JsonWriter w)
        {
            w.Key("keywords").BeginObject();
            if (Taunt) w.Field("taunt", true);
            if (DivineShield) w.Field("divine_shield", true);
            if (Poisonous) w.Field("poisonous", true);
            if (Windfury) w.Field("windfury", true);
            if (MegaWindfury) w.Field("mega_windfury", true);
            if (Reborn) w.Field("reborn", true);
            if (Stealth) w.Field("stealth", true);
            if (Cleave) w.Field("cleave", true);
            if (Rush) w.Field("rush", true);
            w.EndObject();
        }
    }

    // One Tavern slot. During the recruit phase the client parks the shop in the
    // opposing player's PLAY zone and marks every item with HAS_DRAG_TO_BUY, so this
    // is a real entity with a card id, stats and cost — not a UI stub.
    public sealed class ShopSlot
    {
        public int Slot;
        public string Card = "";
        public string Name = "";
        public int Atk = -1;
        public int Hp = -1;
        public int Cost = -1;
        public int Tier = -1;
        public bool Spell;
    }

    public sealed class SideInfo
    {
        public int PlayerId = -1;
        public string Hero = "";
        public int Health = -1;
        public int Armor = -1;
        public int Tier = -1;
        public int Gold = -1;
        public readonly List<MinionInfo> Board = new List<MinionInfo>();
        public readonly List<MinionInfo> Hand = new List<MinionInfo>();

        public void Write(JsonWriter w)
        {
            w.BeginObject();
            if (PlayerId >= 0) w.Field("player", PlayerId);
            w.Field("hero", Hero);
            if (Health >= 0) w.Field("health", Health);
            if (Armor >= 0) w.Field("armor", Armor);
            if (Tier >= 0) w.Field("tier", Tier);
            if (Gold >= 0) w.Field("gold", Gold);
            WriteMinions(w, "board", Board);
            WriteMinions(w, "hand", Hand);
            w.EndObject();
        }

        private static void WriteMinions(JsonWriter w, string name, List<MinionInfo> minions)
        {
            w.Key(name).BeginArray();
            foreach (MinionInfo m in minions)
            {
                w.BeginObject();
                w.Field("entity", m.EntityId);
                w.Field("card", m.Card);
                w.Field("name", m.Name);
                w.Field("atk", m.Atk);
                w.Field("hp", m.Hp);
                if (m.Tier >= 0) w.Field("tier", m.Tier);
                if (m.Golden) w.Field("golden", true);
                if (m.Spell) w.Field("spell", true);
                if (m.HasKeywords) m.WriteKeywords(w);
                if (m.Attached.Count > 0)
                {
                    w.Key("attached").BeginArray();
                    foreach (string a in m.Attached) w.Value(a);
                    w.EndArray();
                }
                w.EndObject();
            }
            w.EndArray();
        }
    }

    // The choice the client is waiting on (Discover, Dark Gift, hero pick, target,
    // Choose One). TavernGym models this as Player::pending; when it is active no
    // other action is legal.
    public sealed class PendingChoice
    {
        public bool Active;
        public int Type = -1;        // CHOICE_TYPE: 0 INVALID, 1 MULLIGAN, 2 GENERAL, 3 TARGET
        public int CountMin = -1;
        public int CountMax = -1;
        public int Source = -1;
        public readonly List<string> Options = new List<string>();

        public void Write(JsonWriter w)
        {
            w.Key("pending").BeginObject();
            w.Field("active", Active);
            if (!Active)
            {
                w.EndObject();
                return;
            }
            if (Type >= 0) w.Field("type", Type);
            if (CountMin >= 0) w.Field("count_min", CountMin);
            if (CountMax >= 0) w.Field("count_max", CountMax);
            if (Source >= 0) w.Field("source", Source);
            w.Key("options").BeginArray();
            foreach (string o in Options) w.Value(o);
            w.EndArray();
            w.EndObject();
        }
    }

    public sealed class Snapshot
    {
        public const int ProtocolVersion = 1;

        public long Seq;
        public string Timestamp = "";
        public string Phase = "unknown";
        public int Turn = -1;
        public int LocalPlayer = -1;
        public bool LocalPlayerConfident;
        public readonly SideInfo Me = new SideInfo();
        public readonly SideInfo Them = new SideInfo();
        public readonly List<ShopSlot> Shop = new List<ShopSlot>();
        public readonly PendingChoice Pending = new PendingChoice();
        public string LogFile = "";
        public long LogLine;
        public string Error = "";

        public string ToJson()
        {
            var w = new JsonWriter();
            w.BeginObject();
            w.Field("v", ProtocolVersion);
            w.Field("seq", (long)Seq);
            w.Field("ts", Timestamp);
            if (!string.IsNullOrEmpty(Error)) w.Field("error", Error);
            w.Key("log").BeginObject();
            w.Field("file", LogFile);
            w.Field("line", LogLine);
            w.EndObject();
            w.Key("game").BeginObject();
            w.Field("phase", Phase);
            if (Turn >= 0) w.Field("turn", Turn);
            if (LocalPlayer >= 0) w.Field("local_player", LocalPlayer);
            w.Field("local_player_confident", LocalPlayerConfident);
            w.EndObject();
            w.Key("me");
            Me.Write(w);
            w.Key("them");
            Them.Write(w);
            w.Key("shop").BeginArray();
            foreach (ShopSlot s in Shop)
            {
                w.BeginObject();
                w.Field("slot", s.Slot);
                if (!string.IsNullOrEmpty(s.Card)) w.Field("card", s.Card);
                if (!string.IsNullOrEmpty(s.Name)) w.Field("name", s.Name);
                if (s.Atk >= 0) w.Field("atk", s.Atk);
                if (s.Hp >= 0) w.Field("hp", s.Hp);
                if (s.Cost >= 0) w.Field("cost", s.Cost);
                if (s.Tier >= 0) w.Field("tier", s.Tier);
                if (s.Spell) w.Field("spell", true);
                w.EndObject();
            }
            w.EndArray();
            Pending.Write(w);
            w.EndObject();
            return w.ToString();
        }

        public string Summary()
        {
            var sb = new StringBuilder();
            sb.Append("turn ").Append(Turn).Append("  ").Append(Phase);
            sb.Append("  me ").Append(Board(Me)).Append(" vs ").Append(Board(Them));
            if (Me.Health >= 0)
            {
                sb.Append("  hp ").Append(Me.Health);
                if (Me.Armor > 0) sb.Append('+').Append(Me.Armor);
            }
            return sb.ToString();
        }

        private static string Board(SideInfo side)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < side.Board.Count; i++)
            {
                if (i > 0) sb.Append(' ');
                MinionInfo m = side.Board[i];
                sb.Append(m.Atk).Append('/').Append(m.Hp);
                if (m.Golden) sb.Append('*');
            }
            return sb.Length == 0 ? "-" : sb.ToString();
        }

        public static string Now()
        {
            return DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture);
        }
    }
}



