// F9 diagnostic dump.
//
// Writes the game's own view of the match next to a screenshot, so a snapshot can
// be checked field by field when something does not line up. The payload that the
// server actually sees is saved separately as `<name>.snapshot.json`.
//
// Output: %USERPROFILE%\Documents\TavernPlugin\dumps\dump_<timestamp>.json
//   summary   phase, turn, local player
//   me/them   hero, health, armor, tier, gold, board (ordered), hand
//   entities  id, cardId, zone, zonePos, controller and a fixed tag set
//   choice    the pending choice, its options and source

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using TavernPlugin;

namespace HSLibrary
{
    public static class GameStateDump
    {
        private static readonly GAME_TAG[] Tags =
        {
            GAME_TAG.ZONE, GAME_TAG.ZONE_POSITION, GAME_TAG.CONTROLLER, GAME_TAG.CARDTYPE,
            GAME_TAG.ATK, GAME_TAG.HEALTH, GAME_TAG.DAMAGE, GAME_TAG.ARMOR,
            GAME_TAG.TECH_LEVEL, GAME_TAG.PLAYER_TECH_LEVEL, GAME_TAG.COST, GAME_TAG.RESOURCES,
            GAME_TAG.TAUNT, GAME_TAG.DIVINE_SHIELD, GAME_TAG.POISONOUS, GAME_TAG.WINDFURY,
            GAME_TAG.MEGA_WINDFURY, GAME_TAG.REBORN, GAME_TAG.STEALTH, GAME_TAG.RUSH,
            GAME_TAG.EXHAUSTED, GAME_TAG.PLAYER_ID, GAME_TAG.NUM_TURNS_IN_PLAY,
            GAME_TAG.IS_BACON_POOL_MINION, GAME_TAG.HAS_DRAG_TO_BUY,
        };

        public static string LastPath = "";

        public static bool Capture(out string error)
        {
            error = null;
            try
            {
                GameState state = GameState.Get();
                if (state == null)
                {
                    error = "GameState.Get() returned null (not in a match)";
                    return false;
                }

                var w = new JsonWriter();
                w.BeginObject();
                w.Field("captured", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff", CultureInfo.InvariantCulture));

                GameEntity game = state.GetGameEntity();
                Player me = state.GetLocalSidePlayer();
                if (me == null) me = state.GetFriendlySidePlayer();

                w.Key("summary").BeginObject();
                if (me != null) w.Field("local_player_id", me.GetPlayerId());
                if (game != null)
                {
                    w.Field("turn", game.GetTag(GAME_TAG.TURN));
                    w.Field("in_combat", game.GetTag(GAME_TAG.BACON_IN_COMBAT_PHASE));
                }
                w.EndObject();

                WritePlayer(w, "me", me);
                WritePlayer(w, "them", me == null ? null : state.GetFirstOpponentPlayer(me));

                w.Key("entities").BeginArray();
                for (int id = 1; id <= 12000; id++)
                {
                    Entity e = state.GetEntity(id);
                    if (e != null) WriteEntity(w, e);
                }
                w.EndArray();

                WriteChoice(w, state);

                w.EndObject();

                string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                                          "TavernPlugin", "dumps");
                Directory.CreateDirectory(dir);
                string path = Path.Combine(dir, "dump_"
                    + DateTime.Now.ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture) + ".json");
                using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
                using (var writer = new StreamWriter(stream, new UTF8Encoding(false)))
                {
                    writer.Write(w.ToString());
                }
                LastPath = path;
                return true;
            }
            catch (Exception exc)
            {
                error = exc.GetType().Name + ": " + exc.Message;
                LastPath = "";
                return false;
            }
        }

        private static void WritePlayer(JsonWriter w, string name, Player p)
        {
            w.Key(name);
            if (p == null)
            {
                w.Null();
                return;
            }
            w.BeginObject();
            w.Field("player_id", p.GetPlayerId());

            Entity hero = p.GetHero();
            if (hero != null)
            {
                w.Field("hero_card", hero.GetCardId());
                w.Field("hero_health", hero.GetTag(GAME_TAG.HEALTH));
                w.Field("hero_armor", hero.GetTag(GAME_TAG.ARMOR));
                w.Field("tavern_tier", hero.GetTag(GAME_TAG.PLAYER_TECH_LEVEL));
            }
            w.Field("gold", p.GetNumAvailableResources());
            WriteZone(w, "board", p.GetBattlefieldZone());
            WriteZone(w, "hand", p.GetHandZone());
            w.EndObject();
        }

        private static void WriteZone(JsonWriter w, string name, Zone zone)
        {
            w.Key(name).BeginArray();
            if (zone != null)
            {
                foreach (Card card in zone.GetCards())
                {
                    Entity e = card == null ? null : card.GetEntity();
                    if (e == null) continue;
                    w.BeginObject();
                    w.Field("card", e.GetCardId());
                    w.Field("id", e.GetEntityId());
                    w.Field("zone_pos", e.GetZonePosition());
                    w.Field("atk", e.GetTag(GAME_TAG.ATK));
                    w.Field("hp", e.GetTag(GAME_TAG.HEALTH));
                    w.EndObject();
                }
            }
            w.EndArray();
        }

        private static void WriteEntity(JsonWriter w, Entity e)
        {
            w.BeginObject();
            w.Field("id", e.GetEntityId());
            w.Field("card", e.GetCardId());
            w.Field("zone", (int)e.GetZone());
            w.Field("zone_pos", e.GetZonePosition());
            w.Field("controller", e.GetControllerId());
            w.Key("tags").BeginObject();
            foreach (GAME_TAG tag in Tags)
            {
                int value = e.GetTag(tag);
                if (value != 0) w.Field(((int)tag).ToString(CultureInfo.InvariantCulture), value);
            }
            w.EndObject();
            w.EndObject();
        }

        // A pending choice (Discover, Dark Gift, hero pick, target, Choose One).
        // TavernGym models these as Player::pending, so the options are what the
        // "did this action align?" check needs.
        private static void WriteChoice(JsonWriter w, GameState state)
        {
            w.Key("choice").BeginObject();
            Network.EntityChoices choice = null;
            try
            {
                choice = state.GetFriendlyEntityChoices();
            }
            catch (Exception exc)
            {
                // a dump without the pending choice is still worth having
                GameLog.Write("[dump] choice read failed: " + exc.GetType().Name);
            }
            if (choice == null)
            {
                w.Field("active", false);
                w.EndObject();
                return;
            }

            w.Field("active", true);
            w.Field("choice_type", (int)choice.ChoiceType);
            w.Field("count_min", choice.CountMin);
            w.Field("count_max", choice.CountMax);
            w.Field("source", choice.Source);
            w.Key("options").BeginArray();
            foreach (int optionId in choice.Entities)
            {
                Entity option = state.GetEntity(optionId);
                w.BeginObject();
                w.Field("id", optionId);
                if (option != null)
                {
                    w.Field("card", option.GetCardId() ?? "");
                    w.Field("atk", option.GetTag(GAME_TAG.ATK));
                    w.Field("hp", option.GetTag(GAME_TAG.HEALTH));
                    w.Field("tier", option.GetTag(GAME_TAG.TECH_LEVEL));
                }
                w.EndObject();
            }
            w.EndArray();
            w.EndObject();
        }
    }
}
