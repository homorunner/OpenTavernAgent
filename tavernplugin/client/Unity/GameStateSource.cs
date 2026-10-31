// The live data source: the game's own GameState, sampled on the main thread.
//
// A warband minion is a PLAY entity at ZONE_POSITION 1..7. In the recruit phase the
// client parks the Tavern in the *opposing* player's PLAY zone, marked with
// GAME_TAG.HAS_DRAG_TO_BUY; a combat phase puts the real opponent there instead.
// Golden minions are separate card ids ending in "_G".
//
// Guards: `Capture` has one catch and it is the whole safety net — if the game
// changes a signature, the capture fails visibly ("state: ..." in the overlay)
// instead of silently yielding half a snapshot. Only the two optional sections
// (attachments, pending choice) get their own catch, because losing those should
// not cost the warband.

using System;
using System.Collections.Generic;
using TavernPlugin;

namespace HSLibrary
{
    public static class GameStateSource
    {
        // Main thread only; never throws.
        public static bool Capture(out Snapshot snapshot, out string error)
        {
            snapshot = null;
            error = null;
            try
            {
                GameState state = GameState.Get();
                if (state == null)
                {
                    error = "no GameState (not in a match)";
                    return false;
                }
                Player me = state.GetLocalSidePlayer();
                if (me == null) me = state.GetFriendlySidePlayer();
                if (me == null)
                {
                    error = "no local player";
                    return false;
                }

                var snap = new Snapshot
                {
                    Timestamp = Snapshot.Now(),
                    LocalPlayer = me.GetPlayerId(),
                    LocalPlayerConfident = true,
                    Phase = "recruit",
                };

                GameEntity game = state.GetGameEntity();
                if (game != null)
                {
                    snap.Turn = game.GetTag(GAME_TAG.TURN);
                    if (game.GetTag(GAME_TAG.BACON_IN_COMBAT_PHASE) == 1) snap.Phase = "combat";
                }

                // The opponent is optional: losing it should cost the Tavern, not the
                // whole snapshot (hero selection has no opponent yet).
                Player them = null;
                try { them = state.GetFirstOpponentPlayer(me); }
                catch (Exception) { }

                FillSide(snap.Me, me);
                if (them != null) FillSide(snap.Them, them);
                CollectShop(snap, them);
                CollectPending(snap, state);

                snapshot = snap;
                return true;
            }
            catch (Exception exc)
            {
                error = exc.GetType().Name + ": " + exc.Message;
                GameLog.Write("[game] capture failed: " + error);
                return false;
            }
        }

        private static void FillSide(SideInfo side, Player player)
        {
            side.PlayerId = player.GetPlayerId();

            Entity hero = player.GetHero();
            if (hero != null)
            {
                side.Hero = hero.GetCardId() ?? "";
                side.Health = hero.GetTag(GAME_TAG.HEALTH);
                side.Armor = hero.GetTag(GAME_TAG.ARMOR);
                side.Tier = hero.GetTag(GAME_TAG.PLAYER_TECH_LEVEL);
            }
            side.Gold = player.GetNumAvailableResources();

            CollectBoard(side, player.GetBattlefieldZone());
            CollectHand(side, player.GetHandZone());
        }

        private static void CollectBoard(SideInfo side, Zone zone)
        {
            if (zone == null) return;
            foreach (Card card in zone.GetCards())
            {
                Entity e = card == null ? null : card.GetEntity();
                if (e == null) continue;
                if (e.HasTag(GAME_TAG.HAS_DRAG_TO_BUY)) continue;   // that is the Tavern
                if (e.GetTag(GAME_TAG.ZONE_POSITION) < 1) continue;
                side.Board.Add(ToMinion(e));
            }
        }

        private static void CollectHand(SideInfo side, Zone zone)
        {
            if (zone == null) return;
            foreach (Card card in zone.GetCards())
            {
                Entity e = card == null ? null : card.GetEntity();
                if (e != null) side.Hand.Add(ToMinion(e));
            }
        }

        // The Tavern: the opposing side's PLAY zone, filtered by HAS_DRAG_TO_BUY.
        private static void CollectShop(Snapshot snap, Player them)
        {
            if (them == null) return;
            Zone zone = them.GetBattlefieldZone();
            if (zone == null) return;

            int slot = 0;
            foreach (Card card in zone.GetCards())
            {
                Entity e = card == null ? null : card.GetEntity();
                if (e == null) continue;
                if (!e.HasTag(GAME_TAG.HAS_DRAG_TO_BUY)) continue;

                bool spell = e.GetTag(GAME_TAG.CARDTYPE) == 42;      // TAVERN_SPELL
                int cost = e.GetTag(GAME_TAG.COST);
                // Minions carry no COST in the Tavern (only the slot actor and tavern
                // spells do), and a plain minion always costs 3.
                if (cost <= 0 && !spell) cost = 3;

                snap.Shop.Add(new TavernPlugin.ShopSlot
                {
                    Slot = slot,
                    Card = e.GetCardId() ?? "",
                    Atk = e.GetTag(GAME_TAG.ATK),
                    Hp = e.GetTag(GAME_TAG.HEALTH),
                    Tier = e.GetTag(GAME_TAG.TECH_LEVEL),
                    Cost = cost,
                    Spell = spell,
                });
                slot++;
            }
        }

        private static MinionInfo ToMinion(Entity e)
        {
            string card = e.GetCardId() ?? "";
            var m = new MinionInfo
            {
                EntityId = e.GetEntityId(),
                Card = card,
                Name = e.GetName() ?? "",
                Atk = e.GetTag(GAME_TAG.ATK),
                Hp = e.GetTag(GAME_TAG.HEALTH) - e.GetTag(GAME_TAG.DAMAGE),
                Tier = e.GetTag(GAME_TAG.TECH_LEVEL),
                Golden = Text.EndsWith(card, "_G"),
                Taunt = e.HasTag(GAME_TAG.TAUNT),
                DivineShield = e.HasTag(GAME_TAG.DIVINE_SHIELD),
                Poisonous = e.HasTag(GAME_TAG.POISONOUS),
                Windfury = e.HasTag(GAME_TAG.WINDFURY),
                MegaWindfury = e.HasTag(GAME_TAG.MEGA_WINDFURY),
                Reborn = e.HasTag(GAME_TAG.REBORN),
                Stealth = e.HasTag(GAME_TAG.STEALTH),
                Rush = e.HasTag(GAME_TAG.RUSH),
                // CARDTYPE 5 = SPELL; hand entries can be spells instead of minions
                Spell = e.GetTag(GAME_TAG.CARDTYPE) == 5 || e.GetTag(GAME_TAG.CARDTYPE) == 42,
            };
            CollectAttachments(m, e);
            return m;
        }

        // Dark Gifts and magnetized cards. The game exposes them as attachments whose
        // card id carries the enchantment's "e" suffix (BG36_MidGameEffect_000t75e);
        // TavernGym keys its scripts on the base id, so the suffix is dropped when the
        // attachment is an enchantment (CARDTYPE 6).
        private static void CollectAttachments(MinionInfo m, Entity e)
        {
            try
            {
                List<Entity> attachments = e.GetAttachments();
                if (attachments == null) return;
                foreach (Entity a in attachments)
                {
                    if (a == null) continue;
                    string id = a.GetCardId();
                    if (string.IsNullOrEmpty(id)) continue;
                    if (a.GetTag(GAME_TAG.CARDTYPE) == 6 && id.Length > 1 && Text.EndsWith(id, "e"))
                        id = id.Substring(0, id.Length - 1);
                    m.Attached.Add(id);
                }
            }
            catch (Exception exc)
            {
                GameLog.Write("[game] attachments read failed: " + exc.GetType().Name);
            }
        }

            private static void CollectPending(Snapshot snap, GameState state)
        {
            try
            {
                Network.EntityChoices choice = state.GetFriendlyEntityChoices();
                if (choice == null) return;

                snap.Pending.Active = true;
                snap.Pending.Type = (int)choice.ChoiceType;
                snap.Pending.CountMin = choice.CountMin;
                snap.Pending.CountMax = choice.CountMax;
                snap.Pending.Source = choice.Source;
                foreach (int optionId in choice.Entities)
                {
                    Entity option = state.GetEntity(optionId);
                    if (option == null) continue;
                    string id = option.GetCardId();
                    if (!string.IsNullOrEmpty(id)) snap.Pending.Options.Add(id);
                }
            }
            catch (Exception exc)
            {
                GameLog.Write("[game] pending choice read failed: " + exc.GetType().Name);
            }
        }
    }
}

