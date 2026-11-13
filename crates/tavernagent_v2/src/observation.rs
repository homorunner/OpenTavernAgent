//! Compact v2 observations and action constraints; engine/v1 schemas are unchanged.
use crate::rules::{ActionHistory, PolicyRules};
use taverngym::actions::{A_CANCEL, A_END as A_END_TURN};
use taverngym::{Game, obs::observe};

pub const MUST_SPEND_GOLD: i32 = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub x: Vec<f32>,
    pub cards: Vec<i32>,
    pub mask: Vec<bool>,
}

pub fn observe_seat(game: &Game, seat: usize) -> Observation {
    observe_seat_with_rules(
        game,
        seat,
        PolicyRules::default(),
        &ActionHistory::default(),
    )
}

pub fn observe_seat_with_rules(
    game: &Game,
    seat: usize,
    rules: PolicyRules,
    history: &ActionHistory,
) -> Observation {
    let mut o = observe(game, seat);
    // Remove the engine's eighth shop slot before remapping the compact card offsets.
    let unused = crate::net::SHOP_FLOAT_START + crate::actions::N_SHOP * taverngym::obs::ITEM_F;
    o.floats.drain(unused..unused + taverngym::obs::ITEM_F);
    o.cards
        .remove(crate::net::SHOP_CARD_START + crate::actions::N_SHOP);
    map_owned_effects(&game.players[seat], &mut o.cards);
    // Combat fixes cap eligibility at recruit start, even if a player dies mid-turn.
    let alive = match game.turn_alive {
        0 => game.alive_count(),
        n => n,
    };
    let player = &game.players[seat];
    let protected = player.alive
        && game
            .cfg
            .damage_cap(game.turn, alive)
            .is_some_and(|cap| player.health.saturating_add(player.armor) > cap);
    // Agent-only extensions leave the engine and v1 observation layout unchanged.
    o.floats.reserve_exact(1 + crate::actions::N_SHOP);
    o.floats.push(protected as u8 as f32);
    for slot in 0..crate::actions::N_SHOP {
        o.floats.push(
            game.slot_price(seat, slot)
                .is_some_and(|(_, health, _)| health) as u8 as f32,
        );
    }
    Observation {
        x: o.floats,
        cards: o.cards,
        mask: agent_mask_with_rules(game, seat, rules, history),
    }
}

/// Compact acquisition sources without moving the engine's actual effects. Orb shells
/// have already fired; only their resulting trinkets occupy a slot. Pending choices are
/// deliberately untouched, including Orb options that have not been bought yet.
fn map_owned_effects(player: &taverngym::state::Player, cards: &mut [i32]) {
    use crate::net::{OWNED_TRINKET_SLOTS, TRINKET_SLOT_START};
    use taverngym::state::TrinketSource;

    let is_orb = |card: taverngym::cards::CardId| {
        card.base() == taverngym::cardid!("BG35_MagicItem_816")
            || card.base() == taverngym::cardid!("BG35_MagicItem_816t")
    };
    let reward_slot = 1 + taverngym::actions::N_POWERS + 2 + 2;
    let reward = player.rewards.last().copied();
    // Ethereal Evidence appends its current pick after the permanent reward shell.
    cards[reward_slot] = reward.map_or(0, |card| card.idx() as i32 + 1);
    cards[TRINKET_SLOT_START..TRINKET_SLOT_START + 8].fill(0);
    for (slot, power) in player
        .powers
        .iter()
        .take(taverngym::actions::N_POWERS)
        .enumerate()
    {
        if is_orb(power.card) {
            cards[1 + slot] = 0;
        }
    }
    let mut ordinary = 0;
    for &trinket in &player.trinkets {
        if is_orb(trinket) {
            continue;
        }
        let source = player.trinket_source(trinket);
        let slot = match source {
            Some(TrinketSource::HeroPower(slot)) => {
                (usize::from(slot) < taverngym::actions::N_POWERS).then_some(1 + usize::from(slot))
            }
            Some(TrinketSource::QuestReward(reward)) => {
                // Only the last owned reward is observed. Hidden rewards must not
                // spill their trinkets into the ordinary owned-trinket slots.
                if player.rewards.last() != Some(&reward) {
                    continue;
                }
                Some(reward_slot)
            }
            None => None,
        };
        let slot = slot.or_else(|| {
            if ordinary == OWNED_TRINKET_SLOTS {
                return None;
            }
            let slot = TRINKET_SLOT_START + ordinary;
            ordinary += 1;
            Some(slot)
        });
        if let Some(slot) = slot {
            cards[slot] = trinket.idx() as i32 + 1;
        }
    }
}

/// The seat's legal actions over the agent's action space, with two training rules applied.
///
/// Play uses seven positions per hand card and automatically magnetizes onto
/// compatible hosts. Move is absent; see `crate::actions` for engine translation.
///
/// Must-spend: End Turn is illegal for a seat holding `MUST_SPEND_GOLD` Gold or more,
/// whatever the shop currently offers. A free Refresh is exactly the case that matters —
/// nothing is affordable now, a Refresh may turn up something that is — so the rule must not
/// try to predict whether Gold *can* be spent. It only ever removes End Turn, and never
/// leaves the mask empty: past the action cap the engine returns End Turn alone, and that
/// one bit is restored.
///
/// No Cancel: a menu the agent opened (a spell's or Hero Power's target, a Choose One) must
/// be resolved by picking an option. The engine only opens a target menu when a target
/// exists, so an option is always there; Cancel is restored only when it is the sole legal
/// action (a paid trinket offer with nothing in it affordable). Without this the policy can repeat
/// open-menu / close-menu as a free no-op pair and spends the turn's action budget on it.
pub fn agent_mask(game: &Game, seat: usize) -> Vec<bool> {
    agent_mask_with_rules(
        game,
        seat,
        PolicyRules::default(),
        &ActionHistory::default(),
    )
}

pub fn agent_mask_with_rules(
    game: &Game,
    seat: usize,
    rules: PolicyRules,
    history: &ActionHistory,
) -> Vec<bool> {
    let mut mask = crate::actions::legal_mask(game, seat);
    let can_freeze = mask[taverngym::actions::A_FREEZE];
    if rules.freeze_at_end {
        // Remove the toggle before must-spend escape detection. Otherwise a free
        // toggle could keep End masked when no productive action remains.
        mask[taverngym::actions::A_FREEZE] = false;
    }
    // Apply before the EndTurn/Cancel escape rules, so blocking a free action cannot
    // strand a seat whose only remaining engine action is EndTurn.
    if rules.block_consecutive_freeze && history.follows_freeze(game.turn) {
        mask[taverngym::actions::A_FREEZE] = false;
    }
    if mask[A_END_TURN] && game.players[seat].gold >= MUST_SPEND_GOLD {
        mask[A_END_TURN] = false;
        if !mask.iter().any(|legal| *legal) {
            mask[A_END_TURN] = true;
        }
    }
    if mask[A_CANCEL] {
        mask[A_CANCEL] = false;
        if !mask.iter().any(|legal| *legal) {
            mask[A_CANCEL] = true;
        }
    }
    if rules.freeze_at_end {
        mask[taverngym::actions::A_FREEZE] = mask[A_END_TURN]
            && can_freeze
            && game.players[seat].shop.iter().any(|slot| !slot.frozen);
    }
    mask
}
