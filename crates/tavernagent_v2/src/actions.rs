//! Compact agent actions. Engine actions and their indices remain unchanged.
use taverngym::actions as engine;
use taverngym::{Action, Game};

pub use engine::{
    A_BUY, A_CANCEL, A_DARK_GIFT, A_END, A_FREEZE, A_HERO_POWER, A_REFRESH, A_UPGRADE, N_BOARD,
    N_CHOICE, N_HAND, N_POWERS,
};
pub const N_SHOP: usize = 7;
pub const A_SELL: usize = A_BUY + N_SHOP;
pub const A_ACTIVATE: usize = A_SELL + N_BOARD;
pub const A_CHOOSE: usize = A_ACTIVATE + N_BOARD;
pub const A_PLAY: usize = A_CHOOSE + N_CHOICE;

pub const N_POS: usize = engine::N_BOARD;
pub const N_ACTIONS: usize = A_PLAY + N_HAND * N_POS;

/// Resolve against the state *before* applying the action. A compatible occupied
/// position means Magnetize; an empty or incompatible position means normal Play.
pub fn engine_action(game: &Game, seat: usize, action: usize) -> Result<Action, &'static str> {
    if action >= N_ACTIONS {
        return Err("bad agent action index");
    }
    if action < A_PLAY {
        let engine_index = if action < A_SELL { action } else { action + 1 };
        return Action::decode(engine_index).ok_or("bad agent action index");
    }
    let k = action - A_PLAY;
    let (hand, pos) = (k / N_POS, k % N_POS);
    Ok(if game.can_magnetize(seat, hand, pos) {
        Action::Magnetize { hand, board: pos }
    } else {
        Action::Play { hand, pos }
    })
}

pub fn legal_mask(game: &Game, seat: usize) -> Vec<bool> {
    let mask = game.legal_mask(seat);
    (0..N_ACTIONS)
        .map(|a| mask[engine_action(game, seat, a).unwrap().encode()])
        .collect()
}

pub fn apply(game: &mut Game, seat: usize, action: usize) -> Result<(), &'static str> {
    let action = engine_action(game, seat, action)?;
    game.apply(seat, action)
}

/// Policy semantics belong to the checkpoint; the simulator action space is unchanged.
pub fn apply_with_rules(
    game: &mut Game,
    seat: usize,
    action: usize,
    rules: crate::rules::PolicyRules,
) -> Result<(), &'static str> {
    apply_with(game, seat, action, rules, |game, seat, action| {
        game.apply(seat, action)
    })
}

/// Shared action expansion for ordinary evaluation and observational replay hooks.
pub(crate) fn apply_with(
    game: &mut Game,
    seat: usize,
    action: usize,
    rules: crate::rules::PolicyRules,
    mut apply: impl FnMut(&mut Game, usize, Action) -> Result<(), &'static str>,
) -> Result<(), &'static str> {
    if rules.freeze_at_end && action == engine::A_FREEZE {
        let legal = game.legal_mask(seat);
        if !legal[engine::A_END] || !legal[engine::A_FREEZE] {
            return Err("freeze-and-end is not legal");
        }
        // An already frozen shop need not be toggled off. Partially frozen shops
        // become fully frozen, matching the engine Freeze button.
        if !game.players[seat].shop.iter().all(|slot| slot.frozen) {
            apply(game, seat, Action::Freeze)?;
        }
        apply(game, seat, Action::EndTurn)
    } else {
        let engine = engine_action(game, seat, action)?;
        apply(game, seat, engine)
    }
}
