//! Compact agent actions. Engine actions and their indices remain unchanged.
use taverngym::actions as engine;
use taverngym::{Action, Game};

pub const N_POS: usize = engine::N_BOARD;
pub const N_ACTIONS: usize = engine::A_PLAY + engine::N_HAND * N_POS;

/// Resolve against the state *before* applying the action. A compatible occupied
/// position means Magnetize; an empty or incompatible position means normal Play.
pub fn engine_action(game: &Game, seat: usize, action: usize) -> Result<Action, &'static str> {
    if action >= N_ACTIONS {
        return Err("bad agent action index");
    }
    if action < engine::A_PLAY {
        return Action::decode(action).ok_or("bad agent action index");
    }
    let k = action - engine::A_PLAY;
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
    if rules.freeze_at_end && action == engine::A_FREEZE {
        let legal = game.legal_mask(seat);
        if !legal[engine::A_END] || !legal[engine::A_FREEZE] {
            return Err("freeze-and-end is not legal");
        }
        // An already frozen shop need not be toggled off. Partially frozen shops
        // become fully frozen, matching the engine Freeze button.
        if !game.players[seat].shop.iter().all(|slot| slot.frozen) {
            game.apply(seat, Action::Freeze)?;
        }
        game.apply(seat, Action::EndTurn)
    } else {
        apply(game, seat, action)
    }
}
