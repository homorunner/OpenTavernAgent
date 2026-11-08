//! Deterministic evaluation teacher: leveling curve, pairs/triples and early Dark Gifts.
//! Adapted from nbshenxm/AutoTavern, bg/client-diff-replay @ 5075b273 (teacher.rs).
//! Copyright (c) 2026 The AutoTavern Authors. MIT license (see repository LICENSE).
//! Uses the tested no-composition variant: no external tables; economy/gift bonuses are zero.

use taverngym::actions::{Action, N_BOARD, N_HAND, N_POWERS};
use taverngym::bot::{self, minion_value};
use taverngym::cards::{CardId, db};
use taverngym::engine::{Game, Phase};
use taverngym::state::{Item, Minion, Pending};
use taverngym::types::{Kw, PlayerId};

/// Value added to a minion that would be the second copy, or the third (a golden minion and a
/// Discover from a higher Tier).
const PAIR: i32 = 8;
const TRIPLE: i32 = 25;
const LEVEL_TURNS: [u32; 5] = [2, 5, 7, 9, 11];
const EARLY_TURNS: u32 = 8;
const KEEP: i32 = PAIR;

fn item_value(item: &Item) -> i32 {
    match item {
        Item::Minion(m) => minion_value(m),
        Item::Spell { card, .. } => 2 + db().get(*card).tier as i32,
    }
}

/// Non-golden copies of `card` (by normal form) on `p`'s warband and in hand.
fn copies(g: &Game, p: PlayerId, card: CardId) -> usize {
    let d = db();
    let normal = d.normal_of(card);
    let pl = &g.players[p];
    pl.board
        .iter()
        .filter(|m| !m.golden && d.normal_of(m.card) == normal)
        .count()
        + pl.hand
            .iter()
            .filter(|h| matches!(&h.item, Item::Minion(m) if !m.golden && d.normal_of(m.card) == normal))
            .count()
}

/// Additional value of completing a pair or triple.
fn bonus(g: &Game, p: PlayerId, item: &Item) -> i32 {
    match item {
        Item::Minion(m) if !m.golden => match copies(g, p, m.card) {
            0 => 0,
            1 => PAIR,
            _ => TRIPLE,
        },
        _ => 0,
    }
}

/// What a warband minion is worth keeping.
fn kept(g: &Game, p: PlayerId, m: &Minion) -> i32 {
    let pair = if !m.golden && copies(g, p, m.card) >= 2 {
        PAIR
    } else {
        0
    };
    minion_value(m) + pair
}

/// Choose one action for `p` (always legal when `game.needs_action(p)`).
pub fn choose_action(g: &mut Game, p: PlayerId) -> Action {
    let saved = g.rng.clone();
    let action = choose(g, p);
    g.rng = saved;
    action
}

fn choose(g: &mut Game, p: PlayerId) -> Action {
    if let Some(top) = g.players[p].pending.last().cloned() {
        // A Discover (a triple's reward, a Dark Gift...): the most valuable option.
        if let Pending::Discover { options, .. } = &top {
            let i = (0..options.len())
                .filter(|i| g.option_legal(p, *i))
                .max_by_key(|i| item_value(&options[*i]) + bonus(g, p, &options[*i]));
            if let Some(i) = i {
                return Action::Choose(i);
            }
        }
        return bot::choose_action(g, p);
    }
    if g.phase != Phase::Recruit {
        return Action::EndTurn;
    }
    let pl = &g.players[p];
    let mask = g.legal_mask(p);
    let legal = |a: Action| mask[a.encode()];
    if pl.actions_this_turn > 60 {
        return Action::EndTurn;
    }
    // 1. Play cards from hand (the plain bot's way).
    for (h, card) in pl.hand.iter().enumerate().take(N_HAND) {
        match &card.item {
            Item::Minion(m) => {
                if m.has(Kw::MAGNETIC)
                    && let Some(b) = (0..pl.board.len().min(N_BOARD))
                        .find(|b| legal(Action::Magnetize { hand: h, board: *b }))
                {
                    return Action::Magnetize { hand: h, board: b };
                }
                let pos = pl.board.len();
                if pos < 8 && legal(Action::Play { hand: h, pos }) {
                    return Action::Play { hand: h, pos };
                }
            }
            Item::Spell { .. } => {
                if legal(Action::Play { hand: h, pos: 0 }) {
                    return Action::Play { hand: h, pos: 0 };
                }
            }
        }
    }
    // 2. Level on the curve, unless low Health asks for warband first.
    let curve_tier = 1 + LEVEL_TURNS.iter().filter(|x| g.turn >= **x).count() as u8;
    let low = pl.health <= 10 && pl.tier >= 4;
    if pl.tier < curve_tier && !low && legal(Action::Upgrade) {
        return Action::Upgrade;
    }
    // 3. The best Tavern card.
    let worth = |s: &Item| item_value(s) + bonus(g, p, s);
    let best = pl
        .shop
        .iter()
        .enumerate()
        .filter(|(i, _)| legal(Action::Buy(*i)))
        .max_by_key(|(_, s)| worth(&s.item));
    let minions_owned = pl.board.len()
        + pl.hand
            .iter()
            .filter(|h| matches!(h.item, Item::Minion(_)))
            .count();
    let room = minions_owned < g.cfg.board_size;
    // Early, a Dark Gift beats a card with nothing beyond its stats.
    let plain = best.is_none_or(|(_, s)| bonus(g, p, &s.item) <= 0);
    if legal(Action::DarkGift) && room && plain && g.turn <= EARLY_TURNS {
        return Action::DarkGift;
    }
    // 4. Full warband: sell the least kept for a clearly better card.
    if pl.board.len() >= g.cfg.board_size
        && let Some((_, s)) = best
        && let Some((wi, wm)) = pl
            .board
            .iter()
            .enumerate()
            .min_by_key(|(_, m)| kept(g, p, m))
        && worth(&s.item) > kept(g, p, wm) + 4
        && pl.gold >= g.minion_price(p)
        && legal(Action::Sell(wi))
    {
        return Action::Sell(wi);
    }
    // 5. Buy (a minion when there is room or it pairs up).
    if let Some((i, s)) = best {
        let pairs = matches!(&s.item, Item::Minion(m) if !m.golden && copies(g, p, m.card) >= 1);
        if !matches!(s.item, Item::Minion(_)) || room || pairs {
            return Action::Buy(i);
        }
    }
    // 6. Dark Gift, Hero Power, Activate.
    if legal(Action::DarkGift) && room {
        return Action::DarkGift;
    }
    for slot in 0..N_POWERS {
        if legal(Action::HeroPower(slot)) {
            return Action::HeroPower(slot);
        }
    }
    for i in 0..pl.board.len().min(N_BOARD) {
        if legal(Action::Activate(i)) {
            return Action::Activate(i);
        }
    }
    // 7. A cheap level off the curve.
    if legal(Action::Upgrade) && g.upgrade_price(p) <= 3 {
        return Action::Upgrade;
    }
    // 8. Keep a card worth having for next turn; otherwise spend the Gold on Refreshes.
    let keep = pl.shop.iter().any(|s| bonus(g, p, &s.item) >= KEEP);
    if keep && !pl.is_frozen() && legal(Action::Freeze) {
        return Action::Freeze;
    }
    if legal(Action::Refresh)
        && !(keep && pl.is_frozen())
        && pl.stats.refreshes < 30 * g.turn as i32
        && pl.actions_this_turn < 50
    {
        return Action::Refresh;
    }
    Action::EndTurn
}
