//! Search information boundary, chance sampling and the ended-state End Turn transition.

use rand::{Rng as _, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde_json::Value;
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{
    self, A_END_TURN, Evaluator, MUST_SPEND_GOLD, Observation, SearchOptions, agent_mask,
    observe_seat, reseeded, rewind_turn,
};
use tavernagent::net::{ENDED_TURN_FEATURE, N_ACTIONS, input_feature_indices};
use tavernagent::rules::Budgets;
use taverngym::Game;
use taverngym::actions::{
    A_BUY, A_CANCEL, A_CHOOSE, A_HERO_POWER, A_MOVE, A_PLAY, A_REFRESH, N_BOARD, N_CHOICE, N_SHOP,
};
use taverngym::cards::{CardId, CardKind, cid, db};
use taverngym::state::{Item, Pending};

fn position(seed: u64) -> Game {
    let mut game = Game::new(seed, Budgets::default().config().unwrap());
    select_heroes(&mut game, seed).unwrap();
    // Give the controlled seat a plain recruit phase, without hero-specific menus.
    let mut state: Value = serde_json::to_value(&game).unwrap();
    let me = &mut state["players"][0];
    me["pending"] = Value::Array(vec![]);
    me["skipping"] = Value::Bool(false);
    me["gold"] = Value::from(30);
    serde_json::from_value(state).unwrap()
}

/// `position` with too little Gold for the must-spend rule to bite.
fn broke_position(seed: u64) -> Game {
    let mut game = position(seed);
    game.players[0].gold = MUST_SPEND_GOLD - 1;
    game
}

fn state_without_rng(game: &Game) -> Value {
    let mut v = serde_json::to_value(game).unwrap();
    v.as_object_mut().unwrap().remove("rng");
    v
}

/// Puts all prior on one action (or the first legal one) with an outcome-dependent value.
struct SingleAction(usize);

impl Evaluator for SingleAction {
    fn evaluate(&self, obs: &Observation) -> Result<(Vec<f32>, f32), String> {
        let mut prior = vec![0.0; N_ACTIONS];
        if obs.mask.iter().any(|m| *m) {
            let action = if obs.mask[self.0] {
                self.0
            } else {
                obs.mask.iter().position(|m| *m).unwrap()
            };
            prior[action] = 1.0;
        }
        // Positive but outcome-dependent, so root exploration keeps the forced action.
        Ok((prior, 0.1 + (obs.cards[30] % 10) as f32 / 20.0))
    }
}

fn opts(sims: usize) -> SearchOptions {
    SearchOptions {
        sims,
        keep_games: true,
        ..SearchOptions::default()
    }
}

#[test]
fn reseeded_copy_changes_only_the_future_rng() {
    let game = position(19);
    let copy = reseeded(&game, 12345);
    assert_eq!(state_without_rng(&copy), state_without_rng(&game));
    assert_ne!(
        serde_json::to_value(&copy).unwrap()["rng"],
        serde_json::to_value(&game).unwrap()["rng"]
    );
    // Different future streams produce different Refresh outcomes.
    let shops: std::collections::HashSet<Vec<i32>> = (0..8u64)
        .map(|seed| {
            let mut sample = reseeded(&game, seed);
            sample.apply_index(0, 1).unwrap();
            observe_seat(&sample, 0).cards
        })
        .collect();
    assert!(shops.len() > 1);
}

#[test]
fn rewind_turn_restores_the_reseeded_copy_after_any_turn_line() {
    // Positions from several games and turns, bots driving all seats to get there; each
    // then plays random legal lines (including End Turn and its effects) on a working copy
    // and rewinds. The rewound copy must be byte-identical to a fresh reseeded copy.
    let mut lines = 0;
    for game_seed in 0..6u64 {
        let mut game = Game::new(game_seed, Budgets::default().config().unwrap());
        select_heroes(&mut game, game_seed).unwrap();
        let mut rng = Pcg64Mcg::seed_from_u64(game_seed);
        let mut visited = 0;
        while !game.is_over() && game.turn <= 14 {
            let seat = match (0..game.players.len()).find(|&s| game.needs_action(s)) {
                Some(s) => s,
                None => break,
            };
            // The first decision of a seat's turn, every other turn: deeper turns bring hero
            // powers, trinkets and buddies into the lines.
            let fresh = game.players[seat].actions_this_turn == 0;
            if fresh && game.turn % 2 == 1 && game.players[seat].pending.is_empty() {
                visited += 1;
                let mut sim = game.clone();
                for line in 0..5u64 {
                    let seed = game_seed * 1000 + visited * 10 + line;
                    rewind_turn(&mut sim, &game, seat, seed);
                    for _ in 0..rng.random_range(1..12) {
                        let mask = agent_mask(&sim, seat);
                        let legal: Vec<usize> = (0..N_ACTIONS).filter(|a| mask[*a]).collect();
                        if legal.is_empty() {
                            break;
                        }
                        let a = legal[rng.random_range(0..legal.len())];
                        if a == A_END_TURN {
                            sim.end_turn_effects(seat).unwrap();
                            break;
                        }
                        tavernagent::actions::apply(&mut sim, seat, a).unwrap();
                    }
                    lines += 1;
                    rewind_turn(&mut sim, &game, seat, seed);
                    let expect = reseeded(&game, seed);
                    assert_eq!(
                        serde_json::to_string(&sim).unwrap(),
                        serde_json::to_string(&expect).unwrap(),
                        "game {game_seed} turn {} seat {seat} line {line}",
                        game.turn
                    );
                }
            }
            let a = mcts::bot_action(&mut game, seat);
            game.apply(seat, a).unwrap();
        }
    }
    assert!(lines > 100, "{lines} lines");
}

#[test]
fn the_agent_cannot_cancel_a_menu_it_opened() {
    // A targeted hand spell (Winner's Bread: give a friendly minion +2/+3) opens a target
    // menu; the engine offers Cancel, the agent is not allowed to take it.
    let mut game = position(19);
    let shop_minion = game.players[0]
        .shop
        .iter()
        .position(|s| s.item.as_minion().is_some())
        .expect("a minion in the tavern");
    game.apply_index(0, taverngym::actions::A_BUY + shop_minion)
        .unwrap();
    let hand = game.players[0].hand.len() - 1;
    game.apply_index(0, A_PLAY + hand * taverngym::actions::N_POS)
        .unwrap();
    assert_eq!(game.players[0].board.len(), 1);
    game.with_recruit(0, |ctx| {
        ctx.add_to_hand(
            0,
            Item::Spell {
                card: cid("BG36_883"),
                data: 0,
            },
        );
    });
    let spell = game.players[0]
        .hand
        .iter()
        .position(|h| matches!(h.item, Item::Spell { .. }))
        .unwrap();
    game.apply_index(0, A_PLAY + spell * taverngym::actions::N_POS)
        .unwrap();
    assert!(matches!(
        game.players[0].pending.last(),
        Some(Pending::Target { .. })
    ));
    assert!(game.legal_mask(0)[A_CANCEL]);
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_CANCEL]);
    let choices: Vec<usize> = (A_CHOOSE..A_CHOOSE + N_CHOICE)
        .filter(|a| mask[*a])
        .collect();
    assert!(!choices.is_empty());
    assert_eq!(
        mask.iter().filter(|m| **m).count(),
        choices.len(),
        "only the menu's options are open"
    );
    let atk = game.players[0].board[0].atk;
    game.apply_index(0, choices[0]).unwrap();
    assert!(game.players[0].pending.is_empty());
    assert_eq!(game.players[0].board[0].atk, atk + 2);

    // Same for a Hero Power's target menu (Lich Baz'hial: steal a Tavern card).
    game.players[0].set_hero_power(0, cid("TB_BaconShop_HP_049"));
    game.apply_index(0, A_HERO_POWER).unwrap();
    assert!(game.legal_mask(0)[A_CANCEL]);
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_CANCEL]);
    assert!((A_CHOOSE..A_CHOOSE + N_CHOICE).any(|a| mask[a]));
}

#[test]
fn cancel_stays_when_it_is_the_only_legal_action() {
    // A paid trinket offer with nothing affordable is the one state where the engine makes
    // Cancel the sole legal action. `offer_trinkets` skips such an offer, so build it by
    // hand: the rule's empty-mask guard must leave Cancel in place.
    let mut game = position(19);
    game.players[0].gold = 0;
    let dear: Vec<_> = (0..db().len() as u16)
        .map(CardId)
        .filter(|c| db().get(*c).kind == CardKind::TrinketGreater && db().get(*c).cost > 0)
        .take(3)
        .collect();
    assert_eq!(dear.len(), 3);
    game.players[0].pending.push(Pending::Trinket {
        options: dear,
        free: false,
        replace: None,
    });
    let legal = game.legal_mask(0);
    assert!(legal[A_CANCEL]);
    assert_eq!(legal.iter().filter(|m| **m).count(), 1);
    assert_eq!(agent_mask(&game, 0), legal[..N_ACTIONS]);
}

#[test]
fn the_agent_has_no_move_actions() {
    // Compact Play replaces both engine Play and Magnetize; Move remains engine-only.
    assert_eq!(N_ACTIONS, 116);
    assert_eq!(taverngym::actions::N_ACTIONS, A_MOVE + N_BOARD * N_BOARD);
    // Two minions on the warband make a Move legal for the engine but not for the agent.
    let mut game = position(7);
    for _ in 0..2 {
        let buy = (A_BUY..A_BUY + N_SHOP)
            .find(|a| {
                game.legal_mask(0)[*a]
                    && matches!(game.players[0].shop[*a - A_BUY].item, Item::Minion(_))
            })
            .unwrap();
        game.apply_index(0, buy).unwrap();
        let play = (A_PLAY..A_MOVE).find(|a| game.legal_mask(0)[*a]).unwrap();
        game.apply_index(0, play).unwrap();
    }
    let legal = game.legal_mask(0);
    assert!(
        legal[A_MOVE..].iter().any(|m| *m),
        "the engine offers a Move"
    );
    let mask = agent_mask(&game, 0);
    assert_eq!(mask.len(), N_ACTIONS);
    // End Turn is hidden at 30 Gold; other entries match their translated engine action.
    for (a, &masked) in mask.iter().enumerate().skip(1) {
        let engine = tavernagent::actions::engine_action(&game, 0, a)
            .unwrap()
            .encode();
        assert_eq!(masked, legal[engine]);
    }
}

#[test]
fn rewind_turn_falls_back_to_a_full_copy_when_the_seat_died() {
    // Lich Baz'hial's power costs 2 Health; at 1 Health it kills the seat mid-turn, which
    // eliminates it (phase, pairings, elimination order, pool) and may end the game.
    let mut game = position(19);
    game.players[0].health = 1;
    game.players[0].armor = 0;
    game.players[0].set_hero_power(0, cid("TB_BaconShop_HP_049"));
    let seed = 77;
    let mut sim = game.clone();
    rewind_turn(&mut sim, &game, 0, seed);
    assert!(agent_mask(&sim, 0)[A_HERO_POWER]);
    sim.apply_index(0, A_HERO_POWER).unwrap();
    // The power asks for a Tavern target: take the first one offered (not Cancel).
    let mut guard = 0;
    while sim.needs_action(0) && !sim.players[0].pending.is_empty() {
        let mask = agent_mask(&sim, 0);
        let a = (A_CANCEL + 1..N_ACTIONS).find(|a| mask[*a]).unwrap();
        tavernagent::actions::apply(&mut sim, 0, a).unwrap();
        guard += 1;
        assert!(guard < 10);
    }
    assert!(!sim.players[0].alive, "the power did not kill the seat");
    assert_ne!(sim.eliminated, game.eliminated);
    rewind_turn(&mut sim, &game, 0, seed);
    assert_eq!(
        serde_json::to_string(&sim).unwrap(),
        serde_json::to_string(&reseeded(&game, seed)).unwrap()
    );
    // And a search from that position (End Turn, Hero Power and the rest all legal) is
    // still a pure function of the seeds, and leaves the real game untouched.
    let before = serde_json::to_string(&game).unwrap();
    let a = mcts::search(
        &game,
        0,
        &SingleAction(A_HERO_POWER),
        &opts(32),
        &mut Pcg64Mcg::seed_from_u64(3),
    )
    .unwrap();
    let b = mcts::search(
        &game,
        0,
        &SingleAction(A_HERO_POWER),
        &opts(32),
        &mut Pcg64Mcg::seed_from_u64(3),
    )
    .unwrap();
    assert_eq!(a.visits, b.visits);
    assert_eq!(serde_json::to_string(&game).unwrap(), before);
}

#[test]
fn search_is_independent_of_the_real_rng_and_does_not_mutate_the_game() {
    let game = position(19);
    let other = reseeded(&game, 987_654_321);
    let before = serde_json::to_string(&game).unwrap();
    let a = mcts::search(
        &game,
        0,
        &SingleAction(1),
        &opts(24),
        &mut Pcg64Mcg::seed_from_u64(1),
    )
    .unwrap();
    let b = mcts::search(
        &other,
        0,
        &SingleAction(1),
        &opts(24),
        &mut Pcg64Mcg::seed_from_u64(1),
    )
    .unwrap();
    assert_eq!(a.visits, b.visits);
    assert_eq!(a.root.cw, b.root.cw);
    assert_eq!(serde_json::to_string(&game).unwrap(), before);
    // The real next draw after a search is identical to an untouched snapshot's.
    let mut game = game;
    let mut reference: Game = serde_json::from_str(&before).unwrap();
    game.apply_index(0, 1).unwrap();
    reference.apply_index(0, 1).unwrap();
    assert_eq!(
        serde_json::to_string(&game).unwrap(),
        serde_json::to_string(&reference).unwrap()
    );
}

#[test]
fn repeated_refresh_visits_sample_several_outcomes_on_one_edge() {
    let game = position(19);
    let result = mcts::search(
        &game,
        0,
        &SingleAction(1),
        &opts(64),
        &mut Pcg64Mcg::seed_from_u64(123),
    )
    .unwrap();
    let index = result.root.actions.iter().position(|a| *a == 1).unwrap();
    let outcomes = &result.root.children[index];
    assert_eq!(result.visits[1], 64.0);
    assert!(outcomes.len() > 8, "{} outcomes", outcomes.len());
    assert!(
        outcomes
            .values()
            .all(|n| n.game.as_ref().unwrap().turn == game.turn && n.value.is_finite())
    );
}

#[test]
fn deterministic_actions_reuse_nodes_despite_new_simulation_rng() {
    let game = position(19);
    let result = mcts::search(
        &game,
        0,
        &SingleAction(2),
        &opts(12),
        &mut Pcg64Mcg::seed_from_u64(9),
    )
    .unwrap();
    let index = result.root.actions.iter().position(|a| *a == 2).unwrap();
    assert_eq!(result.root.children[index].len(), 1);
}

#[test]
fn turn_search_ends_the_turn_without_fighting_or_advancing_other_seats() {
    let mut game = broke_position(19);
    // The controlled seat is the last seat: a real EndTurn would resolve combat.
    for seat in 1..game.players.len() {
        while game.needs_action(seat) {
            let a = mcts::bot_action(&mut game, seat);
            game.apply(seat, a).unwrap();
        }
    }
    let before = serde_json::to_string(&game).unwrap();
    let end = mcts::search(
        &game,
        0,
        &SingleAction(A_END_TURN),
        &opts(16),
        &mut Pcg64Mcg::seed_from_u64(2),
    )
    .unwrap();
    assert_eq!(end.visits[A_END_TURN], 16.0);
    // EndTurn is an ordinary edge: its child is the ended position, a leaf without actions.
    let index = end
        .root
        .actions
        .iter()
        .position(|a| *a == A_END_TURN)
        .unwrap();
    let outcomes = &end.root.children[index];
    assert_eq!(outcomes.len(), 1);
    let child = outcomes.values().next().unwrap();
    let child_game = child.game.as_ref().unwrap();
    assert_eq!(child_game.turn, game.turn);
    assert!(!child_game.is_over());
    assert!(child.actions.is_empty());
    let ended = observe_seat(child_game, 0);
    assert!(!ended.mask.iter().any(|m| *m) && !child_game.needs_action(0));
    assert!((end.root.cw[index] - 16.0 * child.value).abs() < 1e-4);
    // The child is exactly what Game::end_turn_effects gives on a copy.
    let mut preview = game.clone();
    preview.end_turn_effects(0).unwrap();
    assert_eq!(
        mcts::observation_key(&observe_seat(&preview, 0)),
        mcts::observation_key(&ended)
    );
    // The ended flag is a network input.
    assert_eq!(observe_seat(&game, 0).x[ENDED_TURN_FEATURE], 0.0);
    assert_eq!(ended.x[ENDED_TURN_FEATURE], 1.0);
    assert!(input_feature_indices().contains(&ENDED_TURN_FEATURE));
    mcts::search(
        &game,
        0,
        &SingleAction(1),
        &opts(16),
        &mut Pcg64Mcg::seed_from_u64(3),
    )
    .unwrap();
    assert_eq!(serde_json::to_string(&game).unwrap(), before);
    // Changing a current hidden board cannot affect an EndTurn estimate.
    let mut state: Value = serde_json::from_str(&before).unwrap();
    state["players"][1]["board"] = Value::Array(vec![]);
    let changed: Game = serde_json::from_value(state).unwrap();
    let same = mcts::search(
        &changed,
        0,
        &SingleAction(A_END_TURN),
        &opts(16),
        &mut Pcg64Mcg::seed_from_u64(2),
    )
    .unwrap();
    assert_eq!(end.visits, same.visits);
    assert_eq!(end.root.cw, same.root.cw);
}

#[test]
fn search_rejects_impossible_requests_and_bad_options() {
    let game = position(19);
    let mut rng = Pcg64Mcg::seed_from_u64(0);
    let bad = SearchOptions {
        sims: 0,
        ..SearchOptions::default()
    };
    assert!(mcts::search(&game, 0, &SingleAction(1), &bad, &mut rng).is_err());
    let negative = SearchOptions {
        c_puct: -1.0,
        ..SearchOptions::default()
    };
    assert!(mcts::search(&game, 0, &SingleAction(1), &negative, &mut rng).is_err());
    let mut fresh = Game::new(5, Budgets::default().config().unwrap());
    assert!(mcts::is_hero_selection(&fresh));
    assert!(
        mcts::search(
            &fresh,
            0,
            &SingleAction(1),
            &SearchOptions::default(),
            &mut rng
        )
        .is_err()
    );
    select_heroes(&mut fresh, 5).unwrap();
    assert!(!mcts::is_hero_selection(&fresh));
    assert!(
        Budgets {
            action_cap: 0,
            ..Budgets::default()
        }
        .config()
        .is_err()
    );
    assert!(Budgets::default().config().is_ok());
}

#[test]
fn root_noise_keeps_zero_prior_legal_actions_available() {
    let game = position(19);
    let noisy = SearchOptions {
        sims: 1,
        root_noise: 0.35,
        ..SearchOptions::default()
    };
    let result = mcts::search(
        &game,
        0,
        &SingleAction(1),
        &noisy,
        &mut Pcg64Mcg::seed_from_u64(5),
    )
    .unwrap();
    let mask = agent_mask(&game, 0);
    let legal: Vec<usize> = (0..N_ACTIONS).filter(|a| mask[*a]).collect();
    assert_eq!(result.root.actions, legal);
    assert!(legal.iter().all(|a| result.root.prior[*a] > 0.0));
}

#[test]
fn the_must_spend_rule_hides_end_turn_while_gold_is_left() {
    let mut game = position(23);
    assert!(game.legal_mask(0)[A_END_TURN], "the game itself allows it");
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_END_TURN], "30 Gold is not a place to stop");
    assert!(
        mask.iter().any(|m| *m),
        "masking End Turn never leaves the seat without an action"
    );
    // The rule is a plain Gold check, so it releases only once Gold itself is gone.
    let mut steps = 0;
    while !agent_mask(&game, 0)[A_END_TURN] {
        let pending = !game.players[0].pending.is_empty();
        let mask = agent_mask(&game, 0);
        let from = if pending { 0 } else { A_REFRESH };
        let action = (from..N_ACTIONS).find(|a| mask[*a]).unwrap();
        tavernagent::actions::apply(&mut game, 0, action).unwrap();
        steps += 1;
        assert!(steps < 512, "the rule never released End Turn");
    }
    assert!(game.players[0].gold < MUST_SPEND_GOLD);
    assert!(game.legal_mask(0)[A_END_TURN]);
}

#[test]
fn a_free_refresh_does_not_unlock_end_turn() {
    let mut game = position(29);
    let mut state: Value = serde_json::to_value(&game).unwrap();
    let me = &mut state["players"][0];
    me["gold"] = Value::from(MUST_SPEND_GOLD);
    me["shop"] = Value::Array(vec![]);
    me["free_refreshes"] = Value::from(1);
    me["tier"] = Value::from(6);
    me["powers"] = Value::Array(vec![]);
    me["hand"] = Value::Array(vec![]);
    me["board"] = Value::Array(vec![]);
    game = serde_json::from_value(state).unwrap();
    // Nothing is affordable right now, but a free Refresh may turn something up: the rule
    // keys on Gold alone and keeps End Turn masked, so the seat has to take the Refresh.
    let mask = agent_mask(&game, 0);
    assert!(!mask[A_END_TURN], "3 Gold is still Gold");
    assert!(
        mask[A_REFRESH],
        "the free Refresh is what the rule pushes toward"
    );
}

#[test]
fn the_action_cap_overrides_the_must_spend_rule() {
    let mut game = position(31);
    game.players[0].actions_this_turn = game.action_cap_of(0);
    let mask = agent_mask(&game, 0);
    assert!(
        mask[A_END_TURN],
        "past the cap End Turn is the only legal action, so the rule steps aside"
    );
    assert!(mask.iter().any(|m| *m), "the mask is never emptied");
}
