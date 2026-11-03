//! Search information boundary, chance sampling and the ended-state End Turn transition.

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde_json::Value;
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{
    self, A_END_TURN, Evaluator, Horizon, Observation, SearchOptions, observe_seat, reseeded,
};
use tavernagent::net::{ENDED_TURN_FEATURE, N_ACTIONS, input_feature_indices};
use tavernagent::rules::Budgets;
use taverngym::Game;

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

fn opts(sims: usize, horizon: Horizon) -> SearchOptions {
    SearchOptions {
        sims,
        horizon,
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
fn search_is_independent_of_the_real_rng_and_does_not_mutate_the_game() {
    for horizon in [Horizon::Turn, Horizon::Game] {
        let game = position(19);
        let other = reseeded(&game, 987_654_321);
        let before = serde_json::to_string(&game).unwrap();
        let a = mcts::search(
            &game,
            0,
            &SingleAction(1),
            &opts(24, horizon),
            &mut Pcg64Mcg::seed_from_u64(1),
        )
        .unwrap();
        let b = mcts::search(
            &other,
            0,
            &SingleAction(1),
            &opts(24, horizon),
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
}

#[test]
fn repeated_refresh_visits_sample_several_outcomes_on_one_edge() {
    let game = position(19);
    let result = mcts::search(
        &game,
        0,
        &SingleAction(1),
        &opts(64, Horizon::Turn),
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
        &opts(12, Horizon::Turn),
        &mut Pcg64Mcg::seed_from_u64(9),
    )
    .unwrap();
    let index = result.root.actions.iter().position(|a| *a == 2).unwrap();
    assert_eq!(result.root.children[index].len(), 1);
}

#[test]
fn turn_search_ends_the_turn_without_fighting_or_advancing_other_seats() {
    let mut game = position(19);
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
        &opts(16, Horizon::Turn),
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
        &opts(16, Horizon::Turn),
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
        &opts(16, Horizon::Turn),
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
    let mask = game.legal_mask(0);
    let legal: Vec<usize> = (0..N_ACTIONS).filter(|a| mask[*a]).collect();
    assert_eq!(result.root.actions, legal);
    assert!(legal.iter().all(|a| result.root.prior[*a] > 0.0));
}
