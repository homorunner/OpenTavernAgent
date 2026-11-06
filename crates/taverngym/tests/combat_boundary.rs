mod common;
use common::*;
use taverngym::actions::{A_END as A_END_TURN, A_FREEZE};

fn assert_same(a: &Game, b: &Game) {
    // Includes the RNG, action counters, shops, hands and combat records.
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
}

#[test]
fn pause_is_after_end_effects_and_resumes_exactly_once() {
    let mut g = game(4207);
    set_board(&mut g, 7, &["BG36_115"]); // Nightmare Corroder: gain a spell at end of turn.
    for p in 0..7 {
        assert!(!g.apply_index_until_combat(p, A_END_TURN).unwrap());
    }
    let turn = g.turn;
    let mut ordinary = g.clone();
    ordinary.apply_index(7, A_END_TURN).unwrap();
    assert!(g.apply_index_until_combat(7, A_END_TURN).unwrap());
    assert_eq!(g.turn, turn);
    assert!(g.players.iter().all(|p| p.ended_turn));
    assert_eq!(g.players[7].hand.len(), 1);
    assert!(g.last_combats.is_empty());
    assert!(g.apply_index(7, A_END_TURN).is_err());
    // The boundary is clonable and serializable, including deferred action cleanup.
    let mut restored: Game = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    restored.resume_combat().unwrap();
    g.resume_combat().unwrap();
    assert_same(&g, &ordinary);
    assert_same(&restored, &ordinary);
    assert!(g.resume_combat().is_err());
}

#[test]
fn recruit_elimination_can_trigger_the_same_boundary() {
    let mut g = game(4210);
    for p in 0..7 {
        g.apply_index(p, A_END_TURN).unwrap();
    }
    g.players[7].health = 0;
    let mut ordinary = g.clone();
    ordinary.apply_index(7, A_FREEZE).unwrap();
    assert!(g.apply_index_until_combat(7, A_FREEZE).unwrap());
    assert!(!g.players[7].alive);
    assert_eq!(g.turn, 1);
    assert!(g.last_combats.is_empty());
    g.resume_combat().unwrap();
    assert_same(&g, &ordinary);
}

#[test]
fn pausing_every_action_preserves_normal_games() {
    for seed in 20..24 {
        let mut ordinary = game(seed);
        ordinary.cfg.max_turns = 5;
        for p in 0..8 {
            set_board(&mut ordinary, p, &["BG36_115", "BG25_008"]);
        }
        let mut paused = ordinary.clone();
        let mut rng = taverngym::rng::Rng::new(seed);
        while !ordinary.is_over() {
            for seat in 0..8 {
                if !ordinary.needs_action(seat) {
                    continue;
                }
                let mask = ordinary.legal_mask(seat);
                let legal: Vec<_> = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| m.then_some(i))
                    .collect();
                let action = if mask[A_END_TURN] && rng.coin() {
                    A_END_TURN
                } else {
                    legal[rng.below(legal.len())]
                };
                ordinary.apply_index(seat, action).unwrap();
                if paused.apply_index_until_combat(seat, action).unwrap() {
                    paused.resume_combat().unwrap();
                }
                assert_same(&paused, &ordinary);
            }
        }
    }
}
