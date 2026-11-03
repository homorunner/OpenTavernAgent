mod common;
use common::*;
use std::sync::Arc;
use taverngym::obs::*;
use taverngym::state::Scouted;

#[test]
fn stats_are_log_scaled_and_other_features_stay_linear() {
    let mut g = game(4201);
    let mut m = minion(&mut g, 0, "BG25_008");
    m.atk = 10;
    m.hp = 1000;
    m.max_hp = 10000;
    g.players[0].board = vec![m];
    g.players[0].gold = 7;
    g.players[0].health = 20;
    let obs = observe(&g, 0);
    let offset = GLOBAL_F + N_QUESTS * QUEST_F + PENDING_KINDS;
    assert_eq!(obs.floats.len(), 3238);
    assert_eq!(obs.floats[7], 0.7);
    assert!((obs.floats[5] - 20.0_f32.ln_1p() / 40.0_f32.ln_1p()).abs() < 1e-6);
    assert!((obs.floats[offset + 1] - 1.0).abs() < 1e-6);
    assert!((obs.floats[offset + 2] - 1000.0_f32.ln_1p() / 10.0_f32.ln_1p()).abs() < 1e-6);
    assert!((obs.floats[offset + 3] - 10000.0_f32.ln_1p() / 10.0_f32.ln_1p()).abs() < 1e-6);
}

#[test]
fn scout_turn_is_visible_without_revealing_current_opponent_board() {
    let mut g = game(4202);
    g.turn = 9;
    g.players[0].scouted.resize_with(8, Scouted::default);
    // An observed empty board still has a timestamp, unlike an unknown board.
    g.players[0].scouted[1] = Scouted {
        board: Arc::new(vec![]),
        turn: 4,
    };
    let offset =
        GLOBAL_F + N_QUESTS * QUEST_F + PENDING_KINDS + 7 * MINION_F + (10 + 8 + 16) * ITEM_F;
    let obs = observe(&g, 0);
    assert_eq!(obs.floats[offset + 9], 0.2);
    assert_eq!(obs.floats[offset + OPP_F + 9], 0.0);
    set_board(&mut g, 1, &["BG25_008"]);
    assert_eq!(observe(&g, 0).floats, obs.floats);
    let restored: Game = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    assert_eq!(observe(&restored, 0).floats, obs.floats);
}

#[test]
fn remaining_action_budgets_are_visible_and_unlimited_is_distinct() {
    let mut g = game(4206);
    g.cfg.turn_action_caps = vec![100];
    g.cfg.move_actions_per_turn = 10;
    g.cfg.freeze_presses_per_turn = 5;
    g.players[0].actions_this_turn = 90;
    g.players[0].moves_this_turn = 8;
    g.players[0].freezes_this_turn = 5;
    assert_eq!(
        &observe(&g, 0).floats[OBS_FLOATS - ACTION_BUDGET_F..],
        &[0.1, 0.2, 0.0]
    );
    g.players[0].uncapped = true;
    g.cfg.move_actions_per_turn = 0;
    g.cfg.freeze_presses_per_turn = 0;
    assert_eq!(
        &observe(&g, 0).floats[OBS_FLOATS - ACTION_BUDGET_F..],
        &[-1.0, -1.0, -1.0]
    );
}

#[test]
fn end_turn_effects_previews_the_position_handed_to_combat_without_fighting() {
    use taverngym::engine::Phase;
    let mut g = game(4207);
    g.cfg.triple_copies = 99;
    set_board(&mut g, 0, &["BG36_115"]); // Nightmare Corroder: a spell to hand at end of turn
    assert_eq!(observe(&g, 0).floats[ENDED_TURN_F], 0.0);
    // A non-last seat: the preview is exactly what End Turn does.
    let mut real = g.clone();
    real.apply(0, Action::EndTurn).unwrap();
    let mut preview = g.clone();
    preview.end_turn_effects(0).unwrap();
    preview.players[0].actions_this_turn += 1; // `apply` counts the action itself
    assert_eq!(
        serde_json::to_string(&preview).unwrap(),
        serde_json::to_string(&real).unwrap()
    );
    assert_eq!(preview.players[0].hand.len(), 1, "end-of-turn effects ran");
    assert!(preview.players[0].ended_turn && !preview.needs_action(0));
    assert!(preview.legal_actions(0).is_empty());
    assert_eq!(observe(&preview, 0).floats[ENDED_TURN_F], 1.0);
    assert!(preview.end_turn_effects(0).is_err(), "already ended");
    // The last seat to end: the round does not start.
    for p in 1..8 {
        g.apply(p, Action::EndTurn).unwrap();
    }
    let turn = g.turn;
    g.end_turn_effects(0).unwrap();
    assert_eq!(g.turn, turn);
    assert_eq!(g.phase, Phase::Recruit);
    assert!(g.players.iter().all(|p| p.ended_turn));
    assert_eq!(observe(&g, 0).floats[ENDED_TURN_F], 1.0);
}

#[test]
fn an_unaffordable_paid_trinket_offer_can_be_declined() {
    use taverngym::actions::{A_CANCEL, A_CHOOSE};
    let mut g = game(4208);
    g.players[0].gold = 3;
    let greater: Vec<CardId> = db()
        .iter()
        .filter(|d| d.kind == taverngym::cards::CardKind::TrinketGreater && d.cost >= 5)
        .map(|d| d.cid)
        .take(3)
        .collect();
    g.players[0].trinkets.push(cid("BG30_MagicItem_891"));
    g.players[0].pending.push(Pending::Trinket {
        options: greater.clone(),
        free: false,
        replace: Some(cid("BG30_MagicItem_891")),
    });
    assert!(g.needs_action(0));
    let mask = g.legal_mask(0);
    assert!((0..greater.len()).all(|i| !mask[A_CHOOSE + i]));
    assert!(mask[A_CANCEL], "declining is the only legal action");
    assert!(g.legal_actions(0).len() == 1);
    g.apply(0, Action::Cancel).unwrap();
    assert!(g.players[0].pending.is_empty());
    assert!(g.players[0].trinkets.contains(&cid("BG30_MagicItem_891")));
    // With enough Gold the offer cannot be declined.
    g.players[0].gold = 20;
    g.players[0].pending.push(Pending::Trinket {
        options: greater,
        free: false,
        replace: Some(cid("BG30_MagicItem_891")),
    });
    assert!(!g.legal_mask(0)[A_CANCEL]);
    assert!(g.apply(0, Action::Cancel).is_err());
}
