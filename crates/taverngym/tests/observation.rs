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
    assert_eq!(obs.floats.len(), 3237);
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
