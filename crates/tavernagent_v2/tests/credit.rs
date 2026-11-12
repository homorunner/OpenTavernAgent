use tavernagent_v2::{
    actions, critic,
    gae::{self, ValueStep},
    heroes::select_heroes,
    observation::{Observation, agent_mask_with_rules},
    rollout::{Sample, Trajectory, Transition},
    rules::{ActionHistory, PolicyRules},
    train::TrainOptions,
};
use taverngym::{
    Game,
    actions::{A_END, A_FREEZE},
};

fn trajectory(turns: &[u32]) -> Trajectory {
    Trajectory {
        seed: 1,
        seat: 0,
        placement: 1,
        terminal_health: 40.,
        steps: turns
            .iter()
            .enumerate()
            .map(|(i, &turn)| Transition {
                turn,
                health: 40.,
                obs: Observation {
                    x: vec![],
                    cards: vec![],
                    mask: vec![],
                },
                action: 0,
                old_log_prob: 0.,
                old_value: 0.,
                reward: if i + 1 == turns.len() { 1. } else { 0. },
                terminated: i + 1 == turns.len(),
            })
            .collect(),
    }
}

#[test]
fn turn_trace_ignores_inserted_within_turn_decisions_and_counts_skipped_turns() {
    let compact = trajectory(&[1, 2, 4])
        .into_samples_with_trace(1., 0.95, Some(1.))
        .unwrap();
    let expanded = trajectory(&[1, 1, 1, 2, 2, 4])
        .into_samples_with_trace(1., 0.95, Some(1.))
        .unwrap();
    assert_eq!(compact[0].return_, expanded[0].return_);
    assert!((compact[0].return_ - 0.95f32.powi(3)).abs() < 1e-6);
    assert_eq!(expanded[0].return_, expanded[2].return_);
    let old = trajectory(&[1, 1, 1, 2, 2, 4])
        .into_samples(1., 0.95)
        .unwrap();
    assert!(old[0].return_ < expanded[0].return_);
    assert!(
        trajectory(&[2, 1])
            .into_samples_with_trace(1., 0.95, Some(1.))
            .is_err()
    );
}

#[test]
fn terminal_only_lambda_returns_are_bounded_and_lambda_one_is_monte_carlo() {
    for reward in [-1., -0.2, 1.] {
        let values: Vec<_> = (0..101)
            .map(|i| ValueStep {
                value: if i % 2 == 0 { 0.99 } else { -0.99 },
                reward: if i == 100 { reward } else { 0. },
                terminated: i == 100,
            })
            .collect();
        for lambda in [0., 0.95, 0.97, 0.98, 0.99, 1.] {
            let targets = gae::estimate(&values, 0., 1., lambda).unwrap();
            assert!(
                targets
                    .iter()
                    .all(|(_, r)| (-1.000001..=1.000001).contains(r))
            );
            if lambda == 1. {
                assert!(targets.iter().all(|(_, r)| (r - reward).abs() < 1e-6));
            }
        }
    }
    assert!(gae::estimate(&[], 0., 1., f32::NAN).is_err());
}

#[test]
fn health_targets_use_turn_plus_three_and_censor_early_winners() {
    let mut path = trajectory(&[1, 1, 2, 4, 5]);
    for (s, h) in path.steps.iter_mut().zip([30., 25., 20., 10., 5.]) {
        s.health = h;
    }
    let alive = path.clone().into_samples(1., 0.95).unwrap();
    assert!((alive[0].future_health.unwrap() + 2.0 / 3.0).abs() < 1e-6);
    assert_eq!(alive[0].future_health, alive[1].future_health);
    assert!((alive[2].future_health.unwrap() + 5.0 / 6.0).abs() < 1e-6);
    assert!(alive[3].future_health.is_none());
    path.terminal_health = 0.;
    let dead = path.into_samples(1., 0.95).unwrap();
    assert_eq!(dead[3].future_health, Some(-1.));
}

#[test]
fn freeze_and_end_obeys_gold_menu_caps_and_finishes_the_seat() {
    let opts = TrainOptions::default();
    let mut game = Game::new(23, opts.game_config().unwrap());
    select_heroes(&mut game, 23).unwrap();
    for p in &mut game.players {
        p.pending.clear();
        p.powers.clear();
        p.skipping = false;
    }
    let rules = PolicyRules {
        freeze_at_end: true,
        ..Default::default()
    };
    let history = ActionHistory::default();
    game.players[0].gold = 10;
    let mask = agent_mask_with_rules(&game, 0, rules, &history);
    assert!(!mask[A_END] && !mask[A_FREEZE]);
    game.players[0].gold = 0;
    let mask = agent_mask_with_rules(&game, 0, rules, &history);
    assert!(mask[A_END] && mask[A_FREEZE]);
    let mut capped = game.clone();
    capped.players[0].actions_this_turn = capped.action_cap_of(0);
    let mask = agent_mask_with_rules(&capped, 0, rules, &history);
    assert!(mask[A_END] && !mask[A_FREEZE]);
    actions::apply_with_rules(&mut game, 0, A_FREEZE, rules).unwrap();
    assert!(!game.needs_action(0));
    assert!(game.players[0].shop.iter().all(|s| s.frozen));
    assert_eq!(game.players[0].freezes_this_turn, 1);
}

#[test]
fn explained_variance_uses_centered_residuals_and_handles_constant_targets() {
    let mut samples: Vec<Sample> = trajectory(&[1, 1, 2]).into_samples(1., 1.).unwrap();
    for (i, s) in samples.iter_mut().enumerate() {
        s.return_ = i as f32;
        s.terminal_reward = i as f32;
        s.transition.old_value = i as f32 + 0.5;
    }
    let report = critic::summarize(&samples);
    assert_eq!(report["all"]["lambda_return"]["explained_variance"], 1.);
    assert_eq!(report["all"]["lambda_return"]["mse"], 0.25);
    assert!(report["by_turn"][1]["metrics"]["lambda_return"]["explained_variance"].is_null());
}
