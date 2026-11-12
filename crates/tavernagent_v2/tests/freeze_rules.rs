use rand::Rng;
use tavernagent_v2::{
    arena, checkpoint, eval,
    heroes::select_heroes,
    net::PolicyValueNet,
    observation::{agent_mask, agent_mask_with_rules},
    rollout,
    rules::{ActionHistory, PolicyRules},
    train,
};
use taverngym::{
    Game,
    actions::{A_END, A_FREEZE, A_REFRESH},
};

fn enabled() -> PolicyRules {
    PolicyRules {
        block_consecutive_freeze: true,
        ..Default::default()
    }
}

fn position() -> Game {
    let mut game = Game::new(23, train::TrainOptions::default().game_config().unwrap());
    select_heroes(&mut game, 23).unwrap();
    for player in &mut game.players {
        player.pending.clear();
        player.powers.clear();
        player.skipping = false;
        player.gold = 10;
    }
    game
}

#[test]
fn freeze_mask_is_seat_local_and_resets_after_another_action_or_turn() {
    let mut game = position();
    let mut histories = [ActionHistory::default(); 8];
    assert!(agent_mask_with_rules(&game, 0, enabled(), &histories[0])[A_FREEZE]);
    let turn = game.turn;
    game.apply_index(0, A_FREEZE).unwrap();
    histories[0].record(turn, A_FREEZE);
    assert!(!agent_mask_with_rules(&game, 0, enabled(), &histories[0])[A_FREEZE]);
    assert!(agent_mask_with_rules(&game, 1, enabled(), &histories[1])[A_FREEZE]);
    game.apply_index(1, A_REFRESH).unwrap();
    histories[1].record(turn, A_REFRESH);
    assert!(!agent_mask_with_rules(&game, 0, enabled(), &histories[0])[A_FREEZE]);
    assert_eq!(
        agent_mask(&game, 0),
        agent_mask_with_rules(&game, 0, PolicyRules::default(), &histories[0])
    );
    game.apply_index(0, A_REFRESH).unwrap();
    histories[0].record(turn, A_REFRESH);
    assert!(agent_mask_with_rules(&game, 0, enabled(), &histories[0])[A_FREEZE]);
    game.apply_index(0, A_FREEZE).unwrap();
    histories[0].record(turn, A_FREEZE);
    game.turn += 1;
    assert!(agent_mask_with_rules(&game, 0, enabled(), &histories[0])[A_FREEZE]);
}

#[test]
fn freeze_rule_preserves_forced_endturn_escape() {
    let mut game = position();
    let mut history = ActionHistory::default();
    history.record(game.turn, A_FREEZE);
    game.players[0].actions_this_turn = game.action_cap_of(0);
    let mask = agent_mask_with_rules(&game, 0, enabled(), &history);
    assert!(mask[A_END]);
    assert_eq!(mask.iter().filter(|m| **m).count(), 1);
    // Removing the only discretionary action also releases the must-spend escape.
    game.players[0].actions_this_turn = 0;
    game.players[0].gold = 3;
    game.players[0].tier = 6;
    game.players[0].shop.clear();
    game.players[0].hand.clear();
    game.players[0].board.clear();
    game.players[0].free_refreshes = 0;
    game.cfg.refresh_cost = 100;
    let mask = agent_mask_with_rules(&game, 0, enabled(), &history);
    assert!(mask[A_END]);
    assert!(!mask[A_FREEZE]);
    assert!(mask.iter().any(|m| *m));
}

#[test]
fn sampled_rollouts_store_the_actual_rule_mask_and_are_worker_independent() {
    let options = train::TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 3,
        ..Default::default()
    };
    let net = PolicyValueNet::new(42, options.net_config()).unwrap();
    let cfg = options.game_config().unwrap();
    let seeds = [11, 12, 13, 14];
    let run = |workers| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        rollout::collect_with_rules(&net, &seeds, &cfg, &pool, enabled()).unwrap()
    };
    let games = run(1);
    assert_eq!(games, run(4));
    assert!(games.iter().map(|g| g.actions.freeze_actions).sum::<u64>() > 0);
    for game in games {
        assert_eq!(game.actions.consecutive_freeze_actions, 0);
        for trajectory in game.trajectories {
            for step in &trajectory.steps {
                assert!(step.obs.mask[step.action]);
            }
            for pair in trajectory.steps.windows(2) {
                if pair[0].action == A_FREEZE {
                    assert!(!pair[1].obs.mask[A_FREEZE]);
                }
            }
        }
    }
    for sample in [false, true] {
        let g = eval::play_detailed(
            &net,
            10_000_000,
            &cfg,
            sample,
            eval::Opponent::Teacher,
            enabled(),
        )
        .unwrap();
        assert_eq!(g.diagnostics.actions.consecutive_freeze_actions, 0);
    }
}

#[test]
fn fork_preserves_weights_adam_rng_and_counters_and_old_metadata_defaults_off() {
    let dir = std::env::temp_dir().join(format!("ppo-freeze-fork-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let parent = dir.join("parent.npz");
    let child = dir.join("child.npz");
    train::train(train::TrainArgs {
        device: 0,
        iterations: 1,
        workers: 1,
        out: parent.clone(),
        save_every: 0,
        options: train::TrainOptions {
            games: 1,
            hidden: 16,
            blocks: 1,
            embedding_dim: 2,
            max_turns: 2,
            batch: 64,
            ..Default::default()
        },
    })
    .unwrap();
    // Recreate the old schema: policy_rules and backend were absent.
    let mut npz = tavernagent_v2::npz::Npz::load(&parent).unwrap();
    // Preserve the exact u128 RNG state; serde_json::Value would round it via f64.
    let meta = npz
        .get("trainer_meta")
        .unwrap()
        .as_text()
        .unwrap()
        .replacen(
            &format!(
                ",\"policy_rules\":{}",
                serde_json::to_string(&PolicyRules::default()).unwrap()
            ),
            "",
            1,
        )
        .replacen("\"backend\":\"cpu\",", "", 1);
    // Check the options object, since historical diagnostic rows also contain rules.
    let parsed: serde_json::Value = serde_json::from_str(&meta).unwrap();
    assert!(parsed["state"]["options"].get("policy_rules").is_none());
    assert!(parsed["state"]["options"].get("backend").is_none());
    npz.insert("trainer_meta", tavernagent_v2::npz::NpyArray::text(&meta));
    npz.save(&parent).unwrap();
    let (before, old) = checkpoint::load(&parent).unwrap();
    assert!(!old.options.policy_rules.block_consecutive_freeze);
    assert_eq!(old.options.backend, tavernagent_v2::backend::Backend::Cpu);
    train::fork(train::ForkArgs {
        checkpoint: parent.clone(),
        out: child.clone(),
        block_consecutive_freeze: Some(true),
        ..Default::default()
    })
    .unwrap();
    let (after, new) = checkpoint::load(&child).unwrap();
    assert_eq!(before.net.p.tensors, after.net.p.tensors);
    assert_eq!(before.m.tensors, after.m.tensors);
    assert_eq!(before.v.tensors, after.v.tensors);
    assert_eq!(before.optimizer_step, after.optimizer_step);
    assert_eq!(
        (old.iteration, old.games, old.transitions),
        (new.iteration, new.games, new.transitions)
    );
    assert_eq!(
        old.shuffle_rng.clone().random::<u64>(),
        new.shuffle_rng.clone().random::<u64>()
    );
    assert!(new.options.policy_rules.block_consecutive_freeze);
    assert!(
        train::fork(train::ForkArgs {
            checkpoint: parent.clone(),
            out: child.clone(),
            block_consecutive_freeze: Some(false),
            ..Default::default()
        })
        .is_err()
    );
    train::resume(train::ResumeArgs {
        backend: None,
        device: 0,
        checkpoint: child.clone(),
        iterations: 2,
        workers: 2,
        out: None,
        save_every: 0,
    })
    .unwrap();
    let (_, continued) = checkpoint::load(&child).unwrap();
    assert!(continued.options.policy_rules.block_consecutive_freeze);
    assert_eq!(
        continued.history.last().unwrap()["actions"]["consecutive_freeze_actions"],
        0
    );
    let tuned = dir.join("tuned.npz");
    let source_bytes = std::fs::read(&child).unwrap();
    train::fork(train::ForkArgs {
        checkpoint: child.clone(),
        out: tuned.clone(),
        lr: Some(8e-5),
        games: Some(2),
        epochs: Some(2),
        entropy_coef: Some(0.025),
        gae_lambda: Some(0.965),
        gae_within_turn_lambda: Some(0.98),
        freeze_at_end: Some(true),
        terminal_value_coef: Some(0.05),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(source_bytes, std::fs::read(&child).unwrap());
    let source_npz = tavernagent_v2::npz::Npz::load(&child).unwrap();
    let tuned_npz = tavernagent_v2::npz::Npz::load(&tuned).unwrap();
    let (source_trainer, source_state) = checkpoint::load(&child).unwrap();
    let (tuned_trainer, mut tuned_state) = checkpoint::load(&tuned).unwrap();
    assert_eq!(source_trainer.net.p.tensors, tuned_trainer.net.p.tensors);
    assert_eq!(source_trainer.m.tensors, tuned_trainer.m.tensors);
    assert_eq!(source_trainer.v.tensors, tuned_trainer.v.tensors);
    assert_eq!(source_trainer.optimizer_step, tuned_trainer.optimizer_step);
    assert_eq!(
        source_npz.get("meta").unwrap(),
        tuned_npz.get("meta").unwrap()
    );
    assert!(tuned_state.options.policy_rules.block_consecutive_freeze);
    assert!(tuned_state.options.policy_rules.freeze_at_end);
    assert_eq!(tuned_state.options.lr, 8e-5);
    assert_eq!(tuned_state.options.games, 2);
    assert_eq!(tuned_state.options.epochs, 2);
    assert_eq!(tuned_state.options.ppo.entropy_coef, 0.025);
    assert_eq!(tuned_state.options.ppo.terminal_value_coef, 0.05);
    assert_eq!(tuned_state.options.gae_lambda, 0.965);
    assert_eq!(tuned_state.options.gae_within_turn_lambda, Some(0.98));
    tuned_state.options = source_state.options.clone();
    assert_eq!(
        serde_json::to_string(&source_state).unwrap(),
        serde_json::to_string(&tuned_state).unwrap()
    );
    let cleared = dir.join("cleared.npz");
    train::fork(train::ForkArgs {
        checkpoint: tuned,
        out: cleared.clone(),
        clear_gae_within_turn_lambda: true,
        ..Default::default()
    })
    .unwrap();
    let (_, state) = checkpoint::load(&cleared).unwrap();
    assert_eq!(state.options.gae_within_turn_lambda, None);
    assert_eq!(state.options.gae_lambda, 0.965);
    assert!(state.options.policy_rules.block_consecutive_freeze);
    assert!(state.options.policy_rules.freeze_at_end);
    train::resume(train::ResumeArgs {
        backend: None,
        device: 0,
        checkpoint: cleared.clone(),
        iterations: 3,
        workers: 2,
        out: None,
        save_every: 0,
    })
    .unwrap();
    let (_, resized) = checkpoint::load(&cleared).unwrap();
    assert_eq!(resized.iteration, source_state.iteration + 1);
    assert_eq!(resized.games, source_state.games + 2);
    assert_eq!(resized.history.last().unwrap()["games"], 2);
    assert_eq!(resized.options.epochs, 2);
    let invalid = dir.join("invalid.npz");
    assert!(
        train::fork(train::ForkArgs {
            checkpoint: child,
            out: invalid.clone(),
            lr: Some(f64::NAN),
            ..Default::default()
        })
        .is_err()
    );
    assert!(!invalid.exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn identical_policies_have_neutral_paired_arena_score() {
    let options = train::TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 3,
        ..Default::default()
    };
    let net = PolicyValueNet::new(42, options.net_config()).unwrap();
    let cfg = options.game_config().unwrap();
    for offset in 0..4 {
        let a = arena::play(&net, &net, &cfg, enabled(), enabled(), 17, offset).unwrap();
        let b = arena::play(&net, &net, &cfg, enabled(), enabled(), 17, offset + 4).unwrap();
        assert_eq!(a.placements, b.placements);
        assert_eq!(a.candidate_mean_placement + b.candidate_mean_placement, 9.0);
        assert_ne!(a.candidate_won, b.candidate_won);
        assert!(
            a.diagnostics
                .iter()
                .all(|d| d.actions.consecutive_freeze_actions == 0)
        );
    }
}
