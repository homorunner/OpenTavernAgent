//! Training data, the ended-position value rows, replay persistence and exact resume.

use std::collections::HashSet;
use std::path::PathBuf;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{self, A_END_TURN, Evaluator, Observation, SearchOptions, observe_seat};
use tavernagent::net::{
    BUDGET_FEATURE_START, ENDED_TURN_FEATURE, N_ACTIONS, NetConfig, PolicyValueNet,
};
use tavernagent::npz::Npz;
use tavernagent::rules::Budgets;
use tavernagent::selfplay::{
    OptimizerHparams, PlayOptions, ReplayWindow, Sample, Stats, StepOptions, Trainer,
    TrainingState, cosine_lr, ended_sample, load_training_state, play_one, save_training_state,
    subsample,
};
use tavernagent::train::{self, SelfplayArgs};
use taverngym::Game;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tavernagent-train-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn batch() -> (Game, Sample) {
    let mut game = Game::new(79, Budgets::default().config().unwrap());
    select_heroes(&mut game, 79).unwrap();
    let obs = observe_seat(&game, 0);
    let legal = obs.mask.iter().filter(|m| **m).count() as f32;
    let pi = obs
        .mask
        .iter()
        .map(|m| if *m { 1.0 / legal } else { 0.0 })
        .collect();
    let sample = Sample {
        x: obs.x,
        cards: obs.cards,
        mask: obs.mask,
        pi,
        z: 1.0,
        action: -1,
        combat: 0.0,
        combat_known: false,
    };
    (game, sample)
}

fn small() -> NetConfig {
    NetConfig {
        embedding_dim: 4,
        ..NetConfig::default()
    }
}

#[test]
fn ended_sample_is_the_preview_of_end_turn_with_no_policy_target() {
    let (game, decision) = batch();
    let before = serde_json::to_string(&game).unwrap();
    let end = ended_sample(&game, 0).unwrap();
    assert_eq!(
        serde_json::to_string(&game).unwrap(),
        before,
        "the real game is untouched"
    );
    assert!(end.is_ended() && end.pi.iter().all(|p| *p == 0.0) && end.action == -1);
    assert_eq!(decision.x[ENDED_TURN_FEATURE], 0.0);
    assert_eq!(end.x[ENDED_TURN_FEATURE], 1.0);
    // Identical to the real End Turn up to the action budget, which only `apply` counts.
    let mut real = game.clone();
    real.apply_index(0, A_END_TURN).unwrap();
    let real_obs = observe_seat(&real, 0);
    assert_eq!(
        end.x[..BUDGET_FEATURE_START],
        real_obs.x[..BUDGET_FEATURE_START]
    );
    assert_eq!(end.cards, real_obs.cards);
}

#[test]
fn ended_rows_train_the_value_head_but_not_the_policy() {
    let (game, decision) = batch();
    let mut end = ended_sample(&game, 0).unwrap();
    end.z = -1.0;
    let mut trainer = Trainer::new(
        PolicyValueNet::new(4, small()).unwrap(),
        OptimizerHparams {
            wd: 0.0,
            ..Default::default()
        },
    );
    let policy_before = (
        trainer.net.p.tensors[trainer.net.p.wp()].clone(),
        trainer.net.p.tensors[trainer.net.p.bp()].clone(),
    );
    trainer
        .step(
            &[&end],
            0.001,
            StepOptions {
                pg_weight: 0.0,
                smoothing: 0.05,
            },
        )
        .unwrap();
    let m = &trainer.last_metrics;
    assert_eq!(m["ended_samples"], 1.0);
    assert_eq!(m["policy_training_samples"], 0.0);
    assert_eq!(m["policy_head_grad_norm"], 0.0);
    assert!(m["value_head_grad_norm"] > 0.0);
    assert_eq!(m["policy_entropy"], 0.0);
    assert_eq!(m["target_kl"], 0.0);
    assert_eq!(trainer.net.p.tensors[trainer.net.p.wp()], policy_before.0);
    assert_eq!(trainer.net.p.tensors[trainer.net.p.bp()], policy_before.1);
    // Mixed rows: the policy gradient is normalized by the policy rows alone.
    let mut solo = Trainer::new(
        PolicyValueNet::new(4, small()).unwrap(),
        OptimizerHparams {
            wd: 0.0,
            ..Default::default()
        },
    );
    solo.step(&[&decision], 0.001, StepOptions::default())
        .unwrap();
    let mut mixed = Trainer::new(
        PolicyValueNet::new(4, small()).unwrap(),
        OptimizerHparams {
            wd: 0.0,
            ..Default::default()
        },
    );
    mixed
        .step(&[&decision, &end], 0.001, StepOptions::default())
        .unwrap();
    let rel = (mixed.last_metrics["policy_head_grad_norm"]
        - solo.last_metrics["policy_head_grad_norm"])
        .abs()
        / solo.last_metrics["policy_head_grad_norm"];
    assert!(rel < 1e-4, "{rel}");
    assert!(
        (mixed.last_metrics["policy_entropy"] - solo.last_metrics["policy_entropy"]).abs() < 1e-5
    );
    // Validation of the batch itself.
    let mut tagged = end.clone();
    tagged.pi = decision.pi.clone();
    assert!(
        mixed
            .step(&[&tagged], 0.001, StepOptions::default())
            .unwrap_err()
            .contains("ended positions")
    );
    let mut illegal = decision.clone();
    illegal.action = decision.mask.iter().position(|m| !*m).unwrap() as i64;
    assert!(
        mixed
            .step(&[&illegal], 0.001, StepOptions::default())
            .unwrap_err()
            .contains("legal")
    );
}

/// Prefers End Turn when it is legal; otherwise the first legal action.
struct EndOrFirst;

impl Evaluator for EndOrFirst {
    fn evaluate(&self, obs: &Observation) -> Result<(Vec<f32>, f32), String> {
        let mut prior = vec![0.0; N_ACTIONS];
        if obs.mask[A_END_TURN] {
            prior[A_END_TURN] = 1.0;
        } else if let Some(a) = obs.mask.iter().position(|m| *m) {
            prior[a] = 1.0;
        }
        Ok((prior, 0.0))
    }
}

fn play_options(sims: usize, end_explore: f32) -> PlayOptions {
    PlayOptions {
        search: SearchOptions {
            sims,
            ..SearchOptions::default()
        },
        temperature: 0.0,
        end_explore,
        budgets: Budgets::default(),
        combat_sims: 0,
    }
}

#[test]
fn self_play_records_one_ended_position_per_end_turn() {
    let (samples, _) = play_one(&EndOrFirst, 31, &play_options(2, 0.0)).unwrap();
    let ended = samples.iter().filter(|s| s.is_ended()).count();
    let ends = samples
        .iter()
        .filter(|s| s.action == A_END_TURN as i64)
        .count();
    assert_eq!(ended, ends);
    assert!(ended > 0);
    for (i, s) in samples.iter().enumerate() {
        if s.action == A_END_TURN as i64 {
            let next = &samples[i + 1];
            assert!(next.is_ended() && next.z == s.z && next.x[ENDED_TURN_FEATURE] == 1.0);
        } else if !s.is_ended() {
            assert_eq!(s.x[ENDED_TURN_FEATURE], 0.0);
        }
    }
    assert!(samples.iter().all(|s| (-1.0..=1.0).contains(&s.z)));
    assert!(
        play_one(&EndOrFirst, 31, &play_options(1, 1.5))
            .unwrap_err()
            .contains("probability")
    );
    // Exploration changes what is played, not the target.
    let (explored, _) = play_one(&EndOrFirst, 31, &play_options(2, 1.0)).unwrap();
    assert!(
        explored
            .iter()
            .filter(|s| !s.is_ended() && s.mask[A_END_TURN])
            .all(|s| s.action == A_END_TURN as i64)
    );
}

#[test]
fn replay_window_evicts_old_rounds_and_draws_without_replacement() {
    let (_, s) = batch();
    let tagged = |z: f32| Sample { z, ..s.clone() };
    let mut replay = ReplayWindow::new(2).unwrap();
    let mut rng = Pcg64Mcg::seed_from_u64(7);
    replay.add(vec![tagged(1.0), tagged(2.0)]);
    assert_eq!(
        replay
            .draw(2, &mut rng)
            .iter()
            .map(|s| s.z)
            .collect::<Vec<_>>(),
        vec![1.0, 2.0]
    );
    replay.add(vec![tagged(3.0), tagged(4.0)]);
    replay.add(vec![tagged(5.0), tagged(6.0)]);
    let mut drawn: Vec<f32> = replay.draw(4, &mut rng).iter().map(|s| s.z).collect();
    drawn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(drawn, vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(replay.draw(3, &mut rng).len(), 3);
    assert!((cosine_lr(0.001, 0.0001, 199, 100) - 0.0001).abs() < 1e-12);
    assert!(ReplayWindow::new(0).is_err());
    let many: Vec<Sample> = (0..10).map(|i| tagged(i as f32)).collect();
    let kept = subsample(many, 4, 1);
    assert_eq!(kept.len(), 4);
    assert!(
        kept.windows(2).all(|w| w[0].z < w[1].z),
        "order is preserved"
    );
}

#[test]
fn full_trainer_snapshot_restores_optimizer_rng_and_replay_exactly() {
    let dir = tmp("snapshot");
    let (game, decision) = batch();
    let mut end = ended_sample(&game, 0).unwrap();
    end.z = -0.5;
    let hparams = OptimizerHparams {
        wd: 0.02,
        clip: 2.0,
        value_weight: 0.25,
        combat_weight: 0.3,
        ema_decay: 0.995,
    };
    let mut trainer = Trainer::new(PolicyValueNet::new(12, small()).unwrap(), hparams.clone());
    trainer
        .step(&[&decision, &end], 3e-4, StepOptions::default())
        .unwrap();
    let mut rng = Pcg64Mcg::seed_from_u64(10);
    let replay_rng = Pcg64Mcg::seed_from_u64(11);
    let _ = rand::RngCore::next_u64(&mut rng);
    let mut replay = ReplayWindow::new(3).unwrap();
    let labelled = Sample {
        combat: 0.35,
        combat_known: true,
        ..decision.clone()
    };
    replay.add(vec![decision.clone()]);
    replay.add(vec![end.clone(), labelled]);
    let path = dir.join("state.npz");
    let options = serde_json::json!({"replay_rounds": 3});
    save_training_state(
        &path,
        &TrainingState {
            trainer: &trainer,
            step: 1,
            train_rng: &rng,
            replay_rng: &replay_rng,
            history: &[],
            stats: &Stats::default(),
            options,
            replay: Some(&replay),
        },
    )
    .unwrap();
    // A network with another configuration is refused.
    let mut other = Trainer::new(
        PolicyValueNet::new(1, NetConfig::default()).unwrap(),
        hparams.clone(),
    );
    assert!(
        load_training_state(
            &path,
            &mut other,
            &mut rng.clone(),
            &mut replay_rng.clone(),
            None
        )
        .unwrap_err()
        .contains("schema")
    );
    let mut restored = Trainer::new(
        PolicyValueNet::new(99, small()).unwrap(),
        OptimizerHparams::default(),
    );
    let mut rng2 = Pcg64Mcg::seed_from_u64(1);
    let mut replay_rng2 = Pcg64Mcg::seed_from_u64(2);
    let mut restored_replay = ReplayWindow::new(3).unwrap();
    let meta = load_training_state(
        &path,
        &mut restored,
        &mut rng2,
        &mut replay_rng2,
        Some(&mut restored_replay),
    )
    .unwrap();
    assert_eq!(meta.replay_rounds_stored, 2);
    assert_eq!(restored.t, trainer.t);
    assert_eq!(restored.hparams, hparams);
    assert_eq!(rng2, rng);
    assert_eq!(replay_rng2, replay_rng);
    assert_eq!(restored_replay.rounds.len(), 2);
    for (a, b) in replay.rounds.iter().zip(&restored_replay.rounds) {
        assert_eq!(a, b);
    }
    // A smaller window on resume keeps the newest rounds.
    let mut small_window = ReplayWindow::new(1).unwrap();
    load_training_state(
        &path,
        &mut restored,
        &mut rng2,
        &mut replay_rng2,
        Some(&mut small_window),
    )
    .unwrap();
    assert_eq!(small_window.rounds.len(), 1);
    assert_eq!(small_window.rounds[0].len(), 2);
    // The next update is identical.
    trainer
        .step(&[&decision, &end], 3e-4, StepOptions::default())
        .unwrap();
    restored
        .step(&[&decision, &end], 3e-4, StepOptions::default())
        .unwrap();
    assert_eq!(trainer.net.p, restored.net.p);
    assert_eq!(trainer.m, restored.m);
    assert_eq!(trainer.v, restored.v);
    assert_eq!(trainer.ema, restored.ema);
    // Snapshots from another search protocol or format are refused.
    let mut npz = Npz::load(&path).unwrap();
    let meta_text = npz.get("metadata").unwrap().as_text().unwrap();
    let wrong = meta_text.replace(mcts::SEARCH_VERSION, "sampled-prep-v2");
    npz.insert("metadata", tavernagent::npz::NpyArray::text(&wrong));
    let wrong_path = dir.join("wrong.npz");
    npz.save(&wrong_path).unwrap();
    assert!(
        load_training_state(
            &wrong_path,
            &mut restored,
            &mut rng2,
            &mut replay_rng2,
            None
        )
        .unwrap_err()
        .contains("schema")
    );
    let mut truncated = Npz::load(&path).unwrap();
    truncated.arrays.remove("replay/1/combat_known");
    truncated.save(&wrong_path).unwrap();
    assert!(
        load_training_state(
            &wrong_path,
            &mut restored,
            &mut rng2,
            &mut replay_rng2,
            Some(&mut restored_replay)
        )
        .unwrap_err()
        .contains("incomplete replay window")
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

fn selfplay_args(out: PathBuf, steps: usize) -> SelfplayArgs {
    SelfplayArgs {
        steps,
        games: 1,
        sims: 2,
        workers: 1,
        max_samples: 40,
        batch: 16,
        embedding_dim: 4,
        probe_every: 0,
        save_trainer_state: true,
        replay_rounds: 2,
        out,
        ..SelfplayArgs::default()
    }
}

#[test]
fn training_resumes_exactly_across_the_replay_window() {
    let dir = tmp("resume");
    let full = selfplay_args(dir.join("full.npz"), 3);
    train::run(full, &HashSet::new()).unwrap();
    let split = selfplay_args(dir.join("split.npz"), 2);
    train::run(split, &HashSet::new()).unwrap();
    // Resume with a different value on the command line for a non-resumed option: it must
    // be inherited from the snapshot unless given explicitly.
    let mut resumed = selfplay_args(dir.join("ignored.npz"), 3);
    resumed.resume_state = Some(dir.join("split.trainer.npz"));
    resumed.max_samples = 999; // not in `provided`, so the saved 40 applies
    train::run(
        resumed,
        &HashSet::from(["steps".to_string(), "resume_state".to_string()]),
    )
    .unwrap();
    let a = PolicyValueNet::load(&dir.join("full.npz")).unwrap();
    let b = PolicyValueNet::load(&dir.join("split.npz")).unwrap();
    assert_eq!(a.p, b.p);
    let history: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("split.history.json")).unwrap())
            .unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[2]["replay_rounds"], 2);
    assert!(history[2]["ended_samples"].as_f64().unwrap() >= 0.0);
    // Resuming without a target beyond the saved step is an error.
    let mut stale = selfplay_args(dir.join("stale.npz"), 3);
    stale.resume_state = Some(dir.join("split.trainer.npz"));
    assert!(train::run(stale, &HashSet::from(["resume_state".to_string()])).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ema_window_is_defined_in_rounds_and_warms_up() {
    use tavernagent::selfplay::{ema_decay_at, ema_decay_for_window};
    let args = SelfplayArgs {
        max_samples: 65536,
        batch: 128,
        epochs: 1,
        ema_rounds: 2.0,
        ..SelfplayArgs::default()
    };
    assert_eq!(args.updates_per_round(), 512);
    assert!((args.hparams().ema_decay - (1.0 - 1.0 / 1024.0) as f32).abs() < 1e-7);
    assert_eq!(ema_decay_for_window(1024.0), args.hparams().ema_decay);
    // Early updates follow the weights; the target decay takes over later.
    assert!((ema_decay_at(0.999, 1) - 2.0 / 11.0).abs() < 1e-6);
    assert!(ema_decay_at(0.999, 100) < 0.999);
    assert_eq!(ema_decay_at(0.999, 100_000), 0.999);
    // One optimizer step: EMA = d1 * initial + (1 - d1) * updated, with d1 = 2/11.
    let (game, decision) = batch();
    let _ = game;
    let net = PolicyValueNet::new(3, small()).unwrap();
    let initial = net.p.clone();
    let mut trainer = Trainer::new(
        net,
        OptimizerHparams {
            ema_decay: 0.999,
            ..OptimizerHparams::default()
        },
    );
    trainer
        .step(&[&decision], 1e-3, StepOptions::default())
        .unwrap();
    assert!((trainer.last_metrics["ema_decay"] - 2.0 / 11.0).abs() < 1e-6);
    let d1 = 2.0f32 / 11.0;
    for i in 0..initial.tensors.len() {
        for j in 0..initial.tensors[i].data.len() {
            let expected =
                d1 * initial.tensors[i].data[j] + (1.0 - d1) * trainer.net.p.tensors[i].data[j];
            assert!((trainer.ema.tensors[i].data[j] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn combat_labels_are_simulated_matchup_means_and_train_only_the_combat_head_where_known() {
    use tavernagent::mcts::bot_action;
    use tavernagent::selfplay::{combat_label, end_turn_preview};
    // Drive a game with the bot until the last seat is about to end turn 1.
    let mut game = Game::new(41, Budgets::default().config().unwrap());
    select_heroes(&mut game, 41).unwrap();
    let seat = 'outer: loop {
        for seat in 0..game.players.len() {
            if !game.needs_action(seat) {
                continue;
            }
            let action = bot_action(&mut game, seat);
            let others_done = (0..game.players.len())
                .all(|p| p == seat || !game.players[p].alive || game.players[p].ended_turn);
            if action == taverngym::Action::EndTurn && others_done {
                break 'outer seat;
            }
            game.apply(seat, action).unwrap();
        }
    };
    let full = end_turn_preview(&game, seat).unwrap();
    assert!(full.players.iter().all(|p| p.ended_turn));
    let mut rng = Pcg64Mcg::seed_from_u64(3);
    for s in 0..full.players.len() {
        let label = combat_label(&full, s, &full.players[s], 20, &mut rng).unwrap();
        assert!((-1.0..=1.0).contains(&label));
        assert!(
            (label * 20.0).round() == label * 20.0,
            "a mean of 20 outcomes"
        );
    }
    // The label follows the seat's own final state: an empty board always loses to a
    // non-empty one and ties an empty one; the real game is untouched.
    let before = serde_json::to_string(&full).unwrap();
    let (a, b) = full.pairs[0];
    let b = b.expect("eight live players pair up");
    let mut empty = full.players[a].clone();
    empty.board.clear();
    let mut empty_opponent = full.clone();
    empty_opponent.players[b].board.clear();
    assert_eq!(combat_label(&full, a, &empty, 5, &mut rng), Some(-1.0));
    assert_eq!(
        combat_label(&empty_opponent, a, &full.players[a], 5, &mut rng),
        Some(1.0)
    );
    assert_eq!(
        combat_label(&empty_opponent, a, &empty, 5, &mut rng),
        Some(0.0)
    );
    assert_eq!(serde_json::to_string(&full).unwrap(), before);
    assert_eq!(combat_label(&full, a, &full.players[a], 0, &mut rng), None);

    // Self-play attaches the label to every position of a turn whose fight happened, and
    // the End Turn decision shares its label with its ended row.
    let mut opts = play_options(2, 0.0);
    opts.combat_sims = 4;
    let (samples, _) = play_one(&EndOrFirst, 31, &opts).unwrap();
    let known = samples.iter().filter(|s| s.combat_known).count();
    assert!(
        known > samples.len() / 2,
        "{known} of {} labelled",
        samples.len()
    );
    assert!(
        samples
            .iter()
            .filter(|s| s.combat_known)
            .all(|s| (-1.0..=1.0).contains(&s.combat))
    );
    for (i, s) in samples.iter().enumerate() {
        if s.action == A_END_TURN as i64 && s.combat_known {
            assert!(samples[i + 1].is_ended());
            assert_eq!(samples[i + 1].combat, s.combat);
        }
    }

    // Training: rows without a label leave the combat head alone.
    let (_, decision) = batch();
    let mut unknown = Trainer::new(
        PolicyValueNet::new(4, small()).unwrap(),
        OptimizerHparams::default(),
    );
    unknown
        .step(&[&decision], 1e-3, StepOptions::default())
        .unwrap();
    assert_eq!(unknown.last_metrics["combat_training_samples"], 0.0);
    assert_eq!(unknown.last_metrics["combat_head_grad_norm"], 0.0);
    assert_eq!(unknown.last_metrics["combat_loss"], 0.0);
    let labelled = Sample {
        combat: -0.6,
        combat_known: true,
        ..decision.clone()
    };
    let mut known = Trainer::new(
        PolicyValueNet::new(4, small()).unwrap(),
        OptimizerHparams::default(),
    );
    known
        .step(&[&labelled], 1e-3, StepOptions::default())
        .unwrap();
    assert_eq!(known.last_metrics["combat_training_samples"], 1.0);
    assert!(known.last_metrics["combat_head_grad_norm"] > 0.0);
    assert!(known.last_metrics["combat_loss"] > 0.0);
}
