//! Training data, the ended-position value rows, replay persistence and exact resume.

use std::collections::HashSet;
use std::path::PathBuf;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{self, A_END_TURN, Evaluator, Observation, SearchOptions, observe_seat};
use tavernagent::net::{ENDED_TURN_FEATURE, N_ACTIONS, NetConfig, PolicyValueNet};
use tavernagent::npz::Npz;
use tavernagent::rules::Budgets;
use tavernagent::selfplay::{
    OptimizerHparams, PlayOptions, ReplayWindow, Sample, Stats, StepOptions, Trainer,
    TrainingState, cosine_lr, ended_sample, load_training_state, play_one, save_training_state,
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
        weight: 1.0,
        action: -1,
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
    // Identical to the real End Turn.
    let mut real = game.clone();
    real.apply_index(0, A_END_TURN).unwrap();
    let real_obs = observe_seat(&real, 0);
    assert_eq!(end.x, real_obs.x);
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
        search_prob: 1.0,
        split_turn: Vec::new(),
    }
}

/// A self-play game is a function of its seed alone: the pool that runs the ply's
/// concurrent seat searches, and its thread count, change nothing.
#[test]
fn self_play_does_not_depend_on_the_search_pool() {
    let opts = play_options(2, 0.0);
    let global = play_one(&EndOrFirst, 31, &opts, None).unwrap();
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let serial = play_one(&EndOrFirst, 31, &opts, Some(&single)).unwrap();
    assert_eq!(serial.1, global.1);
    assert_eq!(serial.0.len(), global.0.len());
    for (a, b) in serial.0.iter().zip(&global.0) {
        assert_eq!(a.action, b.action);
        assert_eq!(a.mask, b.mask);
        assert_eq!(a.pi, b.pi);
        assert_eq!(a.z, b.z);
        assert!(a.is_ended() || a.mask[a.action as usize]);
    }
    let mut unreachable = opts.clone();
    unreachable.split_turn = vec![u32::MAX];
    assert_eq!(
        play_one(&EndOrFirst, 31, &unreachable, Some(&single)).unwrap(),
        global
    );
    let mut tree = opts;
    tree.split_turn = vec![4, 8];
    assert_eq!(
        play_one(&EndOrFirst, 31, &tree, Some(&single)).unwrap(),
        play_one(&EndOrFirst, 31, &tree, None).unwrap()
    );
}

#[test]
fn self_play_records_one_ended_position_per_end_turn() {
    let (samples, _) = play_one(&EndOrFirst, 31, &play_options(2, 0.0), None).unwrap();
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
        play_one(&EndOrFirst, 31, &play_options(1, 1.5), None)
            .unwrap_err()
            .contains("probability")
    );
    // Exploration changes what is played, not the target.
    let (explored, _) = play_one(&EndOrFirst, 31, &play_options(2, 1.0), None).unwrap();
    assert!(
        explored
            .iter()
            .filter(|s| !s.is_ended() && s.mask[A_END_TURN])
            .all(|s| s.action == A_END_TURN as i64)
    );
}

/// Only searched decisions become training rows; the ended positions are kept either
/// way, and the split is a function of the seed.
#[test]
fn unsearched_decisions_play_the_prior_and_leave_no_policy_row() {
    let full = play_options(2, 0.0);
    let mut none = play_options(2, 0.0);
    none.search_prob = 0.0;
    let mut half = play_options(2, 0.0);
    half.search_prob = 0.5;
    let (all, _) = play_one(&EndOrFirst, 31, &full, None).unwrap();
    let (direct, _) = play_one(&EndOrFirst, 31, &none, None).unwrap();
    let (mixed, _) = play_one(&EndOrFirst, 31, &half, None).unwrap();
    let policy_rows = |s: &[Sample]| s.iter().filter(|r| !r.is_ended()).count();
    let ended_rows = |s: &[Sample]| s.iter().filter(|r| r.is_ended()).count();
    assert!(policy_rows(&all) > 0);
    assert_eq!(policy_rows(&direct), 0);
    assert!(ended_rows(&direct) > 0);
    assert!(policy_rows(&mixed) > 0 && policy_rows(&mixed) < policy_rows(&all));
    assert!(
        mixed
            .iter()
            .all(|s| s.is_ended() || s.mask[s.action as usize])
    );
    let (again, _) = play_one(&EndOrFirst, 31, &half, None).unwrap();
    assert_eq!(again.len(), mixed.len());
    assert!(
        again
            .iter()
            .zip(&mixed)
            .all(|(a, b)| a.action == b.action && a.pi == b.pi)
    );
    let mut bad = play_options(2, 0.0);
    bad.search_prob = 1.5;
    assert!(
        play_one(&EndOrFirst, 31, &bad, None)
            .unwrap_err()
            .contains("search_prob")
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
    for index in [0, 1, 199] {
        assert_eq!(cosine_lr(0.001, 0.0001, index, 0), 0.001);
    }
    assert!(ReplayWindow::new(0).is_err());
}

#[test]
fn full_trainer_snapshot_restores_optimizer_rng_and_replay_exactly() {
    let dir = tmp("snapshot");
    let (game, decision) = batch();
    let mut end = ended_sample(&game, 0).unwrap();
    end.z = -0.5;
    end.weight = 0.55;
    let hparams = OptimizerHparams {
        wd: 0.02,
        clip: 2.0,
        value_weight: 0.25,
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
    replay.add(vec![decision.clone()]);
    replay.add(vec![end.clone(), decision.clone()]);
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
    truncated.arrays.remove("replay/1/action");
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
    let mut missing_weights = Npz::load(&path).unwrap();
    for i in 0..2 {
        missing_weights.arrays.remove(&format!("replay/{i}/weight"));
    }
    missing_weights.save(&wrong_path).unwrap();
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
    // An old format is rejected even with all current arrays present.
    let mut old_format = Npz::load(&path).unwrap();
    let meta = meta_text.replace(
        &format!("\"format\":{}", tavernagent::selfplay::TRAINER_FORMAT),
        "\"format\":7",
    );
    old_format.insert("metadata", tavernagent::npz::NpyArray::text(&meta));
    old_format.save(&wrong_path).unwrap();
    assert!(
        load_training_state(
            &wrong_path,
            &mut restored,
            &mut rng2,
            &mut replay_rng2,
            Some(&mut restored_replay),
        )
        .unwrap_err()
        .contains("schema")
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
        replay_rounds: Some(2),
        out,
        ..SelfplayArgs::default()
    }
}

#[test]
fn scheduled_replay_resumes_exactly_across_a_growth_boundary() {
    let dir = tmp("scheduled-resume");
    let (_, decision) = batch();
    let mut replay = ReplayWindow::new(4).unwrap();
    for _ in 0..4 {
        replay.add(vec![decision.clone()]);
    }
    let args = |name: &str, steps| SelfplayArgs {
        hidden: 16,
        blocks: 1,
        replay_rounds: None,
        ..selfplay_args(dir.join(format!("{name}.npz")), steps)
    };
    let trainer = Trainer::new(
        PolicyValueNet::new(12, args("full", 401).net_config()).unwrap(),
        args("full", 401).hparams(),
    );
    let rng = Pcg64Mcg::seed_from_u64(10);
    let replay_rng = Pcg64Mcg::seed_from_u64(11);
    for name in ["full", "split"] {
        save_training_state(
            &dir.join(format!("{name}.trainer.npz")),
            &TrainingState {
                trainer: &trainer,
                step: 398,
                train_rng: &rng,
                replay_rng: &replay_rng,
                history: &[],
                stats: &Stats::default(),
                options: serde_json::to_value(args(name, 401)).unwrap(),
                replay: Some(&replay),
            },
        )
        .unwrap();
    }
    for (name, steps) in [("full", 401), ("split", 399), ("split", 401)] {
        let mut run = args(name, steps);
        run.resume_state = Some(dir.join(format!("{name}.trainer.npz")));
        train::run(run, &HashSet::from(["steps".into(), "resume_state".into()])).unwrap();
    }
    assert_eq!(
        PolicyValueNet::load(&dir.join("full.npz")).unwrap().p,
        PolicyValueNet::load(&dir.join("split.npz")).unwrap().p
    );
    for name in ["full", "split"] {
        let history: Vec<serde_json::Value> = serde_json::from_slice(
            &std::fs::read(dir.join(format!("{name}.history.json"))).unwrap(),
        )
        .unwrap();
        for (row, expected) in history.iter().zip([4, 5, 5]) {
            assert_eq!(row["replay_capacity"], expected);
            assert_eq!(row["replay_rounds"], expected);
        }
        assert_eq!(history.len(), 3);
    }
    // A fixed smaller override drops oldest rounds; an invalid resize changes nothing.
    replay.set_capacity(2).unwrap();
    assert_eq!(replay.rounds.len(), 2);
    assert!(replay.set_capacity(0).is_err());
    assert_eq!(replay.capacity, 2);
    std::fs::remove_dir_all(dir).unwrap();
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
    // Every self-play decision enters the replay window; `max_samples` (40 here) only
    // bounds the draw each epoch trains on.
    for round in &history {
        assert_eq!(round["fresh_samples"], round["rollout_decisions"]);
        assert!(round["samples"].as_u64().unwrap() <= 40);
    }
    // Resuming without a target beyond the saved step is an error.
    let mut stale = selfplay_args(dir.join("stale.npz"), 3);
    stale.resume_state = Some(dir.join("split.trainer.npz"));
    assert!(train::run(stale, &HashSet::from(["resume_state".to_string()])).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn split_training_resumes_with_saved_schedule_and_weights() {
    let dir = tmp("split-resume");
    let args = |name: &str, steps| SelfplayArgs {
        hidden: 16,
        blocks: 1,
        split_turn: vec![4, 8],
        ..selfplay_args(dir.join(name), steps)
    };
    train::run(args("full.npz", 2), &HashSet::new()).unwrap();
    train::run(args("split.npz", 1), &HashSet::new()).unwrap();
    let mut resumed = args("ignored.npz", 2);
    resumed.resume_state = Some(dir.join("split.trainer.npz"));
    resumed.split_turn.clear(); // Not explicitly provided: inherit the saved 4,8.
    train::run(
        resumed,
        &HashSet::from(["steps".into(), "resume_state".into()]),
    )
    .unwrap();
    let a = Npz::load(&dir.join("full.trainer.npz")).unwrap();
    let b = Npz::load(&dir.join("split.trainer.npz")).unwrap();
    for (key, value) in &a.arrays {
        if key != "metadata" {
            assert_eq!(value, b.get(key).unwrap(), "{key}");
        }
    }
    let meta = tavernagent::selfplay::read_training_meta(&dir.join("split.trainer.npz")).unwrap();
    assert_eq!(meta.options["split_turn"], serde_json::json!([4, 8]));
    let weights = b.get("replay/0/weight").unwrap().as_f32().unwrap();
    assert!(weights.contains(&1.0) && weights.contains(&0.55) && weights.contains(&(0.55 * 0.55)));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn split_turn_cli_and_required_saved_options() {
    use clap::Parser;
    use tavernagent::selfplay::validate_split_turns;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: SelfplayArgs,
    }
    assert_eq!(
        Cli::try_parse_from(["test", "--split-turn", "4,8"])
            .unwrap()
            .args
            .split_turn,
        vec![4, 8]
    );
    assert_eq!(
        Cli::try_parse_from(["test"]).unwrap().args.split_turn,
        vec![4]
    );
    assert!(
        Cli::try_parse_from(["test", "--split-turn", "--steps", "1"])
            .unwrap()
            .args
            .split_turn
            .is_empty()
    );
    assert!(Cli::try_parse_from(["test", "--split-turn", "4,no"]).is_err());
    assert!(validate_split_turns(&[3, 6, 9]).is_ok());
    for turns in [&[0][..], &[4, 4], &[8, 4]] {
        assert!(validate_split_turns(turns).is_err());
    }
    let mut saved = serde_json::to_value(SelfplayArgs::default()).unwrap();
    saved.as_object_mut().unwrap().remove("split_turn");
    assert!(serde_json::from_value::<SelfplayArgs>(saved).is_err());
}

#[test]
fn weighted_losses_and_gradients_match_repeated_samples() {
    let (game, mut first) = batch();
    let action = first.mask.iter().position(|m| *m).unwrap();
    first.action = action as i64;
    let mut second = first.clone();
    second.pi.fill(0.0);
    second.pi[action] = 1.0;
    second.z = -0.5;
    second.weight = 0.5;
    let mut end = ended_sample(&game, 0).unwrap();
    end.z = 0.25;
    end.weight = 0.5;
    let run = |rows: &[&Sample]| {
        let mut trainer = Trainer::new(
            PolicyValueNet::new(
                8,
                NetConfig {
                    hidden: 16,
                    blocks: 1,
                    ..small()
                },
            )
            .unwrap(),
            OptimizerHparams {
                clip: 1000.0,
                wd: 0.0,
                ..Default::default()
            },
        );
        let losses = trainer
            .step(
                rows,
                0.0,
                StepOptions {
                    pg_weight: 0.3,
                    smoothing: 0.05,
                },
            )
            .unwrap();
        (trainer, losses)
    };
    let (weighted, a) = run(&[&first, &second, &end]);
    second.weight = 1.0;
    end.weight = 1.0;
    let (repeated, b) = run(&[&first, &first, &second, &end]);
    assert!((a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5);
    for (a, b) in weighted.m.tensors.iter().zip(&repeated.m.tensors) {
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((x - y).abs() < 1e-6, "{x} != {y}");
        }
    }
    assert_eq!(weighted.last_metrics["sample_weight_sum"], 2.0);
    assert_eq!(weighted.last_metrics["policy_sample_weight_sum"], 1.5);
    let mut trainer = weighted;
    for bad in [f32::NAN, f32::INFINITY, -0.1, 0.0] {
        first.weight = bad;
        assert!(
            trainer
                .step(&[&first], 0.0, StepOptions::default())
                .unwrap_err()
                .contains("weight")
        );
    }
}

#[test]
fn training_step_does_not_depend_on_the_thread_count() {
    use tavernagent::selfplay::STEP_CHUNK_ROWS;
    // Several chunks' worth of rows from a few positions, so the chunked reduction runs.
    let mut samples = Vec::new();
    for seed in 0..(3 * STEP_CHUNK_ROWS as u64 + 5) {
        let mut game = Game::new(100 + seed, Budgets::default().config().unwrap());
        select_heroes(&mut game, seed).unwrap();
        let obs = observe_seat(&game, (seed % 8) as usize);
        let legal = obs.mask.iter().filter(|m| **m).count() as f32;
        let pi = obs
            .mask
            .iter()
            .map(|m| if *m { 1.0 / legal } else { 0.0 })
            .collect();
        samples.push(Sample {
            x: obs.x,
            cards: obs.cards,
            mask: obs.mask,
            pi,
            z: 1.0 - 2.0 * (seed % 8) as f32 / 7.0,
            weight: 0.55f32.powi((seed % 3) as i32),
            action: -1,
        });
    }
    let batch: Vec<&Sample> = samples.iter().collect();
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut trainer = Trainer::new(
                PolicyValueNet::new(5, small()).unwrap(),
                OptimizerHparams {
                    wd: 0.01,
                    clip: 5.0,
                    value_weight: 0.5,
                    ema_decay: 0.99,
                },
            );
            let mut losses = Vec::new();
            for _ in 0..3 {
                losses.push(trainer.step(&batch, 3e-4, StepOptions::default()).unwrap());
            }
            (losses, trainer.net.p, trainer.ema, trainer.last_metrics)
        })
    };
    let (la, pa, ea, ma) = run(1);
    let (lb, pb, eb, mb) = run(7);
    assert_eq!(la, lb);
    assert_eq!(pa, pb);
    assert_eq!(ea, eb);
    assert_eq!(ma, mb);
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
