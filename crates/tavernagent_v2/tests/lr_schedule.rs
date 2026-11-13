use tavernagent_v2::{
    checkpoint,
    train::{self, LrSchedule, TrainOptions},
};

#[test]
fn default_schedule_boundaries_and_legacy_options() {
    let options = TrainOptions::default();
    assert_eq!(options.lr_schedule, LrSchedule::Cosine);
    for (step, expected) in [
        (1, 1e-4),
        (5000, 1e-4),
        (7500, 6e-5),
        (10000, 2e-5),
        (20000, 2e-5),
    ] {
        assert!((options.learning_rate(step) - expected).abs() < 1e-15);
    }
    let rates: Vec<_> = (5000..=10000).map(|i| options.learning_rate(i)).collect();
    assert!(rates.windows(2).all(|w| w[1] <= w[0]));
    let mut legacy = serde_json::to_value(&options).unwrap();
    for field in ["lr_schedule", "lr_min", "lr_decay_start", "lr_decay_end"] {
        legacy.as_object_mut().unwrap().remove(field);
    }
    let old: TrainOptions = serde_json::from_value(legacy).unwrap();
    assert_eq!(old.lr_schedule, LrSchedule::Constant);
    assert_eq!(old.learning_rate(20000), 1e-4);
    for invalid in [
        TrainOptions {
            lr_min: f64::NAN,
            ..options.clone()
        },
        TrainOptions {
            lr_min: 2e-4,
            ..options.clone()
        },
        TrainOptions {
            lr_decay_end: 5000,
            ..options.clone()
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn resumed_schedule_matches_uninterrupted_weights_and_adam() {
    let dir = std::env::temp_dir().join(format!("ppo-lr-schedule-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let full = dir.join("full.npz");
    let split = dir.join("split.npz");
    let options = TrainOptions {
        games: 1,
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 2,
        batch: 64,
        lr_decay_start: 1,
        lr_decay_end: 3,
        ..Default::default()
    };
    for (out, iterations) in [(&full, 4), (&split, 2)] {
        train::train(train::TrainArgs {
            device: 0,
            iterations,
            workers: 1,
            out: out.clone(),
            save_every: 0,
            options: options.clone(),
        })
        .unwrap();
    }
    train::resume(train::ResumeArgs {
        backend: None,
        device: 0,
        checkpoint: split.clone(),
        iterations: 4,
        workers: 1,
        out: None,
        save_every: 0,
    })
    .unwrap();
    let (a, sa) = checkpoint::load(&full).unwrap();
    let (b, sb) = checkpoint::load(&split).unwrap();
    assert_eq!(a.net.p.tensors, b.net.p.tensors);
    assert_eq!(a.m.tensors, b.m.tensors);
    assert_eq!(a.v.tensors, b.v.tensors);
    assert_eq!(a.optimizer_step, b.optimizer_step);
    for (i, row) in sb.history.iter().enumerate() {
        assert!(
            (row["learning_rate"].as_f64().unwrap() - options.learning_rate(i + 1)).abs() < 1e-15
        );
    }
    assert_eq!(sa.iteration, sb.iteration);
    assert_eq!(
        serde_json::to_string(&sa.shuffle_rng).unwrap(),
        serde_json::to_string(&sb.shuffle_rng).unwrap()
    );
    let constant = dir.join("constant.npz");
    train::fork(train::ForkArgs {
        checkpoint: split,
        out: constant.clone(),
        lr_schedule: Some(LrSchedule::Constant),
        lr: Some(5e-5),
        ..Default::default()
    })
    .unwrap();
    let (forked, state) = checkpoint::load(&constant).unwrap();
    assert_eq!(state.options.learning_rate(10000), 5e-5);
    assert_eq!(b.m.tensors, forked.m.tensors);
    assert_eq!(b.v.tensors, forked.v.tensors);
    std::fs::remove_dir_all(dir).unwrap();
}
