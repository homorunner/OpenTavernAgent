use tavernagent_v2::actions as a;
use tavernagent_v2::{
    actions,
    policy::Categorical,
    ppo::{self, PpoOptions},
    train::TrainOptions,
};

#[test]
fn balanced_checkpoint_resume_and_fork_preserve_training_state() {
    use tavernagent_v2::{checkpoint, train};
    let dir = std::env::temp_dir().join(format!("ppo-category-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let full = dir.join("full.npz");
    let split = dir.join("split.npz");
    let options = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        games: 1,
        max_turns: 2,
        batch: 64,
        ppo: PpoOptions {
            category_balanced_entropy: true,
            ..Default::default()
        },
        ..Default::default()
    };
    for (out, iterations) in [(&full, 2), (&split, 1)] {
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
        iterations: 2,
        workers: 1,
        out: None,
        save_every: 0,
    })
    .unwrap();
    let (a, sa) = checkpoint::load(&full).unwrap();
    let (b, sb) = checkpoint::load(&split).unwrap();
    assert!(sb.options.ppo.category_balanced_entropy);
    assert_eq!(a.net.p.tensors, b.net.p.tensors);
    assert_eq!(a.m.tensors, b.m.tensors);
    assert_eq!(a.v.tensors, b.v.tensors);
    assert_eq!(a.optimizer_step, b.optimizer_step);
    assert_eq!(
        serde_json::to_string(&sa.shuffle_rng).unwrap(),
        serde_json::to_string(&sb.shuffle_rng).unwrap()
    );
    let fork = dir.join("fork.npz");
    train::fork(train::ForkArgs {
        checkpoint: split,
        out: fork.clone(),
        category_balanced_entropy: Some(false),
        ..Default::default()
    })
    .unwrap();
    let (f, sf) = checkpoint::load(&fork).unwrap();
    assert!(!sf.options.ppo.category_balanced_entropy);
    assert_eq!(f.net.p.tensors, b.net.p.tensors);
    assert_eq!(f.m.tensors, b.m.tensors);
    assert_eq!(sf.iteration, sb.iteration);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn categories_cover_compact_actions_and_legacy_defaults_off() {
    let mut counts = [0; actions::N_CATEGORIES];
    for action in 0..actions::N_ACTIONS {
        counts[actions::category(action)] += 1;
    }
    assert_eq!(counts, [1, 1, 1, 1, 2, 1, 1, 7, 7, 7, 16, 70]);
    let mut json = serde_json::to_value(TrainOptions::default()).unwrap();
    json["ppo"]
        .as_object_mut()
        .unwrap()
        .remove("category_balanced_entropy");
    let legacy: TrainOptions = serde_json::from_value(json).unwrap();
    assert!(!legacy.ppo.category_balanced_entropy);
    let mut balanced = legacy;
    balanced.ppo.category_balanced_entropy = true;
    let restored: TrainOptions =
        serde_json::from_str(&serde_json::to_string(&balanced).unwrap()).unwrap();
    assert!(restored.ppo.category_balanced_entropy);
}

#[test]
fn balanced_loss_gradient_matches_finite_differences() {
    let mut logits = vec![f32::NAN; actions::N_ACTIONS];
    let mut mask = vec![false; actions::N_ACTIONS];
    let legal = [
        a::A_REFRESH,
        a::A_UPGRADE,
        a::A_PLAY,
        a::A_PLAY + 1,
        a::A_PLAY + 7,
    ];
    for (i, &action) in legal.iter().enumerate() {
        logits[action] = i as f32 * 0.17 - 0.3;
        mask[action] = true;
    }
    let opts = PpoOptions {
        category_balanced_entropy: true,
        ..Default::default()
    };
    let lp = Categorical::new(&logits, &mask).unwrap().log_probabilities[a::A_PLAY];
    for ratio in [0.7_f64, 1.0, 1.4] {
        for advantage in [-0.7, 0.0, 0.7] {
            let loss = |x: &[f32]| {
                ppo::policy_loss_gradient(x, &mask, a::A_PLAY, lp - ratio.ln(), advantage, &opts)
                    .unwrap()
            };
            let (metrics, grad) = loss(&logits);
            assert!(metrics.exploration_bonus <= 0.0);
            for &action in &legal {
                let mut hi = logits.clone();
                let mut lo = logits.clone();
                hi[action] += 1e-3;
                lo[action] -= 1e-3;
                let numeric = (loss(&hi).0.total_loss(&opts) - loss(&lo).0.total_loss(&opts))
                    / (hi[action] as f64 - lo[action] as f64);
                assert!((numeric - grad[action] as f64).abs() < 2e-5);
            }
            assert!(grad.iter().zip(&mask).all(|(&g, &legal)| legal || g == 0.0));
        }
    }
}

#[test]
fn equal_category_mass_is_stationary_and_single_category_matches_entropy() {
    let mut logits = vec![0.0; actions::N_ACTIONS];
    let mut mask = vec![false; actions::N_ACTIONS];
    mask[a::A_REFRESH] = true;
    mask[a::A_PLAY] = true;
    mask[a::A_PLAY + 1] = true;
    // Refresh gets 1/2, the two Play actions get 1/4 each.
    logits[a::A_REFRESH] = 2_f32.ln();
    let balanced = PpoOptions {
        category_balanced_entropy: true,
        ..Default::default()
    };
    let (m, grad) =
        ppo::policy_loss_gradient(&logits, &mask, a::A_REFRESH, -2_f64.ln(), 0.0, &balanced)
            .unwrap();
    assert!(m.exploration_bonus.abs() < 1e-12);
    assert!(grad.iter().all(|g| g.abs() < 1e-8));
    mask[a::A_REFRESH] = false;
    logits[a::A_PLAY] = 0.7;
    let (_, g1) =
        ppo::policy_loss_gradient(&logits, &mask, a::A_PLAY, -1.0, 0.0, &balanced).unwrap();
    let (_, g2) =
        ppo::policy_loss_gradient(&logits, &mask, a::A_PLAY, -1.0, 0.0, &PpoOptions::default())
            .unwrap();
    assert!(g1.iter().zip(g2).all(|(x, y)| (x - y).abs() < 1e-8));
    mask[a::A_PLAY + 1] = false;
    let (m, g) = ppo::policy_loss_gradient(&logits, &mask, a::A_PLAY, 0.0, 0.0, &balanced).unwrap();
    assert_eq!(m.exploration_bonus, 0.0);
    assert!(g.iter().all(|&x| x == 0.0));
}
