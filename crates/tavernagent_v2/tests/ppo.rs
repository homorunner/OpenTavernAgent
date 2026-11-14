use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use tavernagent_v2::{
    checkpoint::{self, RunState},
    gae::{self, ValueStep},
    heroes::select_heroes,
    net::{self, NetConfig, PolicyValueNet},
    observation::observe_seat,
    policy::{self, Categorical},
    ppo::{self, PpoOptions, Trainer},
    rollout::{self, Sample, Transition},
    train::TrainOptions,
};
use taverngym::{Game, RulesConfig};

fn options() -> TrainOptions {
    TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 3,
        budgets: tavernagent_v2::rules::Budgets {
            action_cap: 8,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sample(net: &PolicyValueNet) -> Sample {
    let mut game = Game::new(17, RulesConfig::default());
    select_heroes(&mut game, 17).unwrap();
    let obs = observe_seat(&game, 0);
    let mut rng = Pcg64Mcg::seed_from_u64(1);
    let d = policy::decide(net, &obs, &mut rng, false).unwrap();
    Sample {
        transition: Transition {
            turn: game.turn,
            health: 40.,
            obs,
            action: d.action,
            old_log_prob: d.log_prob,
            old_value: d.value,
            reward: 1.0,
            terminated: true,
        },
        advantage: 0.7,
        return_: 0.8,
        terminal_reward: 1.0,
        future_health: Some(0.),
    }
}

#[test]
fn gae_resets_at_terminal_and_bootstraps_only_unfinished_paths() {
    let steps = [
        ValueStep {
            value: 0.5,
            reward: 0.0,
            terminated: false,
        },
        ValueStep {
            value: 0.5,
            reward: 1.0,
            terminated: true,
        },
        ValueStep {
            value: 0.2,
            reward: 0.0,
            terminated: false,
        },
    ];
    let result = gae::estimate(&steps, 0.3, 1.0, 1.0).unwrap();
    for ((adv, ret), (want_adv, want_ret)) in
        result.iter().zip([(0.5, 1.0), (0.5, 1.0), (0.1, 0.3)])
    {
        assert!((adv - want_adv).abs() < 1e-6 && (ret - want_ret).abs() < 1e-6);
    }
    let steps = [
        ValueStep {
            value: 0.0,
            reward: 0.0,
            terminated: false,
        },
        ValueStep {
            value: 0.0,
            reward: 0.0,
            terminated: false,
        },
        ValueStep {
            value: 0.0,
            reward: 1.0,
            terminated: true,
        },
    ];
    assert_eq!(
        gae::estimate(&steps, 99.0, 1.0, 0.5).unwrap(),
        vec![(0.25, 0.25), (0.5, 0.5), (1.0, 1.0)]
    );
    assert!(gae::estimate(&steps, 0.0, f32::NAN, 0.95).is_err());
}

#[test]
fn masked_categorical_is_stable_and_never_samples_illegal_actions() {
    let dist = Categorical::new(&[1000.0, 999.0, f32::NAN], &[true, true, false]).unwrap();
    assert!((dist.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(dist.probabilities[2], 0.0);
    assert!(dist.entropy.is_finite());
    let mut rng = Pcg64Mcg::seed_from_u64(7);
    for _ in 0..1000 {
        assert!(dist.sample(&mut rng) < 2);
    }
    assert!(Categorical::new(&[0.0], &[false]).is_err());
    assert!(Categorical::new(&[f32::NAN], &[true]).is_err());
    let one = Categorical::new(&[1e20, -1e20], &[false, true]).unwrap();
    assert_eq!(one.log_probabilities[1], 0.0);
    assert_eq!(one.entropy, 0.0);
    assert_eq!(one.sample(&mut rng), 1);
}

#[test]
fn clipping_and_entropy_gradients_match_finite_differences_for_both_advantage_signs() {
    let logits = [0.2, -0.3, 50.0];
    let mask = [true, true, false];
    let opts = PpoOptions {
        entropy_coef: 0.03,
        ..Default::default()
    };
    let lp = Categorical::new(&logits, &mask).unwrap().log_probabilities[0];
    for ratio in [0.7f64, 1.0, 1.4] {
        for advantage in [-0.7, 0.0, 0.7] {
            let old_lp = lp - ratio.ln();
            let (_, grads) =
                ppo::policy_loss_gradient(&logits, &mask, 0, old_lp, advantage, &opts).unwrap();
            for a in 0..logits.len() {
                let mut hi = logits;
                let mut lo = logits;
                hi[a] += 1e-3;
                lo[a] -= 1e-3;
                let loss = |x: &[f32]| {
                    ppo::policy_loss_gradient(x, &mask, 0, old_lp, advantage, &opts)
                        .unwrap()
                        .0
                        .total_loss(&opts)
                };
                let numeric = (loss(&hi) - loss(&lo)) / (hi[a] as f64 - lo[a] as f64);
                assert!(
                    (numeric - grads[a] as f64).abs() < 2e-5,
                    "ratio={ratio}, advantage={advantage}, action={a}: {numeric} vs {}",
                    grads[a]
                );
            }
        }
    }
    let opts = PpoOptions {
        entropy_coef: 0.0,
        ..opts
    };
    for (ratio, advantage) in [(1.4f64, 1.0), (0.7, -1.0)] {
        let (_, grad) =
            ppo::policy_loss_gradient(&logits, &mask, 0, lp - ratio.ln(), advantage, &opts)
                .unwrap();
        assert!(grad.iter().all(|g| *g == 0.0));
        let (_, corrective) =
            ppo::policy_loss_gradient(&logits, &mask, 0, lp - ratio.ln(), -advantage, &opts)
                .unwrap();
        assert!(corrective.iter().any(|g| g.abs() > 0.01));
    }
}

#[test]
fn fp32_behavior_probabilities_match_training_and_full_ppo_gradients() {
    let mut net = PolicyValueNet::new(3, options().net_config()).unwrap();
    let row = sample(&net);
    let opts = PpoOptions::default();
    // Exercise both the single-row path and batched GEMM, including chunk boundaries.
    for n in [1, 5, 65] {
        let (_, metrics) = ppo::loss_gradients(&net, &vec![&row; n], &opts).unwrap();
        assert!((metrics.ratio_mean - 1.0).abs() < 1e-5);
        assert!(metrics.approx_kl < 1e-9);
        assert_eq!(metrics.clip_fraction, 0.0);
    }
    let (grads, _) = ppo::loss_gradients(&net, &[&row], &opts).unwrap();
    let mut probes: Vec<_> = grads
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let j = (0..t.len())
                .max_by(|&a, &b| t.data[a].abs().total_cmp(&t.data[b].abs()))
                .unwrap();
            (i, j)
        })
        .collect();
    probes.push((net::E, 0));
    for (i, j) in probes {
        let original = net.p.tensors[i].data[j];
        let eps = 1e-3;
        net.p.tensors[i].data[j] = original + eps;
        let hi = ppo::loss_gradients(&net, &[&row], &opts)
            .unwrap()
            .1
            .total_loss(&opts);
        net.p.tensors[i].data[j] = original - eps;
        let lo = ppo::loss_gradients(&net, &[&row], &opts)
            .unwrap()
            .1
            .total_loss(&opts);
        net.p.tensors[i].data[j] = original;
        let numeric = (hi - lo) / (2.0 * eps as f64);
        let analytic = grads.tensors[i].data[j] as f64;
        assert!(
            (numeric - analytic).abs() < 0.003 + 0.02 * numeric.abs(),
            "{}[{j}]: numeric={numeric}, analytic={analytic}",
            net.p.names()[i]
        );
    }
}

#[test]
fn terminal_auxiliary_and_clipped_value_gradients_match_the_reported_loss() {
    let mut net = PolicyValueNet::new(3, options().net_config()).unwrap();
    let mut row = sample(&net);
    row.return_ = 1.4;
    row.terminal_reward = -1.0;
    let index = net.p.bv();
    for clip in [false, true] {
        let opts = PpoOptions {
            terminal_value_coef: 0.1,
            clip_value_target: clip,
            ..Default::default()
        };
        let (grads, _) = ppo::loss_gradients(&net, &[&row], &opts).unwrap();
        let original = net.p.tensors[index].data[0];
        net.p.tensors[index].data[0] = original + 1e-3;
        let hi = ppo::loss_gradients(&net, &[&row], &opts)
            .unwrap()
            .1
            .total_loss(&opts);
        net.p.tensors[index].data[0] = original - 1e-3;
        let lo = ppo::loss_gradients(&net, &[&row], &opts)
            .unwrap()
            .1
            .total_loss(&opts);
        net.p.tensors[index].data[0] = original;
        assert!(((hi - lo) / 2e-3 - f64::from(grads.tensors[index].data[0])).abs() < 1e-3);
    }
}

#[test]
fn distribution_and_health_heads_preserve_inference_and_have_correct_gradients() {
    for (distribution, health) in [(true, false), (false, true), (true, true)] {
        let opts = TrainOptions {
            critic_distribution: distribution,
            critic_health: health,
            ..options()
        };
        let mut net = PolicyValueNet::new(13, opts.net_config()).unwrap();
        let row = sample(&net);
        let obs = &row.transition.obs;
        let (logits, values, _) = net.forward(&[&obs.x], &[&obs.cards], &[&obs.mask]).unwrap();
        let (single, value) = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap();
        assert!((value - values[0]).abs() < 1e-5);
        for (a, b) in single.iter().zip(&logits.data) {
            assert!((a - b).abs() < 1e-5);
        }
        assert!((-1.0..=1.0).contains(&value));
        let restored = PolicyValueNet::from_npz(&net.to_npz()).unwrap();
        assert_eq!(restored.cfg, net.cfg);
        assert_eq!(restored.p, net.p);
        let loss_opts = PpoOptions {
            terminal_value_coef: 0.1,
            ..Default::default()
        };
        let (grads, _) = ppo::loss_gradients(&net, &[&row], &loss_opts).unwrap();
        for i in [net.p.wv(), net.p.bv(), net.p.wv_hidden(), net::W0] {
            for j in [0, net.p.tensors[i].len() - 1] {
                let original = net.p.tensors[i].data[j];
                net.p.tensors[i].data[j] = original + 1e-3;
                let hi = ppo::loss_gradients(&net, &[&row], &loss_opts)
                    .unwrap()
                    .1
                    .total_loss(&loss_opts);
                net.p.tensors[i].data[j] = original - 1e-3;
                let lo = ppo::loss_gradients(&net, &[&row], &loss_opts)
                    .unwrap()
                    .1
                    .total_loss(&loss_opts);
                net.p.tensors[i].data[j] = original;
                assert!(
                    ((hi - lo) / 2e-3 - f64::from(grads.tensors[i].data[j])).abs() < 2e-3,
                    "distribution={distribution} health={health} tensor={i} index={j}"
                );
            }
        }
        if distribution {
            let (weight, bias) = (net.p.wv(), net.p.bv());
            net.p.tensors[weight].data.fill(0.0);
            net.p.tensors[bias].data.fill(0.0);
            let (_, metrics) = ppo::loss_gradients(&net, &[&row], &loss_opts).unwrap();
            assert!((metrics.value_loss - 8.0f64.ln()).abs() < 1e-10);
            assert!((metrics.terminal_value_loss - 8.0f64.ln()).abs() < 1e-10);
        }
    }
}

#[test]
fn complete_rollouts_are_per_seat_on_policy_and_worker_independent() {
    let opts = options();
    let net = PolicyValueNet::new(3, opts.net_config()).unwrap();
    let mut cfg = opts.game_config().unwrap();
    cfg.hero_pool = vec!["TB_BaconShop_HERO_34".into()];
    let run = |workers| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        rollout::collect(&net, &[17, 18], &cfg, &pool).unwrap()
    };
    let games = run(1);
    assert_eq!(games, run(2));
    let mut saw_nonterminal_end = false;
    let mut samples = Vec::new();
    for game in games {
        assert_eq!(game.trajectories.len(), 8);
        for trajectory in game.trajectories {
            assert!(!trajectory.steps.is_empty());
            let reward = taverngym::env::placement_reward(trajectory.placement, 8);
            for (i, step) in trajectory.steps.iter().enumerate() {
                assert!(step.obs.mask[step.action]);
                let (logits, value) = net
                    .infer_f32(&step.obs.x, &step.obs.cards, &step.obs.mask)
                    .unwrap();
                assert_eq!(value, step.old_value);
                let dist = Categorical::new(&logits, &step.obs.mask).unwrap();
                assert_eq!(dist.log_probabilities[step.action], step.old_log_prob);
                if i + 1 == trajectory.steps.len() {
                    assert!(step.terminated);
                    assert_eq!(step.reward, reward);
                } else {
                    assert!(!step.terminated);
                    assert_eq!(step.reward, 0.0);
                    saw_nonterminal_end |= step.action == 0;
                }
            }
            let rows = trajectory.into_samples(1.0, 1.0).unwrap();
            for row in &rows {
                assert!((row.return_ - reward).abs() < 1e-5);
            }
            samples.extend(rows);
        }
    }
    assert!(saw_nonterminal_end);
    let returns: Vec<_> = samples.iter().map(|s| s.return_).collect();
    rollout::normalize_advantages(&mut samples).unwrap();
    assert!(
        samples
            .iter()
            .map(|s| s.advantage as f64)
            .sum::<f64>()
            .abs()
            / (samples.len() as f64)
            < 1e-6
    );
    assert_eq!(
        returns,
        samples.iter().map(|s| s.return_).collect::<Vec<_>>()
    );
    let mut trainer = Trainer::new(net);
    let metrics = ppo::update(
        &mut trainer,
        &samples,
        2,
        64,
        1e-4,
        &opts.ppo,
        &mut Pcg64Mcg::seed_from_u64(9),
    )
    .unwrap();
    assert!(metrics.updates > 0);
    assert!(trainer.net.p.all_finite());
}

#[test]
fn target_kl_stops_before_mutating_parameters() {
    let net = PolicyValueNet::new(5, options().net_config()).unwrap();
    let mut row = sample(&net);
    row.transition.old_log_prob -= 2.0;
    let mut trainer = Trainer::new(net);
    let before = trainer.net.p.tensors.clone();
    let metrics = ppo::update(
        &mut trainer,
        &[row],
        3,
        1,
        1e-4,
        &PpoOptions::default(),
        &mut Pcg64Mcg::seed_from_u64(0),
    )
    .unwrap();
    assert!(metrics.early_stop);
    assert_eq!(metrics.updates, 0);
    assert_eq!(metrics.trained_samples, 0);
    assert_eq!(metrics.evaluated_samples, 1);
    assert_eq!(metrics.first_epoch_samples, 0);
    assert_eq!(metrics.effective_epochs, 0.0);
    assert_eq!(trainer.optimizer_step, 0);
    assert_eq!(trainer.net.p.tensors, before);
}

#[test]
fn updates_count_partial_batches_and_are_worker_independent() {
    let net = PolicyValueNet::new(5, options().net_config()).unwrap();
    // Two full gradient chunks plus a tail; W0 also spans multiple parameter chunks.
    let rows = vec![sample(&net); 67];
    let opts = PpoOptions {
        target_kl: 0.0,
        ..Default::default()
    };
    let run = |workers| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut trainer = Trainer::new(net.clone());
        let mut rng = Pcg64Mcg::seed_from_u64(7);
        let metrics = pool
            .install(|| ppo::update(&mut trainer, &rows, 2, 65, 1e-4, &opts, &mut rng))
            .unwrap();
        assert_eq!(metrics.updates, 4);
        assert_eq!(metrics.trained_samples, 134);
        assert_eq!(metrics.evaluated_samples, 134);
        assert_eq!(metrics.first_epoch_samples, 67);
        assert_eq!(metrics.effective_epochs, 2.0);
        assert_eq!(metrics.epochs_completed, 2);
        assert!(!metrics.early_stop);
        (trainer, rng.random::<u64>())
    };
    let (one, rng_one) = run(1);
    let (four, rng_four) = run(4);
    assert_eq!(one.net.p.tensors, four.net.p.tensors);
    assert_eq!(one.m.tensors, four.m.tensors);
    assert_eq!(one.v.tensors, four.v.tensors);
    assert_eq!(rng_one, rng_four);
}

#[test]
fn checkpoint_restores_adam_rng_and_exact_next_update() {
    let opts = options();
    let net = PolicyValueNet::new(opts.seed, opts.net_config()).unwrap();
    let rows = vec![sample(&net); 5];
    let mut trainer = Trainer::new(net);
    let mut state = RunState {
        format: checkpoint::FORMAT,
        algorithm: checkpoint::ALGORITHM.into(),
        iteration: 1,
        games: 1,
        transitions: 5,
        options: opts,
        shuffle_rng: Pcg64Mcg::seed_from_u64(7),
        history: Vec::new(),
    };
    ppo::update(
        &mut trainer,
        &rows,
        1,
        3,
        1e-4,
        &state.options.ppo,
        &mut state.shuffle_rng,
    )
    .unwrap();
    let path =
        std::env::temp_dir().join(format!("tavernagent-v2-{}-resume.npz", std::process::id()));
    checkpoint::save(&path, &trainer, &state).unwrap();
    // Retired experiment settings must not prevent loading historical checkpoints.
    // Preserve the raw RNG integer rather than round-tripping it through JSON Value.
    let mut archive = tavernagent_v2::npz::Npz::load(&path).unwrap();
    let metadata = archive.get("trainer_meta").unwrap().as_text().unwrap();
    assert_eq!(metadata.matches("\"ppo\":{").count(), 1);
    for enabled in [false, true] {
        let historical = metadata.replacen(
            "\"ppo\":{",
            &format!("\"ppo\":{{\"category_balanced_entropy\":{enabled},"),
            1,
        );
        archive.insert(
            "trainer_meta",
            tavernagent_v2::npz::NpyArray::text(&historical),
        );
        archive.save(&path).unwrap();
        let (loaded, loaded_state) = checkpoint::load(&path).unwrap();
        assert_eq!(loaded.net.p.tensors, trainer.net.p.tensors);
        assert_eq!(loaded_state.options.ppo, state.options.ppo);
        let saved_options = serde_json::to_value(&loaded_state.options).unwrap();
        assert!(
            saved_options["ppo"]
                .get("category_balanced_entropy")
                .is_none()
        );
        let (policy, _) = checkpoint::load_for_evaluation(&path).unwrap();
        assert_eq!(policy.p.tensors, trainer.net.p.tensors);
    }
    let (mut restored, mut restored_state) = checkpoint::load(&path).unwrap();
    assert_eq!(restored.net.p.tensors, trainer.net.p.tensors);
    assert_eq!(restored.m.tensors, trainer.m.tensors);
    assert_eq!(restored.v.tensors, trainer.v.tensors);
    assert_eq!(restored.optimizer_step, trainer.optimizer_step);
    assert_eq!(
        restored_state.shuffle_rng.clone().random::<u64>(),
        state.shuffle_rng.clone().random::<u64>()
    );
    ppo::update(
        &mut trainer,
        &rows,
        1,
        3,
        1e-4,
        &state.options.ppo,
        &mut state.shuffle_rng,
    )
    .unwrap();
    ppo::update(
        &mut restored,
        &rows,
        1,
        3,
        1e-4,
        &restored_state.options.ppo,
        &mut restored_state.shuffle_rng,
    )
    .unwrap();
    assert_eq!(restored.net.p.tensors, trainer.net.p.tensors);
    assert_eq!(restored.m.tensors, trainer.m.tensors);
    assert_eq!(restored.v.tensors, trainer.v.tensors);
    // Models remain numpy-readable, and the same archive supports inference loading.
    let loaded = PolicyValueNet::load(&path).unwrap();
    assert!(loaded.p.all_finite());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn observation_and_output_shapes_preserve_the_original_contract() {
    let cfg = NetConfig {
        hidden: 16,
        blocks: 2,
        embedding_dim: 4,
        head_norm: true,
        ..Default::default()
    };
    let net = PolicyValueNet::new(7, cfg).unwrap();
    assert_eq!(net.n_input_features(), 1699);
    assert_eq!(net.n_card_slots(), 56);
    assert_eq!(
        (net::N_FEATURES, net::N_CARD_SLOTS, net::N_ACTIONS),
        (3203, 123, 115)
    );
    for seed in [0, 17, 95] {
        let mut game = Game::new(seed, Default::default());
        select_heroes(&mut game, seed).unwrap();
        for seat in 0..8 {
            let obs = observe_seat(&game, seat);
            let mut old_obs = tavernagent::mcts::observe_seat(&game, seat);
            let unused = net::SHOP_FLOAT_START + 7 * taverngym::obs::ITEM_F;
            old_obs.x.drain(unused..unused + taverngym::obs::ITEM_F);
            old_obs.cards.remove(net::SHOP_CARD_START + 7);
            old_obs.mask.remove(taverngym::actions::A_BUY + 7);
            assert_eq!(&obs.x[..old_obs.x.len()], &old_obs.x);
            assert_eq!(obs.cards, old_obs.cards);
            assert_eq!(obs.mask, old_obs.mask);
            let (logits, value) = net.infer_f32(&obs.x, &obs.cards, &obs.mask).unwrap();
            assert_eq!(logits.len(), net::N_ACTIONS);
            assert!(logits.iter().all(|v| v.is_finite()));
            assert!((-1.0..=1.0).contains(&value));
        }
    }
    assert!(
        PolicyValueNet::from_npz(
            &net::PolicyValueNet::new(0, NetConfig::default())
                .unwrap()
                .to_npz()
        )
        .is_ok()
    );
}
