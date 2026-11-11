use tavernagent_v2::{
    backend::Execution, batched, net::PolicyValueNet, rollout, rules::PolicyRules,
    train::TrainOptions,
};

#[test]
fn batching_independent_games_preserves_every_cpu_transition() {
    let options = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 4,
        ..Default::default()
    };
    let net = PolicyValueNet::new(728, options.net_config()).unwrap();
    let cfg = options.game_config().unwrap();
    for enabled in [false, true] {
        let rules = PolicyRules {
            block_consecutive_freeze: enabled,
        };
        let expected = rollout::collect_game_with_rules(&net, 14, &cfg, rules).unwrap();
        for workers in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let actual =
                batched::collect(&net, &[13, 14, 15], &cfg, rules, &pool, &Execution::Cpu).unwrap();
            assert_eq!(actual[1], expected);
        }
    }
}

#[test]
fn batched_bot_evaluation_preserves_placement_rng_and_diagnostics() {
    use tavernagent_v2::{
        batched_eval,
        eval::{self, Opponent},
    };
    let options = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 5,
        ..Default::default()
    };
    let net = PolicyValueNet::new(872, options.net_config()).unwrap();
    let cfg = options.game_config().unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let seeds = [12, 13, 14, 15, 16];
    for opponent in [Opponent::Scripted, Opponent::Teacher] {
        for sample in [false, true] {
            let rules = PolicyRules {
                block_consecutive_freeze: true,
            };
            let actual = batched_eval::evaluate(
                &net,
                &Execution::Cpu,
                &seeds,
                &cfg,
                rules,
                opponent,
                sample,
                3,
                &pool,
            )
            .unwrap();
            for (&seed, a) in seeds.iter().zip(actual) {
                let e = eval::play_detailed(&net, seed, &cfg, sample, opponent, rules).unwrap();
                assert_eq!(a.placement, e.placement);
                assert_eq!(a.turn_limit_end, e.turn_limit_end);
                assert_eq!(a.truncated_combats, e.truncated_combats);
                assert_eq!(
                    serde_json::to_value(a.diagnostics).unwrap(),
                    serde_json::to_value(e.diagnostics).unwrap()
                );
            }
        }
    }
}

#[test]
fn batched_arena_preserves_paired_seats_and_each_game_result() {
    use tavernagent_v2::{arena, batched_arena};
    let opts = TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 4,
        ..Default::default()
    };
    let a = PolicyValueNet::new(71, opts.net_config()).unwrap();
    let b = PolicyValueNet::new(93, opts.net_config()).unwrap();
    let cfg = opts.game_config().unwrap();
    let rules = PolicyRules {
        block_consecutive_freeze: true,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let seeds = [15, 16, 17];
    let results = batched_arena::evaluate(
        &a,
        &b,
        &Execution::Cpu,
        &Execution::Cpu,
        &cfg,
        rules,
        PolicyRules::default(),
        &seeds,
        3,
        &pool,
    )
    .unwrap();
    for (i, result) in results.into_iter().enumerate() {
        let expected = arena::play(
            &a,
            &b,
            &cfg,
            rules,
            PolicyRules::default(),
            seeds[i / 2],
            (i / 2) % 4 + 4 * (i % 2),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(
            serde_json::to_value(result.diagnostics).unwrap(),
            serde_json::to_value(expected.diagnostics).unwrap()
        );
    }
}
