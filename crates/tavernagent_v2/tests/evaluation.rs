use tavernagent_v2::{checkpoint, eval, net::PolicyValueNet, train};

#[test]
fn iteration_zero_is_a_resumable_untrained_checkpoint() {
    let path = std::env::temp_dir().join(format!("ppo-initial-{}.npz", std::process::id()));
    let options = train::TrainOptions {
        games: 1,
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        max_turns: 3,
        epochs: 1,
        batch: 64,
        ..Default::default()
    };
    let expected = PolicyValueNet::new(options.seed, options.net_config()).unwrap();
    train::train(train::TrainArgs {
        iterations: 0,
        workers: 1,
        out: path.clone(),
        save_every: 0,
        options,
    })
    .unwrap();
    let (trainer, state) = checkpoint::load(&path).unwrap();
    assert_eq!(
        (
            state.iteration,
            state.games,
            state.transitions,
            trainer.optimizer_step
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(trainer.net.p.tensors, expected.p.tensors);
    assert_eq!(trainer.m.sum_squares(), 0.0);
    assert_eq!(trainer.v.sum_squares(), 0.0);
    train::resume(train::ResumeArgs {
        checkpoint: path.clone(),
        iterations: 1,
        workers: 2,
        out: None,
        save_every: 0,
    })
    .unwrap();
    assert_eq!(checkpoint::load(&path).unwrap().1.iteration, 1);
    std::fs::remove_file(path.with_extension("history.json")).unwrap();
    std::fs::remove_file(path).unwrap();
}

#[test]
fn both_opponents_evaluate_reproducibly_across_worker_counts() {
    let options = train::TrainOptions {
        hidden: 16,
        blocks: 1,
        embedding_dim: 2,
        ..Default::default()
    };
    let net = PolicyValueNet::new(17, options.net_config()).unwrap();
    let cfg = options.game_config().unwrap();
    for opponent in [eval::Opponent::Scripted, eval::Opponent::Teacher] {
        for sample in [false, true] {
            let run = |workers| {
                use rayon::prelude::*;
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                pool.install(|| {
                    (10_000_000..10_000_004u64)
                        .into_par_iter()
                        .map(|seed| eval::play_against(&net, seed, &cfg, sample, opponent).unwrap())
                        .collect::<Vec<_>>()
                })
            };
            let one = run(1);
            assert!(one.iter().all(|p| (1..=8).contains(p)));
            assert_eq!(one, run(4));
        }
    }
}
