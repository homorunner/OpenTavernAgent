//! Reproducible inference cost/precision comparison on real bot-driven observations.
//! cargo run --release -p tavernagent --example inference -- --blocks 6,8,10,12
use clap::Parser;
use rayon::prelude::*;
use serde_json::json;
use std::{hint::black_box, path::PathBuf, time::Instant};
use tavernagent::{
    heroes::select_heroes,
    linalg::native_bf16_available,
    mcts::{Observation, bot_action, observe_seat},
    net::{InferenceBackend, InferenceWorkspace, NetConfig, PolicyValueNet, masked_softmax},
    rules::Budgets,
};
use taverngym::Game;
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "6,8,10,12")]
    blocks: Vec<usize>,
    #[arg(long, default_value_t = 2000)]
    iters: usize,
    #[arg(long, default_value_t = 12)]
    games: u64,
    #[arg(long, default_value_t = 1)]
    workers: usize,
}
fn positions(games: u64) -> Vec<Observation> {
    let mut out = Vec::new();
    for seed in 0..games {
        let mut game = Game::new(seed, Budgets::default().config().unwrap());
        select_heroes(&mut game, seed).unwrap();
        let mut last = 0;
        for _ in 0..10000 {
            if game.is_over() || game.turn > 12 {
                break;
            }
            let mut moved = false;
            for seat in 0..8 {
                if !game.needs_action(seat) {
                    continue;
                }
                if seat == 0 && game.turn != last {
                    last = game.turn;
                    if [1, 4, 8, 12].contains(&last) {
                        out.push(observe_seat(&game, seat));
                        let mut ended = game.clone();
                        if ended.end_turn_effects(seat).is_ok() {
                            out.push(observe_seat(&ended, seat));
                        }
                    }
                }
                let a = bot_action(&mut game, seat);
                game.apply(seat, a).unwrap();
                moved = true;
            }
            if !moved {
                break;
            }
        }
    }
    out
}
fn main() {
    let a = Args::parse();
    assert!(a.games > 0 && a.iters > 0 && a.workers > 0);
    let observations = positions(a.games);
    let nets = if let Some(p) = &a.model {
        vec![PolicyValueNet::load(p).unwrap()]
    } else {
        a.blocks
            .iter()
            .map(|&blocks| {
                PolicyValueNet::new(
                    42,
                    NetConfig {
                        blocks,
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.workers)
        .build()
        .unwrap();
    for net in nets {
        for backend in [InferenceBackend::Fma, InferenceBackend::Dot] {
            if backend == InferenceBackend::Dot && !native_bf16_available() {
                continue;
            }
            let mut ws = InferenceWorkspace::default();
            let (mut max_value, mut max_logit, mut kl, mut flips, mut strong_flips, mut sq) =
                (0.0f32, 0.0f32, 0.0f64, 0usize, 0usize, 0.0f64);
            for o in &observations {
                let (reference, rv) = net.infer_f32(&o.x, &o.cards, &o.mask).unwrap();
                let (logits, v) = net
                    .infer_with_workspace(&o.x, &o.cards, &o.mask, backend, &mut ws)
                    .unwrap();
                max_value = max_value.max((rv - v).abs());
                sq += (rv as f64 - v as f64).powi(2);
                for (x, y) in reference.iter().zip(logits) {
                    max_logit = max_logit.max((x - y).abs());
                }
                let p = masked_softmax(&reference, &o.mask);
                let q = masked_softmax(logits, &o.mask);
                kl += p
                    .iter()
                    .zip(&q)
                    .filter(|(x, _)| **x > 0.0)
                    .map(|(&x, &y)| x as f64 * (x as f64 / y.max(1e-30) as f64).ln())
                    .sum::<f64>();
                let mut legal: Vec<_> = (0..p.len()).filter(|&i| o.mask[i]).collect();
                legal.sort_by(|&i, &j| p[j].total_cmp(&p[i]));
                if legal.len() > 1 {
                    let best = legal
                        .iter()
                        .copied()
                        .max_by(|&i, &j| q[i].total_cmp(&q[j]))
                        .unwrap();
                    if best != legal[0] {
                        flips += 1;
                        if p[legal[0]] - p[legal[1]] > 0.01 {
                            strong_flips += 1;
                        }
                    }
                }
            }
            for reuse in [false, true] {
                let mut times = Vec::new();
                for _ in 0..3 {
                    let barrier = std::sync::Barrier::new(a.workers);
                    let durations: Vec<_> = pool.install(|| {
                        (0..a.workers)
                            .into_par_iter()
                            .map(|worker| {
                                let mut ws = InferenceWorkspace::default();
                                for o in &observations {
                                    black_box(
                                        net.infer_with_workspace(
                                            &o.x, &o.cards, &o.mask, backend, &mut ws,
                                        )
                                        .unwrap(),
                                    );
                                }
                                barrier.wait();
                                let start = Instant::now();
                                for i in 0..a.iters {
                                    let o = &observations[(i + worker) % observations.len()];
                                    if reuse {
                                        black_box(
                                            net.infer_with_workspace(
                                                &o.x, &o.cards, &o.mask, backend, &mut ws,
                                            )
                                            .unwrap(),
                                        );
                                    } else {
                                        let mut fresh = InferenceWorkspace::default();
                                        black_box(
                                            net.infer_with_workspace(
                                                &o.x, &o.cards, &o.mask, backend, &mut fresh,
                                            )
                                            .unwrap(),
                                        );
                                    }
                                }
                                start.elapsed().as_secs_f64()
                            })
                            .collect()
                    });
                    times.push(durations.into_iter().fold(0.0f64, f64::max));
                }
                times.sort_by(f64::total_cmp);
                let elapsed = times[1];
                println!(
                    "{}",
                    json!({"blocks":net.cfg.blocks,"params":net.n_params(),"backend":format!("{backend:?}"),"reuse":reuse,"workers":a.workers,"positions":observations.len(),"us_per_thread_eval":elapsed*1e6/a.iters as f64,"evals_s":a.iters as f64*a.workers as f64/elapsed,"max_value_error":max_value,"rms_value_error":(sq/observations.len() as f64).sqrt(),"max_logit_error":max_logit,"mean_policy_kl":kl/observations.len() as f64,"argmax_flips":flips,"argmax_flips_margin_gt_001":strong_flips})
                );
            }
        }
    }
}
