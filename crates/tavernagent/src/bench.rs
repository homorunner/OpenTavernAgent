//! Inference and decision latency on real positions.

use std::time::Instant;

use clap::Args;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use taverngym::Game;

use crate::heroes::select_heroes;
use crate::mcts::{self, SearchOptions, bot_action, observe_seat};
use crate::net::{NetConfig, PolicyValueNet};
use crate::rules::Budgets;

#[derive(Clone, Debug, Args)]
pub struct BenchArgs {
    /// Checkpoint; omit to time a freshly initialized network.
    #[arg(long)]
    pub model: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 2000)]
    pub iters: usize,
    /// p99 budget for one prediction, in milliseconds.
    #[arg(long, default_value_t = 0.5)]
    pub budget_ms: f64,
    #[arg(long, default_value_t = 64)]
    pub sims: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Turns at which a full decision is timed (seat 0 vs bots).
    #[arg(long, value_delimiter = ',', default_values_t = [1u32, 4, 8, 12])]
    pub turns: Vec<u32>,
    #[command(flatten)]
    pub budgets: Budgets,
}

fn timed(mut f: impl FnMut(), iters: usize) -> Vec<f64> {
    for _ in 0..iters.min(50) {
        f();
    }
    (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect()
}

fn percentile(samples: &mut [f64], q: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((samples.len() - 1) as f64 * q).round() as usize;
    samples[i]
}

fn report(name: &str, mut s: Vec<f64>) -> f64 {
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let max = s.iter().cloned().fold(0.0, f64::max);
    let p50 = percentile(&mut s, 0.5);
    let p99 = percentile(&mut s, 0.99);
    println!("{name:<18} {mean:7.3} {p50:7.3} {p99:7.3} {max:7.3}");
    p99
}

/// Seat 0's first decision position at each requested turn, bots driving everything else.
fn positions(seed: u64, budgets: &Budgets, turns: &[u32]) -> Result<Vec<(u32, Game)>, String> {
    let mut game = Game::new(seed, budgets.config()?);
    select_heroes(&mut game, seed)?;
    let mut out = Vec::new();
    let mut captured = Vec::new();
    while !game.is_over() && captured.len() < turns.len() {
        if game.needs_action(0) {
            if turns.contains(&game.turn) && !captured.contains(&game.turn) {
                captured.push(game.turn);
                out.push((game.turn, game.clone()));
            }
            let a = bot_action(&mut game, 0);
            game.apply(0, a).map_err(|e| e.to_string())?;
            continue;
        }
        let mut moved = false;
        for p in 1..game.players.len() {
            if game.needs_action(p) {
                let a = bot_action(&mut game, p);
                game.apply(p, a).map_err(|e| e.to_string())?;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    Ok(out)
}

pub fn run(args: BenchArgs) -> Result<(), String> {
    let net = match &args.model {
        Some(p) => PolicyValueNet::load(p)?,
        None => PolicyValueNet::new(args.seed, NetConfig::default())?,
    };
    let positions = positions(args.seed, &args.budgets, &args.turns)?;
    let (_, first) = positions.first().ok_or("no position reached")?;
    let obs = observe_seat(first, 0);
    println!(
        "net {} params, hidden {}, {} iterations",
        net.n_params(),
        net.cfg.hidden,
        args.iters
    );
    println!(
        "{:<18} {:>7} {:>7} {:>7} {:>7}   (ms)",
        "case", "mean", "p50", "p99", "max"
    );
    report(
        "forward_only",
        timed(
            || drop(net.forward(&[&obs.x], &[&obs.cards]).unwrap()),
            args.iters,
        ),
    );
    let p99 = report(
        "predict",
        timed(
            || drop(net.predict(&obs.x, &obs.cards, &obs.mask).unwrap()),
            args.iters,
        ),
    );
    let (probs, value) = net.predict(&obs.x, &obs.cards, &obs.mask)?;
    let total: f32 = probs.iter().sum();
    if (total - 1.0).abs() > 1e-5
        || probs.iter().zip(&obs.mask).any(|(p, m)| !*m && *p != 0.0)
        || !(-1.0..=1.0).contains(&value)
    {
        return Err("masked policy is not a distribution over legal actions".into());
    }
    println!(
        "\npolicy sums to 1 over {} legal actions, value {value:+.3}\n",
        obs.mask.iter().filter(|m| **m).count()
    );
    let search = SearchOptions {
        sims: args.sims,
        ..SearchOptions::default()
    };
    println!(
        "{:<18} {:>7} {:>7} {:>7} {:>7}   (ms, {} sims)",
        "decision", "mean", "p50", "p99", "max", args.sims
    );
    for (turn, game) in &positions {
        let mut rng = Pcg64Mcg::seed_from_u64(args.seed);
        let legal = game.legal_mask(0).iter().filter(|m| **m).count();
        report(
            &format!("turn {turn} ({legal} legal)"),
            timed(
                || drop(mcts::search(game, 0, &net, &search, &mut rng).unwrap()),
                (args.iters / 40).max(10),
            ),
        );
    }
    let ok = p99 < args.budget_ms;
    println!(
        "\nbudget {} ms: predict p99 {p99:.3} ms -> {}",
        args.budget_ms,
        if ok { "PASS" } else { "FAIL" }
    );
    if ok {
        Ok(())
    } else {
        Err("latency budget exceeded".into())
    }
}
