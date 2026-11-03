//! Checkpoint self-play matches with seat swaps and game-clustered score intervals.
//!
//! The reported Elo-equivalent is 400*log10(score/(1-score)), where score is the
//! fraction of cross-team placement comparisons won. It is not a fitted global Elo.

use std::path::PathBuf;

use clap::Args;
use rayon::prelude::*;
use serde_json::{Value, json};
use taverngym::Game;

use crate::heroes::select_heroes;
use crate::mcts::{self, Horizon, SearchOptions};
use crate::net::PolicyValueNet;
use crate::rules::Budgets;
use crate::selfplay::game_rng;

#[derive(Clone, Debug, Args)]
pub struct ArenaArgs {
    #[arg(long)]
    pub a: PathBuf,
    #[arg(long)]
    pub b: PathBuf,
    #[arg(long, default_value_t = 512)]
    pub games: usize,
    #[arg(long, default_value_t = 21_000_000)]
    pub seed: u64,
    #[arg(long, default_value_t = 8)]
    pub sims: usize,
    #[arg(long, default_value_t = 32)]
    pub workers: usize,
    #[arg(long, default_value = "turn", value_parser = Horizon::parse)]
    pub horizon: Horizon,
    #[arg(long, default_value_t = crate::mcts::DEFAULT_C_PUCT)]
    pub c_puct: f32,
    #[command(flatten)]
    pub budgets: Budgets,
    #[arg(long)]
    pub out: PathBuf,
}

pub fn elo_equivalent(score: f64) -> f64 {
    let p = score.clamp(1e-6, 1.0 - 1e-6);
    400.0 * (p / (1.0 - p)).log10()
}

/// One game: even seats are team A (odd when `swapped`); returns team A's score.
fn play_match(
    nets: (&PolicyValueNet, &PolicyValueNet),
    seed: u64,
    swapped: bool,
    budgets: &Budgets,
    search: &SearchOptions,
) -> Result<Value, String> {
    let mut game = Game::new(seed, budgets.config()?);
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed ^ swapped as u64);
    let n = game.players.len();
    let team_a = |seat: usize| (seat % 2 == 1) == swapped;
    while !game.is_over() {
        let mut acted = false;
        for seat in 0..n {
            if game.needs_action(seat) {
                acted = true;
                let net = if team_a(seat) { nets.0 } else { nets.1 };
                let action = mcts::search(&game, seat, net, search, &mut rng)?.best_action();
                game.apply_index(seat, action).map_err(|e| e.to_string())?;
            }
        }
        if !acted {
            return Err(format!("arena stalled: {seed}"));
        }
    }
    let places: Vec<u8> = game.players.iter().map(|p| p.placement).collect();
    let a: Vec<u8> = (0..n).filter(|s| team_a(*s)).map(|s| places[s]).collect();
    let b: Vec<u8> = (0..n).filter(|s| !team_a(*s)).map(|s| places[s]).collect();
    let wins = a
        .iter()
        .flat_map(|x| b.iter().map(move |y| (x < y) as u32))
        .sum::<u32>();
    Ok(json!({
        "seed": seed,
        "swapped": swapped,
        "placements": places,
        "score": wins as f64 / (a.len() * b.len()) as f64,
    }))
}

pub fn battle(args: &ArenaArgs) -> Result<Value, String> {
    if args.games < 2 || args.games % 2 == 1 {
        return Err("games must be positive and even for seat swaps".into());
    }
    let search = SearchOptions {
        sims: args.sims,
        c_puct: args.c_puct,
        horizon: args.horizon,
        ..SearchOptions::default()
    };
    search.validate()?;
    let a = PolicyValueNet::load(&args.a)?;
    let b = PolicyValueNet::load(&args.b)?;
    let jobs: Vec<(u64, bool)> = (args.seed..args.seed + args.games as u64 / 2)
        .flat_map(|s| [(s, false), (s, true)])
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers.max(1).min(args.games))
        .build()
        .map_err(|e| e.to_string())?;
    let rows: Vec<Value> = pool.install(|| {
        jobs.par_iter()
            .map(|&(seed, swapped)| play_match((&a, &b), seed, swapped, &args.budgets, &search))
            .collect::<Result<_, _>>()
    })?;
    // Each seed's two seat-swapped games form one statistical cluster.
    let scores: Vec<f64> = rows
        .chunks(2)
        .map(|pair| {
            pair.iter()
                .map(|r| r["score"].as_f64().unwrap())
                .sum::<f64>()
                / pair.len() as f64
        })
        .collect();
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let se = if scores.len() > 1 {
        (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (scores.len() - 1) as f64).sqrt()
            / (scores.len() as f64).sqrt()
    } else {
        0.0
    };
    let ci = [(mean - 1.96 * se).max(0.0), (mean + 1.96 * se).min(1.0)];
    Ok(json!({
        "a": args.a.display().to_string(),
        "b": args.b.display().to_string(),
        "games": args.games,
        "seed_clusters": scores.len(),
        "sims": args.sims,
        "horizon": args.horizon.name(),
        "c_puct": args.c_puct,
        "search_version": mcts::SEARCH_VERSION,
        "budgets": args.budgets,
        "score": mean,
        "se": se,
        "ci95": ci,
        "elo_equivalent": elo_equivalent(mean),
        "elo_ci95": [elo_equivalent(ci[0]), elo_equivalent(ci[1])],
        "matches": rows,
    }))
}

pub fn run(args: ArenaArgs) -> Result<(), String> {
    let result = battle(&args)?;
    crate::npz::ensure_parent(&args.out)?;
    std::fs::write(&args.out, serde_json::to_string_pretty(&result).unwrap())
        .map_err(|e| e.to_string())?;
    let mut summary = result.clone();
    summary.as_object_mut().unwrap().remove("matches");
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    Ok(())
}
