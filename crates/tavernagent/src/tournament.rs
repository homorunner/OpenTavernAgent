//! Up to eight checkpoints at one table: every game seats the models by a seeded random
//! permutation, so each model plays every seat equally often. Reports mean placement per
//! model with standard errors.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Args;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use serde_json::{Value, json};
use taverngym::Game;

use crate::heroes::select_heroes;
use crate::mcts::{self, Horizon, SearchOptions};
use crate::net::PolicyValueNet;
use crate::rules::Budgets;
use crate::selfplay::game_rng;

#[derive(Clone, Debug, Args)]
pub struct TournamentArgs {
    /// Checkpoints as name=path (2 to 8; fewer than 8 are seated cyclically).
    #[arg(long = "model", required = true)]
    pub models: Vec<String>,
    #[arg(long, default_value_t = 512)]
    pub games: usize,
    #[arg(long, default_value_t = 31_000_000)]
    pub seed: u64,
    #[arg(long, default_value_t = 64)]
    pub sims: usize,
    #[arg(long, default_value_t = crate::mcts::DEFAULT_C_PUCT)]
    pub c_puct: f32,
    #[arg(long, default_value = "turn", value_parser = Horizon::parse)]
    pub horizon: Horizon,
    #[arg(long, default_value_t = 32)]
    pub workers: usize,
    #[command(flatten)]
    pub budgets: Budgets,
    #[arg(long)]
    pub out: PathBuf,
}

fn play(
    nets: &[PolicyValueNet],
    seating: &[usize],
    seed: u64,
    budgets: &Budgets,
    search: &SearchOptions,
) -> Result<Vec<u8>, String> {
    let mut game = Game::new(seed, budgets.config()?);
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    while !game.is_over() {
        let mut acted = false;
        for seat in 0..game.players.len() {
            if game.needs_action(seat) {
                acted = true;
                let net = &nets[seating[seat]];
                let action = mcts::search(&game, seat, net, search, &mut rng)?.best_action();
                crate::actions::apply(&mut game, seat, action).map_err(|e| e.to_string())?;
            }
        }
        if !acted {
            return Err(format!("tournament stalled: {seed}"));
        }
    }
    Ok(game.players.iter().map(|p| p.placement).collect())
}

fn mean_se(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n.max(1) as f64;
    let se = if n > 1 {
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
            / (n as f64).sqrt()
    } else {
        0.0
    };
    (mean, se)
}

pub fn run(args: TournamentArgs) -> Result<(), String> {
    if args.models.len() < 2 || args.models.len() > 8 {
        return Err("between 2 and 8 models are needed".into());
    }
    if args.games < 1 || args.sims < 1 {
        return Err("games and sims must be positive".into());
    }
    let search = SearchOptions {
        sims: args.sims,
        c_puct: args.c_puct,
        horizon: args.horizon,
        ..SearchOptions::default()
    };
    search.validate()?;
    let mut names = Vec::new();
    let mut nets = Vec::new();
    for spec in &args.models {
        let (name, path) = spec.split_once('=').ok_or("--model expects name=path")?;
        names.push(name.to_string());
        nets.push(PolicyValueNet::load(&PathBuf::from(path))?);
    }
    let n_models = nets.len();
    // Seatings: a random permutation per game, drawn once so results are reproducible.
    let mut seating_rng = Pcg64Mcg::seed_from_u64(args.seed ^ 0x5345_4154);
    let seatings: Vec<Vec<usize>> = (0..args.games)
        .map(|_| {
            let mut seats: Vec<usize> = (0..8).map(|s| s % n_models).collect();
            seats.shuffle(&mut seating_rng);
            seats
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers.max(1).min(args.games))
        .build()
        .map_err(|e| e.to_string())?;
    let results: Vec<Vec<u8>> = pool.install(|| {
        seatings
            .par_iter()
            .enumerate()
            .map(|(i, seating)| play(&nets, seating, args.seed + i as u64, &args.budgets, &search))
            .collect::<Result<_, _>>()
    })?;
    // Per game, a model with several seats contributes its mean placement.
    let mut per_model: Vec<Vec<f64>> = vec![Vec::new(); n_models];
    let mut firsts = vec![0usize; n_models];
    let mut top4 = vec![0usize; n_models];
    let mut seats_played = vec![0usize; n_models];
    for (seating, places) in seatings.iter().zip(&results) {
        let mut sums = vec![(0.0f64, 0usize); n_models];
        for (seat, &m) in seating.iter().enumerate() {
            sums[m].0 += places[seat] as f64;
            sums[m].1 += 1;
            seats_played[m] += 1;
            firsts[m] += (places[seat] == 1) as usize;
            top4[m] += (places[seat] <= 4) as usize;
        }
        for (m, (total, count)) in sums.into_iter().enumerate() {
            if count > 0 {
                per_model[m].push(total / count as f64);
            }
        }
    }
    let mut summary: Vec<(String, Value)> = Vec::new();
    for m in 0..n_models {
        let (mean, se) = mean_se(&per_model[m]);
        summary.push((
            names[m].clone(),
            json!({
                "mean_placement": mean,
                "se": se,
                "first_rate": firsts[m] as f64 / seats_played[m].max(1) as f64,
                "top4_rate": top4[m] as f64 / seats_played[m].max(1) as f64,
                "seats_played": seats_played[m],
            }),
        ));
    }
    summary.sort_by(|a, b| {
        a.1["mean_placement"]
            .as_f64()
            .partial_cmp(&b.1["mean_placement"].as_f64())
            .unwrap()
    });
    println!(
        "{:<24} {:>8} {:>7} {:>7} {:>7}",
        "model", "mean", "se", "first", "top4"
    );
    for (name, s) in &summary {
        println!(
            "{name:<24} {:8.3} {:7.3} {:6.1}% {:6.1}%",
            s["mean_placement"].as_f64().unwrap(),
            s["se"].as_f64().unwrap(),
            100.0 * s["first_rate"].as_f64().unwrap(),
            100.0 * s["top4_rate"].as_f64().unwrap()
        );
    }
    let report = json!({
        "models": names.iter().zip(&args.models).map(|(n, spec)| (n.clone(), spec.split_once('=').unwrap().1.to_string())).collect::<BTreeMap<_, _>>(),
        "games": args.games,
        "seed": args.seed,
        "sims": args.sims,
        "c_puct": args.c_puct,
        "horizon": args.horizon.name(),
        "search_version": mcts::SEARCH_VERSION,
        "budgets": args.budgets,
        "summary": summary.iter().cloned().collect::<BTreeMap<String, Value>>(),
        "ranking": summary.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "seatings": seatings,
        "placements": results,
    });
    crate::npz::ensure_parent(&args.out)?;
    std::fs::write(&args.out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    Ok(())
}
