//! A league of many checkpoints: each round shuffles the models into tables of eight
//! (padding the last table from the start of the shuffled list), so every model plays the
//! same number of games up to the padding, and seats are random. Writes one JSON line
//! per game; `tools/league_elo.py` fits ratings from the placements.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use clap::Args;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use taverngym::Game;

use crate::heroes::select_heroes;
use crate::mcts::{self, Horizon, SearchOptions};
use crate::net::PolicyValueNet;
use crate::rules::Budgets;
use crate::selfplay::game_rng;

#[derive(Clone, Debug, Args)]
pub struct LeagueArgs {
    /// Checkpoints as name=path (at least 8).
    #[arg(long = "model", required = true)]
    pub models: Vec<String>,
    /// Rounds; each round seats every model once (plus padding), i.e. ceil(N/8) games.
    #[arg(long, default_value_t = 100)]
    pub rounds: usize,
    #[arg(long, default_value_t = 41_000_000)]
    pub seed: u64,
    #[arg(long, default_value_t = 64)]
    pub sims: usize,
    #[arg(long, default_value_t = crate::mcts::DEFAULT_C_PUCT)]
    pub c_puct: f32,
    #[arg(long, default_value = "turn", value_parser = Horizon::parse)]
    pub horizon: Horizon,
    #[arg(long, default_value_t = 64)]
    pub workers: usize,
    #[command(flatten)]
    pub budgets: Budgets,
    /// JSON lines: {"game", "seed", "models": [8 names], "placements": [8]} per game,
    /// appended as games finish (so a partial run is usable).
    #[arg(long)]
    pub out: PathBuf,
}

fn play(
    nets: &[PolicyValueNet],
    table: &[usize],
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
                let action =
                    mcts::search(&game, seat, &nets[table[seat]], search, &mut rng)?.best_action();
                game.apply_index(seat, action).map_err(|e| e.to_string())?;
            }
        }
        if !acted {
            return Err(format!("league game stalled: {seed}"));
        }
    }
    Ok(game.players.iter().map(|p| p.placement).collect())
}

pub fn run(args: LeagueArgs) -> Result<(), String> {
    if args.models.len() < 8 {
        return Err("a league needs at least 8 models".into());
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
    let n = nets.len();
    let mut rng = Pcg64Mcg::seed_from_u64(args.seed ^ 0x4C45_4147);
    let mut tables: Vec<Vec<usize>> = Vec::new();
    for _ in 0..args.rounds {
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        let pad = (8 - n % 8) % 8;
        // Pad the last table with models from the start of the shuffle (never a duplicate
        // within a table, since n >= 8).
        let padded: Vec<usize> = order
            .iter()
            .chain(order.iter().take(pad))
            .copied()
            .collect();
        for table in padded.chunks(8) {
            let mut seats = table.to_vec();
            seats.shuffle(&mut rng);
            tables.push(seats);
        }
    }
    crate::npz::ensure_parent(&args.out)?;
    let file = Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.out)
            .map_err(|e| e.to_string())?,
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers.max(1))
        .build()
        .map_err(|e| e.to_string())?;
    let counts: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
    pool.install(|| {
        tables.par_iter().enumerate().try_for_each(|(i, table)| {
            let seed = args.seed + i as u64;
            let places = play(&nets, table, seed, &args.budgets, &search)?;
            let row = serde_json::json!({
                "game": i,
                "seed": seed,
                "models": table.iter().map(|m| names[*m].clone()).collect::<Vec<_>>(),
                "placements": places,
            });
            let mut f = file.lock().unwrap();
            writeln!(f, "{row}").map_err(|e| e.to_string())?;
            let mut c = counts.lock().unwrap();
            for m in table {
                *c.entry(names[*m].clone()).or_insert(0) += 1;
            }
            Ok::<(), String>(())
        })
    })?;
    let counts = counts.into_inner().unwrap();
    let (lo, hi) = (
        counts.values().min().copied().unwrap_or(0),
        counts.values().max().copied().unwrap_or(0),
    );
    println!(
        "{} games, {} models, {lo}-{hi} games per model -> {}",
        tables.len(),
        n,
        args.out.display()
    );
    Ok(())
}
