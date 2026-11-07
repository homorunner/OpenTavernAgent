//! Evaluate checkpoints against the scripted bot on paired seeds.
//!
//! Seat 0 is played by each checkpoint and by the bot over the same seeds; reports mean
//! placement, standard error and paired differences. Paired seeds matter: an 8-player
//! placement has a standard deviation of ~2.2, so small samples can be off by whole
//! placements.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Args;
use rayon::prelude::*;
use serde_json::{Value, json};
use taverngym::Game;

use crate::heroes::select_heroes;
use crate::mcts::{self, Horizon, SearchOptions, bot_action};
use crate::net::PolicyValueNet;
use crate::rules::Budgets;
use crate::selfplay::game_rng;

#[derive(Clone, Debug, Args)]
pub struct EvalArgs {
    #[arg(long, default_value = "models/demo.npz")]
    pub model: PathBuf,
    #[arg(long, default_value_t = 40)]
    pub games: usize,
    #[arg(long, default_value_t = 48)]
    pub sims: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[command(flatten)]
    pub budgets: Budgets,
    #[arg(long, default_value = "turn", value_parser = Horizon::parse)]
    pub horizon: Horizon,
    #[arg(long, default_value_t = crate::mcts::DEFAULT_C_PUCT)]
    pub c_puct: f32,
    /// LCB move selection: stdevs below the mean (0 = plain visit argmax).
    #[arg(long, default_value_t = crate::mcts::DEFAULT_LCB_STDEVS)]
    pub lcb_stdevs: f32,
    /// First-play urgency reduction (negative = unvisited edges at Q = 0).
    #[arg(long, default_value_t = crate::mcts::DEFAULT_FPU_REDUCTION)]
    pub fpu_reduction: f32,
    /// Skip the scripted-bot baseline.
    #[arg(long, default_value_t = false)]
    pub no_baselines: bool,
    #[arg(long, default_value_t = 1)]
    pub workers: usize,
    /// Save per-seed placements and paired statistics as JSON.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Additional checkpoints as name=path.
    #[arg(long = "compare-model")]
    pub compare_model: Vec<String>,
}

pub enum Policy<'a> {
    Net(&'a PolicyValueNet),
    Bot,
}

/// Seat 0 under `policy` against seven scripted bots; returns seat 0's placement.
pub fn play(
    policy: &Policy,
    seed: u64,
    budgets: &Budgets,
    search: &SearchOptions,
) -> Result<u8, String> {
    let mut game = Game::new(seed, budgets.config()?);
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    while !game.is_over() {
        if game.needs_action(0) {
            let action = match policy {
                Policy::Net(net) => {
                    let a = mcts::search(&game, 0, *net, search, &mut rng)?.best_action();
                    crate::actions::engine_action(&game, 0, a).map_err(str::to_string)?
                }
                Policy::Bot => bot_action(&mut game, 0),
            };
            game.apply(0, action).map_err(|e| e.to_string())?;
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
            return Err(format!(
                "evaluation stalled: seed={seed}, turn={}",
                game.turn
            ));
        }
    }
    Ok(game.players[0].placement)
}

pub fn summarize(places: &[u8]) -> Value {
    let n = places.len();
    let mean = places.iter().map(|p| *p as f64).sum::<f64>() / n.max(1) as f64;
    let sd = if n > 1 {
        (places
            .iter()
            .map(|p| (*p as f64 - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    json!({
        "n": n,
        "mean": mean,
        "se": sd / (n.max(1) as f64).sqrt(),
        "first_rate": places.iter().filter(|p| **p == 1).count() as f64 / n.max(1) as f64,
        "top4_rate": places.iter().filter(|p| **p <= 4).count() as f64 / n.max(1) as f64,
    })
}

pub fn run(args: EvalArgs) -> Result<(), String> {
    if args.games < 1 || args.sims < 1 || args.workers < 1 {
        return Err("games, sims and workers must be positive".into());
    }
    let search = SearchOptions {
        sims: args.sims,
        c_puct: args.c_puct,
        horizon: args.horizon,
        lcb_stdevs: args.lcb_stdevs,
        fpu_reduction: args.fpu_reduction,
        ..SearchOptions::default()
    };
    search.validate()?;
    args.budgets.validate()?;
    let mut models: Vec<(String, PathBuf)> = vec![("network".into(), args.model.clone())];
    for spec in &args.compare_model {
        let (name, path) = spec
            .split_once('=')
            .ok_or("--compare-model expects name=path")?;
        models.push((name.into(), PathBuf::from(path)));
    }
    let nets: Vec<(String, PolicyValueNet)> = models
        .iter()
        .map(|(name, path)| Ok((name.clone(), PolicyValueNet::load(path)?)))
        .collect::<Result<_, String>>()?;
    let seeds: Vec<u64> = (args.seed..args.seed + args.games as u64).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers.min(args.games))
        .build()
        .map_err(|e| e.to_string())?;
    let rows: Vec<BTreeMap<String, u8>> = pool.install(|| {
        seeds
            .par_iter()
            .map(|&seed| {
                let mut row = BTreeMap::new();
                for (name, net) in &nets {
                    row.insert(
                        name.clone(),
                        play(&Policy::Net(net), seed, &args.budgets, &search)?,
                    );
                }
                if !args.no_baselines {
                    row.insert(
                        "bot".into(),
                        play(&Policy::Bot, seed, &args.budgets, &search)?,
                    );
                }
                Ok::<_, String>(row)
            })
            .collect::<Result<_, _>>()
    })?;
    let mut placements: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for row in &rows {
        for (name, place) in row {
            placements.entry(name.clone()).or_default().push(*place);
        }
    }
    let mut paired = serde_json::Map::new();
    if let Some(bot) = placements.get("bot") {
        for (name, places) in &placements {
            if name == "bot" {
                continue;
            }
            let delta: Vec<f64> = places
                .iter()
                .zip(bot)
                .map(|(a, b)| *a as f64 - *b as f64)
                .collect();
            let mean = delta.iter().sum::<f64>() / delta.len() as f64;
            let se = if delta.len() > 1 {
                (delta.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (delta.len() - 1) as f64)
                    .sqrt()
                    / (delta.len() as f64).sqrt()
            } else {
                0.0
            };
            paired.insert(
                name.clone(),
                json!({
                    "mean_delta": mean,
                    "se": se,
                    "ci95": [mean - 1.96 * se, mean + 1.96 * se],
                    "significantly_better": mean + 1.96 * se < 0.0,
                }),
            );
        }
    }
    for (name, places) in &placements {
        let s = summarize(places);
        println!(
            "{name:<14} n={:<4} mean {:.3}  se {:.3}  top4 {:.1}%",
            s["n"],
            s["mean"].as_f64().unwrap(),
            s["se"].as_f64().unwrap(),
            100.0 * s["top4_rate"].as_f64().unwrap()
        );
    }
    for (name, stats) in &paired {
        println!(
            "{name} - bot: {:+.3} placements, 95% CI {}, significantly better: {}",
            stats["mean_delta"].as_f64().unwrap(),
            stats["ci95"],
            stats["significantly_better"]
        );
    }
    if let Some(out) = &args.out {
        let result = json!({
            "seeds": seeds,
            "sims": args.sims,
            "horizon": args.horizon.name(),
            "c_puct": args.c_puct,
            "search_version": mcts::SEARCH_VERSION,
            "budgets": args.budgets,
            "models": models.iter().map(|(n, p)| (n.clone(), p.display().to_string())).collect::<BTreeMap<_, _>>(),
            "placements": placements,
            "summary": placements.iter().map(|(n, p)| (n.clone(), summarize(p))).collect::<BTreeMap<_, _>>(),
            "paired_vs_bot": paired,
        });
        crate::npz::ensure_parent(out)?;
        std::fs::write(out, serde_json::to_string_pretty(&result).unwrap())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
