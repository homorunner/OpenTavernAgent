//! Eight distinct checkpoints per table; full rankings for the shared league Elo fitter.
use crate::{
    checkpoint,
    heroes::select_heroes,
    net::PolicyValueNet,
    observation::observe_seat_with_rules,
    policy,
    rollout::game_rng,
    rules::{ActionHistory, PolicyRules},
};
use clap::Args;
use rand::{SeedableRng, seq::SliceRandom};
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use serde::Serialize;
use std::{collections::BTreeSet, io::Write, path::PathBuf, sync::Mutex};
use taverngym::{Game, RulesConfig};

#[derive(Debug, Args)]
pub struct LeagueArgs {
    /// Distinct checkpoint identities, name=path. At least eight are required.
    #[arg(long = "model", required = true)]
    pub models: Vec<String>,
    /// Each round seats every model once, padding the last table without duplicates.
    #[arg(long, default_value_t = 256)]
    pub rounds: usize,
    /// Explicit JSON array of eight-name tables; overrides --rounds.
    #[arg(long)]
    pub tables: Option<PathBuf>,
    #[arg(long, default_value_t = 60_000_000)]
    pub seed: u64,
    #[arg(long, default_value_t = 16)]
    pub workers: usize,
    /// New JSONL output. Published atomically; an interrupted .partial may be retried.
    #[arg(long)]
    pub out: PathBuf,
}

pub struct LeagueModel {
    pub net: PolicyValueNet,
    pub rules: PolicyRules,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct LeagueGame {
    pub placements: Vec<u8>,
    pub turn_limit_end: bool,
    pub truncated_combats: u32,
}

pub fn play(
    models: &[LeagueModel],
    table: &[usize],
    cfg: &RulesConfig,
    seed: u64,
) -> Result<LeagueGame, String> {
    cfg.validate()?;
    if cfg.num_players != 8 || table.len() != 8 || table.iter().any(|&m| m >= models.len()) {
        return Err("league requires eight valid seats".into());
    }
    let mut game = Game::new(seed, cfg.clone());
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    let mut histories = [ActionHistory::default(); 8];
    while !game.is_over() {
        if game.stopped_reason.is_some() {
            return Err(format!("league stopped: {seed}"));
        }
        let mut acted = false;
        for seat in 0..8 {
            if !game.needs_action(seat) {
                continue;
            }
            let model = &models[table[seat]];
            let turn = game.turn;
            let obs = observe_seat_with_rules(&game, seat, model.rules, &histories[seat]);
            let decision = policy::decide(&model.net, &obs, &mut rng, true)?;
            crate::actions::apply_with_rules(&mut game, seat, decision.action, model.rules)
                .map_err(str::to_owned)?;
            histories[seat].record(turn, decision.action);
            acted = true;
        }
        if !acted {
            return Err(format!("league stalled: {seed}"));
        }
    }
    let placements: Vec<_> = game.players.iter().map(|p| p.placement).collect();
    let mut sorted = placements.clone();
    sorted.sort_unstable();
    if sorted != (1..=8).collect::<Vec<_>>() || game.stopped_reason.is_some() {
        return Err(format!("league missing complete ranking: {seed}"));
    }
    Ok(LeagueGame {
        placements,
        turn_limit_end: game.turn_limit_end,
        truncated_combats: game.truncated_combats,
    })
}

fn resolve_tables(
    names: &[String],
    requested: Vec<Vec<String>>,
) -> Result<Vec<Vec<usize>>, String> {
    if requested.is_empty() {
        return Err("league tables must not be empty".into());
    }
    requested
        .into_iter()
        .map(|table| {
            if table.len() != 8 || table.iter().collect::<BTreeSet<_>>().len() != 8 {
                return Err("each league table needs eight distinct model names".into());
            }
            table
                .iter()
                .map(|name| {
                    names
                        .iter()
                        .position(|n| n == name)
                        .ok_or_else(|| format!("unknown league model: {name}"))
                })
                .collect()
        })
        .collect()
}

fn shuffled_tables(n: usize, rounds: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed ^ 0x4C45_4147);
    let mut tables = Vec::new();
    for _ in 0..rounds {
        let mut order: Vec<_> = (0..n).collect();
        order.shuffle(&mut rng);
        let pad = (8 - n % 8) % 8;
        let padded: Vec<_> = order
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
    tables
}

pub fn run(args: LeagueArgs) -> Result<(), String> {
    if args.models.len() < 8 || args.workers == 0 || (args.tables.is_none() && args.rounds == 0) {
        return Err("league needs at least eight models, positive workers and rounds".into());
    }
    if args.out.exists() {
        return Err("league output already exists; use a new output path".into());
    }
    let specs = args
        .models
        .iter()
        .map(|s| {
            let (name, path) = s.split_once('=').ok_or("--model expects name=path")?;
            if name.is_empty() || path.is_empty() {
                return Err("league names and paths must not be empty");
            }
            Ok((name.to_owned(), PathBuf::from(path)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let names: Vec<_> = specs.iter().map(|(n, _)| n.clone()).collect();
    if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
        return Err("league model names must be unique".into());
    }
    let tables = if let Some(path) = &args.tables {
        let requested = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        resolve_tables(&names, requested)?
    } else {
        shuffled_tables(names.len(), args.rounds, args.seed)
    };
    args.seed
        .checked_add((tables.len() - 1) as u64)
        .ok_or("league seed overflow")?;
    let mut models = Vec::new();
    let mut cfg = None;
    let mut identities = Vec::new();
    for (name, path) in &specs {
        let (trainer, state) = checkpoint::load(path)?;
        let model_cfg = state.options.game_config()?;
        if let Some(expected) = &cfg {
            if serde_json::to_value(expected).map_err(|e| e.to_string())?
                != serde_json::to_value(&model_cfg).map_err(|e| e.to_string())?
            {
                return Err("league checkpoints require identical engine rules".into());
            }
        } else {
            cfg = Some(model_cfg);
        }
        identities.push(serde_json::json!({"name":name,"iteration":state.iteration,
            "training_seed":state.options.seed,"policy_rules":state.options.policy_rules}));
        models.push(LeagueModel {
            net: trainer.net,
            rules: state.options.policy_rules,
        });
    }
    let cfg = cfg.unwrap();
    crate::npz::ensure_parent(&args.out)?;
    let partial = args.out.with_extension("jsonl.partial");
    let file = Mutex::new(std::fs::File::create(&partial).map_err(|e| e.to_string())?);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()
        .map_err(|e| e.to_string())?;
    pool.install(|| tables.par_iter().enumerate().try_for_each(|(i, table)| {
        let seed = args.seed + i as u64;
        let result = play(&models, table, &cfg, seed)?;
        let row = serde_json::json!({"game":i,"seed":seed,
            "models":table.iter().map(|&m| &names[m]).collect::<Vec<_>>(),
            "placements":result.placements,"turn_limit_end":result.turn_limit_end,
            "truncated_combats":result.truncated_combats,"backend":"cpu","inference":"fp32","sample":false});
        writeln!(file.lock().map_err(|e| e.to_string())?, "{row}").map_err(|e| e.to_string())
    }))?;
    file.into_inner()
        .map_err(|e| e.to_string())?
        .sync_all()
        .map_err(|e| e.to_string())?;
    std::fs::rename(partial, &args.out).map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::json!({"out":args.out,"games":tables.len(),"models":identities,"backend":"cpu"})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedules_cover_every_model_and_validate_distinct_seats() {
        for n in [8, 9, 15, 16, 21] {
            let tables = shuffled_tables(n, 3, 17);
            assert_eq!(tables, shuffled_tables(n, 3, 17));
            for table in &tables {
                assert_eq!(table.len(), 8);
                assert_eq!(table.iter().collect::<BTreeSet<_>>().len(), 8);
            }
            for model in 0..n {
                assert!(tables.iter().filter(|t| t.contains(&model)).count() >= 3);
            }
        }
        let names: Vec<_> = (0..9).map(|i| format!("m{i}")).collect();
        assert_eq!(
            resolve_tables(&names, vec![names[1..].to_vec()]).unwrap(),
            vec![(1..9).collect::<Vec<_>>()]
        );
        assert!(resolve_tables(&names, vec![]).is_err());
        assert!(resolve_tables(&names, vec![vec!["m0".into(); 8]]).is_err());
        let mut bad = names[..8].to_vec();
        bad[0] = "unknown".into();
        assert!(resolve_tables(&names, vec![bad]).is_err());
    }
}
