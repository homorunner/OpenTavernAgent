//! Direct FP32 policy evaluation against seven fixed scripted or teacher bots.

use clap::{Args, ValueEnum};
use rayon::prelude::*;
use serde_json::json;
use std::path::PathBuf;
use taverngym::{Game, RulesConfig};

use crate::{
    checkpoint,
    diagnostics::{self, GameDiagnostics},
    heroes::select_heroes,
    net::PolicyValueNet,
    observation::observe_seat_with_rules,
    policy,
    rollout::game_rng,
    rules::{ActionHistory, PolicyRules},
};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum Opponent {
    #[default]
    Scripted,
    Teacher,
}

impl Opponent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scripted => "scripted",
            Self::Teacher => "teacher",
        }
    }
}

#[derive(Debug, Args)]
pub struct EvalArgs {
    #[arg(long,value_enum,default_value_t=crate::backend::Backend::Cpu)]
    pub backend: crate::backend::Backend,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t = 64)]
    pub inference_batch: usize,
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long, default_value_t = 128)]
    pub games: usize,
    #[arg(long, default_value_t = 16)]
    pub workers: usize,
    #[arg(long, default_value_t = 10_000_000)]
    pub seed: u64,
    /// Sample from the policy; otherwise choose the most probable legal action.
    #[arg(long)]
    pub sample: bool,
    /// The other seven seats use this fixed policy, in engine action coordinates.
    #[arg(long, value_enum, default_value_t = Opponent::Scripted)]
    pub opponent: Opponent,
    /// Optionally persist the full JSON result, including per-seed placements.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

pub fn play(
    net: &PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
    sample: bool,
) -> Result<u8, String> {
    play_against(net, seed, cfg, sample, Opponent::Scripted)
}

pub fn play_against(
    net: &PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
    sample: bool,
    opponent: Opponent,
) -> Result<u8, String> {
    Ok(play_detailed(net, seed, cfg, sample, opponent, PolicyRules::default())?.placement)
}

#[derive(Debug)]
pub struct GameEvaluation {
    pub placement: u8,
    pub diagnostics: GameDiagnostics,
    pub turn_limit_end: bool,
    pub truncated_combats: u32,
}

pub fn play_detailed(
    net: &PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
    sample: bool,
    opponent: Opponent,
    rules: PolicyRules,
) -> Result<GameEvaluation, String> {
    cfg.validate()?;
    let mut game = Game::new(seed, cfg.clone());
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    let mut history = ActionHistory::default();
    let mut diagnostics = GameDiagnostics::default();
    while !game.is_over() && game.players[0].alive {
        if game.stopped_reason.is_some() {
            return Err(format!("evaluation stopped: seed={seed}"));
        }
        let mut acted = false;
        for seat in 0..game.players.len() {
            if !game.needs_action(seat) {
                continue;
            }
            if seat == 0 {
                let turn = game.turn;
                let decision = policy::decide(
                    net,
                    &observe_seat_with_rules(&game, 0, rules, &history),
                    &mut rng,
                    !sample,
                )?;
                diagnostics.record(&game, 0, &history, decision.action);
                crate::actions::apply(&mut game, 0, decision.action).map_err(str::to_string)?;
                history.record(turn, decision.action);
            } else {
                // Scripted bots use engine coordinates, and their random choices must
                // not move the environment's random stream.
                let saved = game.rng.clone();
                let action = match opponent {
                    Opponent::Scripted => taverngym::bot::choose_action(&mut game, seat),
                    Opponent::Teacher => crate::teacher::choose_action(&mut game, seat),
                };
                game.rng = saved;
                game.apply(seat, action).map_err(str::to_string)?;
            }
            acted = true;
        }
        if !acted {
            return Err(format!("evaluation stalled: seed={seed}"));
        }
    }
    let placement = game.players[0].placement;
    if placement == 0 || game.stopped_reason.is_some() {
        return Err("evaluation has no valid result".into());
    }
    Ok(GameEvaluation {
        placement,
        diagnostics,
        turn_limit_end: game.turn_limit_end,
        truncated_combats: game.truncated_combats,
    })
}

pub fn run(args: EvalArgs) -> Result<(), String> {
    if args.games == 0 || args.workers == 0 {
        return Err("games and workers must be positive".into());
    }
    let (trainer, state) = checkpoint::load(&args.model)?;
    let cfg = state.options.game_config()?;
    let seeds = (0..args.games)
        .map(|i| args.seed.checked_add(i as u64).ok_or("seed overflow"))
        .collect::<Result<Vec<_>, _>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()
        .map_err(|e| e.to_string())?;
    let games = if args.backend == crate::backend::Backend::Cuda {
        let execution = crate::backend::Execution::new(args.backend, &trainer, args.device)?;
        crate::batched_eval::evaluate(
            &trainer.net,
            &execution,
            &seeds,
            &cfg,
            state.options.policy_rules,
            args.opponent,
            args.sample,
            args.inference_batch,
            &pool,
        )?
    } else {
        pool.install(|| {
            seeds
                .par_iter()
                .map(|&seed| {
                    play_detailed(
                        &trainer.net,
                        seed,
                        &cfg,
                        args.sample,
                        args.opponent,
                        state.options.policy_rules,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })?
    };
    let places: Vec<_> = games.iter().map(|g| g.placement).collect();
    let turn_limit_games = games.iter().filter(|g| g.turn_limit_end).count();
    let truncated_combats: u64 = games.iter().map(|g| u64::from(g.truncated_combats)).sum();
    let diagnostics =
        diagnostics::summarize(&games.into_iter().map(|g| g.diagnostics).collect::<Vec<_>>());
    let n = places.len() as f64;
    let mean = places.iter().map(|&p| p as f64).sum::<f64>() / n;
    let standard_error = if n > 1.0 {
        (places
            .iter()
            .map(|&p| (p as f64 - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0)
            / n)
            .sqrt()
    } else {
        0.0
    };
    let result = json!({"model": args.model, "iteration": state.iteration, "games": args.games,
        "seed": args.seed, "sample": args.sample, "inference": "fp32", "backend":args.backend,"opponent": args.opponent.name(),
        "model_seed": state.options.seed, "evaluated_seat": 0, "policy_rules":state.options.policy_rules,
        "diagnostics":diagnostics, "turn_limit_games":turn_limit_games,"truncated_combats":truncated_combats,
        "mean_placement": mean, "standard_error": standard_error,
        "mean_placement_ci95_normal": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "top4_rate": places.iter().filter(|&&p| p <= 4).count() as f64 / n,
        "win_rate": places.iter().filter(|&&p| p == 1).count() as f64 / n, "placements": places});
    if let Some(path) = &args.out {
        crate::npz::ensure_parent(path)?;
        let temp = path.with_extension("json.tmp");
        std::fs::write(
            &temp,
            serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        std::fs::rename(temp, path).map_err(|e| e.to_string())?;
    }
    println!("{result}");
    Ok(())
}
