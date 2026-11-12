//! FP32 evaluation against fixed bots or self-play, with optional full-lobby replay.

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Opponent {
    #[default]
    Scripted,
    Teacher,
    #[value(name = "selfplay")]
    SelfPlay,
}

impl Opponent {
    pub(crate) fn is_model(self, seat: usize) -> bool {
        seat == 0 || self == Self::SelfPlay
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Scripted => "scripted",
            Self::Teacher => "teacher",
            Self::SelfPlay => "selfplay",
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
    /// Other seats use a fixed bot, or this same checkpoint with selfplay.
    #[arg(long, value_enum, default_value_t = Opponent::Scripted)]
    pub opponent: Opponent,
    /// Optionally persist the full JSON result, including per-seed placements.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Export complete lobbies as JSONL for the offline replay viewer.
    #[arg(long)]
    pub replay: Option<PathBuf>,
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
    pub replay: Option<serde_json::Value>,
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
    let mut histories = vec![ActionHistory::default(); game.players.len()];
    let mut diagnostics = GameDiagnostics::default();
    while !game.is_over() && (opponent == Opponent::SelfPlay || game.players[0].alive) {
        if game.stopped_reason.is_some() {
            return Err(format!("evaluation stopped: seed={seed}"));
        }
        let mut acted = false;
        for (seat, history) in histories.iter_mut().enumerate() {
            if !game.needs_action(seat) {
                continue;
            }
            if opponent.is_model(seat) {
                let turn = game.turn;
                let decision = policy::decide(
                    net,
                    &observe_seat_with_rules(&game, seat, rules, history),
                    &mut rng,
                    !sample,
                )?;
                if seat == 0 {
                    diagnostics.record(&game, 0, history, decision.action);
                }
                crate::actions::apply_with_rules(&mut game, seat, decision.action, rules)
                    .map_err(str::to_string)?;
                history.record(turn, decision.action);
            } else {
                // Scripted bots use engine coordinates, and their random choices must
                // not move the environment's random stream.
                let saved = game.rng.clone();
                let action = match opponent {
                    Opponent::Scripted => taverngym::bot::choose_action(&mut game, seat),
                    Opponent::Teacher => crate::teacher::choose_action(&mut game, seat),
                    Opponent::SelfPlay => unreachable!(),
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
        replay: None,
    })
}

pub fn run(args: EvalArgs) -> Result<(), String> {
    if args.games == 0 || args.workers == 0 {
        return Err("games and workers must be positive".into());
    }
    if args.inference_batch == 0 {
        return Err("inference-batch must be positive".into());
    }
    if let Some(path) = &args.replay {
        crate::replay::validate_paths(&args.model, args.out.as_deref(), path)?;
    }
    let (trainer, state) = checkpoint::load(&args.model)?;
    let mut replay_writer = args
        .replay
        .as_deref()
        .map(crate::replay::JsonlWriter::new)
        .transpose()?;
    let cfg = state.options.game_config()?;
    let seeds = (0..args.games)
        .map(|i| args.seed.checked_add(i as u64).ok_or("seed overflow"))
        .collect::<Result<Vec<_>, _>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()
        .map_err(|e| e.to_string())?;
    let games = if args.backend == crate::backend::Backend::Cuda || replay_writer.is_some() {
        let execution = crate::backend::Execution::new(args.backend, &trainer, args.device)?;
        let mut results = Vec::with_capacity(seeds.len());
        // Bound replay memory to at most four complete games, regardless of --games.
        let chunk_size = if replay_writer.is_some() {
            args.inference_batch.min(4)
        } else {
            args.inference_batch
        };
        for chunk in seeds.chunks(chunk_size) {
            let mut chunk_results = crate::batched_eval::evaluate_recorded(
                &trainer.net,
                &execution,
                chunk,
                &cfg,
                state.options.policy_rules,
                args.opponent,
                args.sample,
                chunk_size,
                &pool,
                replay_writer.is_some(),
            )?;
            for result in &mut chunk_results {
                if let Some(mut replay) = result.replay.take() {
                    replay["model"] = json!({"file":args.model.file_name(),"iteration":state.iteration,
                        "training_seed":state.options.seed.to_string(),"policy_rules":state.options.policy_rules});
                    replay["opponent"] = json!(args.opponent.name());
                    replay["sample"] = json!(args.sample);
                    replay["backend"] = json!(args.backend);
                    replay["rules"] = serde_json::to_value(&cfg).map_err(|e| e.to_string())?;
                    replay_writer
                        .as_mut()
                        .ok_or("missing replay writer")?
                        .write(&replay)?;
                }
            }
            results.extend(chunk_results);
        }
        results
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
    if let Some(writer) = replay_writer {
        writer.finish()?;
    }
    println!("{result}");
    Ok(())
}
