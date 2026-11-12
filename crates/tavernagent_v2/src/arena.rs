//! Four candidate policies vs four frozen references, with paired seat complements.
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
use clap::Args;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;
use std::path::PathBuf;
use taverngym::{Game, RulesConfig};

#[derive(Debug, Args)]
pub struct ArenaArgs {
    #[arg(long,value_enum,default_value_t=crate::backend::Backend::Cpu)]
    pub backend: crate::backend::Backend,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t = 64)]
    pub inference_batch: usize,
    #[arg(long)]
    pub candidate: PathBuf,
    #[arg(long)]
    pub reference: PathBuf,
    /// Even number of games: two complementary seat assignments for each environment seed.
    #[arg(long, default_value_t = 1024)]
    pub games: usize,
    #[arg(long, default_value_t = 16)]
    pub workers: usize,
    #[arg(long, default_value_t = 20_000_000)]
    pub seed: u64,
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct ArenaGame {
    pub seed: u64,
    pub offset: usize,
    pub candidate_seats: Vec<usize>,
    pub placements: Vec<u8>,
    pub candidate_mean_placement: f64,
    pub candidate_won: bool,
    pub candidate_top4_share: f64,
    pub turn_limit_end: bool,
    pub truncated_combats: u32,
    #[serde(skip)]
    pub diagnostics: Vec<GameDiagnostics>,
}

pub fn play(
    candidate: &PolicyValueNet,
    reference: &PolicyValueNet,
    cfg: &RulesConfig,
    candidate_rules: PolicyRules,
    reference_rules: PolicyRules,
    seed: u64,
    offset: usize,
) -> Result<ArenaGame, String> {
    cfg.validate()?;
    if cfg.num_players != 8 || offset >= 8 {
        return Err("arena requires eight seats and offset < 8".into());
    }
    let is_candidate = |seat: usize| (seat + offset) % 8 < 4;
    let mut game = Game::new(seed, cfg.clone());
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    let mut histories = [ActionHistory::default(); 8];
    let mut diagnostics = vec![GameDiagnostics::default(); 8];
    while !game.is_over() {
        if game.stopped_reason.is_some() {
            return Err(format!("arena stopped: seed={seed}"));
        }
        let mut acted = false;
        for seat in 0..8 {
            if !game.needs_action(seat) {
                continue;
            }
            let (net, rules) = if is_candidate(seat) {
                (candidate, candidate_rules)
            } else {
                (reference, reference_rules)
            };
            let turn = game.turn;
            let obs = observe_seat_with_rules(&game, seat, rules, &histories[seat]);
            let decision = policy::decide(net, &obs, &mut rng, true)?;
            if is_candidate(seat) {
                diagnostics[seat].record(&game, seat, &histories[seat], decision.action);
            }
            crate::actions::apply_with_rules(&mut game, seat, decision.action, rules)
                .map_err(str::to_string)?;
            histories[seat].record(turn, decision.action);
            acted = true;
        }
        if !acted {
            return Err(format!("arena stalled: seed={seed}"));
        }
    }
    finish(game, offset, diagnostics)
}

pub(crate) fn finish(
    game: Game,
    offset: usize,
    diagnostics: Vec<GameDiagnostics>,
) -> Result<ArenaGame, String> {
    let is_candidate = |seat: usize| (seat + offset) % 8 < 4;
    let seed = game.seed;
    let placements: Vec<_> = game.players.iter().map(|p| p.placement).collect();
    if placements.contains(&0) || game.stopped_reason.is_some() {
        return Err("arena missing placement".into());
    }
    let candidate_seats: Vec<_> = (0..8).filter(|&s| is_candidate(s)).collect();
    let candidate_mean_placement = candidate_seats
        .iter()
        .map(|&s| f64::from(placements[s]))
        .sum::<f64>()
        / 4.0;
    let candidate_won = candidate_seats.iter().any(|&s| placements[s] == 1);
    let candidate_top4_share = candidate_seats
        .iter()
        .filter(|&&s| placements[s] <= 4)
        .count() as f64
        / 4.0;
    Ok(ArenaGame {
        seed,
        offset,
        candidate_mean_placement,
        candidate_won,
        candidate_top4_share,
        candidate_seats,
        placements,
        turn_limit_end: game.turn_limit_end,
        truncated_combats: game.truncated_combats,
        diagnostics: diagnostics
            .into_iter()
            .enumerate()
            .filter_map(|(s, d)| is_candidate(s).then_some(d))
            .collect(),
    })
}

pub fn run(args: ArenaArgs) -> Result<(), String> {
    if args.games == 0 || !args.games.is_multiple_of(2) || args.workers == 0 {
        return Err("arena games must be positive and even, workers positive".into());
    }
    let (candidate, state) = checkpoint::load(&args.candidate)?;
    let (reference, reference_state) = checkpoint::load(&args.reference)?;
    let cfg = state.options.game_config()?;
    if serde_json::to_value(&cfg).map_err(|e| e.to_string())?
        != serde_json::to_value(reference_state.options.game_config()?)
            .map_err(|e| e.to_string())?
    {
        return Err("arena checkpoints require identical engine rules".into());
    }
    let seeds = (0..args.games / 2)
        .map(|i| args.seed.checked_add(i as u64).ok_or("seed overflow"))
        .collect::<Result<Vec<_>, _>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()
        .map_err(|e| e.to_string())?;
    let mut games = if args.backend == crate::backend::Backend::Cuda {
        let a = crate::backend::Execution::new(args.backend, &candidate, args.device)?;
        let b = crate::backend::Execution::new(args.backend, &reference, args.device)?;
        crate::batched_arena::evaluate(
            &candidate.net,
            &reference.net,
            &a,
            &b,
            &cfg,
            state.options.policy_rules,
            reference_state.options.policy_rules,
            &seeds,
            args.inference_batch,
            &pool,
        )?
    } else {
        pool.install(|| {
            (0..args.games)
                .into_par_iter()
                .map(|i| {
                    let offset = (i / 2) % 4 + 4 * (i % 2);
                    play(
                        &candidate.net,
                        &reference.net,
                        &cfg,
                        state.options.policy_rules,
                        reference_state.options.policy_rules,
                        seeds[i / 2],
                        offset,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })?
    };
    let pair_means: Vec<_> = games
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0].candidate_mean_placement + pair[1].candidate_mean_placement) / 2.0)
        .collect();
    let mean = pair_means.iter().sum::<f64>() / pair_means.len() as f64;
    let se = if pair_means.len() > 1 {
        (pair_means.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (pair_means.len() - 1) as f64
            / pair_means.len() as f64)
            .sqrt()
    } else {
        0.0
    };
    let all_diagnostics: Vec<_> = games
        .iter_mut()
        .flat_map(|g| std::mem::take(&mut g.diagnostics))
        .collect();
    let result = json!({"candidate":args.candidate,"reference":args.reference,"iteration":state.iteration,
        "model_seed":state.options.seed,"reference_iteration":reference_state.iteration,
        "policy_rules":state.options.policy_rules,"reference_policy_rules":reference_state.options.policy_rules,
        "games":args.games,"seed":args.seed,"paired_seeds":pair_means.len(),"sample":false,"inference":"fp32",
        "candidate_mean_placement":mean,"candidate_minus_neutral":mean-4.5,
        "paired_seed_standard_error":se,"mean_placement_ci95_normal":[mean-1.96*se,mean+1.96*se],"backend":args.backend,
        "candidate_team_win_rate":games.iter().filter(|g|g.candidate_won).count() as f64/args.games as f64,
        "candidate_top4_share":games.iter().map(|g|g.candidate_top4_share).sum::<f64>()/args.games as f64,
        "turn_limit_games":games.iter().filter(|g|g.turn_limit_end).count(),
        "truncated_combats":games.iter().map(|g|u64::from(g.truncated_combats)).sum::<u64>(),
        "diagnostics":diagnostics::summarize(&all_diagnostics),"matches":games});
    crate::npz::ensure_parent(&args.out)?;
    let temp = args.out.with_extension("json.tmp");
    std::fs::write(
        &temp,
        serde_json::to_vec(&result).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::rename(temp, &args.out).map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"out":args.out,"candidate_mean_placement":mean,"games":args.games})
    );
    Ok(())
}
