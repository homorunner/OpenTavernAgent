//! Whole-game, per-turn action traces and conservative loop diagnostics.
//!
//! Diagnostics use only the controlled seat's visible preparation state. A repeated
//! state is a candidate loop, not proof of a useless action: budgets are deliberately
//! excluded from the signature, while resources, buffs and ability uses are retained.
//! `mean_gold_left` at End Turn is the direct measure of the early-stop failure mode.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use clap::Args;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use taverngym::Game;
use taverngym::actions::{A_BUY, N_SHOP};

use crate::heroes::select_heroes;
use crate::mcts::{self, A_END_TURN, Horizon, SearchOptions, bot_action, observe_seat};
use crate::net::PolicyValueNet;
use crate::probe::family;
use crate::rules::Budgets;
use crate::selfplay::{game_rng, sample_action};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Seat 0 plays greedily against seven scripted bots (like `eval`).
    Eval,
    /// All eight seats follow the training schedule: root noise and temperature.
    Selfplay,
}

#[derive(Clone, Debug, Args, Serialize)]
pub struct PatternsArgs {
    /// Checkpoint; omit to trace the scripted bot.
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    pub games: usize,
    #[arg(long, default_value_t = 54_000_000)]
    pub seed: u64,
    #[arg(long, default_value_t = 32)]
    pub sims: usize,
    #[arg(long, default_value_t = 64)]
    pub workers: usize,
    #[arg(long, default_value_t = crate::mcts::DEFAULT_C_PUCT)]
    pub c_puct: f32,
    #[arg(long, value_enum, default_value_t = Mode::Eval)]
    pub mode: Mode,
    #[arg(long, default_value_t = 0.2)]
    pub noise: f32,
    #[arg(long, default_value_t = 0.3)]
    pub noise_alpha: f32,
    #[arg(long, default_value_t = 0.5)]
    pub temperature: f32,
    #[command(flatten)]
    #[serde(flatten)]
    pub budgets: Budgets,
    #[arg(long)]
    pub out: PathBuf,
    /// Also write every per-action trace next to the report (`<out>.traces.json`).
    #[arg(long, default_value_t = false)]
    pub traces: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Step {
    action: usize,
    family: String,
    gold_before: i32,
    board_before: usize,
    tier_before: u8,
    before_key: u64,
    after_key: u64,
    remaining_before: [Option<u32>; 3],
    remaining_after: [Option<u32>; 3],
    buy_available: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Turn {
    turn: u32,
    initial_key: u64,
    actions: Vec<Step>,
}

/// Visible preparation state of seat 0.
fn material_key(game: &Game) -> u64 {
    let obs = observe_seat(game, 0);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in &obs.x {
        v.to_bits().hash(&mut h);
    }
    obs.cards.hash(&mut h);
    h.finish()
}

fn budgets_of(game: &Game, p: usize) -> [Option<u32>; 3] {
    let pl = &game.players[p];
    let total = game.action_cap_of(p);
    let remaining = |cap: u32, used: u32| {
        if cap == 0 {
            None
        } else {
            Some(cap.saturating_sub(used))
        }
    };
    [
        if pl.uncapped || total == u32::MAX {
            None
        } else {
            Some(total.saturating_sub(pl.actions_this_turn))
        },
        remaining(game.cfg.move_actions_per_turn, pl.moves_this_turn),
        remaining(game.cfg.freeze_presses_per_turn, pl.freezes_this_turn),
    ]
}

fn summarize_turn(turn: &Turn) -> Value {
    let steps = &turn.actions;
    let mut families: BTreeMap<String, usize> = BTreeMap::new();
    let mut seen = vec![turn.initial_key];
    let mut candidates = Vec::new();
    let mut noops = 0usize;
    let mut longest: BTreeMap<&str, usize> = [("Move", 0), ("Freeze", 0)].into_iter().collect();
    let mut streak = 0usize;
    let mut previous: Option<&str> = None;
    for (i, step) in steps.iter().enumerate() {
        *families.entry(step.family.clone()).or_insert(0) += 1;
        let kind = step.family.as_str();
        streak = if previous == Some(kind) {
            streak + 1
        } else {
            1
        };
        previous = Some(kind);
        if let Some(l) = longest.get_mut(kind) {
            *l = (*l).max(streak);
        }
        if kind == "Move" || kind == "Freeze" {
            noops += (step.before_key == step.after_key) as usize;
            if seen.contains(&step.after_key) {
                candidates.push(i);
            }
        }
        seen.push(step.after_key);
    }
    let last = steps.last();
    let ended = last.is_some_and(|s| s.action == A_END_TURN);
    let end = if ended { last } else { None };
    json!({
        "turn": turn.turn,
        "actions": steps.len(),
        "families": families,
        "candidate_cycle_closures": candidates.len(),
        "candidate_indices": candidates,
        "unchanged_move_freeze": noops,
        "longest_streak": longest,
        "gold_left": end.map(|s| s.gold_before),
        "end_board_size": end.map(|s| s.board_before),
        "tier": end.map(|s| s.tier_before),
        "forced_end": end.is_some_and(|s| s.remaining_before[0] == Some(0)),
        "ended_with_buy_available": end.is_some_and(|s| s.buy_available),
        "move_cap_reached": steps.iter().any(|s| s.remaining_after[1] == Some(0)),
        "freeze_cap_reached": steps.iter().any(|s| s.remaining_after[2] == Some(0)),
    })
}

fn trace_game(
    net: Option<&PolicyValueNet>,
    seed: u64,
    args: &PatternsArgs,
) -> Result<Value, String> {
    let mut game = Game::new(seed, args.budgets.config()?);
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    let selfplay = args.mode == Mode::Selfplay;
    let search = SearchOptions {
        sims: args.sims,
        c_puct: args.c_puct,
        horizon: Horizon::Turn,
        root_noise: if selfplay { args.noise } else { 0.0 },
        noise_alpha: args.noise_alpha,
        ..SearchOptions::default()
    };
    let mut turns: BTreeMap<u32, Turn> = BTreeMap::new();
    while !game.is_over() {
        let mut moved = false;
        let seats: Vec<usize> = if selfplay {
            (0..game.players.len()).collect()
        } else {
            vec![0]
        };
        for seat in seats {
            if !game.needs_action(seat) {
                continue;
            }
            moved = true;
            let action = match net {
                None => bot_action(&mut game, seat).encode(),
                Some(net) => {
                    let result = mcts::search(&game, seat, net, &search, &mut rng)?;
                    if selfplay && args.temperature > 0.0 {
                        sample_action(&result.policy, args.temperature, &mut rng)
                    } else {
                        result.best_action()
                    }
                }
            };
            let record = (seat == 0).then(|| {
                let before_key = material_key(&game);
                let mask = game.legal_mask(0);
                let turn = turns.entry(game.turn).or_insert_with(|| Turn {
                    turn: game.turn,
                    initial_key: before_key,
                    actions: Vec::new(),
                });
                let _ = turn;
                Step {
                    action,
                    family: family(action),
                    gold_before: game.players[0].gold,
                    board_before: game.players[0].board.len(),
                    tier_before: game.players[0].tier,
                    before_key,
                    after_key: 0,
                    remaining_before: budgets_of(&game, 0),
                    remaining_after: [None; 3],
                    buy_available: mask[A_BUY..A_BUY + N_SHOP].iter().any(|m| *m),
                }
            });
            let turn_number = game.turn;
            game.apply_index(seat, action).map_err(|e| e.to_string())?;
            if let Some(mut step) = record {
                step.after_key = material_key(&game);
                step.remaining_after = budgets_of(&game, 0);
                turns.get_mut(&turn_number).unwrap().actions.push(step);
            }
        }
        if !selfplay {
            // Match eval: the controlled seat completes its turn before the bots act.
            if game.needs_action(0) {
                continue;
            }
            for seat in 1..game.players.len() {
                if game.needs_action(seat) {
                    let a = bot_action(&mut game, seat);
                    game.apply(seat, a).map_err(|e| e.to_string())?;
                    moved = true;
                }
            }
        }
        if !moved && !game.is_over() {
            return Err(format!("pattern game stalled: seed={seed}"));
        }
    }
    let turns: Vec<Turn> = turns.into_values().collect();
    Ok(json!({
        "seed": seed,
        "placement": game.players[0].placement,
        "summaries": turns.iter().map(summarize_turn).collect::<Vec<_>>(),
        "turns": if args.traces { serde_json::to_value(&turns).unwrap() } else { Value::Null },
    }))
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = values.collect();
    if v.is_empty() {
        None
    } else {
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }
}

fn stats(ts: &[&Value]) -> Value {
    let ended: Vec<&Value> = ts
        .iter()
        .copied()
        .filter(|t| !t["gold_left"].is_null())
        .collect();
    let sum = |key: &str| {
        ts.iter()
            .map(|t| {
                t[key]
                    .as_u64()
                    .unwrap_or(t[key].as_bool().unwrap_or(false) as u64)
            })
            .sum::<u64>()
    };
    let mut families: BTreeMap<String, u64> = BTreeMap::new();
    for t in ts {
        for (k, v) in t["families"].as_object().unwrap() {
            *families.entry(k.clone()).or_insert(0) += v.as_u64().unwrap_or(0);
        }
    }
    json!({
        "n": ts.len(),
        "mean_actions": mean(ts.iter().map(|t| t["actions"].as_f64().unwrap())),
        "mean_gold_left": mean(ended.iter().map(|t| t["gold_left"].as_f64().unwrap())),
        "mean_board_size": mean(ended.iter().map(|t| t["end_board_size"].as_f64().unwrap())),
        "mean_tier": mean(ended.iter().map(|t| t["tier"].as_f64().unwrap())),
        "candidate_cycle_closures": sum("candidate_cycle_closures"),
        "unchanged_move_freeze": sum("unchanged_move_freeze"),
        "forced_end": sum("forced_end"),
        "ended_with_buy_available": sum("ended_with_buy_available"),
        "move_cap_reached": sum("move_cap_reached"),
        "freeze_cap_reached": sum("freeze_cap_reached"),
        "turns_with_candidate_cycle": ts.iter().filter(|t| t["candidate_cycle_closures"].as_u64().unwrap_or(0) > 0).count(),
        "families": families,
    })
}

pub fn aggregate(rows: &[Value]) -> Value {
    let summaries: Vec<&Value> = rows
        .iter()
        .flat_map(|g| g["summaries"].as_array().unwrap())
        .collect();
    let mut per_turn: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
    for t in &summaries {
        per_turn
            .entry(t["turn"].as_u64().unwrap())
            .or_default()
            .push(t);
    }
    json!({
        "games": rows.len(),
        "mean_placement": mean(rows.iter().map(|g| g["placement"].as_f64().unwrap())),
        "families": stats(&summaries)["families"],
        "overall": stats(&summaries),
        "per_turn": per_turn.iter().map(|(t, ts)| (t.to_string(), stats(ts))).collect::<BTreeMap<_, _>>(),
    })
}

pub fn run(args: PatternsArgs) -> Result<(), String> {
    if args.games < 1 || args.sims < 1 || args.workers < 1 {
        return Err("games, sims and workers must be positive".into());
    }
    if args.mode == Mode::Selfplay && args.model.is_none() {
        return Err("selfplay mode needs --model".into());
    }
    let net = args
        .model
        .as_ref()
        .map(|p| PolicyValueNet::load(p))
        .transpose()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers.min(args.games))
        .build()
        .map_err(|e| e.to_string())?;
    let rows: Vec<Value> = pool.install(|| {
        (0..args.games as u64)
            .into_par_iter()
            .map(|i| trace_game(net.as_ref(), args.seed + i, &args))
            .collect::<Result<_, _>>()
    })?;
    crate::npz::ensure_parent(&args.out)?;
    let summary = aggregate(&rows);
    let mut report = json!({
        "model": args.model.as_ref().map(|p| p.display().to_string()),
        "options": serde_json::to_value(&args).unwrap(),
        "search_version": mcts::SEARCH_VERSION,
        "summary": summary,
        "diagnostic_note": "Repeated material states are candidate loops; budgets are excluded; inspect traces before judging utility.",
    });
    if args.traces {
        let traces = args.out.with_extension("traces.json");
        std::fs::write(&traces, serde_json::to_string(&rows).unwrap())
            .map_err(|e| e.to_string())?;
        report["traces"] = json!(traces.display().to_string());
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    println!("{}", report["summary"]["overall"]);
    Ok(())
}
