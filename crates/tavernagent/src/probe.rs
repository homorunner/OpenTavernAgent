//! What the agent does on its first decision of turn 1 and of turn 2.
//!
//! The first two turns are the cheapest sanity check for the whole pipeline: turn 1
//! should buy a minion or a spell (never roll the Tavern away), turn 2 should upgrade,
//! and the failure mode to look for is "spend the gold on rerolls".

use std::collections::BTreeMap;

use rayon::prelude::*;
use taverngym::{Action, Game};

use crate::heroes::select_heroes;
use crate::mcts::{self, SearchOptions, bot_action};
use crate::net::PolicyValueNet;
use crate::rules::Budgets;
use crate::selfplay::game_rng;

#[derive(Clone, Debug)]
pub struct ProbeOptions {
    pub games: usize,
    pub sims: usize,
    pub c_puct: f32,
    pub budgets: Budgets,
    pub seed: u64,
}

/// The compact agent action family (`Buy`, `Refresh`, `Play`, ...).
pub fn family(action: usize) -> String {
    if (taverngym::actions::A_PLAY..crate::actions::N_ACTIONS).contains(&action) {
        return "Play".into();
    }
    let name = Action::decode(action)
        .map(|a| format!("{a:?}"))
        .unwrap_or_else(|| "?".into());
    name.split([' ', '(', '{'])
        .next()
        .unwrap_or("?")
        .to_string()
}

#[derive(Clone, Debug, Default)]
pub struct Behaviour {
    /// Per turn (1 and 2): first-action family counts and Gold held at that decision.
    pub first: BTreeMap<u32, BTreeMap<String, usize>>,
    pub gold: BTreeMap<u32, Vec<i32>>,
}

/// Distribution of seat 0's first action in turns 1 and 2 against seven scripted bots.
pub fn behaviour(net: &PolicyValueNet, opts: &ProbeOptions) -> Result<Behaviour, String> {
    let cfg = opts.budgets.config()?;
    let search = SearchOptions {
        sims: opts.sims,
        c_puct: opts.c_puct,
        ..SearchOptions::default()
    };
    let per_game: Vec<Vec<(u32, String, i32)>> = (0..opts.games as u64)
        .into_par_iter()
        .map(|g| {
            let seed = opts.seed + g;
            let mut game = Game::new(seed, cfg.clone());
            select_heroes(&mut game, seed)?;
            let mut rng = game_rng(seed);
            let mut seen = Vec::new();
            let mut rows = Vec::new();
            while !game.is_over() {
                if game.needs_action(0) {
                    let action = mcts::search(&game, 0, net, &search, &mut rng)?.best_action();
                    let turn = game.turn;
                    if (turn == 1 || turn == 2) && !seen.contains(&turn) {
                        seen.push(turn);
                        rows.push((turn, family(action), game.players[0].gold));
                    }
                    crate::actions::apply(&mut game, 0, action).map_err(|e| e.to_string())?;
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
            Ok::<_, String>(rows)
        })
        .collect::<Result<_, _>>()?;
    let mut out = Behaviour::default();
    for turn in [1, 2] {
        out.first.entry(turn).or_default();
        out.gold.entry(turn).or_default();
    }
    for rows in per_game {
        for (turn, fam, gold) in rows {
            *out.first.entry(turn).or_default().entry(fam).or_insert(0) += 1;
            out.gold.entry(turn).or_default().push(gold);
        }
    }
    Ok(out)
}

pub fn format_turn(b: &Behaviour, turn: u32) -> String {
    let counts = &b.first[&turn];
    let total = counts.values().sum::<usize>().max(1);
    let mut parts: Vec<(&String, &usize)> = counts.iter().collect();
    parts.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let text: Vec<String> = parts
        .iter()
        .map(|(k, v)| format!("{k} {:.0}%", 100.0 * **v as f64 / total as f64))
        .collect();
    let gold = &b.gold[&turn];
    let avg = if gold.is_empty() {
        0.0
    } else {
        gold.iter().sum::<i32>() as f64 / gold.len() as f64
    };
    format!("{}  (avg gold {avg:.1})", text.join(", "))
}

pub fn behaviour_string(net: &PolicyValueNet, opts: &ProbeOptions) -> Result<String, String> {
    let b = behaviour(net, opts)?;
    Ok(format!(
        "T1 [{}]  T2 [{}]",
        format_turn(&b, 1),
        format_turn(&b, 2)
    ))
}
