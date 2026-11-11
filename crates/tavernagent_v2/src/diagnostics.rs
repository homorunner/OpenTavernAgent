//! Full-game policy diagnostics, aggregated independently of placement statistics.
use crate::rules::{ActionCounts, ActionHistory};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use taverngym::{Game, actions::A_UPGRADE};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TurnDiagnostics {
    pub actions: ActionCounts,
    pub last_decision_tier: u8,
    pub last_decision_gold: i32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GameDiagnostics {
    pub actions: ActionCounts,
    pub turns: BTreeMap<u32, TurnDiagnostics>,
    pub first_upgrade_turn: Option<u32>,
}

impl GameDiagnostics {
    pub fn record(&mut self, game: &Game, seat: usize, history: &ActionHistory, action: usize) {
        self.actions.record(history, game.turn, action);
        let turn = self.turns.entry(game.turn).or_default();
        turn.actions.record(history, game.turn, action);
        turn.last_decision_tier = game.players[seat].tier;
        turn.last_decision_gold = game.players[seat].gold;
        if action == A_UPGRADE && self.first_upgrade_turn.is_none() {
            self.first_upgrade_turn = Some(game.turn);
        }
    }
}

pub fn summarize(games: &[GameDiagnostics]) -> Value {
    let mut actions = ActionCounts::default();
    let mut by_turn = BTreeMap::<u32, Vec<&TurnDiagnostics>>::new();
    let mut first_upgrade = BTreeMap::<String, usize>::new();
    for game in games {
        actions.add(&game.actions);
        *first_upgrade
            .entry(
                game.first_upgrade_turn
                    .map_or_else(|| "never".into(), |t| t.to_string()),
            )
            .or_default() += 1;
        for (&turn, row) in &game.turns {
            by_turn.entry(turn).or_default().push(row);
        }
    }
    let turns: Vec<_> = by_turn.into_iter().map(|(turn, rows)| {
        let mut counts = ActionCounts::default();
        for row in &rows {counts.add(&row.actions);}
        let n = rows.len() as f64;
        json!({"turn":turn,"seat_games":rows.len(),"actions":counts,
            "mean_last_decision_tier":rows.iter().map(|r| f64::from(r.last_decision_tier)).sum::<f64>()/n,
            "mean_last_decision_gold":rows.iter().map(|r| f64::from(r.last_decision_gold)).sum::<f64>()/n,
            "freeze_occurrence":rows.iter().filter(|r| r.actions.freeze_actions>0).count() as f64/n,
            "consecutive_freeze_occurrence":rows.iter().filter(|r| r.actions.consecutive_freeze_actions>0).count() as f64/n})
    }).collect();
    json!({"seat_games":games.len(),"actions":actions,
        "actions_per_seat_game":actions.decisions as f64/games.len().max(1) as f64,
        "first_upgrade_turn_histogram":first_upgrade,"by_turn":turns})
}
