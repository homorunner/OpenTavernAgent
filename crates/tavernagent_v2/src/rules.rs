//! Shared host budgets for training, evaluation and interactive play.

use clap::Args;
use serde::{Deserialize, Serialize};
use taverngym::RulesConfig;

pub const ACTION_CAP: u32 = 100;
pub const MOVE_CAP: u32 = 10;
pub const FREEZE_CAP: u32 = 5;

/// Policy-only ablations; old checkpoints deserialize to the unchanged policy rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Args)]
pub struct PolicyRules {
    /// After this seat successfully presses Freeze, mask Freeze until another action or turn.
    #[arg(long)]
    pub block_consecutive_freeze: bool,
    /// Replace the free Freeze toggle with Freeze-and-End; keep the same output slot.
    #[serde(default)]
    #[arg(long)]
    pub freeze_at_end: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActionHistory {
    last_turn: u32,
    last_action: Option<usize>,
}

impl ActionHistory {
    pub fn follows_freeze(&self, turn: u32) -> bool {
        self.last_turn == turn && self.last_action == Some(taverngym::actions::A_FREEZE)
    }

    /// Call only after a successful engine action, passing the turn before that action.
    pub fn record(&mut self, turn: u32, action: usize) {
        self.last_turn = turn;
        self.last_action = Some(action);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionCounts {
    pub decisions: u64,
    pub freeze_actions: u64,
    pub consecutive_freeze_actions: u64,
}

impl ActionCounts {
    pub fn record(&mut self, history: &ActionHistory, turn: u32, action: usize) {
        self.decisions += 1;
        if action == taverngym::actions::A_FREEZE {
            self.freeze_actions += 1;
            self.consecutive_freeze_actions += u64::from(history.follows_freeze(turn));
        }
    }

    pub fn add(&mut self, other: &Self) {
        self.decisions += other.decisions;
        self.freeze_actions += other.freeze_actions;
        self.consecutive_freeze_actions += other.consecutive_freeze_actions;
    }
}

/// Per-turn host limits: total actions, successful Moves and Freeze presses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Args)]
pub struct Budgets {
    /// Total actions a seat may take per turn.
    #[arg(long, default_value_t = ACTION_CAP)]
    pub action_cap: u32,
    /// Successful warband Moves per turn.
    #[arg(long, default_value_t = MOVE_CAP)]
    pub move_cap: u32,
    /// Freeze / Unfreeze presses per turn.
    #[arg(long, default_value_t = FREEZE_CAP)]
    pub freeze_cap: u32,
}

impl Default for Budgets {
    fn default() -> Self {
        Budgets {
            action_cap: ACTION_CAP,
            move_cap: MOVE_CAP,
            freeze_cap: FREEZE_CAP,
        }
    }
}

impl Budgets {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("action_cap", self.action_cap),
            ("move_cap", self.move_cap),
            ("freeze_cap", self.freeze_cap),
        ] {
            if value == 0 {
                return Err(format!("{name} must be a positive integer"));
            }
        }
        Ok(())
    }

    /// The engine rules with these limits (always enabled; the host, not the game, sets them).
    pub fn config(&self) -> Result<RulesConfig, String> {
        self.validate()?;
        let cfg = RulesConfig {
            turn_action_caps: vec![self.action_cap],
            move_actions_per_turn: self.move_cap,
            freeze_presses_per_turn: self.freeze_cap,
            ..RulesConfig::default()
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
