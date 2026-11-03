//! Shared host budgets for training, evaluation and interactive play.

use clap::Args;
use serde::{Deserialize, Serialize};
use taverngym::RulesConfig;

pub const ACTION_CAP: u32 = 100;
pub const MOVE_CAP: u32 = 10;
pub const FREEZE_CAP: u32 = 5;

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
