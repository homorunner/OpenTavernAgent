//! Complete, unsplit games from a fixed policy; each seat owns its own trajectory.

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use taverngym::{Game, RulesConfig};

use crate::gae::{self, ValueStep};
use crate::heroes::select_heroes;
use crate::net::PolicyValueNet;
use crate::observation::{Observation, observe_seat_with_rules};
use crate::policy;
use crate::rules::{ActionCounts, ActionHistory, PolicyRules};

#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    pub turn: u32,
    pub health: f32,
    pub obs: Observation,
    pub action: usize,
    pub old_log_prob: f64,
    pub old_value: f32,
    pub reward: f32,
    pub terminated: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    pub seed: u64,
    pub seat: usize,
    pub placement: u8,
    pub terminal_health: f32,
    pub steps: Vec<Transition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub transition: Transition,
    pub advantage: f32,
    pub return_: f32,
    pub terminal_reward: f32,
    pub future_health: Option<f32>,
}

impl Trajectory {
    pub fn into_samples(self, gamma: f32, lambda: f32) -> Result<Vec<Sample>, String> {
        self.into_samples_with_trace(gamma, lambda, None)
    }

    pub fn into_samples_with_trace(
        self,
        gamma: f32,
        lambda: f32,
        within_turn_lambda: Option<f32>,
    ) -> Result<Vec<Sample>, String> {
        if let Some(last) = self.steps.last()
            && (!last.terminated
                || self.steps[..self.steps.len() - 1]
                    .iter()
                    .any(|s| s.terminated))
        {
            return Err("complete trajectory must end in exactly one terminal transition".into());
        }
        let values: Vec<_> = self
            .steps
            .iter()
            .map(|s| ValueStep {
                value: s.old_value,
                reward: s.reward,
                terminated: s.terminated,
            })
            .collect();
        let terminal_reward = self.steps.last().map_or(0.0, |s| s.reward);
        let mut health_by_turn = std::collections::BTreeMap::new();
        for step in &self.steps {
            health_by_turn.entry(step.turn).or_insert(step.health);
        }
        let future_health: Vec<_> = self
            .steps
            .iter()
            .map(|step| {
                health_by_turn
                    .range(step.turn.saturating_add(3)..)
                    .next()
                    .map(|(_, h)| *h)
                    .or(if self.terminal_health <= 0.0 {
                        Some(0.0)
                    } else {
                        None
                    })
                    .map(|h| 2.0 * (h / 60.0).clamp(0.0, 1.0) - 1.0)
            })
            .collect();
        let targets = if let Some(within) = within_turn_lambda {
            let mut trace = vec![lambda; self.steps.len()];
            for (i, pair) in self.steps.windows(2).enumerate() {
                let elapsed = pair[1]
                    .turn
                    .checked_sub(pair[0].turn)
                    .ok_or("trajectory turns must be monotone")?;
                trace[i] = if elapsed == 0 {
                    within
                } else {
                    lambda.powf(elapsed as f32)
                };
            }
            gae::estimate_with_trace(&values, 0.0, gamma, &trace)?
        } else {
            gae::estimate(&values, 0.0, gamma, lambda)?
        };
        Ok(self
            .steps
            .into_iter()
            .zip(targets)
            .zip(future_health)
            .map(
                |((transition, (advantage, return_)), future_health)| Sample {
                    transition,
                    advantage,
                    return_,
                    terminal_reward,
                    future_health,
                },
            )
            .collect())
    }
}

#[derive(Debug, PartialEq)]
pub struct GameRollout {
    pub trajectories: Vec<Trajectory>,
    pub turns: u32,
    pub turn_limit_end: bool,
    pub truncated_combats: u32,
    pub actions: ActionCounts,
}

pub fn game_rng(seed: u64) -> Pcg64Mcg {
    Pcg64Mcg::seed_from_u64(seed ^ 0x5050_4F32)
}

pub fn collect_game(
    net: &PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
) -> Result<GameRollout, String> {
    collect_game_with_rules(net, seed, cfg, PolicyRules::default())
}

pub fn collect_game_with_rules(
    net: &PolicyValueNet,
    seed: u64,
    cfg: &RulesConfig,
    rules: PolicyRules,
) -> Result<GameRollout, String> {
    cfg.validate()?;
    let mut game = Game::new(seed, cfg.clone());
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    let mut histories = vec![ActionHistory::default(); game.players.len()];
    let mut actions = ActionCounts::default();
    let mut trajectories: Vec<_> = (0..game.players.len())
        .map(|seat| Trajectory {
            seed,
            seat,
            placement: 0,
            terminal_health: 0.0,
            steps: Vec::new(),
        })
        .collect();
    while !game.is_over() {
        if let Some(reason) = &game.stopped_reason {
            return Err(format!("simulator stopped: seed={seed}, reason={reason:?}"));
        }
        let mut acted = false;
        // Observe immediately before each action. Other seats may alter public state.
        for (seat, trajectory) in trajectories.iter_mut().enumerate() {
            if !game.needs_action(seat) {
                continue;
            }
            let turn = game.turn;
            let health = (game.players[seat].health + game.players[seat].armor).max(0) as f32;
            let obs = observe_seat_with_rules(&game, seat, rules, &histories[seat]);
            let decision = policy::decide(net, &obs, &mut rng, false)
                .map_err(|e| format!("seed={seed}, seat={seat}, turn={}: {e}", game.turn))?;
            crate::actions::apply_with_rules(&mut game, seat, decision.action, rules).map_err(
                |e| format!("seed={seed}, seat={seat}, action={}: {e}", decision.action),
            )?;
            actions.record(&histories[seat], turn, decision.action);
            histories[seat].record(turn, decision.action);
            trajectory.steps.push(Transition {
                turn,
                health,
                obs,
                action: decision.action,
                old_log_prob: decision.log_prob,
                old_value: decision.value,
                reward: 0.0,
                terminated: false,
            });
            acted = true;
        }
        if !acted {
            return Err(format!(
                "self-play stalled: seed={seed}, turn={}",
                game.turn
            ));
        }
    }
    if game.stopped_reason.is_some() {
        return Err(format!("simulator stopped: seed={seed}"));
    }
    for trajectory in &mut trajectories {
        let placement = game.players[trajectory.seat].placement;
        if placement == 0 {
            return Err(format!("missing terminal placement: seed={seed}"));
        }
        trajectory.placement = placement;
        let p = &game.players[trajectory.seat];
        trajectory.terminal_health = if p.alive {
            (p.health + p.armor).max(0) as f32
        } else {
            0.0
        };
        // Elimination may happen while this seat is waiting. Its final action receives
        // the reward, irrespective of which seat triggered the combat or elimination.
        if let Some(last) = trajectory.steps.last_mut() {
            last.reward = taverngym::env::placement_reward(placement, game.players.len());
            last.terminated = true;
        }
    }
    Ok(GameRollout {
        trajectories,
        turns: game.turn,
        turn_limit_end: game.turn_limit_end,
        truncated_combats: game.truncated_combats,
        actions,
    })
}

pub fn collect(
    net: &PolicyValueNet,
    seeds: &[u64],
    cfg: &RulesConfig,
    pool: &rayon::ThreadPool,
) -> Result<Vec<GameRollout>, String> {
    collect_with_rules(net, seeds, cfg, pool, PolicyRules::default())
}

pub fn collect_with_rules(
    net: &PolicyValueNet,
    seeds: &[u64],
    cfg: &RulesConfig,
    pool: &rayon::ThreadPool,
    rules: PolicyRules,
) -> Result<Vec<GameRollout>, String> {
    // Indexed collection and a separate RNG per game keep results independent of workers.
    pool.install(|| {
        seeds
            .par_iter()
            .map(|&s| collect_game_with_rules(net, s, cfg, rules))
            .collect()
    })
}

/// Normalize only the actor advantages, once for the entire fresh rollout batch.
pub fn normalize_advantages(samples: &mut [Sample]) -> Result<(), String> {
    if samples.is_empty() || samples.iter().any(|s| !s.advantage.is_finite()) {
        return Err("advantage normalization needs finite, nonempty samples".into());
    }
    let mean = samples.iter().map(|s| s.advantage as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|s| (s.advantage as f64 - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    let scale = variance.sqrt().max(1e-8);
    for sample in samples {
        sample.advantage = ((sample.advantage as f64 - mean) / scale) as f32;
    }
    Ok(())
}
