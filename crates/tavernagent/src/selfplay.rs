//! Self-play: the whole table plays the network, then we train on it.
//!
//! Per round: `games` games in which **all eight seats** use the current weights with a
//! small PUCT search, and every seat's decisions are recorded. Each End Turn also
//! records the ended position (`Game::end_turn_effects` on a copy): no legal actions, so
//! no policy target, but the same final-placement value label. That is the state the
//! search evaluates when it considers ending the turn, so the single value head is
//! trained exactly where it is used. `end_explore` occasionally forces End Turn so that
//! ended positions with unspent Gold also appear in the data.
//!
//! Training is AdamW with a cosine schedule, gradient clipping and an EMA copy, drawing
//! each epoch's samples uniformly from a replay window of the last rounds.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

use rand::Rng;
use rand::SeedableRng;
use rand::seq::index::sample as sample_indices;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use taverngym::Game;
use taverngym::rng::Rng as GameRng;
use taverngym::state::Player;

use crate::heroes::select_heroes;
use crate::linalg::Mat;
use crate::mcts::{self, A_END_TURN, Evaluator, SearchOptions, observe_seat, placement_reward};
use crate::net::{
    self, N_ACTIONS, N_CARD_SLOTS, N_FEATURES, Params, PolicyValueNet, masked_softmax_rows,
};
use crate::npz::{NpyArray, Npz};
use crate::rules::Budgets;

pub const TRAINER_FORMAT: i64 = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub x: Vec<f32>,
    pub cards: Vec<i32>,
    pub mask: Vec<bool>,
    pub pi: Vec<f32>,
    pub z: f32,
    /// The action actually played, or -1 when unknown (ended positions).
    pub action: i64,
    /// Expected result of this turn's fight if the seat stopped here (mean of simulated
    /// combats, -1 loss / 0 tie / +1 win), when `combat_known`.
    pub combat: f32,
    pub combat_known: bool,
}

impl Sample {
    /// An ended position: no legal action, so no policy target.
    pub fn is_ended(&self) -> bool {
        !self.mask.iter().any(|m| *m)
    }
}

/// The position `seat` hands to combat: `end_turn_effects` on an exact copy, so the real
/// game's random stream is untouched and the preview matches what End Turn is about to do.
pub fn end_turn_preview(game: &Game, seat: usize) -> Result<Game, String> {
    let mut preview = game.clone();
    preview.end_turn_effects(seat).map_err(|e| e.to_string())?;
    Ok(preview)
}

/// The training row for an ended position (`end_turn_preview`): a value label, no policy.
pub fn ended_sample_of(preview: &Game, seat: usize) -> Result<Sample, String> {
    let obs = observe_seat(preview, seat);
    if obs.mask.iter().any(|m| *m) {
        return Err("an ended seat must have no legal actions".into());
    }
    Ok(Sample {
        x: obs.x,
        cards: obs.cards,
        mask: obs.mask,
        pi: vec![0.0; N_ACTIONS],
        z: 0.0,
        action: -1,
        combat: 0.0,
        combat_known: false,
    })
}

pub fn ended_sample(game: &Game, seat: usize) -> Result<Sample, String> {
    ended_sample_of(&end_turn_preview(game, seat)?, seat)
}

/// Expected result of `seat`'s fight this turn with `player` as its final state, against
/// the opponent `full` pairs it with: the mean of `sims` simulated combats scored -1 / 0 /
/// +1, each with a fresh random stream. `full` is the position after every seat has ended
/// (boards final, pairings set). None when the seat has no fight this turn.
pub fn combat_label(
    full: &Game,
    seat: usize,
    player: &Player,
    sims: usize,
    rng: &mut impl Rng,
) -> Option<f32> {
    if sims == 0 || !full.players[seat].alive || full.players[seat].health <= 0 {
        return None;
    }
    let pair = full.pairs.iter().find_map(|(a, b)| {
        if *a == seat {
            Some(*b)
        } else if *b == Some(seat) {
            Some(Some(*a))
        } else {
            None
        }
    })?;
    if let Some(o) = pair
        && (!full.players[o].alive || full.players[o].health <= 0)
    {
        return None;
    }
    let ghost_id = full.eliminated.last().copied();
    let ghost = pair.is_none().then(|| {
        ghost_id
            .map(|g| (full.players[g].last_board.clone(), full.players[g].tier))
            .unwrap_or((vec![], 1))
    });
    let mut total = 0i32;
    for _ in 0..sims {
        let mut g = full.clone();
        g.players[seat] = player.clone();
        g.rng = GameRng::new(rng.random::<u64>());
        if pair.is_none() {
            g.ghost_player = ghost_id;
        }
        let out = g.simulate_combat(seat, pair, ghost.clone());
        total += match out.winner {
            0 => 1,
            1 => -1,
            _ => 0,
        };
    }
    Some(total as f32 / sims as f32)
}

#[derive(Clone, Debug)]
pub struct PlayOptions {
    pub search: SearchOptions,
    pub temperature: f32,
    pub end_explore: f32,
    pub budgets: Budgets,
    /// Simulated combats per labelled position (0 disables the combat head's labels).
    pub combat_sims: usize,
}

/// A position of the current turn waiting for its combat label once every seat has ended.
struct PendingCombat {
    samples: Vec<usize>,
    seat: usize,
    player: Player,
}

/// Seed-derived generator for one self-play game: root noise, simulation seeds,
/// temperature sampling and End Turn exploration.
pub fn game_rng(seed: u64) -> Pcg64Mcg {
    Pcg64Mcg::seed_from_u64(seed ^ 0x504C_4159)
}

/// Play one all-seat self-play game; returns every recorded sample with its final label
/// and seat 0's reward.
pub fn play_one<E: Evaluator + ?Sized>(
    net: &E,
    seed: u64,
    opts: &PlayOptions,
) -> Result<(Vec<Sample>, f32), String> {
    if !(0.0..=1.0).contains(&opts.end_explore) {
        return Err("end_explore must be a probability".into());
    }
    let mut game = Game::new(seed, opts.budgets.config()?);
    select_heroes(&mut game, seed)?;
    let mut rng = game_rng(seed);
    let mut samples: Vec<Sample> = Vec::new();
    let mut seats: Vec<usize> = Vec::new();
    let mut pending: Vec<PendingCombat> = Vec::new();
    while !game.is_over() {
        let mut acted = false;
        for seat in 0..game.players.len() {
            if !game.needs_action(seat) {
                continue;
            }
            acted = true;
            let obs = observe_seat(&game, seat);
            let result = mcts::search(&game, seat, net, &opts.search, &mut rng)?;
            let mut action = if opts.temperature <= 0.0 {
                result.best_action()
            } else {
                sample_action(&result.policy, opts.temperature, &mut rng)
            };
            if opts.end_explore > 0.0
                && obs.mask[A_END_TURN]
                && rng.random::<f32>() < opts.end_explore
            {
                // Exploration only changes what is played; the search policy stays the target.
                action = A_END_TURN;
            }
            // The board this position would hand to combat (only when the turn can end
            // here: a pending choice has to be resolved first).
            let preview = if obs.mask[A_END_TURN] && (opts.combat_sims > 0 || action == A_END_TURN)
            {
                Some(end_turn_preview(&game, seat)?)
            } else {
                None
            };
            let mut labelled = vec![samples.len()];
            samples.push(Sample {
                x: obs.x,
                cards: obs.cards,
                mask: obs.mask,
                pi: result.policy,
                z: 0.0,
                action: action as i64,
                combat: 0.0,
                combat_known: false,
            });
            seats.push(seat);
            if action == A_END_TURN {
                let preview = preview
                    .as_ref()
                    .expect("End Turn is legal, so the preview exists");
                labelled.push(samples.len());
                samples.push(ended_sample_of(preview, seat)?);
                seats.push(seat);
            }
            if let Some(preview) = &preview
                && opts.combat_sims > 0
            {
                pending.push(PendingCombat {
                    samples: labelled,
                    seat,
                    player: preview.players[seat].clone(),
                });
            }
            if action == A_END_TURN {
                let last = (0..game.players.len())
                    .all(|p| p == seat || !game.players[p].alive || game.players[p].ended_turn);
                if last {
                    // Every board is final in the preview: label the whole turn's positions.
                    let full = preview
                        .as_ref()
                        .expect("End Turn is legal, so the preview exists");
                    for item in pending.drain(..) {
                        if let Some(label) =
                            combat_label(full, item.seat, &item.player, opts.combat_sims, &mut rng)
                        {
                            for &i in &item.samples {
                                samples[i].combat = label;
                                samples[i].combat_known = true;
                            }
                        }
                    }
                }
            }
            // Plain step: every seat is the network here, so nothing may drive the bots.
            game.apply_index(seat, action).map_err(|e| e.to_string())?;
        }
        if !acted {
            return Err(format!(
                "self-play stalled before game end: seed={seed}, turn={}",
                game.turn
            ));
        }
    }
    for (s, &seat) in samples.iter_mut().zip(&seats) {
        s.z = placement_reward(game.players[seat].placement);
    }
    Ok((samples, placement_reward(game.players[0].placement)))
}

/// Sample an action from a visit distribution at `temperature`, in log space so low
/// temperatures cannot underflow every weight.
pub fn sample_action(policy: &[f32], temperature: f32, rng: &mut impl Rng) -> usize {
    let logits: Vec<(usize, f64)> = policy
        .iter()
        .enumerate()
        .filter(|(_, p)| **p > 0.0)
        .map(|(i, p)| (i, (*p as f64).ln() / temperature as f64))
        .collect();
    let top = logits
        .iter()
        .map(|(_, l)| *l)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|(_, l)| (l - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut u = rng.random::<f64>() * total;
    for ((i, _), w) in logits.iter().zip(&weights) {
        if u < *w {
            return *i;
        }
        u -= w;
    }
    logits.last().map(|(i, _)| *i).unwrap_or(0)
}

pub fn cosine_lr(initial: f64, minimum: f64, index: usize, total: usize) -> f64 {
    let progress = (index as f64 / (total.max(2) - 1) as f64).min(1.0);
    minimum + 0.5 * (initial - minimum) * (1.0 + (std::f64::consts::PI * progress).cos())
}

// ---------------------------------------------------------------- training

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizerHparams {
    pub wd: f32,
    pub clip: f32,
    pub value_weight: f32,
    /// Weight of the auxiliary combat-outcome MSE.
    pub combat_weight: f32,
    /// Per-update EMA decay once warmed up; `1 - 1/decay` updates is the averaging window.
    /// See `ema_decay_for_window`.
    pub ema_decay: f32,
}

impl Default for OptimizerHparams {
    fn default() -> Self {
        OptimizerHparams {
            wd: 0.01,
            clip: 5.0,
            value_weight: 0.5,
            combat_weight: 0.5,
            ema_decay: ema_decay_for_window(1024.0),
        }
    }
}

/// The per-update decay whose averaging window is `updates` optimizer steps.
pub fn ema_decay_for_window(updates: f64) -> f32 {
    (1.0 - 1.0 / updates.max(1.0)) as f32
}

/// Warmed-up EMA decay at optimizer step `t` (1-based): early on the average follows the
/// weights closely instead of remembering the initialization, then settles at `target`.
pub fn ema_decay_at(target: f32, t: u64) -> f32 {
    target.min((1.0 + t as f64) as f32 / (10.0 + t as f64) as f32)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepOptions {
    pub pg_weight: f32,
    pub smoothing: f32,
}

pub type Metrics = BTreeMap<String, f64>;

/// AdamW + gradient clipping on the residual net, with an EMA copy of the weights.
pub struct Trainer {
    pub net: PolicyValueNet,
    pub hparams: OptimizerHparams,
    pub m: Params,
    pub v: Params,
    pub t: u64,
    pub ema: Params,
    pub last_metrics: Metrics,
}

impl Trainer {
    pub fn new(net: PolicyValueNet, hparams: OptimizerHparams) -> Trainer {
        let m = net.p.zeros_like();
        let v = net.p.zeros_like();
        let ema = net.p.clone();
        Trainer {
            net,
            hparams,
            m,
            v,
            t: 0,
            ema,
            last_metrics: Metrics::new(),
        }
    }

    pub fn ema_net(&self) -> PolicyValueNet {
        self.net.with_params(self.ema.clone())
    }

    /// One AdamW step on a batch; returns (policy loss, value loss).
    pub fn step(
        &mut self,
        batch: &[&Sample],
        lr: f64,
        opts: StepOptions,
    ) -> Result<(f32, f32), String> {
        let bsz = batch.len();
        if bsz == 0 {
            return Err("empty training batch".into());
        }
        let xs: Vec<&[f32]> = batch.iter().map(|s| s.x.as_slice()).collect();
        let cards: Vec<&[i32]> = batch.iter().map(|s| s.cards.as_slice()).collect();
        let masks: Vec<&[bool]> = batch.iter().map(|s| s.mask.as_slice()).collect();
        for s in batch {
            if s.action < -1 || s.action >= N_ACTIONS as i64 {
                return Err("expected selected actions, with -1 for unknown".into());
            }
            if s.action >= 0 && !s.mask[s.action as usize] {
                return Err("selected training actions must be legal".into());
            }
            if s.is_ended() && s.pi.iter().any(|p| *p != 0.0) {
                return Err("ended positions cannot carry a policy target".into());
            }
        }
        let (logits, value, combat, cache) = self.net.forward(&xs, &cards)?;
        // A row without legal actions is an ended position: it carries a value label but
        // no policy target, and contributes nothing to the policy loss or its gradient.
        let policy_rows: Vec<bool> = batch.iter().map(|s| !s.is_ended()).collect();
        let n_policy = policy_rows.iter().filter(|p| **p).count();
        let probs = masked_softmax_rows(&logits, &masks);
        let norm_policy = 1.0 / n_policy.max(1) as f32;
        // Targets with optional smoothing over the legal actions.
        let mut pis = Mat::zeros(bsz, N_ACTIONS);
        for (r, s) in batch.iter().enumerate() {
            let row = pis.row_mut(r);
            let legal = s.mask.iter().filter(|m| **m).count().max(1) as f32;
            for (a, target) in row.iter_mut().enumerate() {
                let uniform = if s.mask[a] { 1.0 / legal } else { 0.0 };
                *target = (1.0 - opts.smoothing) * s.pi[a] + opts.smoothing * uniform;
            }
        }
        let mut dlogits = Mat::zeros(bsz, N_ACTIONS);
        for r in 0..bsz {
            let (p, t) = (probs.row(r), pis.row(r));
            let out = dlogits.row_mut(r);
            for a in 0..N_ACTIONS {
                out[a] = (p[a] - t[a]) * norm_policy;
            }
            if opts.pg_weight != 0.0 && batch[r].action >= 0 {
                // REINFORCE uses the value prediction as its baseline.
                let advantage = batch[r].z - value[r];
                let chosen = batch[r].action as usize;
                for a in 0..N_ACTIONS {
                    let onehot = (a == chosen) as u8 as f32;
                    out[a] += opts.pg_weight * advantage * (p[a] - onehot) * norm_policy;
                }
            }
        }
        // Every row, continuing or ended, carries the final placement as its value label.
        let dvalue: Vec<f32> = (0..bsz)
            .map(|r| {
                self.hparams.value_weight
                    * 2.0
                    * (value[r] - batch[r].z)
                    * (1.0 - value[r] * value[r])
                    / bsz as f32
            })
            .collect();
        let value_loss = (0..bsz)
            .map(|r| (value[r] - batch[r].z).powi(2))
            .sum::<f32>()
            / bsz as f32;
        // The combat head is supervised only where this turn's fight could be simulated.
        let n_combat = batch.iter().filter(|s| s.combat_known).count();
        let dcombat: Vec<f32> = (0..bsz)
            .map(|r| {
                if batch[r].combat_known {
                    self.hparams.combat_weight
                        * 2.0
                        * (combat[r] - batch[r].combat)
                        * (1.0 - combat[r] * combat[r])
                        / n_combat.max(1) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let combat_loss = if n_combat > 0 {
            (0..bsz)
                .filter(|r| batch[*r].combat_known)
                .map(|r| (combat[r] - batch[r].combat).powi(2))
                .sum::<f32>()
                / n_combat as f32
        } else {
            0.0
        };
        let mut policy_loss = 0.0f32;
        let mut policy_entropy = 0.0f64;
        let mut target_entropy = 0.0f64;
        let mut target_kl = 0.0f64;
        let mut logit_span = 0.0f64;
        for r in 0..bsz {
            if !policy_rows[r] {
                continue;
            }
            let (p, t) = (probs.row(r), pis.row(r));
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for a in 0..N_ACTIONS {
                let lp = p[a].max(1e-12).ln();
                let lt = t[a].max(1e-12).ln();
                policy_loss -= t[a] * lp;
                policy_entropy -= (p[a] * lp) as f64;
                target_entropy -= (t[a] * lt) as f64;
                target_kl += (t[a] * (lt - lp)) as f64;
                if batch[r].mask[a] {
                    lo = lo.min(logits.row(r)[a]);
                    hi = hi.max(logits.row(r)[a]);
                }
            }
            logit_span += (hi - lo) as f64;
        }
        let policy_loss = policy_loss * norm_policy;
        let grads = self.net.backward(&cache, &dlogits, &dvalue, &dcombat);

        let norm = grads.sum_squares().sqrt();
        if !norm.is_finite() || !(policy_loss + value_loss + combat_loss).is_finite() {
            return Err("nonfinite training loss or gradient".into());
        }
        let scale = if norm > self.hparams.clip as f64 {
            self.hparams.clip as f64 / norm
        } else {
            1.0
        };
        let head_norm = |indices: &[usize]| -> f64 {
            indices
                .iter()
                .map(|&i| grads.tensors[i].sum_squares())
                .sum::<f64>()
                .sqrt()
        };
        let per_policy = |total: f64| {
            if n_policy > 0 {
                total / n_policy as f64
            } else {
                0.0
            }
        };
        let value_mean = value.iter().sum::<f32>() as f64 / bsz as f64;
        let value_std = (value
            .iter()
            .map(|v| (*v as f64 - value_mean).powi(2))
            .sum::<f64>()
            / bsz as f64)
            .sqrt();
        let target_mean = batch.iter().map(|s| s.z as f64).sum::<f64>() / bsz as f64;
        let adv: Vec<f64> = (0..bsz).map(|r| (batch[r].z - value[r]) as f64).collect();
        let adv_mean = adv.iter().sum::<f64>() / bsz as f64;
        let adv_std = (adv.iter().map(|a| (a - adv_mean).powi(2)).sum::<f64>() / bsz as f64).sqrt();
        let mut metrics = Metrics::new();
        metrics.insert("grad_norm".into(), norm);
        metrics.insert(
            "policy_head_grad_norm".into(),
            head_norm(&[self.net.p.wp(), self.net.p.bp()]),
        );
        metrics.insert(
            "value_head_grad_norm".into(),
            head_norm(&[self.net.p.wv(), self.net.p.bv()]),
        );
        metrics.insert("policy_training_samples".into(), n_policy as f64);
        metrics.insert("combat_training_samples".into(), n_combat as f64);
        metrics.insert("combat_loss".into(), combat_loss as f64);
        metrics.insert(
            "combat_head_grad_norm".into(),
            head_norm(&[self.net.p.wc(), self.net.p.bc()]),
        );
        metrics.insert("ended_samples".into(), (bsz - n_policy) as f64);
        metrics.insert("clipped_fraction".into(), (scale < 1.0) as u8 as f64);
        metrics.insert("policy_entropy".into(), per_policy(policy_entropy));
        metrics.insert("target_entropy".into(), per_policy(target_entropy));
        metrics.insert("target_kl".into(), per_policy(target_kl));
        metrics.insert("legal_logit_span".into(), per_policy(logit_span));
        metrics.insert("value_mean".into(), value_mean);
        metrics.insert("value_std".into(), value_std);
        metrics.insert("target_value_mean".into(), target_mean);
        metrics.insert("advantage_std".into(), adv_std);
        metrics.insert("policy_gradient_norm".into(), dlogits.sum_squares().sqrt());
        self.last_metrics = metrics;

        self.t += 1;
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
        let bias1 = 1.0 - b1.powi(self.t as i32);
        let bias2 = 1.0 - b2.powi(self.t as i32);
        let wd = self.hparams.wd as f64;
        let ema_decay = ema_decay_at(self.hparams.ema_decay, self.t);
        self.last_metrics
            .insert("ema_decay".into(), ema_decay as f64);
        for (i, grad) in grads.tensors.iter().enumerate() {
            let decay = self.net.p.is_weight(i);
            let m = &mut self.m.tensors[i].data;
            let v = &mut self.v.tensors[i].data;
            let p = &mut self.net.p.tensors[i].data;
            let e = &mut self.ema.tensors[i].data;
            for j in 0..grad.data.len() {
                let g = grad.data[j] as f64 * scale;
                m[j] = (b1 * m[j] as f64 + (1.0 - b1) * g) as f32;
                v[j] = (b2 * v[j] as f64 + (1.0 - b2) * g * g) as f32;
                let mh = m[j] as f64 / bias1;
                let vh = v[j] as f64 / bias2;
                let mut update = lr * mh / (vh.sqrt() + eps);
                if decay {
                    update += lr * wd * p[j] as f64;
                }
                p[j] = (p[j] as f64 - update) as f32;
                e[j] = ema_decay * e[j] + (1.0 - ema_decay) * p[j];
            }
        }
        Ok((policy_loss, value_loss))
    }
}

/// One pass over `samples` in random minibatches; returns mean losses and mean metrics.
pub fn train_epoch(
    trainer: &mut Trainer,
    samples: &[&Sample],
    batch_size: usize,
    lr: f64,
    opts: StepOptions,
    rng: &mut impl Rng,
) -> Result<(f32, f32, Metrics), String> {
    let mut order: Vec<usize> = (0..samples.len()).collect();
    shuffle(&mut order, rng);
    let (mut pl, mut vl) = (0.0f32, 0.0f32);
    let mut batches = 0usize;
    let mut sums = Metrics::new();
    for chunk in order.chunks(batch_size) {
        let batch: Vec<&Sample> = chunk.iter().map(|&i| samples[i]).collect();
        let (a, b) = trainer.step(&batch, lr, opts)?;
        for (k, v) in &trainer.last_metrics {
            *sums.entry(k.clone()).or_insert(0.0) += v;
        }
        pl += a;
        vl += b;
        batches += 1;
    }
    let n = batches.max(1) as f32;
    let means = sums
        .into_iter()
        .map(|(k, v)| (k, v / batches.max(1) as f64))
        .collect();
    Ok((pl / n, vl / n, means))
}

fn shuffle(order: &mut [usize], rng: &mut impl Rng) {
    for i in (1..order.len()).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }
}

/// The last `capacity` self-play rounds; each epoch trains on a uniform draw from them.
pub struct ReplayWindow {
    pub capacity: usize,
    pub rounds: VecDeque<Vec<Sample>>,
}

impl ReplayWindow {
    pub const FIELDS: [&'static str; 8] = [
        "x",
        "cards",
        "mask",
        "pi",
        "z",
        "action",
        "combat",
        "combat_known",
    ];

    pub fn new(capacity: usize) -> Result<ReplayWindow, String> {
        if capacity < 1 {
            return Err("replay window must be positive".into());
        }
        Ok(ReplayWindow {
            capacity,
            rounds: VecDeque::new(),
        })
    }

    pub fn add(&mut self, samples: Vec<Sample>) {
        if self.rounds.len() == self.capacity {
            self.rounds.pop_front();
        }
        self.rounds.push_back(samples);
    }

    pub fn len(&self) -> usize {
        self.rounds.iter().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Up to `limit` samples drawn uniformly without replacement from every stored round.
    pub fn draw(&self, limit: usize, rng: &mut impl Rng) -> Vec<&Sample> {
        let pool: Vec<&Sample> = self.rounds.iter().flatten().collect();
        if pool.len() <= limit {
            return pool;
        }
        sample_indices(rng, pool.len(), limit)
            .into_iter()
            .map(|i| pool[i])
            .collect()
    }

    /// Stacked per-round arrays (`replay/<round>/<field>`) for an exact snapshot.
    pub fn write_arrays(&self, npz: &mut Npz) {
        for (i, batch) in self.rounds.iter().enumerate() {
            let n = batch.len();
            let mut x = Vec::with_capacity(n * N_FEATURES);
            let mut cards = Vec::with_capacity(n * N_CARD_SLOTS);
            let mut mask = Vec::with_capacity(n * N_ACTIONS);
            let mut pi = Vec::with_capacity(n * N_ACTIONS);
            let mut z = Vec::with_capacity(n);
            let mut action = Vec::with_capacity(n);
            let mut combat = Vec::with_capacity(n);
            let mut known = Vec::with_capacity(n);
            for s in batch {
                x.extend_from_slice(&s.x);
                cards.extend_from_slice(&s.cards);
                mask.extend_from_slice(&s.mask);
                pi.extend_from_slice(&s.pi);
                z.push(s.z);
                action.push(s.action);
                combat.push(s.combat);
                known.push(s.combat_known);
            }
            npz.insert(
                format!("replay/{i}/x"),
                NpyArray::f32(vec![n, N_FEATURES], x),
            );
            npz.insert(
                format!("replay/{i}/cards"),
                NpyArray::i32(vec![n, N_CARD_SLOTS], cards),
            );
            npz.insert(
                format!("replay/{i}/mask"),
                NpyArray::bool(vec![n, N_ACTIONS], mask),
            );
            npz.insert(
                format!("replay/{i}/pi"),
                NpyArray::f32(vec![n, N_ACTIONS], pi),
            );
            npz.insert(format!("replay/{i}/z"), NpyArray::f32(vec![n], z));
            npz.insert(format!("replay/{i}/action"), NpyArray::i64(vec![n], action));
            npz.insert(format!("replay/{i}/combat"), NpyArray::f32(vec![n], combat));
            npz.insert(
                format!("replay/{i}/combat_known"),
                NpyArray::bool(vec![n], known),
            );
        }
    }

    /// Refill from `write_arrays` output, oldest round first; a smaller window keeps the newest.
    pub fn restore(&mut self, npz: &Npz, n_rounds: usize) -> Result<(), String> {
        self.rounds.clear();
        for i in 0..n_rounds {
            let get = |field: &str| {
                npz.get(&format!("replay/{i}/{field}"))
                    .map_err(|_| "incomplete replay window in trainer snapshot".to_string())
            };
            let x = get("x")?.as_f32()?;
            let cards = get("cards")?.as_i32()?;
            let mask = get("mask")?.as_bool()?;
            let pi = get("pi")?.as_f32()?;
            let z = get("z")?.as_f32()?;
            let action = get("action")?.as_i64()?;
            let combat = get("combat")?.as_f32()?;
            let known = get("combat_known")?.as_bool()?;
            let n = z.len();
            if x.len() != n * N_FEATURES
                || cards.len() != n * N_CARD_SLOTS
                || mask.len() != n * N_ACTIONS
                || pi.len() != n * N_ACTIONS
                || action.len() != n
                || combat.len() != n
                || known.len() != n
            {
                return Err("inconsistent replay window in trainer snapshot".into());
            }
            let batch = (0..n)
                .map(|j| Sample {
                    x: x[j * N_FEATURES..(j + 1) * N_FEATURES].to_vec(),
                    cards: cards[j * N_CARD_SLOTS..(j + 1) * N_CARD_SLOTS].to_vec(),
                    mask: mask[j * N_ACTIONS..(j + 1) * N_ACTIONS].to_vec(),
                    pi: pi[j * N_ACTIONS..(j + 1) * N_ACTIONS].to_vec(),
                    z: z[j],
                    action: action[j],
                    combat: combat[j],
                    combat_known: known[j],
                })
                .collect();
            self.add(batch);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- state

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    pub games: u64,
    pub decisions: u64,
    pub policy_loss: f32,
    pub value_loss: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainerMeta {
    pub format: i64,
    pub search_version: String,
    pub step: usize,
    pub optimizer_step: u64,
    pub optimizer_hparams: OptimizerHparams,
    pub replay_rounds_stored: usize,
    pub card_fingerprint: String,
    pub net: net::NetConfig,
    pub train_rng: Pcg64Mcg,
    pub replay_rng: Pcg64Mcg,
    pub history: Vec<serde_json::Value>,
    pub stats: Stats,
    pub options: serde_json::Value,
}

pub struct TrainingState<'a> {
    pub trainer: &'a Trainer,
    pub step: usize,
    pub train_rng: &'a Pcg64Mcg,
    pub replay_rng: &'a Pcg64Mcg,
    pub history: &'a [serde_json::Value],
    pub stats: &'a Stats,
    pub options: serde_json::Value,
    pub replay: Option<&'a ReplayWindow>,
}

/// Atomic optimizer/RNG/replay snapshot for exact continued training.
///
/// The replay window is stored in full (float32 observations), so a snapshot is roughly
/// `replay_rounds x max_samples x 15 KB` at the defaults.
pub fn save_training_state(path: &Path, state: &TrainingState) -> Result<(), String> {
    let trainer = state.trainer;
    let meta = TrainerMeta {
        format: TRAINER_FORMAT,
        search_version: mcts::SEARCH_VERSION.into(),
        step: state.step,
        optimizer_step: trainer.t,
        optimizer_hparams: trainer.hparams.clone(),
        replay_rounds_stored: state.replay.map_or(0, |r| r.rounds.len()),
        card_fingerprint: trainer.net.card_fingerprint.to_string(),
        net: trainer.net.cfg.clone(),
        train_rng: state.train_rng.clone(),
        replay_rng: state.replay_rng.clone(),
        history: state.history.to_vec(),
        stats: state.stats.clone(),
        options: state.options.clone(),
    };
    let mut npz = Npz::new();
    let names = trainer.net.p.names();
    for (prefix, params) in [
        ("net", &trainer.net.p),
        ("m", &trainer.m),
        ("v", &trainer.v),
        ("ema", &trainer.ema),
    ] {
        for (name, t) in names.iter().zip(&params.tensors) {
            npz.insert(
                format!("{prefix}/{name}"),
                NpyArray::f32(vec![t.rows, t.cols], t.data.clone()),
            );
        }
    }
    if let Some(replay) = state.replay {
        replay.write_arrays(&mut npz);
    }
    npz.insert(
        "metadata",
        NpyArray::text(&serde_json::to_string(&meta).map_err(|e| e.to_string())?),
    );
    npz.save(path)
}

/// Read only the metadata of a snapshot (to inherit options before building the network).
pub fn read_training_meta(path: &Path) -> Result<TrainerMeta, String> {
    let npz = Npz::load(path)?;
    parse_meta(&npz)
}

fn parse_meta(npz: &Npz) -> Result<TrainerMeta, String> {
    let text = npz.get("metadata")?.as_text()?;
    serde_json::from_str(&text)
        .map_err(|e| format!("incomplete trainer format v{TRAINER_FORMAT} metadata: {e}"))
}

/// Restore optimizer, weights, EMA, RNGs and (if given) the replay window.
pub fn load_training_state(
    path: &Path,
    trainer: &mut Trainer,
    train_rng: &mut Pcg64Mcg,
    replay_rng: &mut Pcg64Mcg,
    replay: Option<&mut ReplayWindow>,
) -> Result<TrainerMeta, String> {
    let npz = Npz::load(path)?;
    let meta = parse_meta(&npz)?;
    if meta.format != TRAINER_FORMAT
        || meta.search_version != mcts::SEARCH_VERSION
        || meta.card_fingerprint != trainer.net.card_fingerprint.to_string()
        || meta.net != trainer.net.cfg
    {
        return Err("trainer snapshot does not match network schema".into());
    }
    let names = trainer.net.p.names();
    for (prefix, params) in [
        ("net", &mut trainer.net.p),
        ("m", &mut trainer.m),
        ("v", &mut trainer.v),
        ("ema", &mut trainer.ema),
    ] {
        for (name, t) in names.iter().zip(params.tensors.iter_mut()) {
            let data = npz.get(&format!("{prefix}/{name}"))?.as_f32()?;
            if data.len() != t.len() {
                return Err(format!("trainer parameter shape mismatch: {prefix}/{name}"));
            }
            t.data.copy_from_slice(data);
        }
    }
    trainer.t = meta.optimizer_step;
    trainer.hparams = meta.optimizer_hparams.clone();
    *train_rng = meta.train_rng.clone();
    *replay_rng = meta.replay_rng.clone();
    if let Some(window) = replay {
        window.restore(&npz, meta.replay_rounds_stored)?;
    }
    Ok(meta)
}

/// Subsample `samples` down to `cap` uniformly (deterministic in `seed`).
pub fn subsample(samples: Vec<Sample>, cap: usize, seed: u64) -> Vec<Sample> {
    if samples.len() <= cap {
        return samples;
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed ^ 0x4341_5000);
    let mut keep = sample_indices(&mut rng, samples.len(), cap).into_vec();
    keep.sort_unstable();
    let mut wanted = keep.into_iter().peekable();
    samples
        .into_iter()
        .enumerate()
        .filter(|(i, _)| {
            if wanted.peek() == Some(i) {
                wanted.next();
                true
            } else {
                false
            }
        })
        .map(|(_, s)| s)
        .collect()
}
