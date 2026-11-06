//! Self-play: the whole table plays the network, then we train on it.
//!
//! Per round: `games` games in which **all eight seats** use the current weights. A
//! `search_prob` fraction of the decisions run the PUCT search and are recorded with the
//! visit distribution as policy target (KataGo's full/cheap split); the others sample
//! the network prior at the play temperature and leave no sample. Each End Turn also
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
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use taverngym::Game;

use crate::heroes::select_heroes;
use crate::linalg::Mat;
use crate::mcts::{
    self, A_END_TURN, Evaluator, Observation, SearchOptions, agent_mask, observe_seat,
    placement_reward,
};
use crate::net::{
    self, N_ACTIONS, N_CARD_SLOTS, N_FEATURES, Params, PolicyValueNet, masked_softmax_rows,
};
use crate::npz::{NpyArray, Npz};
use crate::rules::Budgets;

pub const TRAINER_FORMAT: i64 = 7;
/// Per-child sample weight multiplier. Label expectations always use 1/2 instead.
pub const SPLIT_SAMPLE_WEIGHT: f32 = 0.55;

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub x: Vec<f32>,
    pub cards: Vec<i32>,
    pub mask: Vec<bool>,
    pub pi: Vec<f32>,
    pub z: f32,
    /// Relative weight in both value and policy objectives.
    pub weight: f32,
    /// The action actually played, or -1 when unknown (ended positions).
    pub action: i64,
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
        weight: 1.0,
        action: -1,
    })
}

pub fn ended_sample(game: &Game, seat: usize) -> Result<Sample, String> {
    ended_sample_of(&end_turn_preview(game, seat)?, seat)
}

#[derive(Clone, Debug)]
pub struct PlayOptions {
    pub search: SearchOptions,
    pub temperature: f32,
    pub end_explore: f32,
    pub budgets: Budgets,
    /// Probability that a decision is searched (and recorded); the rest play the prior.
    pub search_prob: f32,
    /// Split after these turns' end effects, immediately before combat.
    pub split_turn: Vec<u32>,
}

/// How one decision was made: a full search (recorded) or a direct prior sample.
#[derive(Clone)]
pub enum Decision {
    Searched(Box<mcts::SearchResult>),
    Direct(Vec<f32>),
}

impl Decision {
    /// The distribution the played action is drawn from.
    pub fn policy(&self) -> &[f32] {
        match self {
            Decision::Searched(r) => &r.policy,
            Decision::Direct(prior) => prior,
        }
    }
    fn choose(&self, temperature: f32, rng: &mut impl Rng) -> usize {
        match self {
            Decision::Searched(r) if temperature <= 0.0 => r.best_action(),
            _ if temperature <= 0.0 => mcts::argmax(self.policy()),
            _ => sample_action(self.policy(), temperature, rng),
        }
    }
}

/// Search `seat` (seeded by `seed`) when `full`, otherwise evaluate the prior once.
fn decide<E: Evaluator + ?Sized>(
    game: &Game,
    seat: usize,
    obs: &Observation,
    net: &E,
    opts: &PlayOptions,
    full: bool,
    seed: u64,
) -> Result<Decision, String> {
    if full {
        let mut seat_rng = Pcg64Mcg::seed_from_u64(seed);
        Ok(Decision::Searched(Box::new(mcts::search(
            game,
            seat,
            net,
            &opts.search,
            &mut seat_rng,
        )?)))
    } else {
        Ok(Decision::Direct(net.evaluate(obs)?.0))
    }
}

/// Per decision: whether it is searched, and the seed of its search generator.
fn draw_decision(opts: &PlayOptions, rng: &mut Pcg64Mcg) -> (bool, u64) {
    let full = opts.search_prob >= 1.0 || rng.random::<f32>() < opts.search_prob;
    (full, rng.random::<u64>())
}

/// Search of one ply: the searches of all seats that need an action run concurrently
/// (rayon) on the same read-only starting position, each with its own generator seeded
/// from `rng`; the actions are then applied in seat order by `play_one`. Returns, per
/// seat, its observation and search result (None = did not act). A ply with a single
/// acting seat is left to the caller's sequential search.
fn presearch_ply<E: Evaluator + ?Sized>(
    game: &Game,
    net: &E,
    opts: &PlayOptions,
    rng: &mut Pcg64Mcg,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<Option<(Observation, Decision)>>, String> {
    use rayon::prelude::*;
    let active: Vec<usize> = (0..game.players.len())
        .filter(|&s| game.needs_action(s))
        .collect();
    if active.len() < 2 {
        return Ok(Vec::new());
    }
    let draws: Vec<(bool, u64)> = active.iter().map(|_| draw_decision(opts, rng)).collect();
    let search_all = || {
        active
            .par_iter()
            .zip(draws.par_iter())
            .map(|(&seat, &(full, seed))| {
                let obs = observe_seat(game, seat);
                if !obs.mask.iter().any(|m| *m) {
                    return Err(format!("seat {seat} has no legal action"));
                }
                let result = decide(game, seat, &obs, net, opts, full, seed)?;
                Ok((seat, obs, result))
            })
            .collect::<Result<Vec<_>, String>>()
    };
    let found = match pool {
        Some(pool) => pool.install(search_all)?,
        None => search_all()?,
    };
    let mut out: Vec<Option<(Observation, Decision)>> =
        (0..game.players.len()).map(|_| None).collect();
    for (seat, obs, result) in found {
        out[seat] = Some((obs, result));
    }
    Ok(out)
}

/// Seed-derived generator for one self-play game: root noise, simulation seeds,
/// temperature sampling and End Turn exploration.
pub fn game_rng(seed: u64) -> Pcg64Mcg {
    Pcg64Mcg::seed_from_u64(seed ^ 0x504C_4159)
}

/// Validate the chronological split schedule for both the CLI and library callers.
pub fn validate_split_turns(turns: &[u32]) -> Result<(), String> {
    if turns.contains(&0) || turns.windows(2).any(|w| w[0] >= w[1]) {
        return Err("split-turn must contain strictly increasing positive turns (e.g. 4,8)".into());
    }
    Ok(())
}

/// Keep the seat cursor and outstanding searches across combat, including a split.
/// This preserves the ordinary ply order even if combat starts before the final seat.
#[derive(Clone)]
struct RolloutState {
    game: Game,
    rng: Pcg64Mcg,
    presearched: Vec<Option<(Observation, Decision)>>,
    next_seat: usize,
    acted: bool,
}

impl RolloutState {
    fn new(game: Game, rng: Pcg64Mcg) -> Self {
        let next_seat = game.players.len();
        Self {
            game,
            rng,
            presearched: Vec::new(),
            next_seat,
            acted: true,
        }
    }
}

/// Play one root and all its binary continuations. Shared prefix rows occur once;
/// their labels are conditional means of the two child subtrees. The returned seat-0
/// reward uses the same probability weights, independent of training sample weights.
/// Each ply's searches run concurrently in `pool` (see `presearch_ply`); the driver
/// must run outside it. `None` uses rayon's global pool.
pub fn play_one<E: Evaluator + ?Sized>(
    net: &E,
    seed: u64,
    opts: &PlayOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<(Vec<Sample>, f32), String> {
    if !(0.0..=1.0).contains(&opts.end_explore) {
        return Err("end_explore must be a probability".into());
    }
    if !(0.0..=1.0).contains(&opts.search_prob) {
        return Err("search_prob must be a probability".into());
    }
    validate_split_turns(&opts.split_turn)?;
    let mut game = Game::new(seed, opts.budgets.config()?);
    select_heroes(&mut game, seed)?;
    let state = RolloutState::new(game, game_rng(seed));
    let (samples, rewards) = play_branch(net, state, opts, pool, 1.0)?;
    Ok((samples, rewards[0]))
}

fn play_branch<E: Evaluator + ?Sized>(
    net: &E,
    mut state: RolloutState,
    opts: &PlayOptions,
    pool: Option<&rayon::ThreadPool>,
    weight: f32,
) -> Result<(Vec<Sample>, Vec<f32>), String> {
    let seed = state.game.seed;
    let mut samples: Vec<Sample> = Vec::new();
    let mut seats: Vec<usize> = Vec::new();
    let mut descendants = Vec::new();
    let rewards = loop {
        if state.game.is_over() {
            break state
                .game
                .players
                .iter()
                .map(|p| placement_reward(p.placement))
                .collect();
        }
        if state.next_seat == state.game.players.len() {
            if !state.acted {
                return Err(format!(
                    "self-play stalled before game end: seed={seed}, turn={}",
                    state.game.turn
                ));
            }
            state.presearched = presearch_ply(&state.game, net, opts, &mut state.rng, pool)?;
            state.next_seat = 0;
            state.acted = false;
        }
        let seat = state.next_seat;
        state.next_seat += 1;
        if !state.game.needs_action(seat) {
            continue;
        }
        state.acted = true;
        // A pre-searched seat is used only while its legal actions are still those
        // of the ply's start; otherwise it is searched again on the live position.
        let pre = state
            .presearched
            .get_mut(seat)
            .and_then(|p| p.take())
            .filter(|(o, _)| o.mask == agent_mask(&state.game, seat));
        let (obs, pre_result) = match pre {
            Some((o, r)) => (o, Some(r)),
            None => (observe_seat(&state.game, seat), None),
        };
        if !obs.mask.iter().any(|m| *m) {
            // An engine state the policy cannot act in; keep it for diagnosis.
            if let Ok(dir) = std::env::var("TAVERNAGENT_DUMP_DIR") {
                let path = format!(
                    "{dir}/no_legal_action_seed{seed}_seat{seat}_turn{}.json",
                    state.game.turn
                );
                let _ = std::fs::write(
                    &path,
                    serde_json::to_string(&state.game).unwrap_or_default(),
                );
            }
            return Err(format!(
                "seat {seat} needs an action but has no legal one: seed={seed}, turn={}, pending={:?}",
                state.game.turn,
                state.game.players[seat].pending.last()
            ));
        }
        let decision = match pre_result {
            Some(r) => r,
            None => {
                let (full, seed) = draw_decision(opts, &mut state.rng);
                decide(&state.game, seat, &obs, net, opts, full, seed)?
            }
        };
        let mut action = decision.choose(opts.temperature, &mut state.rng);
        if opts.end_explore > 0.0
            && obs.mask[A_END_TURN]
            && state.rng.random::<f32>() < opts.end_explore
        {
            // Exploration only changes what is played; the search policy stays the target.
            action = A_END_TURN;
        }
        // Only searched decisions are training rows (their visit distribution is
        // the policy target); the ended position below is recorded either way.
        if let Decision::Searched(result) = decision {
            samples.push(Sample {
                x: obs.x,
                cards: obs.cards,
                mask: obs.mask,
                pi: result.policy,
                z: 0.0,
                weight,
                action: action as i64,
            });
            seats.push(seat);
        }
        if action == A_END_TURN {
            let preview = end_turn_preview(&state.game, seat)?;
            let mut ended = ended_sample_of(&preview, seat)?;
            ended.weight = weight;
            samples.push(ended);
            seats.push(seat);
        }
        // Pause only on configured turns; all end effects precede the fork and
        // even this turn's combat randomness belongs to the children.
        let split = if opts.split_turn.binary_search(&state.game.turn).is_ok() {
            state
                .game
                .apply_index_until_combat(seat, action)
                .map_err(str::to_string)?
        } else {
            state
                .game
                .apply_index(seat, action)
                .map_err(str::to_string)?;
            false
        };
        if split {
            let mut mean = vec![0.0; state.game.players.len()];
            // Draw both stream seeds for each child from the parent, never from a
            // completed sibling. Scheduling and subtree size cannot change them.
            let seeds: [(u64, u64); 2] =
                std::array::from_fn(|_| (state.rng.random(), state.rng.random()));
            for (environment_seed, policy_seed) in seeds {
                let mut child = state.clone();
                child.game.rng = taverngym::rng::Rng::new(environment_seed);
                child.rng = Pcg64Mcg::seed_from_u64(policy_seed);
                child.game.resume_combat().map_err(str::to_string)?;
                let (rows, rewards) =
                    play_branch(net, child, opts, pool, weight * SPLIT_SAMPLE_WEIGHT)?;
                for (m, r) in mean.iter_mut().zip(rewards) {
                    *m += 0.5 * r;
                }
                descendants.extend(rows);
            }
            break mean;
        }
    };
    for (s, seat) in samples.iter_mut().zip(seats) {
        s.z = rewards[seat];
    }
    samples.extend(descendants);
    Ok((samples, rewards))
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
    ///
    /// The batch is processed in fixed chunks of `STEP_CHUNK_ROWS` rows, each running the
    /// forward and backward pass on its own thread; chunk gradients are summed in chunk
    /// order and the optimizer update is split over the parameter vector. The chunk size is
    /// a constant, so the result does not depend on the number of threads.
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
        for s in batch {
            if !s.weight.is_finite() || s.weight < 0.0 {
                return Err("sample weights must be finite and nonnegative".into());
            }
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
        // A row without legal actions is an ended position: it carries a value label but
        // no policy target, and contributes nothing to the policy loss or its gradient.
        let n_policy = batch.iter().filter(|s| !s.is_ended()).count();
        let weight_sum: f32 = batch.iter().map(|s| s.weight).sum();
        let policy_weight_sum: f32 = batch
            .iter()
            .filter(|s| !s.is_ended())
            .map(|s| s.weight)
            .sum();
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err("training batch must have positive finite total sample weight".into());
        }
        let norm_policy = if policy_weight_sum > 0.0 {
            1.0 / policy_weight_sum
        } else {
            0.0
        };
        let scales = LossScales {
            norm_policy,
            weight_sum,
            value_weight: self.hparams.value_weight,
        };
        let net = &self.net;
        let parts: Vec<ChunkOutput> = batch
            .par_chunks(STEP_CHUNK_ROWS)
            .map(|rows| chunk_step(net, rows, opts, &scales))
            .collect::<Result<_, String>>()?;

        // Chunk gradients summed in chunk order, split over the parameter vector; one
        // parallel region for all tensors.
        let mut grads = self.net.p.zeros_like();
        let tasks: Vec<(usize, usize, &mut [f32])> = grads
            .tensors
            .iter_mut()
            .enumerate()
            .flat_map(|(i, t)| {
                t.data
                    .chunks_mut(OPTIMIZER_CHUNK)
                    .enumerate()
                    .map(move |(c, out)| (i, c * OPTIMIZER_CHUNK, out))
            })
            .collect();
        tasks.into_par_iter().for_each(|(i, lo, out)| {
            for part in &parts {
                let src = &part.grads.tensors[i].data[lo..lo + out.len()];
                for (o, v) in out.iter_mut().zip(src) {
                    *o += v;
                }
            }
        });
        let value: Vec<f32> = parts.iter().flat_map(|p| p.value.iter().copied()).collect();
        let policy_loss: f32 = parts.iter().map(|p| p.policy_loss).sum::<f32>() * norm_policy;
        let value_loss: f32 = parts.iter().map(|p| p.value_loss).sum::<f32>() / weight_sum;
        let policy_entropy: f64 = parts.iter().map(|p| p.policy_entropy).sum();
        let target_entropy: f64 = parts.iter().map(|p| p.target_entropy).sum();
        let target_kl: f64 = parts.iter().map(|p| p.target_kl).sum();
        let logit_span: f64 = parts.iter().map(|p| p.logit_span).sum();
        let dlogits_sq: f64 = parts.iter().map(|p| p.dlogits_sq).sum();
        drop(parts);

        let norm = grads.sum_squares().sqrt();
        if !norm.is_finite() || !(policy_loss + value_loss).is_finite() {
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
            if policy_weight_sum > 0.0 {
                total / policy_weight_sum as f64
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
        metrics.insert("ended_samples".into(), (bsz - n_policy) as f64);
        metrics.insert("sample_weight_sum".into(), weight_sum as f64);
        metrics.insert("policy_sample_weight_sum".into(), policy_weight_sum as f64);
        metrics.insert("clipped_fraction".into(), (scale < 1.0) as u8 as f64);
        metrics.insert("policy_entropy".into(), per_policy(policy_entropy));
        metrics.insert("target_entropy".into(), per_policy(target_entropy));
        metrics.insert("target_kl".into(), per_policy(target_kl));
        metrics.insert("legal_logit_span".into(), per_policy(logit_span));
        metrics.insert("value_mean".into(), value_mean);
        metrics.insert("value_std".into(), value_std);
        metrics.insert("target_value_mean".into(), target_mean);
        metrics.insert("advantage_std".into(), adv_std);
        metrics.insert("policy_gradient_norm".into(), dlogits_sq.sqrt());
        self.last_metrics = metrics;

        self.t += 1;
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8f64);
        let bias1 = 1.0 - b1.powi(self.t as i32);
        let bias2 = 1.0 - b2.powi(self.t as i32);
        let wd = self.hparams.wd as f64;
        let ema_decay = ema_decay_at(self.hparams.ema_decay, self.t);
        self.last_metrics
            .insert("ema_decay".into(), ema_decay as f64);
        // One parallel region over every (tensor, slice) task.
        let decay_of: Vec<bool> = (0..grads.tensors.len())
            .map(|i| self.net.p.is_weight(i))
            .collect();
        fn split(params: &mut Params) -> Vec<&mut [f32]> {
            params
                .tensors
                .iter_mut()
                .flat_map(|t| t.data.chunks_mut(OPTIMIZER_CHUNK))
                .collect()
        }
        let g_parts: Vec<(bool, &[f32])> = grads
            .tensors
            .iter()
            .enumerate()
            .flat_map(|(i, t)| {
                let decay = decay_of[i];
                t.data.chunks(OPTIMIZER_CHUNK).map(move |c| (decay, c))
            })
            .collect();
        let m_parts = split(&mut self.m);
        let v_parts = split(&mut self.v);
        let p_parts = split(&mut self.net.p);
        let e_parts = split(&mut self.ema);
        g_parts
            .into_par_iter()
            .zip(m_parts)
            .zip(v_parts)
            .zip(p_parts)
            .zip(e_parts)
            .for_each(|(((((decay, grad), m), v), p), e)| {
                for j in 0..grad.len() {
                    let g = grad[j] as f64 * scale;
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
            });
        self.net.params_changed();
        Ok((policy_loss, value_loss))
    }
}

/// Rows per parallel forward/backward chunk of a training step. A constant, not a function
/// of the thread count, so that training results are reproducible on any machine.
pub const STEP_CHUNK_ROWS: usize = 32;
/// Parameters per parallel task of the gradient reduction and the optimizer update.
const OPTIMIZER_CHUNK: usize = 16_384;

/// Batch-level normalisations shared by every chunk of a step.
struct LossScales {
    norm_policy: f32,
    weight_sum: f32,
    value_weight: f32,
}

/// Gradients and loss terms of one chunk of rows; sums are unnormalised.
struct ChunkOutput {
    grads: Params,
    value: Vec<f32>,
    policy_loss: f32,
    value_loss: f32,
    policy_entropy: f64,
    target_entropy: f64,
    target_kl: f64,
    logit_span: f64,
    dlogits_sq: f64,
}

fn chunk_step(
    net: &PolicyValueNet,
    rows: &[&Sample],
    opts: StepOptions,
    scales: &LossScales,
) -> Result<ChunkOutput, String> {
    let n = rows.len();
    let xs: Vec<&[f32]> = rows.iter().map(|s| s.x.as_slice()).collect();
    let cards: Vec<&[i32]> = rows.iter().map(|s| s.cards.as_slice()).collect();
    let masks: Vec<&[bool]> = rows.iter().map(|s| s.mask.as_slice()).collect();
    let (logits, value, cache) = net.forward(&xs, &cards)?;
    let probs = masked_softmax_rows(&logits, &masks);
    // Targets with optional smoothing over the legal actions.
    let mut pis = Mat::zeros(n, N_ACTIONS);
    for (r, s) in rows.iter().enumerate() {
        let row = pis.row_mut(r);
        let legal = s.mask.iter().filter(|m| **m).count().max(1) as f32;
        for (a, target) in row.iter_mut().enumerate() {
            let uniform = if s.mask[a] { 1.0 / legal } else { 0.0 };
            *target = (1.0 - opts.smoothing) * s.pi[a] + opts.smoothing * uniform;
        }
    }
    let mut dlogits = Mat::zeros(n, N_ACTIONS);
    for r in 0..n {
        let norm_policy = scales.norm_policy * rows[r].weight;
        let (p, t) = (probs.row(r), pis.row(r));
        let out = dlogits.row_mut(r);
        for a in 0..N_ACTIONS {
            out[a] = (p[a] - t[a]) * norm_policy;
        }
        if opts.pg_weight != 0.0 && rows[r].action >= 0 {
            // REINFORCE uses the value prediction as its baseline.
            let advantage = rows[r].z - value[r];
            let chosen = rows[r].action as usize;
            for a in 0..N_ACTIONS {
                let onehot = (a == chosen) as u8 as f32;
                out[a] += opts.pg_weight * advantage * (p[a] - onehot) * norm_policy;
            }
        }
    }
    // Every row, continuing or ended, carries the final placement as its value label.
    let dvalue: Vec<f32> = (0..n)
        .map(|r| {
            scales.value_weight
                * 2.0
                * (value[r] - rows[r].z)
                * (1.0 - value[r] * value[r])
                * rows[r].weight
                / scales.weight_sum
        })
        .collect();
    let value_loss = (0..n)
        .map(|r| rows[r].weight * (value[r] - rows[r].z).powi(2))
        .sum::<f32>();
    let mut policy_loss = 0.0f32;
    let mut policy_entropy = 0.0f64;
    let mut target_entropy = 0.0f64;
    let mut target_kl = 0.0f64;
    let mut logit_span = 0.0f64;
    for (r, sample) in rows.iter().enumerate() {
        if sample.is_ended() {
            continue;
        }
        let (p, t) = (probs.row(r), pis.row(r));
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for a in 0..N_ACTIONS {
            let lp = p[a].max(1e-12).ln();
            let lt = t[a].max(1e-12).ln();
            policy_loss -= sample.weight * t[a] * lp;
            policy_entropy -= sample.weight as f64 * (p[a] * lp) as f64;
            target_entropy -= sample.weight as f64 * (t[a] * lt) as f64;
            target_kl += sample.weight as f64 * (t[a] * (lt - lp)) as f64;
            if sample.mask[a] {
                lo = lo.min(logits.row(r)[a]);
                hi = hi.max(logits.row(r)[a]);
            }
        }
        logit_span += sample.weight as f64 * (hi - lo) as f64;
    }
    let dlogits_sq = dlogits.sum_squares();
    let grads = net.backward(&cache, &dlogits, &dvalue);
    Ok(ChunkOutput {
        grads,
        value,
        policy_loss,
        value_loss,
        policy_entropy,
        target_entropy,
        target_kl,
        logit_span,
        dlogits_sq,
    })
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
    pub const FIELDS: [&'static str; 7] = ["x", "cards", "mask", "pi", "z", "action", "weight"];

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
            let mut weight = Vec::with_capacity(n);
            for s in batch {
                x.extend_from_slice(&s.x);
                cards.extend_from_slice(&s.cards);
                mask.extend_from_slice(&s.mask);
                pi.extend_from_slice(&s.pi);
                z.push(s.z);
                action.push(s.action);
                weight.push(s.weight);
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
            npz.insert(format!("replay/{i}/weight"), NpyArray::f32(vec![n], weight));
        }
    }

    /// Refill from `write_arrays` output, oldest round first; a smaller window keeps the newest.
    pub fn restore(&mut self, npz: &Npz, n_rounds: usize) -> Result<(), String> {
        self.restore_version(npz, n_rounds, false)
    }

    fn restore_version(&mut self, npz: &Npz, n_rounds: usize, legacy: bool) -> Result<(), String> {
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
            let n = z.len();
            // Format 6 predates sample weights. Format 7 must store them explicitly.
            let weight = if legacy {
                vec![1.0; n]
            } else {
                get("weight")?.as_f32()?.to_vec()
            };
            if x.len() != n * N_FEATURES
                || cards.len() != n * N_CARD_SLOTS
                || mask.len() != n * N_ACTIONS
                || pi.len() != n * N_ACTIONS
                || action.len() != n
                || weight.len() != n
            {
                return Err("inconsistent replay window in trainer snapshot".into());
            }
            if weight.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err("invalid sample weights in trainer snapshot".into());
            }
            let batch = (0..n)
                .map(|j| Sample {
                    x: x[j * N_FEATURES..(j + 1) * N_FEATURES].to_vec(),
                    cards: cards[j * N_CARD_SLOTS..(j + 1) * N_CARD_SLOTS].to_vec(),
                    mask: mask[j * N_ACTIONS..(j + 1) * N_ACTIONS].to_vec(),
                    pi: pi[j * N_ACTIONS..(j + 1) * N_ACTIONS].to_vec(),
                    z: z[j],
                    weight: weight[j],
                    action: action[j],
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
/// The replay window is stored in full (float32 observations): every decision of the last
/// `replay_rounds` rounds, about 9 KB each, so roughly 6 GB at the defaults.
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
    if ![6, TRAINER_FORMAT].contains(&meta.format)
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
        window.restore_version(&npz, meta.replay_rounds_stored, meta.format == 6)?;
    }
    Ok(meta)
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use taverngym::cards::cid;

    struct EndOnly;
    impl Evaluator for EndOnly {
        fn evaluate(&self, obs: &Observation) -> Result<(Vec<f32>, f32), String> {
            assert!(obs.mask[A_END_TURN]);
            let mut pi = vec![0.0; N_ACTIONS];
            pi[A_END_TURN] = 1.0;
            Ok((pi, 0.0))
        }
    }

    fn fixture(max_turns: u32) -> (Game, PlayOptions) {
        let opts = PlayOptions {
            search: SearchOptions::default(),
            temperature: 0.0,
            end_explore: 0.0,
            search_prob: 0.0,
            budgets: Budgets::default(),
            split_turn: vec![1, 2],
        };
        let mut cfg = opts.budgets.config().unwrap();
        cfg.max_turns = max_turns;
        let mut game = Game::new(31, cfg);
        select_heroes(&mut game, 31).unwrap();
        // This fixture isolates combat: force End Turn as the sole recruit action.
        game.cfg.turn_action_caps = vec![0];
        for seat in 0..8 {
            let board = game.with_recruit(seat, |c| {
                (0..3)
                    .map(|i| {
                        let mut m = c.new_minion(0, cid("BG25_008"));
                        m.atk = 1 + (seat + i) as i32 % 4;
                        m.hp = 2 + (seat * 3 + i) as i32 % 5;
                        m.max_hp = m.hp;
                        m
                    })
                    .collect()
            });
            let p = &mut game.players[seat];
            p.hero = Some(cid("TB_BaconShop_HERO_34"));
            p.powers.clear();
            p.pending.clear();
            p.effects.clear();
            p.quests.clear();
            p.counters.clear();
            p.hand.clear();
            p.hand_queue.clear();
            p.skipping = false;
            p.health = 100;
            p.armor = 0;
            p.board = board;
        }
        (game, opts)
    }

    #[test]
    fn nested_splits_store_prefix_once_and_average_children_per_seat() {
        let (game, opts) = fixture(3);
        let (rows, rewards) = play_branch(
            &EndOnly,
            RolloutState::new(game, game_rng(31)),
            &opts,
            None,
            1.0,
        )
        .unwrap();
        // Depth-first traversal: 1 + 2 + 4 segments, each has eight ended rows.
        assert_eq!(rows.len(), 7 * 8);
        for (segment, turn) in [1, 2, 3, 3, 2, 3, 3].into_iter().enumerate() {
            for row in &rows[segment * 8..(segment + 1) * 8] {
                assert!(row.is_ended());
                assert_eq!(row.x[0], turn as f32 / 20.0);
                assert_eq!(row.weight, SPLIT_SAMPLE_WEIGHT.powi(turn - 1));
            }
        }
        for seat in 0..8 {
            let z = |segment: usize| rows[segment * 8 + seat].z;
            assert_eq!(z(1), 0.5 * (z(2) + z(3)));
            assert_eq!(z(4), 0.5 * (z(5) + z(6)));
            assert_eq!(z(0), 0.5 * (z(1) + z(4)));
            assert_eq!(rewards[seat], z(0));
        }
        assert!(
            (0..8).any(|seat| rows[16 + seat].cards != rows[24 + seat].cards),
            "independent child streams should produce different shop draws"
        );
    }

    #[test]
    fn split_combat_can_finish_a_child_without_any_descendant_rows() {
        let (game, mut opts) = fixture(1);
        opts.split_turn = vec![1, 2];
        // Independently settle the two copies of the first combat. With cheap,
        // deterministic End Turn decisions, each of eight seats consumes one
        // probability draw and one search seed before the split seeds are drawn.
        let mut rng = game_rng(31);
        for _ in 0..8 {
            let _: f32 = rng.random();
            let _: u64 = rng.random();
        }
        let seeds: [(u64, u64); 2] = std::array::from_fn(|_| (rng.random(), rng.random()));
        let mut boundary = game.clone();
        for seat in 0..8 {
            boundary.end_turn_effects(seat).unwrap();
        }
        let mut expected = [0.0; 8];
        for (seed, _) in seeds {
            let mut child = boundary.clone();
            child.rng = taverngym::rng::Rng::new(seed);
            child.combat_round();
            assert!(child.is_over());
            for (e, p) in expected.iter_mut().zip(&child.players) {
                *e += 0.5 * placement_reward(p.placement);
            }
        }
        let (rows, rewards) = play_branch(
            &EndOnly,
            RolloutState::new(game, game_rng(31)),
            &opts,
            None,
            1.0,
        )
        .unwrap();
        assert_eq!(rows.len(), 8); // Neither child reaches turn 2.
        assert_eq!(rewards, expected);
        for (row, z) in rows.iter().zip(expected) {
            assert_eq!(row.weight, 1.0);
            assert_eq!(row.z, z);
        }
    }

    #[test]
    fn unequal_subtree_sizes_use_path_probabilities_not_leaf_counts() {
        let (mut game, opts) = fixture(2);
        for (seat, p) in game.players.iter_mut().enumerate() {
            p.alive = seat < 2;
            p.placement = if seat < 2 { 0 } else { seat as u8 + 1 };
            p.health = 1 + seat as i32;
            p.board.truncate(2);
            for (i, m) in p.board.iter_mut().enumerate() {
                // No combat hooks: depending on targets, these boards tie or seat 0 wins.
                // A tie at the turn cap favors seat 1's higher hero Health.
                m.card = cid("BG26_135");
                let (atk, hp) = if seat == 0 {
                    [(1, 1), (4, 4)][i]
                } else {
                    [(1, 3), (3, 1)][i]
                };
                m.atk = atk;
                m.hp = hp;
                m.max_hp = m.hp;
            }
        }
        game.pairs = vec![(0, Some(1))];
        // A tie keeps both players alive; any winner ends the game. Independently
        // enumerate combats to find a tree with one terminal child and two grandchildren.
        let settle = |mut g: Game, mut rng: Pcg64Mcg| {
            for seat in 0..2 {
                let _: f32 = rng.random();
                let _: u64 = rng.random();
                g.end_turn_effects(seat).unwrap();
            }
            std::array::from_fn::<_, 2, _>(|_| {
                let mut child = g.clone();
                child.rng = taverngym::rng::Rng::new(rng.random());
                let child_rng = Pcg64Mcg::seed_from_u64(rng.random());
                child.combat_round();
                (child, child_rng)
            })
        };
        for seed in 0..100 {
            let children = settle(game.clone(), game_rng(seed));
            if children.iter().filter(|(g, _)| g.is_over()).count() != 1 {
                continue;
            }
            let mut leaves = Vec::new();
            for (g, rng) in children {
                if g.is_over() {
                    leaves.push((0.5, g));
                } else {
                    for (leaf, _) in settle(g, rng) {
                        assert!(leaf.is_over());
                        leaves.push((0.25, leaf));
                    }
                }
            }
            let expected: Vec<f32> = (0..8)
                .map(|seat| {
                    leaves
                        .iter()
                        .map(|(p, g)| p * placement_reward(g.players[seat].placement))
                        .sum()
                })
                .collect();
            let flat: f32 = leaves
                .iter()
                .map(|(_, g)| placement_reward(g.players[0].placement))
                .sum::<f32>()
                / 3.0;
            if (expected[0] - flat).abs() < 1e-6 {
                continue;
            }
            let (rows, rewards) = play_branch(
                &EndOnly,
                RolloutState::new(game.clone(), game_rng(seed)),
                &opts,
                None,
                1.0,
            )
            .unwrap();
            assert_eq!(rows.len(), 4); // Two shared rows + two rows on the surviving child.
            for seat in 0..8 {
                assert!((rewards[seat] - expected[seat]).abs() < 1e-6);
            }
            assert_eq!(rows[0].z, rewards[0]);
            assert_eq!(rows[1].z, rewards[1]);
            assert!(rows[..2].iter().all(|r| r.weight == 1.0));
            assert!(rows[2..].iter().all(|r| r.weight == SPLIT_SAMPLE_WEIGHT));
            return;
        }
        panic!("fixture must produce an unequal tree with distinct returns");
    }
}
