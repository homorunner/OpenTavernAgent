//! PPO-Clip, entropy regularization, value regression and AdamW. No replay or search targets.

use clap::Args;
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::linalg::Mat;
use crate::net::{N_ACTIONS, Params, PolicyValueNet};
use crate::policy::Categorical;
use crate::rollout::Sample;

/// Fixed chunks make gradient reduction independent of worker count.
const CHUNK_ROWS: usize = 32;
// Splitting by tensor alone leaves the wide input projection on one worker.
// These chunks change only scheduling; each element retains its original sum order.
const PARAM_CHUNK: usize = 4096;

#[derive(Clone, Debug, PartialEq, Args, Serialize, Deserialize)]
pub struct PpoOptions {
    #[arg(long, default_value_t = 0.2)]
    pub clip_range: f64,
    #[arg(long, default_value_t = 0.5)]
    pub value_coef: f64,
    /// Additional regression to the observed terminal placement reward (same scalar head).
    #[serde(default)]
    #[arg(long, default_value_t = 0.0)]
    pub terminal_value_coef: f64,
    /// Clip only the critic target; actor advantages retain the original GAE estimate.
    #[serde(default)]
    #[arg(long)]
    pub clip_value_target: bool,
    /// Auxiliary future-health loss, active only with --critic-health.
    #[serde(default = "default_health_coef")]
    #[arg(long, default_value_t = 0.1)]
    pub health_value_coef: f64,
    #[arg(long, default_value_t = 0.03)]
    pub entropy_coef: f64,
    /// Replace entropy bonus with -KL(policy || uniform legal categories then actions).
    #[serde(default)]
    #[arg(long)]
    pub category_balanced_entropy: bool,
    #[arg(long, default_value_t = 0.5)]
    pub max_grad_norm: f64,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f64,
    /// Stop before a minibatch update whose sampled approximate KL exceeds this (0 disables).
    #[arg(long, default_value_t = 0.02)]
    pub target_kl: f64,
}

impl Default for PpoOptions {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            value_coef: 0.5,
            terminal_value_coef: 0.0,
            clip_value_target: false,
            health_value_coef: 0.1,
            entropy_coef: 0.03,
            category_balanced_entropy: false,
            max_grad_norm: 0.5,
            weight_decay: 0.01,
            target_kl: 0.02,
        }
    }
}

impl PpoOptions {
    pub fn validate(&self) -> Result<(), String> {
        if !self.clip_range.is_finite() || self.clip_range <= 0.0 || self.clip_range >= 1.0 {
            return Err("clip-range must be in (0, 1)".into());
        }
        for (name, value) in [
            ("value-coef", self.value_coef),
            ("terminal-value-coef", self.terminal_value_coef),
            ("health-value-coef", self.health_value_coef),
            ("entropy-coef", self.entropy_coef),
            ("weight-decay", self.weight_decay),
            ("target-kl", self.target_kl),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and nonnegative"));
            }
        }
        if !self.max_grad_norm.is_finite() || self.max_grad_norm <= 0.0 {
            return Err("max-grad-norm must be finite and positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LossMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub terminal_value_loss: f64,
    pub health_value_loss: f64,
    pub entropy: f64,
    /// Actual exploration bonus: entropy, or negative category-reference KL.
    #[serde(default)]
    pub exploration_bonus: f64,
    pub approx_kl: f64,
    pub clip_fraction: f64,
    pub ratio_mean: f64,
}

impl LossMetrics {
    pub(crate) fn add_scaled(&mut self, other: &Self, scale: f64) {
        self.policy_loss += other.policy_loss * scale;
        self.value_loss += other.value_loss * scale;
        self.terminal_value_loss += other.terminal_value_loss * scale;
        self.health_value_loss += other.health_value_loss * scale;
        self.entropy += other.entropy * scale;
        self.exploration_bonus += other.exploration_bonus * scale;
        self.approx_kl += other.approx_kl * scale;
        self.clip_fraction += other.clip_fraction * scale;
        self.ratio_mean += other.ratio_mean * scale;
    }

    pub fn total_loss(&self, opts: &PpoOptions) -> f64 {
        self.policy_loss
            + opts.value_coef * self.value_loss
            + opts.terminal_value_coef * self.terminal_value_loss
            + opts.health_value_coef * self.health_value_loss
            - opts.entropy_coef
                * if opts.category_balanced_entropy {
                    self.exploration_bonus
                } else {
                    self.entropy
                }
    }
}

/// Gradient with respect to logits, including the entropy term. The clipped branch
/// depends on the advantage's sign; clipping is not a blanket zero outside the interval.
pub fn policy_loss_gradient(
    logits: &[f32],
    mask: &[bool],
    action: usize,
    old_log_prob: f64,
    advantage: f32,
    opts: &PpoOptions,
) -> Result<(LossMetrics, Vec<f32>), String> {
    opts.validate()?;
    if action >= mask.len()
        || !mask[action]
        || !old_log_prob.is_finite()
        || old_log_prob > 1e-6
        || !advantage.is_finite()
    {
        return Err("invalid PPO action, behavior log probability or advantage".into());
    }
    let dist = Categorical::new(logits, mask)?;
    let mut log_reference = vec![0.0; logits.len()];
    if opts.category_balanced_entropy {
        use crate::actions::{N_CATEGORIES, category};
        if mask.len() != N_ACTIONS {
            return Err(
                "category-balanced entropy requires the complete agent action space".into(),
            );
        }
        let mut counts = [0usize; N_CATEGORIES];
        for (a, &legal) in mask.iter().enumerate() {
            if legal {
                counts[category(a)] += 1;
            }
        }
        let groups = counts.iter().filter(|&&n| n > 0).count();
        for (a, &legal) in mask.iter().enumerate() {
            if legal {
                log_reference[a] = -((groups * counts[category(a)]) as f64).ln();
            }
        }
    }
    let exploration_bonus = if opts.category_balanced_entropy {
        -dist
            .probabilities
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(a, p)| p * (dist.log_probabilities[a] - log_reference[a]))
            .sum::<f64>()
    } else {
        dist.entropy
    };
    let log_ratio = dist.log_probabilities[action] - old_log_prob;
    let ratio = log_ratio.exp();
    if !ratio.is_finite() {
        return Err("nonfinite PPO importance ratio".into());
    }
    let advantage = advantage as f64;
    let clipped_ratio = ratio.clamp(1.0 - opts.clip_range, 1.0 + opts.clip_range);
    let clipped_branch = (advantage > 0.0 && ratio > 1.0 + opts.clip_range)
        || (advantage < 0.0 && ratio < 1.0 - opts.clip_range);
    let coeff = if clipped_branch {
        0.0
    } else {
        ratio * advantage
    };
    let mut grad = vec![0.0; logits.len()];
    for (a, &legal) in mask.iter().enumerate() {
        if legal {
            let p = dist.probabilities[a];
            grad[a] = (coeff * (p - f64::from(a == action))
                + opts.entropy_coef
                    * p
                    * (dist.log_probabilities[a] - log_reference[a] + exploration_bonus))
                as f32;
        }
    }
    let metrics = LossMetrics {
        policy_loss: -(ratio * advantage).min(clipped_ratio * advantage),
        entropy: dist.entropy,
        exploration_bonus,
        approx_kl: (log_ratio.exp_m1() - log_ratio).max(0.0),
        clip_fraction: f64::from((ratio - 1.0).abs() > opts.clip_range),
        ratio_mean: ratio,
        ..Default::default()
    };
    Ok((metrics, grad))
}

fn chunk_gradients(
    net: &PolicyValueNet,
    rows: &[&Sample],
    opts: &PpoOptions,
) -> Result<(Params, LossMetrics), String> {
    let xs: Vec<_> = rows.iter().map(|s| s.transition.obs.x.as_slice()).collect();
    let cards: Vec<_> = rows
        .iter()
        .map(|s| s.transition.obs.cards.as_slice())
        .collect();
    let masks: Vec<_> = rows
        .iter()
        .map(|s| s.transition.obs.mask.as_slice())
        .collect();
    let (logits, values, cache) = net.forward(&xs, &cards, &masks)?;
    let (dlogits, dvalue, metrics) =
        output_gradients(rows, &logits, &values, opts, &net.cfg, &cache.value_logits)?;
    Ok((net.backward(&cache, &dlogits, &dvalue), metrics))
}

/// Shared FP64 masked distribution/loss arithmetic for both network backends.
pub(crate) fn output_gradients(
    rows: &[&Sample],
    logits: &Mat,
    values: &[f32],
    opts: &PpoOptions,
    config: &crate::net::NetConfig,
    raw_values: &Mat,
) -> Result<(Mat, Vec<f32>, LossMetrics), String> {
    let mut dlogits = Mat::zeros(rows.len(), N_ACTIONS);
    let width = config.value_outputs();
    let mut dvalue = vec![0.0; rows.len() * width];
    let mut metrics = LossMetrics::default();
    for (r, sample) in rows.iter().enumerate() {
        if !sample.return_.is_finite() || !values[r].is_finite() {
            return Err("nonfinite PPO value or return".into());
        }
        let (mut row_metrics, grad) = policy_loss_gradient(
            logits.row(r),
            &sample.transition.obs.mask,
            sample.transition.action,
            sample.transition.old_log_prob,
            sample.advantage,
            opts,
        )?;
        dlogits.row_mut(r).copy_from_slice(&grad);
        let target = if opts.clip_value_target {
            sample.return_.clamp(-1.0, 1.0)
        } else {
            sample.return_
        };
        let error = values[r] as f64 - target as f64;
        if !sample.terminal_reward.is_finite() {
            return Err("nonfinite terminal reward".into());
        }
        let terminal_error = values[r] as f64 - sample.terminal_reward as f64;
        if config.critic_distribution {
            let probabilities = crate::net::placement_probabilities(raw_values.row(r));
            let lambda_target = project_placement(sample.return_);
            let terminal_target = project_placement(sample.terminal_reward);
            for k in 0..8 {
                row_metrics.value_loss -= lambda_target[k] * probabilities[k].max(1e-300).ln();
                row_metrics.terminal_value_loss -=
                    terminal_target[k] * probabilities[k].max(1e-300).ln();
                dvalue[r * width + k] = (opts.value_coef * (probabilities[k] - lambda_target[k])
                    + opts.terminal_value_coef * (probabilities[k] - terminal_target[k]))
                    as f32;
            }
        } else {
            row_metrics.value_loss = error * error;
            row_metrics.terminal_value_loss = terminal_error * terminal_error;
            // backward() consumes the derivative of the pre-tanh value.
            dvalue[r * width] = (2.0
                * (opts.value_coef * error + opts.terminal_value_coef * terminal_error)
                * (1.0 - (values[r] as f64).powi(2))) as f32;
        }
        if config.critic_health
            && let Some(target) = sample.future_health
        {
            let prediction = raw_values.row(r)[width - 1].tanh() as f64;
            let error = prediction - f64::from(target);
            row_metrics.health_value_loss = error * error;
            dvalue[r * width + width - 1] =
                (2.0 * opts.health_value_coef * error * (1.0 - prediction * prediction)) as f32;
        }
        metrics.add_scaled(&row_metrics, 1.0);
    }
    Ok((dlogits, dvalue, metrics))
}

fn default_health_coef() -> f64 {
    0.1
}

/// Linear projection preserves the expectation on the eight placement-reward bins.
fn project_placement(value: f32) -> [f64; 8] {
    let position = (1.0 - f64::from(value).clamp(-1.0, 1.0)) * 3.5;
    let low = position.floor() as usize;
    let high = (low + 1).min(7);
    let fraction = position - low as f64;
    let mut result = [0.0; 8];
    result[low] += 1.0 - fraction;
    result[high] += fraction;
    result
}

/// Mean batch loss and gradients; old probabilities, advantages and targets are constants.
pub fn loss_gradients(
    net: &PolicyValueNet,
    rows: &[&Sample],
    opts: &PpoOptions,
) -> Result<(Params, LossMetrics), String> {
    let (grads, metrics, _, _) = timed_loss_gradients(net, rows, opts)?;
    Ok((grads, metrics))
}

fn timed_loss_gradients(
    net: &PolicyValueNet,
    rows: &[&Sample],
    opts: &PpoOptions,
) -> Result<(Params, LossMetrics, f64, f64), String> {
    opts.validate()?;
    if rows.is_empty() {
        return Err("empty PPO batch".into());
    }
    let start = Instant::now();
    let parts: Vec<_> = rows
        .par_chunks(CHUNK_ROWS)
        .map(|chunk| chunk_gradients(net, chunk, opts))
        .collect::<Result<_, _>>()?;
    let gradient_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut grads = net.p.zeros_like();
    let mut metrics = LossMetrics::default();
    let norm = 1.0 / rows.len() as f64;
    for (_, loss) in &parts {
        metrics.add_scaled(loss, norm);
    }
    grads
        .tensors
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| {
            out.data
                .par_chunks_mut(PARAM_CHUNK)
                .enumerate()
                .for_each(|(chunk, dst)| {
                    let start = chunk * PARAM_CHUNK;
                    let end = start + dst.len();
                    for (part, _) in &parts {
                        for (value, add) in dst.iter_mut().zip(&part.tensors[i].data[start..end]) {
                            *value += add;
                        }
                    }
                    for value in dst {
                        *value *= norm as f32;
                    }
                });
        });
    if !grads.all_finite()
        || !metrics.total_loss(opts).is_finite()
        || !metrics.approx_kl.is_finite()
    {
        return Err("nonfinite PPO loss or gradient".into());
    }
    Ok((
        grads,
        metrics,
        gradient_seconds,
        start.elapsed().as_secs_f64(),
    ))
}

#[derive(Clone, Debug)]
pub struct Trainer {
    pub net: PolicyValueNet,
    pub m: Params,
    pub v: Params,
    pub optimizer_step: u64,
}

impl Trainer {
    pub fn new(net: PolicyValueNet) -> Self {
        Self {
            m: net.p.zeros_like(),
            v: net.p.zeros_like(),
            net,
            optimizer_step: 0,
        }
    }

    pub fn apply_gradients(
        &mut self,
        grads: &Params,
        lr: f64,
        opts: &PpoOptions,
    ) -> Result<f64, String> {
        opts.validate()?;
        if !lr.is_finite() || lr <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        if grads.tensors.len() != self.net.p.tensors.len()
            || grads
                .tensors
                .iter()
                .zip(&self.net.p.tensors)
                .any(|(a, b)| a.rows != b.rows || a.cols != b.cols)
        {
            return Err("gradient shape mismatch".into());
        }
        let norm = grads.sum_squares().sqrt();
        if !norm.is_finite() {
            return Err("nonfinite gradient norm".into());
        }
        let scale = if norm > opts.max_grad_norm {
            opts.max_grad_norm / norm
        } else {
            1.0
        };
        self.optimizer_step = self
            .optimizer_step
            .checked_add(1)
            .ok_or("optimizer step overflow")?;
        let (b1, b2) = (0.9f64, 0.999f64);
        let bias1 = 1.0 - b1.powf(self.optimizer_step as f64);
        let bias2 = 1.0 - b2.powf(self.optimizer_step as f64);
        let decay: Vec<_> = (0..self.net.p.tensors.len())
            .map(|i| self.net.p.is_weight(i))
            .collect();
        self.net
            .p
            .tensors
            .par_iter_mut()
            .zip(self.m.tensors.par_iter_mut())
            .zip(self.v.tensors.par_iter_mut())
            .zip(grads.tensors.par_iter())
            .enumerate()
            .for_each(|(i, (((p, m), v), g))| {
                p.data
                    .par_chunks_mut(PARAM_CHUNK)
                    .zip(m.data.par_chunks_mut(PARAM_CHUNK))
                    .zip(v.data.par_chunks_mut(PARAM_CHUNK))
                    .zip(g.data.par_chunks(PARAM_CHUNK))
                    .for_each(|(((p, m), v), g)| {
                        for j in 0..p.len() {
                            let grad = g[j] as f64 * scale;
                            m[j] = (b1 * m[j] as f64 + (1.0 - b1) * grad) as f32;
                            v[j] = (b2 * v[j] as f64 + (1.0 - b2) * grad * grad) as f32;
                            let update = (m[j] as f64 / bias1)
                                / ((v[j] as f64 / bias2).sqrt() + 1e-8)
                                + if decay[i] {
                                    opts.weight_decay * p[j] as f64
                                } else {
                                    0.0
                                };
                            p[j] = (p[j] as f64 - lr * update) as f32;
                        }
                    });
            });
        if !self.net.p.all_finite() || !self.m.all_finite() || !self.v.all_finite() {
            return Err("nonfinite optimizer state".into());
        }
        Ok(norm)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct UpdateMetrics {
    #[serde(flatten)]
    pub loss: LossMetrics,
    pub updates: usize,
    pub epochs_completed: usize,
    pub early_stop: bool,
    pub max_approx_kl: f64,
    pub mean_grad_norm: f64,
    /// Rows actually used by AdamW, excluding the minibatch rejected by the KL guard.
    pub trained_samples: usize,
    pub evaluated_samples: usize,
    pub first_epoch_samples: usize,
    pub effective_epochs: f64,
    pub gradient_seconds: f64,
    pub reduction_seconds: f64,
    pub optimizer_seconds: f64,
}

/// Reuse only this fresh batch for a bounded number of shuffled PPO epochs.
pub fn update(
    trainer: &mut Trainer,
    samples: &[Sample],
    epochs: usize,
    batch_size: usize,
    lr: f64,
    opts: &PpoOptions,
    rng: &mut impl Rng,
) -> Result<UpdateMetrics, String> {
    opts.validate()?;
    if samples.is_empty() || epochs == 0 || batch_size == 0 || !lr.is_finite() || lr <= 0.0 {
        return Err("PPO needs samples, positive epochs, batch size and learning rate".into());
    }
    let mut order: Vec<_> = (0..samples.len()).collect();
    let mut summary = UpdateMetrics::default();
    let mut seen = 0;
    'epochs: for epoch in 0..epochs {
        order.shuffle(rng);
        for chunk in order.chunks(batch_size) {
            let rows: Vec<_> = chunk.iter().map(|&i| &samples[i]).collect();
            let (grads, metrics, gradient_seconds, reduction_seconds) =
                timed_loss_gradients(&trainer.net, &rows, opts)?;
            summary.gradient_seconds += gradient_seconds;
            summary.reduction_seconds += reduction_seconds;
            summary.evaluated_samples += rows.len();
            summary.max_approx_kl = summary.max_approx_kl.max(metrics.approx_kl);
            if opts.target_kl > 0.0 && metrics.approx_kl > opts.target_kl {
                summary.early_stop = true;
                break 'epochs;
            }
            let start = Instant::now();
            summary.mean_grad_norm += trainer.apply_gradients(&grads, lr, opts)?;
            summary.optimizer_seconds += start.elapsed().as_secs_f64();
            summary.loss.add_scaled(&metrics, rows.len() as f64);
            seen += rows.len();
            if epoch == 0 {
                summary.first_epoch_samples += rows.len();
            }
            summary.updates += 1;
        }
        summary.epochs_completed += 1;
    }
    let sums = summary.loss.clone();
    summary.loss = LossMetrics::default();
    summary.loss.add_scaled(&sums, 1.0 / seen.max(1) as f64);
    summary.mean_grad_norm /= summary.updates.max(1) as f64;
    summary.trained_samples = seen;
    summary.effective_epochs = seen as f64 / samples.len() as f64;
    Ok(summary)
}
