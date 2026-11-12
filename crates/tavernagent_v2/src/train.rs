//! Synchronous on-policy training: complete games, per-seat GAE, then PPO epochs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Args;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use serde_json::json;
use taverngym::RulesConfig;

use crate::backend::{Backend, Execution};
use crate::checkpoint::{self, RunState};
use crate::net::{Activation, NetConfig, PolicyValueNet};
use crate::ppo::{self, PpoOptions, Trainer};
use crate::rollout;
use crate::rules::{ActionCounts, Budgets, PolicyRules};

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
pub struct TrainOptions {
    /// CPU reference or FP32 CUDA network/optimizer; stored for consistent resume.
    #[serde(default)]
    #[arg(long,value_enum,default_value_t=Backend::Cpu)]
    pub backend: Backend,
    /// Complete eight-seat self-play games collected with each fixed policy.
    #[arg(long, default_value_t = 64)]
    pub games: usize,
    #[arg(long, default_value_t = 1)]
    pub epochs: usize,
    #[arg(long, default_value_t = 1024)]
    pub batch: usize,
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
    /// Discount per decision of the same seat, not per global engine step.
    #[arg(long, default_value_t = 1.0)]
    pub gamma: f32,
    #[arg(long, default_value_t = 0.95)]
    pub gae_lambda: f32,
    /// If set, gae-lambda applies across turns and this coefficient within a turn.
    #[serde(default)]
    #[arg(long)]
    pub gae_within_turn_lambda: Option<f32>,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = crate::net::DEFAULT_HIDDEN)]
    pub hidden: usize,
    #[arg(long, default_value_t = crate::net::DEFAULT_BLOCKS)]
    pub blocks: usize,
    #[arg(long, default_value_t = crate::net::DEFAULT_EMBEDDING_DIM)]
    pub embedding_dim: usize,
    #[arg(long, default_value = "gelu", value_parser = Activation::parse)]
    pub activation: Activation,
    #[arg(long, default_value_t = 1.0)]
    pub residual_scale: f32,
    #[arg(long)]
    pub head_norm: bool,
    /// Eight placement bins instead of a scalar tanh critic; backbone unchanged.
    #[serde(default)]
    #[arg(long)]
    pub critic_distribution: bool,
    /// Add a scalar prediction of health+armor at the first decision of turn t+3.
    #[serde(default)]
    #[arg(long)]
    pub critic_health: bool,
    /// Engine turn-limit adjudication; lower only for smoke tests.
    #[arg(long, default_value_t = RulesConfig::default().max_turns)]
    pub max_turns: u32,
    #[command(flatten)]
    pub budgets: Budgets,
    #[command(flatten)]
    pub ppo: PpoOptions,
    #[serde(default)]
    #[command(flatten)]
    pub policy_rules: PolicyRules,
}

impl Default for TrainOptions {
    fn default() -> Self {
        #[derive(clap::Parser)]
        struct Defaults {
            #[command(flatten)]
            options: TrainOptions,
        }
        <Defaults as clap::Parser>::parse_from(["tavernagent_v2"]).options
    }
}

impl TrainOptions {
    pub fn net_config(&self) -> NetConfig {
        NetConfig {
            hidden: self.hidden,
            blocks: self.blocks,
            embedding_dim: self.embedding_dim,
            activation: self.activation,
            residual_scale: self.residual_scale,
            head_norm: self.head_norm,
            critic_distribution: self.critic_distribution,
            critic_health: self.critic_health,
        }
    }

    pub fn game_config(&self) -> Result<RulesConfig, String> {
        let mut cfg = self.budgets.config()?;
        if self.max_turns == 0 {
            return Err("max-turns must be positive".into());
        }
        cfg.max_turns = self.max_turns;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.games == 0
            || self.epochs == 0
            || self.batch == 0
            || !self.lr.is_finite()
            || self.lr <= 0.0
        {
            return Err("games, epochs, batch and lr must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) || !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err("gamma and gae-lambda must be in [0, 1]".into());
        }
        self.ppo.validate()?;
        if self
            .gae_within_turn_lambda
            .is_some_and(|v| !(0.0..=1.0).contains(&v))
        {
            return Err("gae-within-turn-lambda must be in [0, 1]".into());
        }
        self.game_config()?;
        Ok(())
    }
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    /// Zero writes an untrained checkpoint for baseline evaluation.
    #[arg(long, default_value_t = 200)]
    pub iterations: usize,
    #[arg(long, default_value_t = 16)]
    pub workers: usize,
    #[arg(long, default_value = "runs/tavernagent_v2/model.npz")]
    pub out: PathBuf,
    /// Also keep an immutable checkpoint every N iterations (0 disables archives).
    #[arg(long, default_value_t = 0)]
    pub save_every: usize,
    #[command(flatten)]
    pub options: TrainOptions,
}

#[derive(Debug, Args)]
pub struct ResumeArgs {
    /// Defaults to the backend saved in the checkpoint. Switching changes numerical arithmetic.
    #[arg(long, value_enum)]
    pub backend: Option<Backend>,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Cumulative target; saved network, PPO options, optimizer and RNG are reused exactly.
    #[arg(long)]
    pub iterations: usize,
    #[arg(long, default_value_t = 16)]
    pub workers: usize,
    /// Defaults to overwriting the supplied checkpoint after each completed iteration.
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub save_every: usize,
}

#[derive(Debug, Default, Args)]
pub struct ForkArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Must be a new path; source weights, optimizer, counters and RNG are preserved.
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, action = clap::ArgAction::Set)]
    pub block_consecutive_freeze: Option<bool>,
    #[arg(long, action = clap::ArgAction::Set)]
    pub freeze_at_end: Option<bool>,
    #[arg(long)]
    pub lr: Option<f64>,
    /// Complete games per future iteration; existing cumulative counts are preserved.
    #[arg(long)]
    pub games: Option<usize>,
    /// PPO passes over each future rollout; KL early stopping still applies.
    #[arg(long)]
    pub epochs: Option<usize>,
    #[arg(long)]
    pub entropy_coef: Option<f64>,
    #[arg(long)]
    pub gae_lambda: Option<f32>,
    #[arg(long)]
    pub gae_within_turn_lambda: Option<f32>,
    /// Return to per-decision GAE; otherwise inherit the parent's trace mode.
    #[arg(long, conflicts_with = "gae_within_turn_lambda")]
    pub clear_gae_within_turn_lambda: bool,
    #[arg(long)]
    pub value_coef: Option<f64>,
    #[arg(long)]
    pub terminal_value_coef: Option<f64>,
    #[arg(long)]
    pub health_value_coef: Option<f64>,
}

pub fn fork(args: ForkArgs) -> Result<(), String> {
    if args.out.exists() {
        return Err("fork output already exists; choose a new checkpoint path".into());
    }
    let (trainer, mut state) = checkpoint::load(&args.checkpoint)?;
    let previous = state.options.clone();
    if let Some(value) = args.block_consecutive_freeze {
        state.options.policy_rules.block_consecutive_freeze = value;
    }
    if let Some(value) = args.freeze_at_end {
        state.options.policy_rules.freeze_at_end = value;
    }
    if let Some(value) = args.lr {
        state.options.lr = value;
    }
    if let Some(value) = args.games {
        state.options.games = value;
    }
    if let Some(value) = args.epochs {
        state.options.epochs = value;
    }
    if let Some(value) = args.entropy_coef {
        state.options.ppo.entropy_coef = value;
    }
    if let Some(value) = args.gae_lambda {
        state.options.gae_lambda = value;
    }
    if args.clear_gae_within_turn_lambda {
        if args.gae_within_turn_lambda.is_some() {
            return Err("cannot both set and clear gae-within-turn-lambda".into());
        }
        state.options.gae_within_turn_lambda = None;
    } else if let Some(value) = args.gae_within_turn_lambda {
        state.options.gae_within_turn_lambda = Some(value);
    }
    if let Some(value) = args.value_coef {
        state.options.ppo.value_coef = value;
    }
    if let Some(value) = args.terminal_value_coef {
        state.options.ppo.terminal_value_coef = value;
    }
    if let Some(value) = args.health_value_coef {
        state.options.ppo.health_value_coef = value;
    }
    state.options.validate()?;
    checkpoint::save(&args.out, &trainer, &state)?;
    println!(
        "{}",
        json!({"event":"fork", "parent":args.checkpoint, "out":args.out,
        "iteration":state.iteration,"previous_policy_rules":previous.policy_rules,"policy_rules":state.options.policy_rules,
        "previous_options":previous,"options":state.options})
    );
    Ok(())
}

pub fn train(args: TrainArgs) -> Result<(), String> {
    args.options.validate()?;
    let net = PolicyValueNet::new(args.options.seed, args.options.net_config())?;
    let state = RunState {
        format: checkpoint::FORMAT,
        algorithm: checkpoint::ALGORITHM.into(),
        iteration: 0,
        games: 0,
        transitions: 0,
        shuffle_rng: Pcg64Mcg::seed_from_u64(args.options.seed ^ 0x5348_5546),
        options: args.options,
        history: Vec::new(),
    };
    let trainer = Trainer::new(net);
    if args.workers == 0 {
        return Err("workers must be positive".into());
    }
    if args.iterations == 0 {
        checkpoint::save(&args.out, &trainer, &state)?;
        println!("saved {} (iteration 0, untrained)", args.out.display());
        return Ok(());
    }
    run(
        trainer,
        state,
        args.iterations,
        args.workers,
        &args.out,
        args.save_every,
        args.device,
    )
}

pub fn resume(args: ResumeArgs) -> Result<(), String> {
    let (trainer, mut state) = checkpoint::load(&args.checkpoint)?;
    if let Some(backend) = args.backend {
        state.options.backend = backend;
    }
    run(
        trainer,
        state,
        args.iterations,
        args.workers,
        args.out.as_deref().unwrap_or(&args.checkpoint),
        args.save_every,
        args.device,
    )
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    path.with_file_name(format!(
        "{}{suffix}",
        path.file_stem().unwrap_or_default().to_string_lossy()
    ))
}

fn run(
    mut trainer: Trainer,
    mut state: RunState,
    iterations: usize,
    workers: usize,
    out: &Path,
    save_every: usize,
    device: usize,
) -> Result<(), String> {
    if workers == 0 || iterations <= state.iteration {
        return Err(
            "workers must be positive and iterations must exceed the saved iteration".into(),
        );
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| e.to_string())?;
    let cfg = state.options.game_config()?;
    let mut execution = Execution::new(state.options.backend, &trainer, device)?;
    println!(
        "PPO-Clip + GAE: {} parameters, {} games/iteration, {} workers, FP32 inference, {:?} backend",
        trainer.net.n_params(),
        state.options.games,
        workers,
        state.options.backend
    );
    for iteration in state.iteration + 1..=iterations {
        let iteration_start = Instant::now();
        let start = Instant::now();
        let opts = &state.options;
        let seeds = (0..opts.games)
            .map(|i| {
                opts.seed
                    .checked_add(state.games)
                    .and_then(|s| s.checked_add(i as u64))
                    .ok_or_else(|| "game seed overflow".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let games = match opts.backend {
            Backend::Cpu => {
                rollout::collect_with_rules(&trainer.net, &seeds, &cfg, &pool, opts.policy_rules)?
            }
            Backend::Cuda => crate::batched::collect(
                &trainer.net,
                &seeds,
                &cfg,
                opts.policy_rules,
                &pool,
                &execution,
            )?,
        };
        let rollout_seconds = start.elapsed().as_secs_f64();
        let mean_turns = games.iter().map(|g| g.turns as f64).sum::<f64>() / games.len() as f64;
        let turn_limit_games = games.iter().filter(|g| g.turn_limit_end).count();
        let truncated_combats: u64 = games.iter().map(|g| g.truncated_combats as u64).sum();
        let seat0_mean_placement = games
            .iter()
            .map(|g| g.trajectories[0].placement as f64)
            .sum::<f64>()
            / games.len() as f64;
        let mut samples = Vec::new();
        let mut actions = ActionCounts::default();
        for game in games {
            actions.add(&game.actions);
            for trajectory in game.trajectories {
                samples.extend(trajectory.into_samples_with_trace(
                    opts.gamma,
                    opts.gae_lambda,
                    opts.gae_within_turn_lambda,
                )?);
            }
        }
        let samples_count = samples.len();
        let mut critic = crate::critic::summarize(&samples);
        // Keep full turn buckets every 50 iterations without bloating every checkpoint.
        if !iteration.is_multiple_of(50) {
            critic.as_object_mut().unwrap().remove("by_turn");
        }
        rollout::normalize_advantages(&mut samples)?;
        let collect_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let metrics = if opts.backend == Backend::Cpu {
            pool.install(|| {
                ppo::update(
                    &mut trainer,
                    &samples,
                    opts.epochs,
                    opts.batch,
                    opts.lr,
                    &opts.ppo,
                    &mut state.shuffle_rng,
                )
            })?
        } else {
            execution.update(
                &mut trainer,
                &samples,
                opts.epochs,
                opts.batch,
                opts.lr,
                &opts.ppo,
                &mut state.shuffle_rng,
            )?
        };
        let train_seconds = start.elapsed().as_secs_f64();
        // No buffer, pending seat transition, or old behavior policy survives this boundary.
        let start = Instant::now();
        drop(samples);
        let cleanup_seconds = start.elapsed().as_secs_f64();
        state.iteration = iteration;
        state.games += opts.games as u64;
        state.transitions += samples_count as u64;
        let mut entry = json!({"iteration": iteration, "games": opts.games, "transitions": samples_count,
            "backend":opts.backend,
            "total_games": state.games, "total_transitions": state.transitions,
            "mean_turns": mean_turns, "seat0_mean_placement": seat0_mean_placement,
            "turn_limit_games": turn_limit_games, "truncated_combats": truncated_combats,
            "collect_seconds": collect_seconds, "train_seconds": train_seconds,
            "rollout_seconds": rollout_seconds, "prepare_seconds": collect_seconds - rollout_seconds,
            "cleanup_seconds": cleanup_seconds,
            "actions": actions,
            "policy_rules": opts.policy_rules,
            "critic": critic,
            "ppo": metrics});
        state.history.push(entry.clone());
        let start = Instant::now();
        checkpoint::save(out, &trainer, &state)?;
        if save_every > 0 && iteration.is_multiple_of(save_every) {
            checkpoint::save(
                &sibling(out, &format!(".iter{iteration}.npz")),
                &trainer,
                &state,
            )?;
        }
        let checkpoint_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let history = sibling(out, ".history.json");
        let temp = history.with_extension("json.tmp");
        std::fs::write(
            &temp,
            serde_json::to_vec_pretty(&state.history).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        std::fs::rename(temp, history).map_err(|e| e.to_string())?;
        // These timings are emitted after persistence, so they include its real cost.
        // They live in stdout; the snapshot's history necessarily predates its own save.
        entry["checkpoint_seconds"] = checkpoint_seconds.into();
        entry["history_seconds"] = start.elapsed().as_secs_f64().into();
        entry["iteration_seconds"] = iteration_start.elapsed().as_secs_f64().into();
        println!("{entry}");
    }
    println!("saved {} (iteration {})", out.display(), state.iteration);
    Ok(())
}
