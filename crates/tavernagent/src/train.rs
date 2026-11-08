//! The self-play training loop: collect a round of games in parallel, train, checkpoint.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Args;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::mcts::{self, SearchOptions};
use crate::net::{self, Activation, NetConfig, PolicyValueNet};
use crate::probe;
use crate::rules::Budgets;
use crate::selfplay::{
    OptimizerHparams, PlayOptions, ReplayWindow, Sample, Stats, StepOptions, Trainer,
    TrainingState, cosine_lr, ema_decay_for_window, load_training_state, play_one,
    read_training_meta, save_training_state, train_epoch, validate_split_turns,
};

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
pub struct SelfplayArgs {
    /// Cumulative target number of collect+train rounds.
    #[arg(long, default_value_t = 200)]
    pub steps: usize,
    /// Self-play roots per round (each can have multiple split continuations).
    #[arg(long, default_value_t = 128)]
    pub games: usize,
    /// MCTS simulations per searched decision.
    #[arg(long, default_value_t = 256)]
    pub sims: usize,
    /// Fraction of decisions that are searched and recorded; the others sample the
    /// network prior at the play temperature and are not training rows.
    #[arg(long, default_value_t = 0.33)]
    pub search_prob: f32,
    /// Binary splits after these turns' end effects, before combat (e.g. 4,8).
    /// Pass --split-turn without values to disable splitting.
    #[arg(long, value_delimiter = ',', num_args = 0.., default_value = "4")]
    pub split_turn: Vec<u32>,
    /// Training passes per round; each draws `max_samples` from the replay window.
    #[arg(long, default_value_t = 2)]
    pub epochs: usize,
    #[arg(long, default_value_t = 3e-5)]
    pub lr: f64,
    /// Final learning rate when --lr-decay-steps is positive; otherwise ignored.
    #[arg(long, default_value_t = 3e-5)]
    pub lr_min: f64,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f32,
    #[arg(long, default_value_t = 512)]
    pub batch: usize,
    /// Threads running the self-play searches (and evaluating probes).
    #[arg(long, default_value_t = 10)]
    pub workers: usize,
    /// Size of the uniform draw from the replay window that each epoch trains on.
    #[arg(long, default_value_t = 80_000)]
    pub max_samples: usize,
    #[command(flatten)]
    #[serde(flatten)]
    pub budgets: Budgets,
    #[arg(long, default_value_t = 1.1)]
    pub root_temperature: f32,
    #[arg(long, default_value_t = 0.5)]
    pub temperature: f32,
    /// Temperature for the first `warm_steps` rounds.
    #[arg(long, default_value_t = 0.5)]
    pub temperature_warm: f32,
    #[arg(long, default_value_t = 5)]
    pub warm_steps: usize,
    /// Fraction of the policy target spread over legal actions.
    #[arg(long, default_value_t = 0.05)]
    pub policy_smoothing: f32,
    /// Weight of the advantage-weighted policy gradient term.
    #[arg(long, default_value_t = 0.0)]
    pub pg_weight: f32,
    /// Dirichlet mixing at the search root during self-play.
    #[arg(long, default_value_t = 0.20)]
    pub noise: f32,
    #[arg(long, default_value_t = 0.3)]
    pub noise_alpha: f32,
    /// LCB move selection: stdevs below the mean (0 = plain visit argmax).
    #[arg(long, default_value_t = crate::mcts::DEFAULT_LCB_STDEVS)]
    pub lcb_stdevs: f32,
    /// Fraction of the top visit count an action needs to be an LCB candidate.
    #[arg(long, default_value_t = crate::mcts::DEFAULT_LCB_MIN_VISIT_PROP)]
    pub lcb_min_visit_prop: f32,
    /// First-play urgency: unvisited edges valued at the parent mean minus this
    /// (negative = the original rule, unvisited edges at Q = 0).
    #[arg(long, default_value_t = crate::mcts::DEFAULT_FPU_REDUCTION)]
    pub fpu_reduction: f32,
    /// Fraction of games scheduling one forced End Turn: uniform seat and action
    /// index 1..=32 (deferred until legal). Discard its earlier intervention-path rows.
    #[arg(long, default_value_t = 0.25)]
    pub end_explore: f32,
    #[arg(long, default_value_t = crate::mcts::DEFAULT_C_PUCT)]
    pub c_puct: f32,
    #[arg(long, default_value = "gelu", value_parser = Activation::parse)]
    pub activation: Activation,
    #[arg(long, default_value_t = 1.0)]
    pub residual_scale: f32,
    /// Fixed LayerNorm before the policy/value heads.
    #[arg(long, default_value_t = false)]
    pub head_norm: bool,
    #[arg(long, default_value_t = net::DEFAULT_EMBEDDING_DIM)]
    pub embedding_dim: usize,
    /// Trunk width.
    #[arg(long, default_value_t = net::DEFAULT_HIDDEN)]
    pub hidden: usize,
    /// Residual blocks in the trunk.
    #[arg(long, default_value_t = net::DEFAULT_BLOCKS)]
    pub blocks: usize,
    #[arg(long, default_value_t = 0.5)]
    pub value_weight: f32,
    #[arg(long, default_value_t = 5.0)]
    pub clip: f32,
    /// Averaging window of the EMA weights, in self-play rounds (converted to a
    /// per-minibatch decay from `max_samples`, `batch` and `epochs`; warmed up early on).
    #[arg(long, default_value_t = 2.0)]
    pub ema_rounds: f64,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value = "models/sp.npz")]
    pub out: PathBuf,
    /// Log the turn-1/turn-2 behaviour every N rounds (0 = off).
    #[arg(long, default_value_t = 5)]
    pub probe_every: usize,
    #[arg(long, default_value_t = 16)]
    pub probe_games: usize,
    /// Start from these weights (optimizer and step count start over).
    #[arg(long)]
    pub resume: Option<PathBuf>,
    /// Fixed replay window override; otherwise min(6, 3 + step / 200) rounds.
    #[arg(long)]
    pub replay_rounds: Option<usize>,
    /// Length of the cosine schedule in rounds (0 = constant learning rate).
    #[arg(long, default_value_t = 0)]
    pub lr_decay_steps: usize,
    /// Keep an immutable checkpoint every N rounds (0 = off).
    #[arg(long, default_value_t = 0)]
    pub save_every: usize,
    /// Write the full trainer state (optimizer, RNGs, replay window) after every round.
    #[arg(long, default_value_t = false)]
    pub save_trainer_state: bool,
    /// Continue exactly from a trainer state; saved options apply unless given again.
    #[arg(long)]
    pub resume_state: Option<PathBuf>,
}

impl Default for SelfplayArgs {
    /// The command-line defaults.
    fn default() -> Self {
        #[derive(clap::Parser)]
        struct Wrapper {
            #[command(flatten)]
            inner: SelfplayArgs,
        }
        <Wrapper as clap::Parser>::parse_from(["tavernagent"]).inner
    }
}

impl SelfplayArgs {
    /// Capacity at the cumulative training round (integer division).
    pub fn replay_capacity(&self, step: usize) -> usize {
        self.replay_rounds.unwrap_or(3 + (step / 200).min(3))
    }

    pub fn net_config(&self) -> NetConfig {
        NetConfig {
            hidden: self.hidden,
            blocks: self.blocks,
            embedding_dim: self.embedding_dim,
            activation: self.activation,
            residual_scale: self.residual_scale,
            head_norm: self.head_norm,
        }
    }

    /// Optimizer updates per self-play round at these settings.
    pub fn updates_per_round(&self) -> usize {
        self.max_samples.div_ceil(self.batch.max(1)) * self.epochs
    }

    pub fn hparams(&self) -> OptimizerHparams {
        OptimizerHparams {
            wd: self.weight_decay,
            clip: self.clip,
            value_weight: self.value_weight,
            ema_decay: ema_decay_for_window(self.ema_rounds * self.updates_per_round() as f64),
        }
    }

    fn validate(&self) -> Result<(), String> {
        validate_split_turns(&self.split_turn)?;
        if [
            self.steps,
            self.games,
            self.epochs,
            self.workers,
            self.max_samples,
            self.batch,
            self.sims,
        ]
        .iter()
        .any(|v| *v < 1)
        {
            return Err(
                "steps/games/epochs/workers/max-samples/batch/sims must be positive".into(),
            );
        }
        for (name, v) in [
            ("noise", self.noise),
            ("policy-smoothing", self.policy_smoothing),
            ("end-explore", self.end_explore),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("{name} must be in [0, 1]"));
            }
        }
        if !self.lr.is_finite() || self.lr < 0.0 {
            return Err("lr must be finite and nonnegative".into());
        }
        if !self.lr_min.is_finite()
            || self.lr_min < 0.0
            || (self.lr_decay_steps > 0 && self.lr_min > self.lr)
        {
            return Err(
                "lr-min must be finite and nonnegative, and <= lr when decay is enabled".into(),
            );
        }
        if self.replay_rounds == Some(0) {
            return Err("replay-rounds must be positive".into());
        }
        if !(self.ema_rounds.is_finite() && self.ema_rounds > 0.0) {
            return Err("ema-rounds must be positive".into());
        }
        if self.resume.is_some() && self.resume_state.is_some() {
            return Err("choose --resume (weights only) or --resume-state (full state)".into());
        }
        self.budgets.validate()
    }

    /// Saved options overlaid with the ones given on this command line.
    fn inherit(&self, saved: &Value, provided: &HashSet<String>) -> Result<SelfplayArgs, String> {
        let mut merged = saved.clone();
        let current = serde_json::to_value(self).map_err(|e| e.to_string())?;
        let (Some(map), Some(cur)) = (merged.as_object_mut(), current.as_object()) else {
            return Err("saved options are not an object".into());
        };
        for (key, value) in cur {
            let forced = matches!(key.as_str(), "steps" | "resume" | "resume_state");
            if forced || provided.contains(key) {
                map.insert(key.clone(), value.clone());
            }
        }
        serde_json::from_value(merged).map_err(|e| format!("saved options do not parse: {e}"))
    }
}

fn with_stem(out: &Path, suffix: &str) -> PathBuf {
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    out.with_file_name(format!("{stem}{suffix}"))
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_string_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

/// Run self-play training. `provided` lists the option ids given explicitly on the
/// command line (they override a resumed state's saved options).
pub fn run(mut args: SelfplayArgs, provided: &HashSet<String>) -> Result<(), String> {
    let mut first_step = 1usize;
    let mut restored_meta = None;
    if let Some(state_path) = args.resume_state.clone() {
        let meta = read_training_meta(&state_path)?;
        if !provided.contains("steps") || args.steps <= meta.step {
            return Err("--resume-state requires a target --steps beyond the saved step".into());
        }
        args = args.inherit(&meta.options, provided)?;
        args.save_trainer_state = true;
        first_step = meta.step + 1;
        restored_meta = Some((state_path, meta));
    }
    args.validate()?;
    let out = args.out.clone();
    crate::npz::ensure_parent(&out)?;

    let net = match &args.resume {
        Some(path) => {
            let net = PolicyValueNet::load(path)?;
            let saved = &net.cfg;
            let wanted = args.net_config();
            for (name, differs) in [
                ("activation", wanted.activation != saved.activation),
                (
                    "residual_scale",
                    wanted.residual_scale != saved.residual_scale,
                ),
                ("head_norm", wanted.head_norm != saved.head_norm),
                ("embedding_dim", wanted.embedding_dim != saved.embedding_dim),
                ("hidden", wanted.hidden != saved.hidden),
                ("blocks", wanted.blocks != saved.blocks),
            ] {
                if provided.contains(name) && differs {
                    return Err(format!(
                        "--resume uses the checkpoint's {name}; conflicting option supplied"
                    ));
                }
            }
            args.activation = saved.activation;
            args.residual_scale = saved.residual_scale;
            args.head_norm = saved.head_norm;
            args.embedding_dim = saved.embedding_dim;
            args.hidden = saved.hidden;
            args.blocks = saved.blocks;
            net
        }
        None => PolicyValueNet::new(args.seed, args.net_config())?,
    };
    let mut trainer = Trainer::new(net, args.hparams());
    // Restore the whole window at the saved round, before advancing the schedule.
    let mut replay = ReplayWindow::new(args.replay_capacity(first_step - 1))?;
    let mut train_rng = Pcg64Mcg::seed_from_u64(args.seed ^ 0x5452_4149);
    let mut replay_rng = Pcg64Mcg::seed_from_u64(args.seed ^ 0x5245_504C);
    let mut history: Vec<Value> = Vec::new();
    let mut stats = Stats::default();
    if let Some((state_path, _)) = &restored_meta {
        let meta = load_training_state(
            state_path,
            &mut trainer,
            &mut train_rng,
            &mut replay_rng,
            Some(&mut replay),
        )?;
        history = meta.history;
        stats = meta.stats;
        // Optimizer settings given explicitly on this command line win over the snapshot.
        if provided.contains("weight_decay") {
            trainer.hparams.wd = args.weight_decay;
        }
        if provided.contains("clip") {
            trainer.hparams.clip = args.clip;
        }
        if provided.contains("value_weight") {
            trainer.hparams.value_weight = args.value_weight;
        }
        if ["ema_rounds", "max_samples", "batch", "epochs"]
            .iter()
            .any(|k| provided.contains(*k))
        {
            trainer.hparams.ema_decay = args.hparams().ema_decay;
        }
        args.weight_decay = trainer.hparams.wd;
        args.clip = trainer.hparams.clip;
        args.value_weight = trainer.hparams.value_weight;
    } else {
        trainer.net.save(&with_stem(&out, ".initial.npz"))?;
    }
    let options = serde_json::to_value(&args).map_err(|e| e.to_string())?;
    let mut config = options.clone();
    config["search_version"] = json!(mcts::SEARCH_VERSION);
    config["split_sample_weight"] = json!(crate::selfplay::SPLIT_SAMPLE_WEIGHT);
    write_json(&with_stem(&out, ".config.json"), &config)?;
    println!(
        "net {} params, {} games x {} steps, sims {}, workers {}, ema decay {:.5} ({} updates/round)",
        trainer.net.n_params(),
        args.games,
        args.steps,
        args.sims,
        args.workers,
        trainer.hparams.ema_decay,
        args.updates_per_round()
    );

    // The games run on their own threads and only hand each ply's searches to this pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()
        .map_err(|e| e.to_string())?;
    for step in first_step..=args.steps {
        let t0 = Instant::now();
        let temperature = if step <= args.warm_steps {
            args.temperature_warm
        } else {
            args.temperature
        };
        let play = PlayOptions {
            search: SearchOptions {
                sims: args.sims,
                c_puct: args.c_puct,
                max_depth: mcts::MAX_DEPTH,
                root_noise: args.noise,
                noise_alpha: args.noise_alpha,
                root_temperature: args.root_temperature,
                lcb_stdevs: args.lcb_stdevs,
                lcb_min_visit_prop: args.lcb_min_visit_prop,
                fpu_reduction: args.fpu_reduction,
                keep_games: false,
            },
            temperature,
            end_explore: args.end_explore,
            budgets: args.budgets,
            search_prob: args.search_prob,
            split_turn: args.split_turn.clone(),
        };
        let seeds: Vec<u64> = (0..args.games as u64)
            .map(|i| args.seed + step as u64 * 10_000 + i)
            .collect();
        let net_ref = &trainer.net;
        let one_game = |seed: u64| -> Result<(Vec<Sample>, f32, usize), String> {
            let (samples, place) = play_one(net_ref, seed, &play, Some(&pool))
                .map_err(|e| format!("self-play seed={seed}: {e}"))?;
            let decisions = samples.len();
            Ok((samples, place, decisions))
        };
        // Every game walks on its own thread and only hands its searches to the pool.
        // Games as pool tasks would let a worker that waits for its ply's searches steal
        // a whole other game, and resume its own only after that one ends.
        let results: Vec<(Vec<Sample>, f32, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = seeds
                .iter()
                .map(|&seed| {
                    let one_game = &one_game;
                    scope.spawn(move || one_game(seed))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| "a self-play thread panicked".to_string())?
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let mut fresh: Vec<Sample> = Vec::new();
        let mut rollout_decisions = 0usize;
        let mut places = Vec::with_capacity(results.len());
        for (samples, place, decisions) in results {
            fresh.extend(samples);
            places.push(place);
            rollout_decisions += decisions;
        }
        let fresh_count = fresh.len();
        replay.set_capacity(args.replay_capacity(step))?;
        replay.add(fresh);
        let collect_seconds = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let (mut pl, mut vl, mut lr, mut epoch_metrics) =
            (0.0f32, 0.0f32, args.lr, Default::default());
        let mut trained = 0usize;
        for epoch in 0..args.epochs {
            lr = cosine_lr(
                args.lr,
                args.lr_min,
                (step - 1) * args.epochs + epoch,
                args.lr_decay_steps * args.epochs,
            );
            let draw = replay.draw(args.max_samples, &mut replay_rng);
            trained = draw.len();
            let (a, b, metrics) = train_epoch(
                &mut trainer,
                &draw,
                args.batch,
                lr,
                StepOptions {
                    pg_weight: args.pg_weight,
                    smoothing: args.policy_smoothing,
                },
                &mut train_rng,
            )?;
            pl = a;
            vl = b;
            epoch_metrics = metrics;
        }
        stats.policy_loss = pl;
        stats.value_loss = vl;
        stats.games += places.len() as u64;
        stats.decisions += trained as u64;
        let train_seconds = t1.elapsed().as_secs_f64();
        let batches = trained.div_ceil(args.batch) * args.epochs;
        println!(
            "step {step:3}: {trained:6} samples from {} games (gen {collect_seconds:5.1}s, train {train_seconds:5.1}s, {batches} batches)  policy {pl:.3}  value {vl:.3}  lr {lr:.6}",
            places.len()
        );

        let probe_start = Instant::now();
        if args.probe_every > 0 && step % args.probe_every == 0 {
            let summary = pool.install(|| {
                probe::behaviour_string(
                    &trainer.net,
                    &probe::ProbeOptions {
                        games: args.probe_games,
                        sims: args.sims,
                        c_puct: args.c_puct,
                        budgets: args.budgets,
                        seed: 9000,
                    },
                )
            })?;
            println!("  probe {summary}");
        }
        let mut entry = json!({
            "step": step,
            "games": places.len(),
            "samples": trained,
            "policy_loss": pl,
            "value_loss": vl,
            "lr": lr,
            "fresh_samples": fresh_count,
            "replay_rounds": replay.rounds.len(),
            "replay_capacity": replay.capacity,
            "replay_size": replay.len(),
            "collect_seconds": collect_seconds,
            "rollout_decisions": rollout_decisions,
            "train_seconds": train_seconds,
            "probe_seconds": probe_start.elapsed().as_secs_f64(),
            "mean_placement_seat0": 1.0 - 3.5 * (places.iter().sum::<f32>() / places.len().max(1) as f32 - 1.0),
        });
        for (k, v) in &epoch_metrics {
            entry[k] = json!(v);
        }
        history.push(entry);
        write_json(
            &with_stem(&out, ".history.json"),
            &Value::Array(history.clone()),
        )?;
        trainer.net.save(&out)?;
        let ema = trainer.ema_net();
        ema.save(&with_stem(&out, ".ema.npz"))?;
        if args.save_every > 0 && step % args.save_every == 0 {
            trainer
                .net
                .save(&with_stem(&out, &format!(".step{step}.npz")))?;
            ema.save(&with_stem(&out, &format!(".step{step}.ema.npz")))?;
        }
        if args.save_trainer_state {
            let state = TrainingState {
                trainer: &trainer,
                step,
                train_rng: &train_rng,
                replay_rng: &replay_rng,
                history: &history,
                stats: &stats,
                options: options.clone(),
                replay: Some(&replay),
            };
            save_training_state(&with_stem(&out, ".trainer.npz"), &state)?;
            if args.save_every > 0 && step % args.save_every == 0 {
                save_training_state(
                    &with_stem(&out, &format!(".step{step}.trainer.npz")),
                    &state,
                )?;
            }
        }
    }
    println!(
        "saved {} and {}",
        out.display(),
        with_stem(&out, ".ema.npz").display()
    );
    write_json(
        &with_stem(&out, ".stats.json"),
        &json!({
            "games": stats.games,
            "decisions": stats.decisions,
            "policy_loss": stats.policy_loss,
            "value_loss": stats.value_loss,
            "params": trainer.net.n_params(),
            "sims": args.sims,
            "steps": args.steps,
            "games_per_step": args.games,
            "lr": args.lr,
            "wd": args.weight_decay,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_schedule_uses_cumulative_rounds_and_allows_a_fixed_override() {
        let mut args = SelfplayArgs::default();
        assert_eq!(args.max_samples, 80_000);
        assert_eq!(args.lr, 3e-5);
        assert_eq!(args.updates_per_round(), 314);
        for (step, capacity) in [
            (1, 3),
            (199, 3),
            (200, 4),
            (399, 4),
            (400, 5),
            (599, 5),
            (600, 6),
            (usize::MAX, 6),
        ] {
            assert_eq!(args.replay_capacity(step), capacity);
        }
        args.replay_rounds = Some(2);
        assert_eq!(args.replay_capacity(600), 2);
        let saved = serde_json::to_value(&args).unwrap();
        let inherited = SelfplayArgs::default()
            .inherit(&saved, &HashSet::new())
            .unwrap();
        assert_eq!(inherited.replay_capacity(600), 2);
        args.replay_rounds = Some(0);
        assert!(args.validate().is_err());
    }

    #[test]
    fn constant_lr_allows_rates_below_the_unused_cosine_minimum() {
        let mut args = SelfplayArgs {
            lr: 1e-5,
            ..SelfplayArgs::default()
        };
        assert!(args.validate().is_ok());
        args.lr_decay_steps = 200;
        assert!(args.validate().is_err());
        args.lr_min = 1e-6;
        assert!(args.validate().is_ok());
    }

    #[test]
    fn resume_preserves_explicit_split_schedule_and_requires_complete_options() {
        let args = SelfplayArgs::default();
        let mut saved = serde_json::to_value(&args).unwrap();
        saved["split_turn"] = json!([]);
        assert!(
            args.inherit(&saved, &HashSet::new())
                .unwrap()
                .split_turn
                .is_empty()
        );
        assert_eq!(
            args.inherit(&saved, &HashSet::from(["split_turn".into()]))
                .unwrap()
                .split_turn,
            vec![4]
        );
        saved.as_object_mut().unwrap().remove("split_turn");
        assert!(args.inherit(&saved, &HashSet::new()).is_err());
    }
}
