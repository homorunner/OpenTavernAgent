//! Diagnostics use frozen behavior-policy predictions, before fitting their targets.
use crate::{
    checkpoint,
    rollout::{self, Sample},
};
use clap::Args;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Default)]
struct Moments {
    n: usize,
    y: f64,
    yy: f64,
    error: f64,
    error2: f64,
}
impl Moments {
    fn add(&mut self, prediction: f32, target: f32) {
        let y = f64::from(target);
        let error = y - f64::from(prediction);
        self.n += 1;
        self.y += y;
        self.yy += y * y;
        self.error += error;
        self.error2 += error * error;
    }
    fn report(&self) -> Value {
        let n = self.n.max(1) as f64;
        let variance = (self.yy / n - (self.y / n).powi(2)).max(0.0);
        let residual_variance = (self.error2 / n - (self.error / n).powi(2)).max(0.0);
        json!({"samples":self.n,"target_mean":self.y/n,"target_variance":variance,
            "mse":self.error2/n,"bias":-self.error/n,
            "explained_variance":if variance>1e-12 {Some(1.0-residual_variance/variance)} else {None}})
    }
}

#[derive(Default)]
struct Bucket {
    lambda: Moments,
    terminal: Moments,
    outside: usize,
}
impl Bucket {
    fn add(&mut self, row: &Sample) {
        self.lambda.add(row.transition.old_value, row.return_);
        self.terminal
            .add(row.transition.old_value, row.terminal_reward);
        self.outside += usize::from(row.return_.abs() > 1.0 + 1e-6);
    }
    fn report(&self) -> Value {
        json!({"lambda_return":self.lambda.report(),"terminal_reward":self.terminal.report(),
            "target_outside_unit_interval":self.outside})
    }
}

pub fn summarize(samples: &[Sample]) -> Value {
    let mut all = Bucket::default();
    let mut turns = BTreeMap::<u32, Bucket>::new();
    for row in samples {
        all.add(row);
        turns.entry(row.transition.turn).or_default().add(row);
    }
    json!({"all":all.report(),"by_turn":turns.into_iter().map(|(turn,bucket)|
        json!({"turn":turn,"metrics":bucket.report()})).collect::<Vec<_>>()})
}

#[derive(Args)]
pub struct DiagnoseArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub games: usize,
    #[arg(long, default_value_t = 16)]
    pub workers: usize,
    #[arg(long, default_value_t = 70_000_000)]
    pub seed: u64,
    #[arg(long)]
    pub gae_lambda: Option<f32>,
    #[arg(long)]
    pub gae_within_turn_lambda: Option<f32>,
    #[arg(long)]
    pub out: PathBuf,
}

pub fn run(args: DiagnoseArgs) -> Result<(), String> {
    if args.games == 0 || args.workers == 0 {
        return Err("games and workers must be positive".into());
    }
    let (trainer, state) = checkpoint::load(&args.model)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()
        .map_err(|e| e.to_string())?;
    let seeds = (0..args.games)
        .map(|i| args.seed.checked_add(i as u64).ok_or("seed overflow"))
        .collect::<Result<Vec<_>, _>>()?;
    let games = rollout::collect_with_rules(
        &trainer.net,
        &seeds,
        &state.options.game_config()?,
        &pool,
        state.options.policy_rules,
    )?;
    let lambda = args.gae_lambda.unwrap_or(state.options.gae_lambda);
    let within = args
        .gae_within_turn_lambda
        .or(state.options.gae_within_turn_lambda);
    let mut rows = Vec::new();
    let mut decisions = BTreeMap::<u32, Vec<usize>>::new();
    for game in games {
        for trajectory in game.trajectories {
            let mut counts = BTreeMap::<u32, usize>::new();
            for step in &trajectory.steps {
                *counts.entry(step.turn).or_default() += 1;
            }
            for (turn, count) in counts {
                decisions.entry(turn).or_default().push(count);
            }
            rows.extend(trajectory.into_samples_with_trace(state.options.gamma, lambda, within)?);
        }
    }
    let report = json!({"iteration":state.iteration,"model_seed":state.options.seed,"seed":args.seed,
        "games":args.games,"gamma":state.options.gamma,"gae_lambda":lambda,"within_turn_lambda":within,
        "critic":summarize(&rows),"decisions_by_turn":decisions.into_iter().map(|(turn,counts)|
            json!({"turn":turn,"seat_games":counts.len(),"mean_decisions":counts.iter().sum::<usize>() as f64/counts.len() as f64,
                "mean_decision_trace_weight":counts.iter().map(|&n|f64::from(lambda).powf(n as f64)).sum::<f64>()/counts.len() as f64})).collect::<Vec<_>>()});
    crate::npz::ensure_parent(&args.out)?;
    std::fs::write(
        args.out,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
