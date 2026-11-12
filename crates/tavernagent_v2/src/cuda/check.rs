//! Explicit GPU validation command; never runs during ordinary CPU tests.
use super::CudaNet;
use crate::{net::PolicyValueNet, ppo, rollout, train::TrainOptions};
use clap::Args;
use rand::SeedableRng;
use serde_json::json;

#[derive(Debug, Args)]
pub struct CheckArgs {
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t = 256)]
    pub hidden: usize,
    #[arg(long, default_value_t = 6)]
    pub blocks: usize,
    #[arg(long, default_value_t = 64)]
    pub batch: usize,
    #[arg(long)]
    pub head_norm: bool,
    #[arg(long)]
    pub critic_distribution: bool,
    #[arg(long)]
    pub critic_health: bool,
    #[arg(long,default_value="gelu",value_parser=crate::net::Activation::parse)]
    pub activation: crate::net::Activation,
    #[arg(long, default_value_t = 1.0)]
    pub residual_scale: f32,
}

fn difference(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0., f64::max);
    let square = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>();
    let norm = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
    (max, (square / norm.max(1e-20)).sqrt())
}

pub fn check(args: CheckArgs) -> Result<(), String> {
    if args.batch == 0 {
        return Err("batch must be positive".into());
    }
    let options = TrainOptions {
        hidden: args.hidden,
        blocks: args.blocks,
        head_norm: args.head_norm,
        critic_distribution: args.critic_distribution,
        critic_health: args.critic_health,
        activation: args.activation,
        residual_scale: args.residual_scale,
        max_turns: 3,
        ..Default::default()
    };
    let net = PolicyValueNet::new(734, options.net_config())?;
    let mut cpu = ppo::Trainer::new(net);
    let mut gpu = CudaNet::new(&cpu, args.device)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .map_err(|e| e.to_string())?;
    let games = rollout::collect(&cpu.net, &[91, 92], &options.game_config()?, &pool)?;
    let mut samples = Vec::new();
    for game in games {
        for t in game.trajectories {
            samples.extend(t.into_samples(options.gamma, options.gae_lambda)?);
        }
    }
    rollout::normalize_advantages(&mut samples)?;
    let rows: Vec<_> = (0..args.batch)
        .map(|i| &samples[i % samples.len()])
        .collect();
    let obs: Vec<_> = rows.iter().map(|s| &s.transition.obs).collect();
    let (cpu_logits, cpu_values, _) = cpu.net.forward(
        &obs.iter().map(|o| o.x.as_slice()).collect::<Vec<_>>(),
        &obs.iter().map(|o| o.cards.as_slice()).collect::<Vec<_>>(),
        &obs.iter().map(|o| o.mask.as_slice()).collect::<Vec<_>>(),
    )?;
    let (gpu_logits, gpu_values) = gpu.infer(&cpu.net, &obs)?;
    let logits = difference(&cpu_logits.data, &gpu_logits.data);
    let values = difference(&cpu_values, &gpu_values);
    let (cpu_grad, cpu_loss) =
        pool.install(|| ppo::loss_gradients(&cpu.net, &rows, &options.ppo))?;
    let (gpu_grad, gpu_loss) = gpu.gradients(&cpu.net, &rows, &options.ppo)?;
    let downloaded = gpu.download_gradients(&cpu.net, &gpu_grad)?;
    let gradients: Vec<_> = cpu_grad
        .names()
        .iter()
        .zip(cpu_grad.tensors.iter().zip(&downloaded.tensors))
        .map(|(name, (a, b))| {
            let (abs, rel) = difference(&a.data, &b.data);
            json!({"tensor":name,"max_abs":abs,"relative_l2":rel})
        })
        .collect();
    let all_cpu: Vec<_> = cpu_grad
        .tensors
        .iter()
        .flat_map(|m| m.data.iter().copied())
        .collect();
    let all_gpu: Vec<_> = downloaded
        .tensors
        .iter()
        .flat_map(|m| m.data.iter().copied())
        .collect();
    let gradient = difference(&all_cpu, &all_gpu);
    let cpu_norm = pool.install(|| cpu.apply_gradients(&cpu_grad, options.lr, &options.ppo))?;
    let gpu_norm = gpu.apply(&cpu.net, &gpu_grad, options.lr, &options.ppo)?;
    let mut gpu_state = cpu.clone();
    gpu.download(&mut gpu_state)?;
    let cpu_weights: Vec<_> = cpu
        .net
        .p
        .tensors
        .iter()
        .flat_map(|m| m.data.iter().copied())
        .collect();
    let gpu_weights: Vec<_> = gpu_state
        .net
        .p
        .tensors
        .iter()
        .flat_map(|m| m.data.iter().copied())
        .collect();
    let update = difference(&cpu_weights, &gpu_weights);
    // Device -> checkpoint-layout state -> new device context must preserve the next update.
    let mut restored = CudaNet::new(&gpu_state, args.device)?;
    let (first, _) = gpu.gradients(&gpu_state.net, &rows, &options.ppo)?;
    gpu.apply(&gpu_state.net, &first, options.lr, &options.ppo)?;
    let (second, _) = restored.gradients(&gpu_state.net, &rows, &options.ppo)?;
    restored.apply(&gpu_state.net, &second, options.lr, &options.ppo)?;
    let mut continued = gpu_state.clone();
    let mut resumed = gpu_state.clone();
    gpu.download(&mut continued)?;
    restored.download(&mut resumed)?;
    let restore_exact = continued.net.p == resumed.net.p
        && continued.m == resumed.m
        && continued.v == resumed.v
        && continued.optimizer_step == resumed.optimizer_step;
    let mut rejected: Vec<_> = rows.iter().map(|s| (*s).clone()).collect();
    for s in &mut rejected {
        s.transition.old_log_prob -= 5.;
    }
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(17);
    let rejected_metrics = gpu.update(
        &continued.net,
        &rejected,
        2,
        args.batch,
        options.lr,
        &options.ppo,
        &mut rng,
    )?;
    let mut after_reject = continued.clone();
    gpu.download(&mut after_reject)?;
    let kl_preserves_state = rejected_metrics.early_stop
        && rejected_metrics.updates == 0
        && after_reject.net.p == continued.net.p
        && after_reject.m == continued.m
        && after_reject.v == continued.v
        && after_reject.optimizer_step == continued.optimizer_step;
    let passed = logits.0 < 2e-4
        && values.0 < 2e-4
        && gradient.1 < 2e-3
        && update.1 < 2e-5
        && restore_exact
        && kl_preserves_state;
    println!(
        "{}",
        json!({"passed":passed,"hidden":args.hidden,"blocks":args.blocks,"batch":args.batch,
        "critic_distribution":args.critic_distribution,"critic_health":args.critic_health,
        "precision":"fp32-pedantic","logits_max_abs":logits.0,"values_max_abs":values.0,"restore_exact":restore_exact,"kl_preserves_state":kl_preserves_state,
        "gradient_max_abs":gradient.0,"gradient_relative_l2":gradient.1,"update_max_abs":update.0,"update_relative_l2":update.1,
        "cpu_gradient_norm":cpu_norm,"gpu_gradient_norm":gpu_norm,"cpu_loss":cpu_loss,"gpu_loss":gpu_loss,"gradients":gradients})
    );
    if !passed {
        return Err("CPU/CUDA numerical validation failed".into());
    }
    Ok(())
}
