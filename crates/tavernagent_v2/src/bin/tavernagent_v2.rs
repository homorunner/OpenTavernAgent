use clap::{Parser, Subcommand};
use std::process::ExitCode;
use tavernagent_v2::{arena, eval, train};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Parser)]
#[command(
    name = "tavernagent_v2",
    version,
    about = "FP32 PPO-Clip + GAE self-play"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a new policy from scratch, checkpointing after every complete iteration.
    Train(Box<train::TrainArgs>),
    /// Continue a v2 checkpoint with its saved hyperparameters and optimizer state.
    Resume(train::ResumeArgs),
    /// Fork a checkpoint for a policy-rule ablation, preserving optimizer and RNG.
    Fork(train::ForkArgs),
    /// Evaluate the policy directly against seven scripted or teacher bots.
    Eval(eval::EvalArgs),
    /// Evaluate four candidate copies against four frozen references with paired seats.
    Arena(arena::ArenaArgs),
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Train(args) => train::train(*args),
        Command::Resume(args) => train::resume(args),
        Command::Fork(args) => train::fork(args),
        Command::Eval(args) => eval::run(args),
        Command::Arena(args) => arena::run(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
