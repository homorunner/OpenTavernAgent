use clap::{Parser, Subcommand};
use std::process::ExitCode;
use tavernagent_v2::{arena, eval, league, train};

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
    /// Change selected training options in a new checkpoint, preserving optimizer, RNG and counters.
    Fork(train::ForkArgs),
    /// Evaluate against bots or the same checkpoint; optionally export full JSONL replays.
    Eval(eval::EvalArgs),
    /// Evaluate four candidate copies against four frozen references with paired seats.
    Arena(arena::ArenaArgs),
    /// Mix eight distinct checkpoints per table; write rankings for league Elo.
    League(league::LeagueArgs),
    /// Measure frozen critic predictions against lambda returns and terminal rewards.
    Diagnose(tavernagent_v2::critic::DiagnoseArgs),
    /// Validate CUDA forward, backward and AdamW against the CPU reference.
    #[cfg(feature = "cuda")]
    CudaCheck(tavernagent_v2::cuda::CheckArgs),
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Train(args) => train::train(*args),
        Command::Resume(args) => train::resume(args),
        Command::Fork(args) => train::fork(args),
        Command::Eval(args) => eval::run(args),
        Command::Arena(args) => arena::run(args),
        Command::League(args) => league::run(args),
        Command::Diagnose(args) => tavernagent_v2::critic::run(args),
        #[cfg(feature = "cuda")]
        Command::CudaCheck(args) => tavernagent_v2::cuda::check(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
