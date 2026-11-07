//! `tavernagent`: self-play training, evaluation and diagnostics for TavernGym.

use std::collections::HashSet;
use std::process::ExitCode;

use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, Subcommand};
use tavernagent::{arena, bench, eval, league, patterns, play, tournament, train};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Parser)]
#[command(name = "tavernagent", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// All-seat self-play training with checkpoints and exact resume.
    Selfplay(Box<train::SelfplayArgs>),
    /// Paired evaluation of checkpoints against the scripted bot.
    Eval(eval::EvalArgs),
    /// Two checkpoints against each other with seat swaps.
    Arena(arena::ArenaArgs),
    /// Up to eight checkpoints at one table, seats permuted per game.
    Tournament(tournament::TournamentArgs),
    /// Many checkpoints in shuffled tables of eight; JSON lines for rating fits.
    League(league::LeagueArgs),
    /// Turn-1 / turn-2 behaviour of a checkpoint.
    Probe(ProbeCli),
    /// Per-turn action traces and loop diagnostics.
    Patterns(patterns::PatternsArgs),
    /// Inference and decision latency.
    Bench(bench::BenchArgs),
    /// Play games against the bots and print placements.
    Play(play::PlayArgs),
}

#[derive(clap::Args)]
struct ProbeCli {
    /// Checkpoint; omit to probe a freshly initialized network.
    #[arg(long)]
    model: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 32)]
    games: usize,
    #[arg(long, default_value_t = 8)]
    sims: usize,
    #[arg(long, default_value_t = tavernagent::mcts::DEFAULT_C_PUCT)]
    c_puct: f32,
    #[arg(long, default_value_t = 9000)]
    seed: u64,
    #[arg(long, default_value_t = 8)]
    workers: usize,
    #[command(flatten)]
    budgets: tavernagent::rules::Budgets,
}

fn provided_ids(matches: &ArgMatches) -> HashSet<String> {
    matches
        .ids()
        .filter(|id| matches.value_source(id.as_str()) == Some(ValueSource::CommandLine))
        .map(|id| id.as_str().to_string())
        .collect()
}

fn run() -> Result<(), String> {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).map_err(|e| e.to_string())?;
    match cli.command {
        Command::Selfplay(args) => {
            let (_, sub) = matches.subcommand().expect("subcommand");
            train::run(*args, &provided_ids(sub))
        }
        Command::Eval(args) => eval::run(args),
        Command::Arena(args) => arena::run(args),
        Command::Tournament(args) => tournament::run(args),
        Command::League(args) => league::run(args),
        Command::Probe(p) => {
            let net = match &p.model {
                Some(path) => tavernagent::net::PolicyValueNet::load(path)?,
                None => tavernagent::net::PolicyValueNet::new(0, Default::default())?,
            };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(p.workers.max(1))
                .build()
                .map_err(|e| e.to_string())?;
            let opts = tavernagent::probe::ProbeOptions {
                games: p.games,
                sims: p.sims,
                c_puct: p.c_puct,
                budgets: p.budgets,
                seed: p.seed,
            };
            let b = pool.install(|| tavernagent::probe::behaviour(&net, &opts))?;
            println!(
                "{}  ({} games, {} sims)",
                p.model
                    .as_ref()
                    .map_or("initial net".to_string(), |m| m.display().to_string()),
                p.games,
                p.sims
            );
            for turn in [1, 2] {
                println!(
                    "  turn {turn}: {}",
                    tavernagent::probe::format_turn(&b, turn)
                );
            }
            Ok(())
        }
        Command::Patterns(args) => patterns::run(args),
        Command::Bench(args) => bench::run(args),
        Command::Play(args) => play::run(args),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
