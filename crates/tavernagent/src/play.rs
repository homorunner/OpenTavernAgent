//! Play games with a checkpoint against the scripted bots; prints placements and an
//! action histogram.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Args;
use taverngym::Game;

use crate::heroes::select_heroes;
use crate::mcts::{self, SearchOptions};
use crate::net::PolicyValueNet;
use crate::probe::family;
use crate::rules::Budgets;
use crate::selfplay::{game_rng, sample_action};

#[derive(Clone, Debug, Args)]
pub struct PlayArgs {
    #[arg(long, default_value = "models/demo.npz")]
    pub model: PathBuf,
    #[arg(long, default_value_t = 6)]
    pub games: usize,
    #[arg(long, default_value_t = 48)]
    pub sims: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[command(flatten)]
    pub budgets: Budgets,
    #[arg(long, default_value_t = crate::mcts::DEFAULT_C_PUCT)]
    pub c_puct: f32,
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
}

pub fn run(args: PlayArgs) -> Result<(), String> {
    let net = PolicyValueNet::load(&args.model)?;
    println!("{}: {} params", args.model.display(), net.n_params());
    let search = SearchOptions {
        sims: args.sims,
        c_puct: args.c_puct,
        ..SearchOptions::default()
    };
    search.validate()?;
    let cfg = args.budgets.config()?;
    let mut places = Vec::new();
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for g in 0..args.games as u64 {
        let seed = args.seed + g;
        let mut game = Game::new(seed, cfg.clone());
        select_heroes(&mut game, seed)?;
        let mut rng = game_rng(seed);
        while !game.is_over() {
            if !game.needs_action(0) {
                let mut moved = false;
                for p in 1..game.players.len() {
                    if game.needs_action(p) {
                        let a = mcts::bot_action(&mut game, p);
                        game.apply(p, a).map_err(|e| e.to_string())?;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
                continue;
            }
            let result = mcts::search(&game, 0, &net, &search, &mut rng)?;
            let action = if args.temperature > 0.0 {
                sample_action(&result.policy, args.temperature, &mut rng)
            } else {
                result.best_action()
            };
            *hist.entry(family(action)).or_insert(0) += 1;
            crate::actions::apply(&mut game, 0, action).map_err(str::to_string)?;
        }
        let placement = game.players[0].placement;
        places.push(placement);
        println!("game {}: placed {placement}  (turn {})", g + 1, game.turn);
    }
    let mean = places.iter().map(|p| *p as f64).sum::<f64>() / places.len().max(1) as f64;
    println!("\nmean placement {mean:.2}  placements {places:?}");
    let mut top: Vec<_> = hist.into_iter().collect();
    top.sort_by_key(|a| std::cmp::Reverse(a.1));
    println!("actions: {:?}", top.into_iter().take(8).collect::<Vec<_>>());
    Ok(())
}
