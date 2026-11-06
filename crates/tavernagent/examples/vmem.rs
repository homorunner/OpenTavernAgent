//! vmem: is the value head keyed on game identity? For a list of seeds, build the T1
//! position (seed-determined, network-independent), take V for all 8 seats, and print the
//! per-game std and mean of those 8 values. Real placement rewards of one game are a
//! permutation of {1, 5/7, ..., -1}: std 0.655, mean 0. Usage: vmem <model.npz> <seed_a> <seed_b> ...
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{Evaluator, observe_seat};
use tavernagent::net::PolicyValueNet;
use tavernagent::rules::Budgets;
use taverngym::Game;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = PolicyValueNet::load(std::path::Path::new(&args[1])).unwrap();
    let mut stds = Vec::new();
    let mut means = Vec::new();
    let mut spread = Vec::new();
    for s in &args[2..] {
        let seed: u64 = s.parse().unwrap();
        let mut g = Game::new(seed, Budgets::default().config().unwrap());
        select_heroes(&mut g, seed).unwrap();
        let v: Vec<f32> = (0..g.players.len())
            .map(|s| net.evaluate(&observe_seat(&g, s)).unwrap().1)
            .collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        stds.push(sd);
        means.push(m);
        spread.extend(v);
    }
    let n = stds.len() as f32;
    let all_m = spread.iter().sum::<f32>() / spread.len() as f32;
    let all_sd =
        (spread.iter().map(|x| (x - all_m).powi(2)).sum::<f32>() / spread.len() as f32).sqrt();
    println!(
        "games {}  within-game std of the 8 T1 values: mean {:.3}  (true rewards: 0.655)   \
         within-game mean: {:+.3} ± {:.3}  overall std {:.3}",
        stds.len(),
        stds.iter().sum::<f32>() / n,
        means.iter().sum::<f32>() / n,
        (means
            .iter()
            .map(|x| (x - means.iter().sum::<f32>() / n).powi(2))
            .sum::<f32>()
            / n)
            .sqrt(),
        all_sd
    );
}
