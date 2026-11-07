//! vpred: does the value head's T1 estimate (before any action) predict the self-play
//! outcome? Prints the correlation and the mean final placement of the seat the net rated
//! best. Usage: vpred <model.npz> [games] [sims]
use rand::Rng as _;
use rayon::prelude::*;
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{self, A_END_TURN, Evaluator, SearchOptions, observe_seat};
use tavernagent::net::PolicyValueNet;
use tavernagent::rules::Budgets;
use tavernagent::selfplay::{game_rng, sample_action};
use taverngym::Game;

fn play(net: &PolicyValueNet, seed: u64, search: &SearchOptions) -> (Vec<f32>, Vec<u8>) {
    let mut game = Game::new(seed, Budgets::default().config().unwrap());
    select_heroes(&mut game, seed).unwrap();
    let v0: Vec<f32> = (0..game.players.len())
        .map(|s| net.evaluate(&observe_seat(&game, s)).unwrap().1)
        .collect();
    let mut rng = game_rng(seed);
    while !game.is_over() {
        for seat in 0..game.players.len() {
            if !game.needs_action(seat) {
                continue;
            }
            let obs = observe_seat(&game, seat);
            let full = rng.random::<f32>() < 0.33;
            let sseed: u64 = rng.random();
            let policy = if full {
                let mut srng = rand_pcg::Pcg64Mcg::new(sseed as u128);
                mcts::search(&game, seat, net, search, &mut srng)
                    .unwrap()
                    .policy
            } else {
                net.evaluate(&obs).unwrap().0
            };
            let mut action = sample_action(&policy, 1.0, &mut rng);
            if obs.mask[A_END_TURN] && rng.random::<f32>() < 0.02 {
                action = A_END_TURN;
            }
            tavernagent::actions::apply(&mut game, seat, action).unwrap();
        }
    }
    let places = (0..game.players.len())
        .map(|s| game.players[s].placement)
        .collect();
    (v0, places)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = PolicyValueNet::load(std::path::Path::new(&args[1])).unwrap();
    let games: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let sims: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let search = SearchOptions {
        sims,
        root_noise: 0.2,
        fpu_reduction: mcts::DEFAULT_FPU_REDUCTION,
        ..SearchOptions::default()
    };
    let rows: Vec<(Vec<f32>, Vec<u8>)> = (0..games)
        .into_par_iter()
        .map(|i| play(&net, 5_000_000 + i, &search))
        .collect();
    let (mut xs, mut ys, mut top) = (Vec::new(), Vec::new(), Vec::new());
    for (v0, places) in &rows {
        for (s, p) in places.iter().enumerate() {
            xs.push(v0[s]);
            ys.push(mcts::placement_reward(*p));
        }
        let best = (0..v0.len())
            .max_by(|a, b| v0[*a].partial_cmp(&v0[*b]).unwrap())
            .unwrap();
        top.push(places[best] as f32);
    }
    let n = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let cov = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f32>()
        / n;
    let sx = (xs.iter().map(|x| (x - mx).powi(2)).sum::<f32>() / n).sqrt();
    let sy = (ys.iter().map(|y| (y - my).powi(2)).sum::<f32>() / n).sqrt();
    println!(
        "games {}  corr(V at T1 before any action, final reward) = {:.3}   std(V_T1) {:.3}   \
         mean placement of the seat rated best at T1: {:.2} (random 4.5)",
        rows.len(),
        cov / (sx * sy),
        sx,
        top.iter().sum::<f32>() / top.len() as f32
    );
}
