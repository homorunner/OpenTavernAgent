//! Where a turn-horizon search spends its time: per-simulation breakdown on the bench
//! positions (clone/rewind, select, engine apply, observe+hash, net predict), plus how deep
//! the simulations go. Run: `tavernagent_profile [sims] [games]`.
use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use tavernagent::heroes::select_heroes;
use tavernagent::mcts::{
    self, A_END_TURN, Evaluator, Observation, SearchOptions, bot_action, observation_key,
    observe_seat, reseeded, rewind_turn,
};
use tavernagent::net::{N_ACTIONS, NetConfig, PolicyValueNet};
use tavernagent::rules::Budgets;
use taverngym::Game;

#[derive(Default)]
struct T {
    clone: Duration,
    rewind: Duration,
    apply: Duration,
    end: Duration,
    observe: Duration,
    hash: Duration,
    predict: Duration,
    sims: u64,
    plies: u64,
    expands: u64,
}

fn positions(seed: u64, turns: &[u32]) -> Vec<(u32, Game)> {
    let mut game = Game::new(seed, Budgets::default().config().unwrap());
    select_heroes(&mut game, seed).unwrap();
    let mut out = Vec::new();
    while !game.is_over() && out.len() < turns.len() {
        if game.needs_action(0) {
            if turns.contains(&game.turn) && !out.iter().any(|(t, _)| *t == game.turn) {
                out.push((game.turn, game.clone()));
            }
            let a = bot_action(&mut game, 0);
            game.apply(0, a).unwrap();
            continue;
        }
        let mut moved = false;
        for p in 1..game.players.len() {
            if game.needs_action(p) {
                let a = bot_action(&mut game, p);
                game.apply(p, a).unwrap();
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    out
}

/// The search loop of `mcts::search`, horizon Turn, with timers; same tree logic.
fn profiled_search(game: &Game, seat: usize, net: &PolicyValueNet, sims: usize, t: &mut T) {
    struct N {
        prior: Vec<f32>,
        actions: Vec<usize>,
        children: Vec<HashMap<u64, N>>,
        n: u32,
        cn: Vec<f32>,
        cw: Vec<f32>,
        value: f32,
    }
    fn expand(net: &PolicyValueNet, obs: &Observation, t: &mut T) -> N {
        let t0 = Instant::now();
        let (prior, value) = net.evaluate(obs).unwrap();
        t.predict += t0.elapsed();
        t.expands += 1;
        let actions: Vec<usize> = (0..N_ACTIONS).filter(|a| obs.mask[*a]).collect();
        let k = actions.len();
        N {
            prior,
            actions,
            children: (0..k).map(|_| HashMap::new()).collect(),
            n: 0,
            cn: vec![0.0; k],
            cw: vec![0.0; k],
            value,
        }
    }
    fn select(node: &N, c_puct: f32, fpu: f32) -> usize {
        let sqrt_n = ((node.n + 1) as f32).sqrt();
        let parent = if node.n == 0 {
            node.value
        } else {
            node.cw.iter().sum::<f32>() / node.n as f32
        };
        let unvisited = parent - fpu;
        let mut best = 0;
        let mut best_u = f32::NEG_INFINITY;
        for (i, &a) in node.actions.iter().enumerate() {
            let q = if node.cn[i] > 0.0 {
                node.cw[i] / node.cn[i]
            } else {
                unvisited
            };
            let u = q + c_puct * node.prior[a] * sqrt_n / (1.0 + node.cn[i]);
            if u > best_u {
                best_u = u;
                best = i;
            }
        }
        best
    }
    let opts = SearchOptions::default();
    let mut rng = Pcg64Mcg::seed_from_u64(7);
    let obs = observe_seat(game, seat);
    let mut root = expand(net, &obs, t);
    let t0 = Instant::now();
    let mut sim = game.clone();
    t.clone += t0.elapsed();
    for _ in 0..sims {
        t.sims += 1;
        let t0 = Instant::now();
        rewind_turn(&mut sim, game, seat, rand::RngCore::next_u64(&mut rng));
        t.rewind += t0.elapsed();
        let mut path: Vec<(usize, u64)> = Vec::new();
        let value;
        {
            let mut node = &mut root;
            loop {
                if node.actions.is_empty() {
                    value = node.value;
                    break;
                }
                t.plies += 1;
                let index = select(node, opts.c_puct, opts.fpu_reduction);
                let action = node.actions[index];
                let t0 = Instant::now();
                if action == A_END_TURN {
                    sim.end_turn_effects(seat).unwrap();
                    t.end += t0.elapsed();
                } else {
                    sim.apply_index(seat, action).unwrap();
                    t.apply += t0.elapsed();
                }
                if mcts::terminal_value(&sim, seat).is_some() {
                    path.push((index, 0));
                    value = 0.0;
                    break;
                }
                let t0 = Instant::now();
                let child_obs = observe_seat(&sim, seat);
                t.observe += t0.elapsed();
                let t0 = Instant::now();
                let key = observation_key(&child_obs);
                t.hash += t0.elapsed();
                path.push((index, key));
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    node.children[index].entry(key)
                {
                    let child = expand(net, &child_obs, t);
                    value = child.value;
                    slot.insert(child);
                    break;
                }
                node = node.children[index].get_mut(&key).unwrap();
            }
        }
        let mut node = &mut root;
        for &(index, key) in &path {
            node.n += 1;
            node.cn[index] += 1.0;
            node.cw[index] += value;
            match node.children[index].get_mut(&key) {
                Some(c) => node = c,
                None => break,
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sims: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let net = PolicyValueNet::new(1, NetConfig::default()).unwrap();
    for (turn, game) in positions(1, &[1, 4, 8, 12]) {
        let mut t = T::default();
        // Full-clone cost for reference.
        let t0 = Instant::now();
        for i in 0..reps {
            let _ = reseeded(&game, i as u64);
        }
        let full_clone = t0.elapsed() / reps as u32;
        let t0 = Instant::now();
        for _ in 0..reps {
            profiled_search(&game, 0, &net, sims, &mut t);
        }
        let total = t0.elapsed();
        // The real `mcts::search` on the same position, for reference.
        let opts = SearchOptions {
            sims,
            ..SearchOptions::default()
        };
        let t0 = Instant::now();
        for i in 0..reps {
            let mut rng = Pcg64Mcg::seed_from_u64(i as u64);
            mcts::search(&game, 0, &net, &opts, &mut rng).unwrap();
        }
        let real = t0.elapsed();
        let per = |d: Duration| d.as_secs_f64() * 1e3 / reps as f64;
        let timed = t.clone + t.rewind + t.apply + t.end + t.observe + t.hash + t.predict;
        println!(
            "turn {turn:>2} ({} legal): {:.2} ms/search (real mcts::search {:.2} ms)  sims {}  plies/sim {:.2}  expands/sim {:.2}  full clone {:.1} us",
            observe_seat(&game, 0).mask.iter().filter(|m| **m).count(),
            per(total),
            per(real),
            sims,
            t.plies as f64 / t.sims as f64,
            t.expands as f64 / t.sims as f64,
            full_clone.as_secs_f64() * 1e6
        );
        for (name, d) in [
            ("clone", t.clone),
            ("rewind", t.rewind),
            ("apply", t.apply),
            ("end_turn", t.end),
            ("observe", t.observe),
            ("hash", t.hash),
            ("predict", t.predict),
        ] {
            println!(
                "   {name:<9} {:7.2} ms  {:5.1}%  {:6.1} us/ply",
                per(d),
                100.0 * d.as_secs_f64() / total.as_secs_f64(),
                d.as_secs_f64() * 1e6 / t.plies as f64
            );
        }
        println!(
            "   untimed   {:7.2} ms  {:5.1}%  (select, tree, alloc)",
            per(total - timed),
            100.0 * (total - timed).as_secs_f64() / total.as_secs_f64()
        );
    }
}
