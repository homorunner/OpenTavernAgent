//! PUCT with sampled chance outcomes and an independent random stream per simulation.
//!
//! The default `Turn` horizon ends at the position a seat hands to combat. End Turn is
//! an ordinary transition: `Game::end_turn_effects` runs the end-of-turn triggers and
//! marks the seat as ended without simulating opponents or the fight, and the resulting
//! state is a leaf (no legal actions) evaluated by the value network like any other
//! state. `Game` is an explicit full-state experimental mode; it advances other seats
//! with the scripted bot and is not information-limited.
//!
//! Action edges aggregate visits across distinct observable chance outcomes. Exact
//! `Game::clone` remains available for replay; search always supplies a new seed.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use rand_distr::{Distribution, Gamma};
use taverngym::actions::A_END;
use taverngym::obs::observe;
use taverngym::{Game, bot, rng::Rng};

use crate::net::{N_ACTIONS, PolicyValueNet};

pub const SEARCH_VERSION: &str = "ended-state-v3";
pub const MAX_PLIES: usize = 2000;
pub const MAX_DEPTH: usize = 100;
pub const A_END_TURN: usize = A_END;
/// Exploration constant. 1.3–1.9 were indistinguishable at 200 rounds across seeds; at
/// 1000 rounds 1.9 led 1.5 narrowly and 1.7 clearly, so 1.9 is the default.
pub const DEFAULT_C_PUCT: f32 = 1.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Horizon {
    Turn,
    Game,
}

impl Horizon {
    pub fn parse(s: &str) -> Result<Horizon, String> {
        match s {
            "turn" => Ok(Horizon::Turn),
            "game" => Ok(Horizon::Game),
            other => Err(format!("unknown horizon {other:?}")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Horizon::Turn => "turn",
            Horizon::Game => "game",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub sims: usize,
    pub c_puct: f32,
    pub horizon: Horizon,
    pub max_depth: usize,
    pub root_noise: f32,
    pub noise_alpha: f32,
    /// Keep a snapshot of the position in every node (for inspection; costs a clone each).
    pub keep_games: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            sims: 64,
            c_puct: DEFAULT_C_PUCT,
            horizon: Horizon::Turn,
            max_depth: MAX_DEPTH,
            root_noise: 0.0,
            noise_alpha: 0.3,
            keep_games: false,
        }
    }
}

impl SearchOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.sims < 1 || self.max_depth < 1 {
            return Err("sims and max_depth must be positive".into());
        }
        if !self.c_puct.is_finite() || self.c_puct < 0.0 {
            return Err("c_puct must be finite and nonnegative".into());
        }
        if !(0.0..=1.0).contains(&self.root_noise)
            || !self.noise_alpha.is_finite()
            || self.noise_alpha <= 0.0
        {
            return Err("noise must be in [0, 1] and noise_alpha must be positive".into());
        }
        Ok(())
    }
}

/// A decision node. Children are keyed by the observable outcome of an action.
pub struct Node {
    pub prior: Vec<f32>,
    pub actions: Vec<usize>,
    pub children: Vec<HashMap<u64, Node>>,
    pub n: u32,
    pub w: f32,
    pub cn: Vec<f32>,
    pub cw: Vec<f32>,
    pub value: f32,
    /// Representative snapshot of the position (never used to sample a transition).
    pub game: Option<Box<Game>>,
}

impl Node {
    fn new(value: f32, prior: Vec<f32>, actions: Vec<usize>, game: Option<Box<Game>>) -> Node {
        let n = actions.len();
        Node {
            prior,
            actions,
            children: (0..n).map(|_| HashMap::new()).collect(),
            n: 0,
            w: 0.0,
            cn: vec![0.0; n],
            cw: vec![0.0; n],
            value,
            game,
        }
    }
    pub fn q(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.w / self.n as f32
        }
    }
}

pub struct SearchResult {
    /// Root visit distribution over all actions (the prior if nothing was visited).
    pub policy: Vec<f32>,
    pub value: f32,
    pub visits: Vec<f32>,
    pub root: Node,
}

impl SearchResult {
    /// Most visited action; ties keep the lowest index.
    pub fn best_action(&self) -> usize {
        let source = if self.visits.iter().any(|v| *v > 0.0) {
            &self.visits
        } else {
            &self.policy
        };
        argmax(source)
    }
}

pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

pub fn placement_reward(placement: u8) -> f32 {
    1.0 - (placement as f32 - 1.0) * (2.0 / 7.0)
}

pub fn terminal_value(game: &Game, seat: usize) -> Option<f32> {
    let place = game.players[seat].placement;
    if place > 0 {
        Some(placement_reward(place))
    } else {
        None
    }
}

pub fn is_hero_selection(game: &Game) -> bool {
    game.phase == taverngym::engine::Phase::HeroSelect
}

/// Full-game mode only: apply an action and drive the bots until `seat` can act again.
pub fn advance(game: &mut Game, seat: usize, action: usize) -> Result<(), String> {
    game.apply_index(seat, action).map_err(|e| e.to_string())?;
    for _ in 0..MAX_PLIES {
        if terminal_value(game, seat).is_some() || game.needs_action(seat) {
            return Ok(());
        }
        let mut moved = false;
        for p in 0..game.players.len() {
            if p != seat && game.needs_action(p) {
                let action = bot_action(game, p);
                game.apply(p, action).map_err(|e| e.to_string())?;
                moved = true;
            }
        }
        if !moved {
            return Ok(());
        }
    }
    Err("search bot rollout exceeded its action guard".into())
}

/// The scripted bot's choice for `p` without moving the game's random stream.
pub fn bot_action(game: &mut Game, p: usize) -> taverngym::Action {
    let saved = game.rng.clone();
    let action = bot::choose_action(game, p);
    game.rng = saved;
    action
}

pub struct Observation {
    pub x: Vec<f32>,
    pub cards: Vec<i32>,
    pub mask: Vec<bool>,
}

pub fn observe_seat(game: &Game, seat: usize) -> Observation {
    let o = observe(game, seat);
    Observation {
        x: o.floats,
        cards: o.cards,
        mask: game.legal_mask(seat),
    }
}

/// Observable outcome key: the observation and legal mask, never the hidden future RNG.
pub fn observation_key(obs: &Observation) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    let (pairs, rest) = obs.x.as_chunks::<2>();
    for [lo, hi] in pairs {
        h.write_u64((lo.to_bits() as u64) << 32 | hi.to_bits() as u64);
    }
    if let Some(v) = rest.first() {
        h.write_u32(v.to_bits());
    }
    obs.cards.hash(&mut h);
    obs.mask.hash(&mut h);
    h.finish()
}

/// Anything that turns an observation into action priors and a value.
pub trait Evaluator: Sync {
    fn evaluate(&self, obs: &Observation) -> Result<(Vec<f32>, f32), String>;
}

impl Evaluator for PolicyValueNet {
    fn evaluate(&self, obs: &Observation) -> Result<(Vec<f32>, f32), String> {
        self.predict(&obs.x, &obs.cards, &obs.mask)
    }
}

fn expand<E: Evaluator + ?Sized>(
    net: &E,
    obs: &Observation,
    game: Option<Box<Game>>,
) -> Result<Node, String> {
    let (prior, value) = net.evaluate(obs)?;
    let actions: Vec<usize> = obs
        .mask
        .iter()
        .enumerate()
        .filter(|(_, m)| **m)
        .map(|(i, _)| i)
        .collect();
    Ok(Node::new(value, prior, actions, game))
}

fn select(node: &Node, c_puct: f32) -> usize {
    let sqrt_n = ((node.n + 1) as f32).sqrt();
    let mut best = 0;
    let mut best_u = f32::NEG_INFINITY;
    for (i, &a) in node.actions.iter().enumerate() {
        let q = node.cw[i] / node.cn[i].max(1.0);
        let u = q + c_puct * node.prior[a] * sqrt_n / (1.0 + node.cn[i]);
        if u > best_u {
            best_u = u;
            best = i;
        }
    }
    best
}

fn add_root_noise(root: &mut Node, epsilon: f32, alpha: f32, rng: &mut impl rand::Rng) {
    if epsilon == 0.0 || root.actions.is_empty() {
        return;
    }
    let noise = dirichlet(alpha, root.actions.len(), rng);
    for (i, &a) in root.actions.iter().enumerate() {
        root.prior[a] = (1.0 - epsilon) * root.prior[a] + epsilon * noise[i];
    }
}

/// A symmetric Dirichlet(alpha) sample of length `n`: normalized Gamma(alpha, 1) draws.
pub fn dirichlet(alpha: f32, n: usize, rng: &mut impl rand::Rng) -> Vec<f32> {
    let gamma = Gamma::new(alpha as f64, 1.0).expect("valid Gamma");
    let mut draws: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
    let total: f64 = draws.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0 / n as f32; n];
    }
    for d in &mut draws {
        *d /= total;
    }
    draws.into_iter().map(|d| d as f32).collect()
}

pub fn simulation_seed(rng: &mut impl rand::Rng) -> u64 {
    rng.random::<u64>()
}

/// Search independent sampled trajectories without changing the real game.
///
/// `rng` drives the root noise and the seeds of the simulation copies.
pub fn search<E: Evaluator + ?Sized>(
    game: &Game,
    seat: usize,
    net: &E,
    opts: &SearchOptions,
    rng: &mut impl rand::Rng,
) -> Result<SearchResult, String> {
    search_with_seeds(game, seat, net, opts, rng, None)
}

/// `search` with the simulation seeds taken from `seeds` (root copy first) instead of `rng`;
/// used to reproduce a search exactly.
pub fn search_with_seeds<E: Evaluator + ?Sized>(
    game: &Game,
    seat: usize,
    net: &E,
    opts: &SearchOptions,
    rng: &mut impl rand::Rng,
    seeds: Option<&[u64]>,
) -> Result<SearchResult, String> {
    if game.is_over() {
        return Err("search requires an active game; terminal rewards bypass the network".into());
    }
    if is_hero_selection(game) {
        return Err("select heroes with an external policy before searching".into());
    }
    if seat >= game.players.len()
        || !game.needs_action(seat)
        || terminal_value(game, seat).is_some()
    {
        return Err("search requires a seat that can act".into());
    }
    opts.validate()?;
    let mut seed_iter = seeds.map(|s| s.iter().copied());
    let mut next_seed = |rng: &mut dyn rand::RngCore| -> u64 {
        match seed_iter.as_mut() {
            Some(it) => it.next().expect("enough simulation seeds"),
            None => rng.next_u64(),
        }
    };
    let observation = observe_seat(game, seat);
    let root_seed = next_seed(rng);
    let root_copy = opts.keep_games.then(|| Box::new(reseeded(game, root_seed)));
    let mut root = expand(net, &observation, root_copy)?;
    let root_value = root.value;
    add_root_noise(&mut root, opts.root_noise, opts.noise_alpha, rng);

    for _ in 0..opts.sims {
        let mut simulation = reseeded(game, next_seed(rng));
        // Path of (node pointer, action index); nodes are owned by the tree, so walk by
        // re-descending with indices instead of holding references.
        let mut path: Vec<(usize, u64)> = Vec::new(); // (action index, outcome key) per level
        let value;
        {
            let mut node: &mut Node = &mut root;
            let mut depth = 0;
            loop {
                if depth >= opts.max_depth {
                    value = node.value;
                    break;
                }
                depth += 1;
                if node.actions.is_empty() {
                    // An ended seat (or a full-game dead end): reuse the leaf's evaluation.
                    value = node.value;
                    break;
                }
                let index = select(node, opts.c_puct);
                let action = node.actions[index];
                match opts.horizon {
                    Horizon::Game => advance(&mut simulation, seat, action)?,
                    Horizon::Turn if action == A_END_TURN => {
                        // Triggers and random end-of-turn choices run; the fight never starts.
                        simulation
                            .end_turn_effects(seat)
                            .map_err(|e| e.to_string())?;
                    }
                    Horizon::Turn => simulation
                        .apply_index(seat, action)
                        .map_err(|e| e.to_string())?,
                }
                if let Some(v) = terminal_value(&simulation, seat) {
                    path.push((index, 0));
                    value = v;
                    break;
                }
                let child_obs = observe_seat(&simulation, seat);
                let key = observation_key(&child_obs);
                path.push((index, key));
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    node.children[index].entry(key)
                {
                    let snapshot = opts.keep_games.then(|| Box::new(simulation.clone()));
                    let child = expand(net, &child_obs, snapshot)?;
                    value = child.value;
                    slot.insert(child);
                    break;
                }
                node = node.children[index].get_mut(&key).unwrap();
            }
        }
        backup(&mut root, &path, value);
    }

    let mut visits = vec![0.0f32; N_ACTIONS];
    for (i, &a) in root.actions.iter().enumerate() {
        visits[a] = root.cn[i];
    }
    let total: f32 = visits.iter().sum();
    let policy = if total > 0.0 {
        visits.iter().map(|v| v / total).collect()
    } else {
        root.prior.clone()
    };
    Ok(SearchResult {
        policy,
        value: root_value,
        visits,
        root,
    })
}

fn backup(root: &mut Node, path: &[(usize, u64)], value: f32) {
    let mut node = root;
    for &(index, key) in path {
        node.n += 1;
        node.w += value;
        node.cn[index] += 1.0;
        node.cw[index] += value;
        match node.children[index].get_mut(&key) {
            Some(child) => node = child,
            None => break, // terminal transition: no decision node below
        }
    }
}

/// An exact copy whose future random stream is replaced.
pub fn reseeded(game: &Game, seed: u64) -> Game {
    let mut copy = game.clone();
    copy.rng = Rng::new(seed);
    copy
}
