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
use taverngym::actions::{A_CANCEL, A_END};
use taverngym::obs::observe;
use taverngym::{Game, bot, rng::Rng};

use crate::net::{N_ACTIONS, PolicyValueNet};

pub const SEARCH_VERSION: &str = "ended-state-v9-compact-play";
pub const MAX_PLIES: usize = 2000;
pub const MAX_DEPTH: usize = 100;
pub const A_END_TURN: usize = A_END;
/// Training rule, not a game rule: a seat holding this much Gold may not end its turn
/// while it could still spend some. It shortens the early game (every unspent Gold piece
/// is wasted anyway, since Gold does not carry over) and keeps the value head from
/// learning to stop with a full purse.
pub const MUST_SPEND_GOLD: i32 = 3;
/// Exploration constant. 1.3–1.9 were indistinguishable at 200 rounds across seeds; at
/// 1000 rounds 1.9 led 1.5 narrowly and 1.7 clearly, so 1.9 is the default.
pub const DEFAULT_C_PUCT: f32 = 1.9;
/// LCB selection is opt-in; by default play the most visited root action. Paired
/// evaluations of the e2 checkpoints favored visit argmax over LCB at 64 and 256 sims.
/// When enabled, candidates need `DEFAULT_LCB_MIN_VISIT_PROP` of the top visit count.
pub const DEFAULT_LCB_STDEVS: f32 = 0.0;
pub const DEFAULT_LCB_MIN_VISIT_PROP: f32 = 0.15;
/// First-play urgency: an unvisited edge is valued at the parent's mean value minus this
/// amount. Negative restores the original rule (unvisited edges at Q = 0), which on a
/// [-1, 1] value spreads the simulations over every untried action whenever the parent is
/// below 0. Two 1,000-round checkpoints, the same 2,048 games at 64 sims: Q = 0 scored
/// 2.702 / 4.006, a reduction of 0.4 2.638 / 3.772, 1.0 2.490 / 3.651, 1.5 2.392 / 3.544.
/// That is evaluation only. In training, 0.2 scored 2.14 / 2.67 (two seeds) at 1,000
/// rounds against 2.54 without it; training with 1 is untested.
// TODO: compute it the way KataGo does: parent value - c * sqrt(prior mass of the visited
// children), with a smaller c at the root, instead of a fixed reduction.
pub const DEFAULT_FPU_REDUCTION: f32 = 1.5;
/// Range of the value head's output; the variance prior at low visit counts.
const VALUE_RANGE_RADIUS: f32 = 1.0;

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
    /// Prior temperature at the root (1 = unchanged); applied before the Dirichlet noise.
    pub root_temperature: f32,
    /// Stdevs for the LCB used by `best_action` (0 disables LCB, keeping the visit argmax).
    pub lcb_stdevs: f32,
    pub lcb_min_visit_prop: f32,
    /// First-play urgency reduction (see `DEFAULT_FPU_REDUCTION`).
    pub fpu_reduction: f32,
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
            root_temperature: 1.0,
            lcb_stdevs: DEFAULT_LCB_STDEVS,
            lcb_min_visit_prop: DEFAULT_LCB_MIN_VISIT_PROP,
            fpu_reduction: DEFAULT_FPU_REDUCTION,
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
        if !self.root_temperature.is_finite() || self.root_temperature <= 0.0 {
            return Err("root_temperature must be positive".into());
        }
        if !self.lcb_stdevs.is_finite()
            || self.lcb_stdevs < 0.0
            || !(0.0..=1.0).contains(&self.lcb_min_visit_prop)
        {
            return Err("lcb_stdevs must be nonnegative and lcb_min_visit_prop in [0, 1]".into());
        }
        Ok(())
    }
}

/// A decision node. Children are keyed by the observable outcome of an action.
#[derive(Clone)]
pub struct Node {
    pub prior: Vec<f32>,
    pub actions: Vec<usize>,
    pub children: Vec<HashMap<u64, Node>>,
    pub n: u32,
    pub w: f32,
    pub cn: Vec<f32>,
    pub cw: Vec<f32>,
    /// Sum of squared backed-up values per edge (for the LCB's variance estimate).
    pub cw2: Vec<f32>,
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
            cw2: vec![0.0; n],
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
    /// Lower confidence bound of edge `i`'s mean value: `mean - stdevs * sqrt(var / n)`,
    /// with a small prior towards the maximum variance so that one or two visits do not
    /// produce a confident bound (KataGo's `getSelfUtilityLCBAndRadius`).
    pub fn edge_lcb(&self, i: usize, stdevs: f32) -> f32 {
        let n = self.cn[i];
        if n <= 0.0 {
            return -2.0 * VALUE_RANGE_RADIUS * stdevs;
        }
        let mean = self.cw[i] / n;
        let mut sq_mean = (self.cw2[i] / n).max(mean * mean + 1e-8);
        let prior_weight = 1.0 / (n * n);
        sq_mean = (sq_mean * n
            + (sq_mean + VALUE_RANGE_RADIUS * VALUE_RANGE_RADIUS) * prior_weight)
            / (n + prior_weight);
        let variance = (sq_mean - mean * mean).max(0.0);
        mean - stdevs * (variance / (n + prior_weight)).sqrt()
    }
}

#[derive(Clone)]
pub struct SearchResult {
    /// Root visit distribution over all actions (the prior if nothing was visited).
    pub policy: Vec<f32>,
    pub value: f32,
    pub visits: Vec<f32>,
    pub root: Node,
    /// The LCB settings the search ran with (used by `best_action`).
    pub lcb_stdevs: f32,
    pub lcb_min_visit_prop: f32,
}

impl SearchResult {
    /// The action to play: the most visited one, unless LCB selection is on and another
    /// action with at least `lcb_min_visit_prop` of the top visit count has a higher lower
    /// confidence bound on its value (then that one). Ties keep the lowest index.
    pub fn best_action(&self) -> usize {
        if !self.visits.iter().any(|v| *v > 0.0) {
            return argmax(&self.policy);
        }
        let top = argmax(&self.visits);
        if self.lcb_stdevs <= 0.0 {
            return top;
        }
        let threshold = self.lcb_min_visit_prop * self.visits[top];
        let root = &self.root;
        let mut best = top;
        let mut best_lcb = f32::NEG_INFINITY;
        for (i, &a) in root.actions.iter().enumerate() {
            if root.cn[i] <= 0.0 || root.cn[i] < threshold {
                continue;
            }
            let lcb = root.edge_lcb(i, self.lcb_stdevs);
            if lcb > best_lcb {
                best_lcb = lcb;
                best = a;
            }
        }
        best
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
    crate::actions::apply(game, seat, action).map_err(|e| e.to_string())?;
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

#[derive(Clone)]
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
        mask: agent_mask(game, seat),
    }
}

/// The seat's legal actions over the agent's action space, with two training rules applied.
///
/// Play uses seven positions per hand card and automatically magnetizes onto
/// compatible hosts. Move is absent; see `crate::actions` for engine translation.
///
/// Must-spend: End Turn is illegal for a seat holding `MUST_SPEND_GOLD` Gold or more,
/// whatever the shop currently offers. A free Refresh is exactly the case that matters —
/// nothing is affordable now, a Refresh may turn up something that is — so the rule must not
/// try to predict whether Gold *can* be spent. It only ever removes End Turn, and never
/// leaves the mask empty: past the action cap the engine returns End Turn alone, and that
/// one bit is restored.
///
/// No Cancel: a menu the agent opened (a spell's or Hero Power's target, a Choose One) must
/// be resolved by picking an option. The engine only opens a target menu when a target
/// exists, so an option is always there; Cancel is restored only when it is the sole legal
/// action (a paid trinket offer with nothing in it affordable). Without this the search treats
/// open-menu / close-menu as a free no-op pair and spends the turn's action budget on it.
pub fn agent_mask(game: &Game, seat: usize) -> Vec<bool> {
    let mut mask = crate::actions::legal_mask(game, seat);
    if mask[A_END_TURN] && game.players[seat].gold >= MUST_SPEND_GOLD {
        mask[A_END_TURN] = false;
        if !mask.iter().any(|legal| *legal) {
            mask[A_END_TURN] = true;
        }
    }
    if mask[A_CANCEL] {
        mask[A_CANCEL] = false;
        if !mask.iter().any(|legal| *legal) {
            mask[A_CANCEL] = true;
        }
    }
    mask
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

fn select(node: &Node, c_puct: f32, fpu_reduction: f32) -> usize {
    let sqrt_n = ((node.n + 1) as f32).sqrt();
    // Value of an edge that has not been tried: the parent's mean (its own network value
    // before any visit) minus the reduction, or 0 under the original rule.
    let unvisited_q = if fpu_reduction < 0.0 {
        0.0
    } else {
        let parent = if node.n == 0 { node.value } else { node.q() };
        parent - fpu_reduction
    };
    let mut best = 0;
    let mut best_u = f32::NEG_INFINITY;
    for (i, &a) in node.actions.iter().enumerate() {
        let q = if node.cn[i] > 0.0 {
            node.cw[i] / node.cn[i]
        } else {
            unvisited_q
        };
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

/// Raise the root prior to `1 / temperature` over the legal actions and renormalise.
fn apply_root_temperature(root: &mut Node, temperature: f32) {
    if temperature == 1.0 || root.actions.is_empty() {
        return;
    }
    let top = root
        .actions
        .iter()
        .map(|&a| root.prior[a])
        .fold(0.0f32, f32::max);
    if top <= 0.0 {
        return;
    }
    let mut total = 0.0f32;
    for &a in &root.actions {
        let p = root.prior[a];
        root.prior[a] = if p > 0.0 {
            ((p.ln() - top.ln()) / temperature).exp()
        } else {
            0.0
        };
        total += root.prior[a];
    }
    for &a in &root.actions {
        root.prior[a] /= total;
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
    apply_root_temperature(&mut root, opts.root_temperature);
    add_root_noise(&mut root, opts.root_noise, opts.noise_alpha, rng);

    // One working copy for the whole search. A turn-horizon simulation only ever changes
    // the searched seat, the pool and a few counters, so `rewind_turn` restores those and
    // leaves the other seats alone; a game-horizon simulation moves every seat and fights,
    // so it gets a fresh copy.
    let mut simulation = game.clone();
    for _ in 0..opts.sims {
        match opts.horizon {
            Horizon::Turn => rewind_turn(&mut simulation, game, seat, next_seed(rng)),
            Horizon::Game => simulation = reseeded(game, next_seed(rng)),
        }
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
                let index = select(node, opts.c_puct, opts.fpu_reduction);
                let action = node.actions[index];
                match opts.horizon {
                    Horizon::Game => advance(&mut simulation, seat, action)?,
                    Horizon::Turn if action == A_END_TURN => {
                        // Triggers and random end-of-turn choices run; the fight never starts.
                        simulation
                            .end_turn_effects(seat)
                            .map_err(|e| e.to_string())?;
                    }
                    Horizon::Turn => crate::actions::apply(&mut simulation, seat, action)
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
        lcb_stdevs: opts.lcb_stdevs,
        lcb_min_visit_prop: opts.lcb_min_visit_prop,
    })
}

fn backup(root: &mut Node, path: &[(usize, u64)], value: f32) {
    let mut node = root;
    for &(index, key) in path {
        node.n += 1;
        node.w += value;
        node.cn[index] += 1.0;
        node.cw[index] += value;
        node.cw2[index] += value * value;
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

/// Make `sim` equal to `reseeded(root, seed)` again after a turn-horizon simulation of
/// `seat`, restoring only what such a simulation can change. Recruit actions and
/// end-of-turn effects act on their owner alone (the cross-seat scripts fire at turn start
/// or on elimination), so the other seats, the phase, the pairings and the lobby tables in
/// `sim` are still the root's — unless the simulation killed the seat (its own power or
/// effect took its last Health): elimination rewrites the phase, the pairings, the
/// elimination order and, through Mr. Bigglesworth, other seats, so that case falls back
/// to a full copy. `sim` must have started as a copy of `root`.
pub fn rewind_turn(sim: &mut Game, root: &Game, seat: usize, seed: u64) {
    if sim.players[seat].alive != root.players[seat].alive {
        *sim = reseeded(root, seed);
        return;
    }
    sim.players[seat].clone_from(&root.players[seat]);
    sim.pool.counts.clone_from(&root.pool.counts);
    sim.rng = Rng::new(seed);
    sim.next_eid = root.next_eid;
    sim.ending_turn = root.ending_turn;
    sim.truncated_recruit_contexts = root.truncated_recruit_contexts;
    sim.log.clone_from(&root.log);
    sim.last_frames.clear();
}
