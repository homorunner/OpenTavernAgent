//! A small pre-norm residual policy/value network with a shared card embedding.
//!
//! ```text
//! numeric features + turn/tier one-hots + card embeddings + legal mask
//!     -> 256 -> pre-norm FFN residual blocks
//!     -> policy: Linear(256, 116)
//!     -> value: Linear(256, 128) -> GELU -> Linear(128, 1) -> tanh
//! ```
//!
//! Forward passes return their cache instead of storing it, so one network can serve
//! many rollout threads at once. Backward is written by hand and checked numerically
//! in the crate tests.

use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use taverngym::actions as a;
use taverngym::cards::{CardKind, db};
use taverngym::obs as o;

use crate::linalg::Mat;
use crate::npz::{NpyArray, Npz};

/// Compact policy layout; resolved to engine actions using the current state.
pub const N_ACTIONS: usize = crate::actions::N_ACTIONS;
/// Agent-only flag appended to the unchanged engine observation.
pub const COMBAT_PROTECTED_FEATURE: usize = o::OBS_FLOATS;
pub const N_FEATURES: usize = o::OBS_FLOATS + 1;
pub const N_CARD_SLOTS: usize = o::OBS_CARDS;
pub const DEFAULT_HIDDEN: usize = 256;
pub const DEFAULT_EMBEDDING_DIM: usize = 32;
pub const DEFAULT_BLOCKS: usize = 6;
pub const MODEL_VERSION: i64 = 1;
pub const TURN_BINS: usize = 16;
pub const TIER_BINS: usize = 6;
pub const ONE_HOT_FEATURES: usize = TURN_BINS + TIER_BINS;
pub const VALUE_HIDDEN: usize = 128;
pub const N_OPP_BOARD_SLOTS: usize = o::N_OPP * a::N_BOARD;
pub const OPP_BOARD_FEATURE_START: usize = o::OBS_FLOATS - N_OPP_BOARD_SLOTS * o::MINION_F;
/// The ended-turn flag: the seat has finished recruiting and has no legal action.
pub const ENDED_TURN_FEATURE: usize = o::ENDED_TURN_F;
/// Global flags the network never needs: hero selection (1), game over (2) and the
/// Tavern-wide freeze flag (15, encoded per shop item instead).
const EXCLUDED_GLOBALS: [usize; 3] = [1, 2, 15];

/// Retain raw turn/tier scalars alongside their one-hot encodings and the safety flag.
/// Historical opponent warbands, extra quests and three redundant global flags are omitted.
pub fn input_feature_indices() -> Vec<usize> {
    (0..OPP_BOARD_FEATURE_START)
        .filter(|i| !EXCLUDED_GLOBALS.contains(i))
        .filter(|i| !(o::GLOBAL_F + o::QUEST_F..o::GLOBAL_F + o::N_QUESTS * o::QUEST_F).contains(i))
        .chain(std::iter::once(COMBAT_PROTECTED_FEATURE))
        .collect()
}

/// Raw observation offsets; only two ordinary owned trinkets enter the network.
pub const TRINKET_SLOT_START: usize = N_CARD_SLOTS - N_OPP_BOARD_SLOTS - o::N_OPP - 8;
pub const OWNED_TRINKET_SLOTS: usize = 2;

/// Keep the first quest, its target reward and the owned-reward slot (three slots).
/// Also omit Deity, secrets, surplus trinket slots and historical warbands.
pub fn card_slot_indices() -> Vec<usize> {
    let quest_start = 1 + a::N_POWERS + 2;
    let secret_start = quest_start + 3 * o::N_QUESTS;
    (0..N_CARD_SLOTS)
        .filter(|&i| {
            i != 3
                && !(quest_start + 3..secret_start).contains(&i)
                && !(secret_start..secret_start + o::N_SECRETS).contains(&i)
                && !(TRINKET_SLOT_START + OWNED_TRINKET_SLOTS..TRINKET_SLOT_START + 8).contains(&i)
                && i < N_CARD_SLOTS - N_OPP_BOARD_SLOTS
        })
        .collect()
}

/// Card indices (CardId + 1; 0 = empty) whose kinds are embedded; others map to empty.
pub fn allowed_cards() -> Vec<bool> {
    let d = db();
    let mut allowed = vec![false; d.len() + 1];
    for c in d.iter() {
        allowed[c.cid.0 as usize + 1] = matches!(
            c.kind,
            CardKind::Minion
                | CardKind::Spell
                | CardKind::TavernSpell
                | CardKind::Hero
                | CardKind::HeroPower
                | CardKind::TrinketLesser
                | CardKind::TrinketGreater
                | CardKind::Quest
                | CardKind::QuestReward
        );
    }
    allowed
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Gelu,
    Silu,
    Relu,
}

impl Activation {
    pub fn parse(s: &str) -> Result<Activation, String> {
        match s {
            "gelu" => Ok(Activation::Gelu),
            "silu" => Ok(Activation::Silu),
            "relu" => Ok(Activation::Relu),
            other => Err(format!("unknown activation {other:?}")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Activation::Gelu => "gelu",
            Activation::Silu => "silu",
            Activation::Relu => "relu",
        }
    }
    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Gelu => gelu(x),
            Activation::Silu => x * sigmoid(x),
            Activation::Relu => x.max(0.0),
        }
    }
    #[inline]
    fn grad(self, x: f32) -> f32 {
        match self {
            Activation::Gelu => gelu_grad(x),
            Activation::Silu => {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            }
            Activation::Relu => (x > 0.0) as u8 as f32,
        }
    }
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)
const GELU_K: f32 = 0.044_715;

#[inline]
fn gelu(x: f32) -> f32 {
    let inner = GELU_C * (x + GELU_K * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
fn gelu_grad(x: f32) -> f32 {
    let inner = GELU_C * (x + GELU_K * x * x * x);
    let t = inner.tanh();
    let d_inner = GELU_C * (1.0 + 3.0 * GELU_K * x * x);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + (0.5 * x).tanh())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetConfig {
    /// Width of the trunk.
    pub hidden: usize,
    /// Number of residual blocks.
    pub blocks: usize,
    pub embedding_dim: usize,
    pub activation: Activation,
    pub residual_scale: f32,
    pub head_norm: bool,
    #[serde(default)]
    pub critic_distribution: bool,
    #[serde(default)]
    pub critic_health: bool,
}

impl Default for NetConfig {
    fn default() -> Self {
        NetConfig {
            hidden: DEFAULT_HIDDEN,
            blocks: DEFAULT_BLOCKS,
            embedding_dim: DEFAULT_EMBEDDING_DIM,
            activation: Activation::Gelu,
            residual_scale: 1.0,
            head_norm: false,
            critic_distribution: false,
            critic_health: false,
        }
    }
}

impl NetConfig {
    pub fn value_outputs(&self) -> usize {
        (if self.critic_distribution { 8 } else { 1 }) + usize::from(self.critic_health)
    }
    /// Fixed 3/2 expansion; rounded up for odd widths used in small tests.
    pub fn ff_hidden(&self) -> usize {
        self.hidden + self.hidden.div_ceil(2)
    }
}

/// Named tensors in a fixed order: E, W0, b0, g0, k0, (W_up, b_up, W_down, b_down, g, k) per block, Wp, bp,
/// Wv, bv.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Params {
    pub blocks: usize,
    pub tensors: Vec<Mat>,
}

pub const E: usize = 0;
pub const W0: usize = 1;
pub const B0: usize = 2;
pub const G0: usize = 3;
pub const K0: usize = 4;
const fn block(i: usize) -> usize {
    5 + 6 * i
}
pub const fn w_up(i: usize) -> usize {
    block(i)
}
pub const fn b_up(i: usize) -> usize {
    block(i) + 1
}
pub const fn w_down(i: usize) -> usize {
    block(i) + 2
}
pub const fn b_down(i: usize) -> usize {
    block(i) + 3
}
pub const fn g(i: usize) -> usize {
    block(i) + 4
}
pub const fn k(i: usize) -> usize {
    block(i) + 5
}

pub fn tensor_names(blocks: usize) -> Vec<String> {
    let mut names = vec![
        "E".into(),
        "W0".into(),
        "b0".into(),
        "g0".into(),
        "k0".into(),
    ];
    for i in 1..=blocks {
        names.extend([
            format!("W{i}_up"),
            format!("b{i}_up"),
            format!("W{i}_down"),
            format!("b{i}_down"),
            format!("g{i}"),
            format!("k{i}"),
        ]);
    }
    names.extend([
        "Wp".into(),
        "bp".into(),
        "Wv_hidden".into(),
        "bv_hidden".into(),
        "Wv".into(),
        "bv".into(),
    ]);
    names
}

impl Params {
    /// Policy projection and the two value-head layers.
    pub fn wp(&self) -> usize {
        block(self.blocks)
    }
    pub fn bp(&self) -> usize {
        block(self.blocks) + 1
    }
    pub fn wv_hidden(&self) -> usize {
        block(self.blocks) + 2
    }
    pub fn bv_hidden(&self) -> usize {
        block(self.blocks) + 3
    }
    pub fn wv(&self) -> usize {
        block(self.blocks) + 4
    }
    pub fn bv(&self) -> usize {
        block(self.blocks) + 5
    }
    pub fn names(&self) -> Vec<String> {
        tensor_names(self.blocks)
    }
    pub fn zeros_like(&self) -> Params {
        Params {
            blocks: self.blocks,
            tensors: self.tensors.iter().map(Mat::zeros_like).collect(),
        }
    }
    pub fn n_params(&self) -> usize {
        self.tensors.iter().map(Mat::len).sum()
    }
    pub fn sum_squares(&self) -> f64 {
        self.tensors.iter().map(Mat::sum_squares).sum()
    }
    pub fn all_finite(&self) -> bool {
        self.tensors.iter().all(Mat::all_finite)
    }
    /// Weight matrices (names starting with W) receive weight decay; vectors and E do not.
    pub fn is_weight(&self, index: usize) -> bool {
        index == W0
            || index == self.wp()
            || index == self.wv_hidden()
            || index == self.wv()
            || (0..self.blocks).any(|i| index == w_up(i) || index == w_down(i))
    }
}

/// `x * w`, taking the sparse single-row path for single-observation evaluations.
fn linear(x: &Mat, w: &Mat) -> Mat {
    if x.rows == 1 {
        Mat::from_vec(1, w.cols, w.vecmat(x.row(0)))
    } else {
        x.matmul(w)
    }
}

struct LnCache {
    xhat: Mat,
    inv: Vec<f32>,
}

#[inline]
fn add_bias(x: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(x.len(), bias.len());
    for (v, b) in x.iter_mut().zip(bias) {
        *v += b;
    }
}

/// `layer_norm` of one row in place, without the backward cache; identical arithmetic.
fn layer_norm_row(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let eps = 1e-5f32;
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let s = 1.0 / (var + eps).sqrt();
    for j in 0..x.len() {
        x[j] = (x[j] - mean) * s * gamma[j] + beta[j];
    }
}

fn layer_norm(x: &Mat, gamma: &[f32], beta: &[f32]) -> (Mat, LnCache) {
    let eps = 1e-5f32;
    let mut out = x.zeros_like();
    let mut xhat = x.zeros_like();
    let mut inv = vec![0.0f32; x.rows];
    let n = x.cols as f32;
    for (r, inv_r) in inv.iter_mut().enumerate() {
        let row = x.row(r);
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let s = 1.0 / (var + eps).sqrt();
        *inv_r = s;
        let xh = xhat.row_mut(r);
        let o = out.row_mut(r);
        for j in 0..x.cols {
            xh[j] = (row[j] - mean) * s;
            o[j] = xh[j] * gamma[j] + beta[j];
        }
    }
    (out, LnCache { xhat, inv })
}

fn layer_norm_backward(grad: &Mat, cache: &LnCache, gamma: &[f32]) -> (Mat, Vec<f32>, Vec<f32>) {
    let cols = grad.cols;
    let mut dgamma = vec![0.0f32; cols];
    let mut dbeta = vec![0.0f32; cols];
    let mut dx = grad.zeros_like();
    let n = cols as f32;
    for r in 0..grad.rows {
        let gr = grad.row(r);
        let xh = cache.xhat.row(r);
        let mut dxhat = vec![0.0f32; cols];
        let mut mean_dxhat = 0.0f32;
        let mut mean_dxhat_xhat = 0.0f32;
        for j in 0..cols {
            dgamma[j] += gr[j] * xh[j];
            dbeta[j] += gr[j];
            dxhat[j] = gr[j] * gamma[j];
            mean_dxhat += dxhat[j];
            mean_dxhat_xhat += dxhat[j] * xh[j];
        }
        mean_dxhat /= n;
        mean_dxhat_xhat /= n;
        let out = dx.row_mut(r);
        for j in 0..cols {
            out[j] = cache.inv[r] * (dxhat[j] - mean_dxhat - xh[j] * mean_dxhat_xhat);
        }
    }
    (dx, dgamma, dbeta)
}

struct StemCache {
    n: Mat,
    ln: LnCache,
}

struct BlockCache {
    normed: Mat,
    pre_activation: Mat,
    activated: Mat,
    ln: LnCache,
}

/// Everything backward needs from a forward pass.
pub struct Cache {
    x_in: Mat,
    cards: Vec<i32>, // batch x used slots, after the allowed-kind filter
    stem: StemCache,
    blocks: Vec<BlockCache>,
    h_final: Mat,
    value_pre: Mat,
    value_hidden: Mat,
    head_ln: Option<LnCache>,
    value: Vec<f32>,
    pub(crate) value_logits: Mat,
}

#[derive(Clone, Debug)]
pub struct PolicyValueNet {
    pub cfg: NetConfig,
    pub p: Params,
    input_features: Vec<usize>,
    card_slots: Vec<usize>,
    allowed: Vec<bool>,
    pub card_fingerprint: u64,
}

impl PolicyValueNet {
    pub fn new(seed: u64, cfg: NetConfig) -> Result<PolicyValueNet, String> {
        if cfg.embedding_dim < 1 || cfg.hidden < 1 || cfg.blocks < 1 {
            return Err("embedding_dim, hidden and blocks must be positive".into());
        }
        if !(cfg.residual_scale > 0.0 && cfg.residual_scale <= 1.0) {
            return Err("residual_scale must be in (0, 1]".into());
        }
        if o::OBS_FLOATS != 3235 || N_CARD_SLOTS != 124 {
            return Err("observation schema changed; update the network".into());
        }
        let input_features = input_feature_indices();
        let card_slots = card_slot_indices();
        let n_cards = db().len() + 1;
        let d = cfg.embedding_dim;
        let h = cfg.hidden;
        let m = cfg.ff_hidden();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut randn = |rows: usize, cols: usize, scale: f32| {
            let data = (0..rows * cols)
                .map(|_| {
                    let z: f32 = StandardNormal.sample(&mut rng);
                    z * scale
                })
                .collect();
            Mat::from_vec(rows, cols, data)
        };
        let he = |fan_in: usize| (2.0 / fan_in as f32).sqrt();
        let in_dim = ONE_HOT_FEATURES + input_features.len() + card_slots.len() * d + N_ACTIONS;
        let mut tensors = Vec::with_capacity(11 + 6 * cfg.blocks);
        let mut e = randn(n_cards, d, 0.1);
        for v in e.row_mut(0) {
            *v = 0.0;
        }
        tensors.push(e);
        tensors.push(randn(in_dim, h, he(in_dim)));
        tensors.push(Mat::zeros(1, h));
        tensors.push(Mat::filled(1, h, 1.0));
        tensors.push(Mat::zeros(1, h));
        for _ in 0..cfg.blocks {
            tensors.push(randn(h, m, he(h)));
            tensors.push(Mat::zeros(1, m));
            // Keep the initial residual variance bounded as depth increases.
            tensors.push(randn(m, h, he(m) / (2.0 * cfg.blocks as f32).sqrt()));
            tensors.push(Mat::zeros(1, h));
            tensors.push(Mat::filled(1, h, 1.0));
            tensors.push(Mat::zeros(1, h));
        }
        tensors.push(randn(h, N_ACTIONS, he(h) * 0.1));
        tensors.push(Mat::zeros(1, N_ACTIONS));
        tensors.push(randn(h, VALUE_HIDDEN, he(h)));
        tensors.push(Mat::zeros(1, VALUE_HIDDEN));
        tensors.push(randn(
            VALUE_HIDDEN,
            cfg.value_outputs(),
            he(VALUE_HIDDEN) * 0.1,
        ));
        tensors.push(Mat::zeros(1, cfg.value_outputs()));
        Ok(PolicyValueNet {
            p: Params {
                blocks: cfg.blocks,
                tensors,
            },
            cfg,
            input_features,
            card_slots,
            allowed: allowed_cards(),
            card_fingerprint: db().fingerprint,
        })
    }

    pub fn n_params(&self) -> usize {
        self.p.n_params()
    }

    pub fn n_input_features(&self) -> usize {
        ONE_HOT_FEATURES + self.input_features.len()
    }

    pub fn n_card_slots(&self) -> usize {
        self.card_slots.len()
    }

    /// Same architecture and configuration, with the given tensors.
    pub fn with_params(&self, p: Params) -> PolicyValueNet {
        PolicyValueNet {
            cfg: self.cfg.clone(),
            p,
            input_features: self.input_features.clone(),
            card_slots: self.card_slots.clone(),
            allowed: self.allowed.clone(),
            card_fingerprint: self.card_fingerprint,
        }
    }

    // ---- forward -----------------------------------------------------------

    fn write_input(
        &self,
        x: &[f32],
        cards: &[i32],
        mask: &[bool],
        row: &mut [f32],
        mut used: Option<&mut Vec<i32>>,
    ) -> Result<(), String> {
        if x.len() != N_FEATURES || cards.len() != N_CARD_SLOTS || mask.len() != N_ACTIONS {
            return Err("observation does not match the schema".into());
        }
        let d = self.cfg.embedding_dim;
        let n = self.n_input_features();
        let e = &self.p.tensors[E];
        // The engine keeps turn / 20 in x[0]. Round before converting to avoid
        // float truncation at integer boundaries; pre-game states use the T1 bin.
        let turn = (x[0] * 20.0).round().clamp(1.0, TURN_BINS as f32) as usize;
        row[..ONE_HOT_FEATURES].fill(0.0);
        row[turn - 1] = 1.0;
        // Tier 7 shares the final one-hot bin with tier 6; tier / 7 still distinguishes it.
        let tier = (x[10] * 7.0).round().clamp(1.0, TIER_BINS as f32) as usize;
        row[TURN_BINS + tier - 1] = 1.0;
        for (dst, &i) in row[ONE_HOT_FEATURES..n]
            .iter_mut()
            .zip(&self.input_features)
        {
            *dst = x[i];
        }
        for (s, &slot) in self.card_slots.iter().enumerate() {
            let idx = cards[slot];
            if idx < 0 || idx as usize >= e.rows {
                return Err("card index outside the current card table".into());
            }
            let idx = if self.allowed[idx as usize] { idx } else { 0 };
            if let Some(v) = &mut used {
                v.push(idx);
            }
            let dst = &mut row[n + s * d..n + (s + 1) * d];
            if idx == 0 {
                dst.fill(0.0);
            } else {
                dst.copy_from_slice(e.row(idx as usize));
            }
        }
        for (dst, &legal) in row[n + self.card_slots.len() * d..].iter_mut().zip(mask) {
            *dst = u8::from(legal) as f32;
        }
        Ok(())
    }

    /// Gather the network input for a batch of raw observations.
    pub(crate) fn assemble(
        &self,
        xs: &[&[f32]],
        cards: &[&[i32]],
        masks: &[&[bool]],
    ) -> Result<(Mat, Vec<i32>), String> {
        let bsz = xs.len();
        if cards.len() != bsz || masks.len() != bsz {
            return Err("feature, card and mask batches differ in length".into());
        }
        let mut x_in = Mat::zeros(bsz, self.p.tensors[W0].rows);
        let mut used = Vec::with_capacity(bsz * self.card_slots.len());
        for r in 0..bsz {
            self.write_input(xs[r], cards[r], masks[r], x_in.row_mut(r), Some(&mut used))?;
        }
        Ok((x_in, used))
    }

    /// Logits (batch x actions), values (batch) and the cache for `backward`.
    pub fn forward(
        &self,
        xs: &[&[f32]],
        cards: &[&[i32]],
        masks: &[&[bool]],
    ) -> Result<(Mat, Vec<f32>, Cache), String> {
        let (x_in, used) = self.assemble(xs, cards, masks)?;
        let p = &self.p.tensors;
        let act = self.cfg.activation;
        let mut z0 = linear(&x_in, &p[W0]);
        z0.add_row(&p[B0].data);
        let (n0, ln0) = layer_norm(&z0, &p[G0].data, &p[K0].data);
        let mut h = n0.clone();
        for v in &mut h.data {
            *v = act.apply(*v);
        }
        let stem = StemCache { n: n0, ln: ln0 };
        let mut blocks = Vec::with_capacity(self.cfg.blocks);
        for i in 0..self.cfg.blocks {
            let (normed, ln) = layer_norm(&h, &p[g(i)].data, &p[k(i)].data);
            let mut pre_activation = linear(&normed, &p[w_up(i)]);
            pre_activation.add_row(&p[b_up(i)].data);
            let mut activated = pre_activation.clone();
            for v in &mut activated.data {
                *v = act.apply(*v);
            }
            let mut residual = linear(&activated, &p[w_down(i)]);
            residual.add_row(&p[b_down(i)].data);
            for (hv, rv) in h.data.iter_mut().zip(&residual.data) {
                *hv += self.cfg.residual_scale * rv;
            }
            blocks.push(BlockCache {
                normed,
                pre_activation,
                activated,
                ln,
            });
        }
        let mut head_ln = None;
        if self.cfg.head_norm {
            let ones = vec![1.0f32; self.cfg.hidden];
            let zeros = vec![0.0f32; self.cfg.hidden];
            let (normed, cache) = layer_norm(&h, &ones, &zeros);
            h = normed;
            head_ln = Some(cache);
        }
        let (wp, bp, wv, bv) = (self.p.wp(), self.p.bp(), self.p.wv(), self.p.bv());
        let mut logits = linear(&h, &p[wp]);
        logits.add_row(&p[bp].data);
        let mut value_pre = linear(&h, &p[self.p.wv_hidden()]);
        value_pre.add_row(&p[self.p.bv_hidden()].data);
        let mut value_hidden = value_pre.clone();
        for v in &mut value_hidden.data {
            *v = gelu(*v);
        }
        let mut value_mat = linear(&value_hidden, &p[wv]);
        value_mat.add_row(&p[bv].data);
        let value: Vec<f32> = (0..value_mat.rows)
            .map(|r| critic_prediction(value_mat.row(r), &self.cfg))
            .collect();
        Ok((
            logits,
            value.clone(),
            Cache {
                x_in,
                cards: used,
                stem,
                blocks,
                h_final: h,
                value_pre,
                value_hidden,
                head_ln,
                value,
                value_logits: value_mat,
            },
        ))
    }

    /// FP32 logits and value, using the same operations as the training forward pass.
    pub fn infer_f32(
        &self,
        x: &[f32],
        cards: &[i32],
        mask: &[bool],
    ) -> Result<(Vec<f32>, f32), String> {
        let (x_in, _) = self.assemble(&[x], &[cards], &[mask])?;
        let p = &self.p.tensors;
        let act = self.cfg.activation;
        let mut h = p[W0].vecmat(x_in.row(0));
        add_bias(&mut h, &p[B0].data);
        layer_norm_row(&mut h, &p[G0].data, &p[K0].data);
        for v in &mut h {
            *v = act.apply(*v);
        }
        for i in 0..self.cfg.blocks {
            let mut n = h.clone();
            layer_norm_row(&mut n, &p[g(i)].data, &p[k(i)].data);
            let mut up = p[w_up(i)].vecmat(&n);
            add_bias(&mut up, &p[b_up(i)].data);
            for v in &mut up {
                *v = act.apply(*v);
            }
            let mut down = p[w_down(i)].vecmat(&up);
            add_bias(&mut down, &p[b_down(i)].data);
            for (hv, rv) in h.iter_mut().zip(&down) {
                *hv += self.cfg.residual_scale * rv;
            }
        }
        if self.cfg.head_norm {
            let ones = vec![1.0f32; self.cfg.hidden];
            let zeros = vec![0.0f32; self.cfg.hidden];
            layer_norm_row(&mut h, &ones, &zeros);
        }
        let mut logits = p[self.p.wp()].vecmat(&h);
        add_bias(&mut logits, &p[self.p.bp()].data);
        let mut vh = p[self.p.wv_hidden()].vecmat(&h);
        add_bias(&mut vh, &p[self.p.bv_hidden()].data);
        for v in &mut vh {
            *v = gelu(*v);
        }
        let mut value_logits = p[self.p.wv()].vecmat(&vh);
        add_bias(&mut value_logits, &p[self.p.bv()].data);
        let value = critic_prediction(&value_logits, &self.cfg);
        Ok((logits, value))
    }

    // ---- backward ----------------------------------------------------------

    /// Parameter gradients for policy logits and flattened raw critic outputs.
    /// The caller includes tanh/softmax and auxiliary-loss derivatives.
    pub fn backward(&self, cache: &Cache, dlogits: &Mat, dvalue: &[f32]) -> Params {
        let p = &self.p.tensors;
        let act = self.cfg.activation;
        let bsz = cache.h_final.rows;
        let dvalue = Mat::from_vec(bsz, self.cfg.value_outputs(), dvalue.to_vec());
        let hidden = self.cfg.hidden;
        let (wp, bp, wv, bv) = (self.p.wp(), self.p.bp(), self.p.wv(), self.p.bv());
        let mut grads = self.p.zeros_like();
        grads.tensors[wp] = cache.h_final.t_matmul(dlogits);
        grads.tensors[bp] = Mat::from_vec(1, N_ACTIONS, dlogits.sum_rows());
        grads.tensors[wv] = cache.value_hidden.t_matmul(&dvalue);
        grads.tensors[bv] = Mat::from_vec(1, self.cfg.value_outputs(), dvalue.sum_rows());
        let mut dh = dlogits.matmul_t(&p[wp]);
        let mut dvh = dvalue.matmul_t(&p[wv]);
        for (d, v) in dvh.data.iter_mut().zip(&cache.value_pre.data) {
            *d *= gelu_grad(*v);
        }
        grads.tensors[self.p.wv_hidden()] = cache.h_final.t_matmul(&dvh);
        grads.tensors[self.p.bv_hidden()] = Mat::from_vec(1, VALUE_HIDDEN, dvh.sum_rows());
        dh.add_assign(&dvh.matmul_t(&p[self.p.wv_hidden()]));
        if let Some(head) = &cache.head_ln {
            let ones = vec![1.0f32; hidden];
            dh = layer_norm_backward(&dh, head, &ones).0;
        }
        for i in (0..self.cfg.blocks).rev() {
            let layer = &cache.blocks[i];
            let mut dr = dh.clone();
            dr.scale(self.cfg.residual_scale);
            grads.tensors[w_down(i)] = layer.activated.t_matmul(&dr);
            grads.tensors[b_down(i)] = Mat::from_vec(1, hidden, dr.sum_rows());
            let mut du = dr.matmul_t(&p[w_down(i)]);
            for (d, v) in du.data.iter_mut().zip(&layer.pre_activation.data) {
                *d *= act.grad(*v);
            }
            grads.tensors[w_up(i)] = layer.normed.t_matmul(&du);
            grads.tensors[b_up(i)] = Mat::from_vec(1, self.cfg.ff_hidden(), du.sum_rows());
            let dn = du.matmul_t(&p[w_up(i)]);
            let (dx, dg, dk) = layer_norm_backward(&dn, &layer.ln, &p[g(i)].data);
            grads.tensors[g(i)] = Mat::from_vec(1, hidden, dg);
            grads.tensors[k(i)] = Mat::from_vec(1, hidden, dk);
            dh.add_assign(&dx);
        }
        let first = &cache.stem;
        let mut dn0 = dh;
        for (d, nv) in dn0.data.iter_mut().zip(&first.n.data) {
            *d *= act.grad(*nv);
        }
        let (dz0, dg0, dk0) = layer_norm_backward(&dn0, &first.ln, &p[G0].data);
        grads.tensors[G0] = Mat::from_vec(1, hidden, dg0);
        grads.tensors[K0] = Mat::from_vec(1, hidden, dk0);
        grads.tensors[W0] = cache.x_in.t_matmul(&dz0);
        grads.tensors[B0] = Mat::from_vec(1, hidden, dz0.sum_rows());
        // Embedding gradient: the input gradient's embedding columns, scatter-added per card.
        let dx_in = dz0.matmul_t(&p[W0]);
        let n_in = self.n_input_features();
        let d = self.cfg.embedding_dim;
        let n_slots = self.card_slots.len();
        let ge = &mut grads.tensors[E];
        for r in 0..bsz {
            let row = dx_in.row(r);
            for s in 0..n_slots {
                let idx = cache.cards[r * n_slots + s];
                if idx == 0 {
                    continue;
                }
                let target = ge.row_mut(idx as usize);
                for (t, v) in target
                    .iter_mut()
                    .zip(&row[n_in + s * d..n_in + (s + 1) * d])
                {
                    *t += v;
                }
            }
        }
        for v in ge.row_mut(0) {
            *v = 0.0;
        }
        grads
    }

    /// The values a forward pass produced.
    pub fn cached_values(cache: &Cache) -> &[f32] {
        &cache.value
    }

    // ---- persistence -------------------------------------------------------

    pub fn to_npz(&self) -> Npz {
        let mut npz = Npz::new();
        for (name, t) in self.p.names().iter().zip(&self.p.tensors) {
            let shape = if t.rows == 1 && !name.starts_with('W') && name != "E" {
                vec![t.cols]
            } else {
                vec![t.rows, t.cols]
            };
            npz.insert(name.clone(), NpyArray::f32(shape, t.data.clone()));
        }
        let meta = CheckpointMeta {
            version: MODEL_VERSION,
            kind: "tavernagent_v2".into(),
            features: N_FEATURES,
            input_features: self.n_input_features(),
            card_fingerprint: self.card_fingerprint.to_string(),
            cfg: self.cfg.clone(),
        };
        npz.insert(
            "meta",
            NpyArray::text(&serde_json::to_string(&meta).unwrap()),
        );
        npz
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.to_npz().save(path)
    }

    pub fn from_npz(npz: &Npz) -> Result<PolicyValueNet, String> {
        let meta: CheckpointMeta = serde_json::from_str(&npz.get("meta")?.as_text()?)
            .map_err(|e| format!("unsupported checkpoint format: {e}"))?;
        if meta.version != MODEL_VERSION || meta.kind != "tavernagent_v2" {
            return Err(format!(
                "unsupported checkpoint format v{}; expected model format v{MODEL_VERSION}",
                meta.version
            ));
        }
        let net = PolicyValueNet::new(0, meta.cfg.clone())?;
        if meta.features != N_FEATURES
            || meta.input_features != net.n_input_features()
            || meta.card_fingerprint != net.card_fingerprint.to_string()
        {
            return Err(
                "checkpoint does not match the current observation/card/network schema".into(),
            );
        }
        let mut p = net.p.clone();
        let names = p.names();
        for (name, t) in names.iter().zip(p.tensors.iter_mut()) {
            let array = npz.get(name)?;
            let data = array.as_f32()?;
            if data.len() != t.len() {
                return Err(format!("checkpoint parameter shape mismatch: {name}"));
            }
            t.data.copy_from_slice(data);
        }
        for v in p.tensors[E].row_mut(0) {
            *v = 0.0;
        }
        Ok(net.with_params(p))
    }

    pub fn load(path: &Path) -> Result<PolicyValueNet, String> {
        PolicyValueNet::from_npz(&Npz::load(path)?)
    }
}

#[derive(Serialize, Deserialize)]
struct CheckpointMeta {
    kind: String,
    version: i64,
    features: usize,
    input_features: usize,
    card_fingerprint: String,
    #[serde(flatten)]
    cfg: NetConfig,
}

/// Softmax over the legal actions only; with no legal action every entry is zero.
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let mut out = vec![0.0f32; logits.len()];
    let top = logits
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(z, _)| *z as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return out;
    }
    let mut total = 0.0f64;
    for (i, (z, m)) in logits.iter().zip(mask).enumerate() {
        if *m {
            let e = (*z as f64 - top).exp();
            out[i] = e as f32;
            total += e;
        }
    }
    for v in &mut out {
        *v = (*v as f64 / total) as f32;
    }
    out
}

/// Softmax of every row over its legal entries; rows without legal entries are zero.
pub fn masked_softmax_rows(logits: &Mat, masks: &[&[bool]]) -> Mat {
    let mut out = logits.zeros_like();
    for (r, mask) in masks.iter().enumerate().take(logits.rows) {
        out.row_mut(r)
            .copy_from_slice(&masked_softmax(logits.row(r), mask));
    }
    out
}

/// FP64 softmax over the eight terminal-placement bins, shared by CPU and CUDA hosts.
pub(crate) fn placement_probabilities(logits: &[f32]) -> [f64; 8] {
    let top = logits[..8]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut result = [0.0; 8];
    for (i, p) in result.iter_mut().enumerate() {
        *p = (f64::from(logits[i]) - top).exp();
    }
    let sum: f64 = result.iter().sum();
    for p in &mut result {
        *p /= sum;
    }
    result
}

pub(crate) fn placement_support(index: usize) -> f64 {
    1.0 - 2.0 * index as f64 / 7.0
}

pub(crate) fn critic_prediction(logits: &[f32], cfg: &NetConfig) -> f32 {
    if cfg.critic_distribution {
        placement_probabilities(logits)
            .iter()
            .enumerate()
            .map(|(i, p)| p * placement_support(i))
            .sum::<f64>() as f32
    } else {
        logits[0].tanh()
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;

    #[test]
    fn tier_one_hot_keeps_scalar_and_safety_flag() {
        let net = PolicyValueNet::new(1, NetConfig::default()).unwrap();
        let mut x = vec![0.0; N_FEATURES];
        let cards = vec![0; N_CARD_SLOTS];
        let mask = vec![false; N_ACTIONS];
        let mut row = vec![99.0; net.p.tensors[W0].rows];
        for tier in 1..=7 {
            x[10] = tier as f32 / 7.0;
            for protected in [0.0, 1.0] {
                x[COMBAT_PROTECTED_FEATURE] = protected;
                net.write_input(&x, &cards, &mask, &mut row, None).unwrap();
                for (index, &value) in row[TURN_BINS..ONE_HOT_FEATURES].iter().enumerate() {
                    assert_eq!(value, (index == tier.min(6) - 1) as u8 as f32);
                }
                for feature in [10, COMBAT_PROTECTED_FEATURE] {
                    let offset = net
                        .input_features
                        .iter()
                        .position(|&i| i == feature)
                        .unwrap();
                    assert_eq!(row[ONE_HOT_FEATURES + offset], x[feature]);
                }
            }
        }
    }

    #[test]
    fn turn_one_hot_caps_at_sixteen_and_keeps_the_original_scalar() {
        let net = PolicyValueNet::new(1, NetConfig::default()).unwrap();
        let mut x = vec![0.0; N_FEATURES];
        x[7] = 0.8; // Other numeric features must keep their value and offset.
        let cards = vec![0; N_CARD_SLOTS];
        let mask = vec![false; N_ACTIONS];
        let mut row = vec![99.0; net.p.tensors[W0].rows];
        for turn in (0..=32).chain([100, 1000]) {
            x[0] = turn as f32 / 20.0;
            net.write_input(&x, &cards, &mask, &mut row, None).unwrap();
            let expected = turn.clamp(1, 16) as usize - 1;
            for (index, value) in row[..TURN_BINS].iter().enumerate() {
                assert_eq!(*value, if index == expected { 1.0 } else { 0.0 });
            }
            assert_eq!(
                row[ONE_HOT_FEATURES + net.input_features.iter().position(|&i| i == 0).unwrap()],
                turn as f32 / 20.0
            );
            assert_eq!(
                row[ONE_HOT_FEATURES + net.input_features.iter().position(|&i| i == 7).unwrap()],
                0.8
            );
        }
        let infer = |turn| {
            let mut x = x.clone();
            x[0] = turn as f32 / 20.0;
            net.infer_f32(&x, &cards, &mask).unwrap()
        };
        assert_ne!(infer(5), infer(6));
        assert_ne!(infer(16), infer(17));
        assert_ne!(infer(16), infer(100));
    }
}
