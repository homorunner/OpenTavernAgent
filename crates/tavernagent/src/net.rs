//! A small pre-norm residual policy/value network with a shared card embedding.
//!
//! ```text
//! numeric features + flattened card embeddings -> 192 -> residual blocks
//!     -> {196 policy logits, 1 tanh value}
//! ```
//!
//! Forward passes return their cache instead of storing it, so one network can serve
//! many search threads at once. Backward is written by hand and checked numerically
//! in the crate tests.

use std::path::Path;
use std::sync::OnceLock;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use taverngym::actions as a;
use taverngym::cards::{CardKind, db};
use taverngym::obs as o;

use crate::linalg::{Bf16Mat, Mat};
use crate::npz::{NpyArray, Npz};

/// The agent's action space: the engine's flat layout up to and excluding the Move block
/// (`A_MOVE..N_ACTIONS`), which is the trailing block. Warband ordering is left to a
/// separate model; this network never sees, predicts or trains on Move actions.
pub const N_ACTIONS: usize = a::A_MOVE;
pub const N_FEATURES: usize = o::OBS_FLOATS;
pub const N_CARD_SLOTS: usize = o::OBS_CARDS;
pub const DEFAULT_HIDDEN: usize = 192;
pub const DEFAULT_BLOCKS: usize = 6;
pub const MODEL_VERSION: i64 = 7;
pub const N_OPP_BOARD_SLOTS: usize = o::N_OPP * a::N_BOARD;
pub const OPP_BOARD_FEATURE_START: usize = N_FEATURES - N_OPP_BOARD_SLOTS * o::MINION_F;
/// The ended-turn flag: the seat has finished recruiting and has no legal action.
pub const ENDED_TURN_FEATURE: usize = o::ENDED_TURN_F;
/// Global flags the network never needs: hero selection (1), game over (2) and the
/// Tavern-wide freeze flag (15, encoded per shop item instead).
const EXCLUDED_GLOBALS: [usize; 3] = [1, 2, 15];

/// Observation features that enter the network: everything but the historical opponent
/// warbands and three redundant global flags.
pub fn input_feature_indices() -> Vec<usize> {
    (0..OPP_BOARD_FEATURE_START)
        .filter(|i| !EXCLUDED_GLOBALS.contains(i))
        .collect()
}

/// Card slots that enter the network: not the Deity, the secrets or the historical boards.
pub fn card_slot_indices() -> Vec<usize> {
    let quest_start = 1 + a::N_POWERS + 2;
    let secret_start = quest_start + 3 * o::N_QUESTS;
    (0..N_CARD_SLOTS)
        .filter(|&i| {
            i != 3
                && !(secret_start..secret_start + o::N_SECRETS).contains(&i)
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

/// tanh as a clamped rational polynomial (Eigen's float `ptanh`): arithmetic only, so a
/// loop over it vectorises; absolute error below 1e-6. Used by inference, not training.
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    const BOUND: f32 = 7.905_311;
    let x = x.clamp(-BOUND, BOUND);
    let x2 = x * x;
    let p = -2.760_768_5e-16f32;
    let p = p * x2 + 2.000_188e-13;
    let p = p * x2 - 8.604_672e-11;
    let p = p * x2 + 5.122_297_1e-8;
    let p = p * x2 + 1.485_722_4e-5;
    let p = p * x2 + 6.372_619_3e-4;
    let p = p * x2 + 4.893_524_6e-3;
    let p = p * x;
    let q = 1.198_258_4e-6f32;
    let q = q * x2 + 1.185_347_1e-4;
    let q = q * x2 + 2.268_434_6e-3;
    let q = q * x2 + 4.893_525e-3;
    p / q
}

/// `Activation::apply` over a slice with `fast_tanh`; one loop per activation so that the
/// compiler vectorises it.
fn apply_fast(act: Activation, h: &mut [f32]) {
    match act {
        Activation::Gelu => {
            for v in h.iter_mut() {
                let x = *v;
                let inner = GELU_C * (x + GELU_K * x * x * x);
                *v = 0.5 * x * (1.0 + fast_tanh(inner));
            }
        }
        Activation::Silu => {
            for v in h.iter_mut() {
                let x = *v;
                *v = x * 0.5 * (1.0 + fast_tanh(0.5 * x));
            }
        }
        Activation::Relu => {
            for v in h.iter_mut() {
                *v = v.max(0.0);
            }
        }
    }
}

/// `h += scale * act(z)` with `fast_tanh`.
fn add_residual_fast(act: Activation, h: &mut [f32], z: &mut [f32], scale: f32) {
    apply_fast(act, z);
    for (hv, nv) in h.iter_mut().zip(z.iter()) {
        *hv += scale * nv;
    }
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
}

impl Default for NetConfig {
    fn default() -> Self {
        NetConfig {
            hidden: DEFAULT_HIDDEN,
            blocks: DEFAULT_BLOCKS,
            embedding_dim: 16,
            activation: Activation::Gelu,
            residual_scale: 1.0,
            head_norm: false,
        }
    }
}

/// Named tensors in a fixed order: E, W0, b0, g0, k0, (Wi, bi, gi, ki) per block, Wp, bp,
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
    5 + 4 * i
}
/// Weight of residual block `i`.
pub const fn w(i: usize) -> usize {
    block(i)
}
pub const fn b(i: usize) -> usize {
    block(i) + 1
}
pub const fn g(i: usize) -> usize {
    block(i) + 2
}
pub const fn k(i: usize) -> usize {
    block(i) + 3
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
        for prefix in ["W", "b", "g", "k"] {
            names.push(format!("{prefix}{i}"));
        }
    }
    names.extend(["Wp".into(), "bp".into(), "Wv".into(), "bv".into()]);
    names
}

impl Params {
    /// Policy head weight / bias and value head weight / bias indices.
    pub fn wp(&self) -> usize {
        block(self.blocks)
    }
    pub fn bp(&self) -> usize {
        block(self.blocks) + 1
    }
    pub fn wv(&self) -> usize {
        block(self.blocks) + 2
    }
    pub fn bv(&self) -> usize {
        block(self.blocks) + 3
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
            || index == self.wv()
            || (0..self.blocks).any(|i| index == w(i))
    }
}

/// `x * w`, taking the sparse single-row path for search-time evaluations.
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

struct Layer {
    h_in: Mat,
    n: Mat,
    ln: LnCache,
}

/// Everything backward needs from a forward pass.
pub struct Cache {
    x_in: Mat,
    cards: Vec<i32>, // batch x used slots, after the allowed-kind filter
    layers: Vec<Layer>,
    h_final: Mat,
    head_ln: Option<LnCache>,
    value: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct PolicyValueNet {
    pub cfg: NetConfig,
    pub p: Params,
    input_features: Vec<usize>,
    card_slots: Vec<usize>,
    allowed: Vec<bool>,
    pub card_fingerprint: u64,
    /// bfloat16 copies of the weight matrices for search-time inference, built on first
    /// use and dropped by `params_changed`.
    fast: OnceLock<FastWeights>,
}

/// The weight matrices in bfloat16 (biases and LayerNorm parameters stay f32 in `p`).
#[derive(Clone, Debug)]
struct FastWeights {
    w0: Bf16Mat,
    blocks: Vec<Bf16Mat>,
    wp: Bf16Mat,
    wv: Bf16Mat,
}

impl FastWeights {
    fn new(p: &Params) -> FastWeights {
        FastWeights {
            w0: Bf16Mat::from_mat(&p.tensors[W0]),
            blocks: (0..p.blocks)
                .map(|i| Bf16Mat::from_mat(&p.tensors[w(i)]))
                .collect(),
            wp: Bf16Mat::from_mat(&p.tensors[p.wp()]),
            wv: Bf16Mat::from_mat(&p.tensors[p.wv()]),
        }
    }
}

impl PolicyValueNet {
    pub fn new(seed: u64, cfg: NetConfig) -> Result<PolicyValueNet, String> {
        if cfg.embedding_dim < 1 || cfg.hidden < 1 || cfg.blocks < 1 {
            return Err("embedding_dim, hidden and blocks must be positive".into());
        }
        if !(cfg.residual_scale > 0.0 && cfg.residual_scale <= 1.0) {
            return Err("residual_scale must be in (0, 1]".into());
        }
        if N_FEATURES != 3235 || N_CARD_SLOTS != 124 {
            return Err("observation schema changed; update the network".into());
        }
        let input_features = input_feature_indices();
        let card_slots = card_slot_indices();
        let n_cards = db().len() + 1;
        let d = cfg.embedding_dim;
        let h = cfg.hidden;
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
        let in_dim = input_features.len() + card_slots.len() * d;
        let mut tensors = Vec::with_capacity(9 + 4 * cfg.blocks);
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
            tensors.push(randn(h, h, he(h) * 0.6));
            tensors.push(Mat::zeros(1, h));
            tensors.push(Mat::filled(1, h, 1.0));
            tensors.push(Mat::zeros(1, h));
        }
        tensors.push(randn(h, N_ACTIONS, he(h) * 0.1));
        tensors.push(Mat::zeros(1, N_ACTIONS));
        tensors.push(randn(h, 1, he(h) * 0.1));
        tensors.push(Mat::zeros(1, 1));
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
            fast: OnceLock::new(),
        })
    }

    /// Forget the bfloat16 inference weights; call after writing to `p` so that the next
    /// search evaluation converts the new values.
    pub fn params_changed(&mut self) {
        self.fast.take();
    }

    fn fast(&self) -> &FastWeights {
        self.fast.get_or_init(|| FastWeights::new(&self.p))
    }

    pub fn n_params(&self) -> usize {
        self.p.n_params()
    }

    pub fn n_input_features(&self) -> usize {
        self.input_features.len()
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
            fast: OnceLock::new(),
        }
    }

    // ---- forward -----------------------------------------------------------

    /// Gather the network input for a batch of raw observations.
    fn assemble(&self, xs: &[&[f32]], cards: &[&[i32]]) -> Result<(Mat, Vec<i32>), String> {
        let bsz = xs.len();
        if cards.len() != bsz {
            return Err("feature and card batches differ in length".into());
        }
        let d = self.cfg.embedding_dim;
        let n_in = self.input_features.len();
        let n_slots = self.card_slots.len();
        let mut x_in = Mat::zeros(bsz, n_in + n_slots * d);
        let mut used = Vec::with_capacity(bsz * n_slots);
        let e = &self.p.tensors[E];
        for (r, (x, c)) in xs.iter().zip(cards).enumerate() {
            if x.len() != N_FEATURES || c.len() != N_CARD_SLOTS {
                return Err("observation does not match the schema".into());
            }
            let row = x_in.row_mut(r);
            for (j, &i) in self.input_features.iter().enumerate() {
                row[j] = x[i];
            }
            for (s, &slot) in self.card_slots.iter().enumerate() {
                let idx = c[slot];
                if idx < 0 || idx as usize >= e.rows {
                    return Err("card index outside the current card table".into());
                }
                let idx = if self.allowed[idx as usize] { idx } else { 0 };
                used.push(idx);
                if idx != 0 {
                    row[n_in + s * d..n_in + (s + 1) * d].copy_from_slice(e.row(idx as usize));
                }
            }
        }
        Ok((x_in, used))
    }

    /// Logits (batch x actions), values (batch) and the cache for `backward`.
    pub fn forward(
        &self,
        xs: &[&[f32]],
        cards: &[&[i32]],
    ) -> Result<(Mat, Vec<f32>, Cache), String> {
        let (x_in, used) = self.assemble(xs, cards)?;
        let p = &self.p.tensors;
        let act = self.cfg.activation;
        let mut z0 = linear(&x_in, &p[W0]);
        z0.add_row(&p[B0].data);
        let (n0, ln0) = layer_norm(&z0, &p[G0].data, &p[K0].data);
        let mut h = n0.clone();
        for v in &mut h.data {
            *v = act.apply(*v);
        }
        let mut layers = vec![Layer {
            h_in: x_in.clone(),
            n: n0,
            ln: ln0,
        }];
        for i in 0..self.cfg.blocks {
            let h_in = h.clone();
            let mut z = linear(&h_in, &p[w(i)]);
            z.add_row(&p[b(i)].data);
            let (n, ln) = layer_norm(&z, &p[g(i)].data, &p[k(i)].data);
            for (hv, nv) in h.data.iter_mut().zip(&n.data) {
                *hv += self.cfg.residual_scale * act.apply(*nv);
            }
            layers.push(Layer { h_in, n, ln });
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
        let mut value_mat = linear(&h, &p[wv]);
        value_mat.add_row(&p[bv].data);
        let value: Vec<f32> = value_mat.data.iter().map(|v| v.tanh()).collect();
        Ok((
            logits,
            value.clone(),
            Cache {
                x_in,
                cards: used,
                layers,
                h_final: h,
                head_ln,
                value,
            },
        ))
    }

    /// Masked policy over legal actions and the value of one observation.
    pub fn predict(
        &self,
        x: &[f32],
        cards: &[i32],
        mask: &[bool],
    ) -> Result<(Vec<f32>, f32), String> {
        let (logits, value) = self.infer(x, cards)?;
        Ok((masked_softmax(&logits, mask), value))
    }

    /// Logits and value of one observation for search: bfloat16 weight matrices with f32
    /// accumulation, a polynomial tanh in the activations, no training cache. The weights
    /// are converted once per parameter update (`params_changed`); training itself stays
    /// in f32 with the exact functions (`forward`/`backward`).
    pub fn infer(&self, x: &[f32], cards: &[i32]) -> Result<(Vec<f32>, f32), String> {
        let (x_in, _) = self.assemble(&[x], &[cards])?;
        let p = &self.p.tensors;
        let fast = self.fast();
        let act = self.cfg.activation;
        let mut h = fast.w0.vecmat(x_in.row(0));
        add_bias(&mut h, &p[B0].data);
        layer_norm_row(&mut h, &p[G0].data, &p[K0].data);
        apply_fast(act, &mut h);
        for i in 0..self.cfg.blocks {
            let mut z = fast.blocks[i].vecmat(&h);
            add_bias(&mut z, &p[b(i)].data);
            layer_norm_row(&mut z, &p[g(i)].data, &p[k(i)].data);
            add_residual_fast(act, &mut h, &mut z, self.cfg.residual_scale);
        }
        if self.cfg.head_norm {
            let ones = vec![1.0f32; self.cfg.hidden];
            let zeros = vec![0.0f32; self.cfg.hidden];
            layer_norm_row(&mut h, &ones, &zeros);
        }
        let mut logits = fast.wp.vecmat(&h);
        add_bias(&mut logits, &p[self.p.bp()].data);
        let value = (fast.wv.vecmat(&h)[0] + p[self.p.bv()].data[0]).tanh();
        Ok((logits, value))
    }

    /// `infer` in f32: the same operations in the same order as `forward`, so the results
    /// are bit-identical to the training forward pass; for reference and tests.
    pub fn infer_f32(&self, x: &[f32], cards: &[i32]) -> Result<(Vec<f32>, f32), String> {
        let (x_in, _) = self.assemble(&[x], &[cards])?;
        let p = &self.p.tensors;
        let act = self.cfg.activation;
        let mut h = p[W0].vecmat(x_in.row(0));
        add_bias(&mut h, &p[B0].data);
        layer_norm_row(&mut h, &p[G0].data, &p[K0].data);
        for v in &mut h {
            *v = act.apply(*v);
        }
        for i in 0..self.cfg.blocks {
            let mut z = p[w(i)].vecmat(&h);
            add_bias(&mut z, &p[b(i)].data);
            layer_norm_row(&mut z, &p[g(i)].data, &p[k(i)].data);
            for (hv, nv) in h.iter_mut().zip(&z) {
                *hv += self.cfg.residual_scale * act.apply(*nv);
            }
        }
        if self.cfg.head_norm {
            let ones = vec![1.0f32; self.cfg.hidden];
            let zeros = vec![0.0f32; self.cfg.hidden];
            layer_norm_row(&mut h, &ones, &zeros);
        }
        let mut logits = p[self.p.wp()].vecmat(&h);
        add_bias(&mut logits, &p[self.p.bp()].data);
        let value = (p[self.p.wv()].vecmat(&h)[0] + p[self.p.bv()].data[0]).tanh();
        Ok((logits, value))
    }

    // ---- backward ----------------------------------------------------------

    /// Parameter gradients for upstream gradients of the logits (batch x actions) and of
    /// the tanh value (batch).
    pub fn backward(&self, cache: &Cache, dlogits: &Mat, dvalue: &[f32]) -> Params {
        let p = &self.p.tensors;
        let act = self.cfg.activation;
        let bsz = cache.h_final.rows;
        let dvalue = Mat::from_vec(bsz, 1, dvalue.to_vec());
        let hidden = self.cfg.hidden;
        let (wp, bp, wv, bv) = (self.p.wp(), self.p.bp(), self.p.wv(), self.p.bv());
        let mut grads = self.p.zeros_like();
        grads.tensors[wp] = cache.h_final.t_matmul(dlogits);
        grads.tensors[bp] = Mat::from_vec(1, N_ACTIONS, dlogits.sum_rows());
        grads.tensors[wv] = cache.h_final.t_matmul(&dvalue);
        grads.tensors[bv] = Mat::from_vec(1, 1, dvalue.sum_rows());
        let mut dh = dlogits.matmul_t(&p[wp]);
        dh.add_assign(&dvalue.matmul_t(&p[wv]));
        if let Some(head) = &cache.head_ln {
            let ones = vec![1.0f32; hidden];
            dh = layer_norm_backward(&dh, head, &ones).0;
        }
        for i in (0..self.cfg.blocks).rev() {
            let layer = &cache.layers[i + 1];
            let mut dn = dh.clone();
            for (d, nv) in dn.data.iter_mut().zip(&layer.n.data) {
                *d *= self.cfg.residual_scale * act.grad(*nv);
            }
            let (dz, dg, dk) = layer_norm_backward(&dn, &layer.ln, &p[g(i)].data);
            grads.tensors[g(i)] = Mat::from_vec(1, hidden, dg);
            grads.tensors[k(i)] = Mat::from_vec(1, hidden, dk);
            grads.tensors[w(i)] = layer.h_in.t_matmul(&dz);
            grads.tensors[b(i)] = Mat::from_vec(1, hidden, dz.sum_rows());
            dh.add_assign(&dz.matmul_t(&p[w(i)]));
        }
        let first = &cache.layers[0];
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
        let n_in = self.input_features.len();
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
            features: N_FEATURES,
            input_features: self.input_features.len(),
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
        if meta.version != MODEL_VERSION {
            return Err(format!(
                "unsupported checkpoint format v{}; expected model format v{MODEL_VERSION}",
                meta.version
            ));
        }
        let net = PolicyValueNet::new(0, meta.cfg.clone())?;
        if meta.features != N_FEATURES
            || meta.input_features != net.input_features.len()
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
