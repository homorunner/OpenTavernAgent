//! FP32 CUDA network and optimizer. Checkpoints retain the ordinary CPU tensor layout.
mod check;
mod ffi;
use crate::{
    linalg::Mat,
    net::{self, Activation, Params, PolicyValueNet},
    observation::Observation,
    ppo::{self, LossMetrics, PpoOptions, Trainer},
    rollout::Sample,
};
pub use check::{CheckArgs, check};
use ffi::{
    Arg::{self, Double, Float, Int, Ptr},
    Gpu, Matrix,
};
use rand::{Rng, seq::SliceRandom};
use std::rc::Rc;
use std::time::Instant;

struct Norm {
    xhat: Matrix,
    inv: Matrix,
}
struct Block {
    normed: Matrix,
    pre: Matrix,
    activated: Matrix,
    norm: Norm,
}
pub struct Cache {
    input: Matrix,
    cards: Vec<i32>,
    stem: Matrix,
    stem_norm: Norm,
    blocks: Vec<Block>,
    head: Matrix,
    head_norm: Option<Norm>,
    value_pre: Matrix,
    value_hidden: Matrix,
}

pub struct CudaNet {
    config: net::NetConfig,
    card_fingerprint: u64,
    gpu: Rc<Gpu>,
    weights: Vec<Matrix>,
    m: Vec<Matrix>,
    v: Vec<Matrix>,
    pub optimizer_step: u64,
}

fn kind(a: Activation) -> i32 {
    match a {
        Activation::Gelu => 0,
        Activation::Silu => 1,
        Activation::Relu => 2,
    }
}
fn blocks(n: usize) -> usize {
    n.div_ceil(256)
}
fn shape(m: &Matrix) -> Arg {
    Int(m.len() as i32)
}

impl CudaNet {
    pub fn new(trainer: &Trainer, device: usize) -> Result<Self, String> {
        let gpu = Gpu::new(device)?;
        let upload = |p: &Params| {
            p.tensors
                .iter()
                .map(|m| gpu.from_f32(m.rows, m.cols, &m.data))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            config: trainer.net.cfg.clone(),
            card_fingerprint: trainer.net.card_fingerprint,
            weights: upload(&trainer.net.p)?,
            m: upload(&trainer.m)?,
            v: upload(&trainer.v)?,
            optimizer_step: trainer.optimizer_step,
            gpu,
        })
    }
    fn bias(&self, x: &Matrix, index: usize) -> Result<(), String> {
        self.gpu.launch(
            "bias",
            blocks(x.len()),
            &mut [
                Ptr(x.ptr()),
                Ptr(self.weights[index].ptr()),
                shape(x),
                Int(x.cols as i32),
            ],
        )
    }
    fn linear(&self, x: &Matrix, weight: usize, bias: usize) -> Result<Matrix, String> {
        let y = self.gpu.gemm(x, false, &self.weights[weight], false)?;
        self.bias(&y, bias)?;
        Ok(y)
    }
    fn activate(&self, x: &Matrix, activation: Activation) -> Result<Matrix, String> {
        let y = self.gpu.matrix(x.rows, x.cols)?;
        self.gpu.launch(
            "activation",
            blocks(x.len()),
            &mut [Ptr(x.ptr()), Ptr(y.ptr()), shape(x), Int(kind(activation))],
        )?;
        Ok(y)
    }
    fn act_grad(&self, dx: &Matrix, x: &Matrix, activation: Activation) -> Result<(), String> {
        self.gpu.launch(
            "activation_grad",
            blocks(dx.len()),
            &mut [
                Ptr(dx.ptr()),
                Ptr(x.ptr()),
                shape(dx),
                Int(kind(activation)),
            ],
        )
    }
    fn norm(&self, x: &Matrix, params: Option<(usize, usize)>) -> Result<(Matrix, Norm), String> {
        let y = self.gpu.matrix(x.rows, x.cols)?;
        let xhat = self.gpu.matrix(x.rows, x.cols)?;
        let inv = self.gpu.matrix(x.rows, 1)?;
        let (g, b) = params
            .map(|(g, b)| (self.weights[g].ptr(), self.weights[b].ptr()))
            .unwrap_or((0, 0));
        self.gpu.launch(
            "layer_norm",
            x.rows,
            &mut [
                Ptr(x.ptr()),
                Ptr(g),
                Ptr(b),
                Ptr(y.ptr()),
                Ptr(xhat.ptr()),
                Ptr(inv.ptr()),
                Int(x.rows as i32),
                Int(x.cols as i32),
            ],
        )?;
        Ok((y, Norm { xhat, inv }))
    }
    fn norm_grad(
        &self,
        dy: &Matrix,
        n: &Norm,
        g: Option<usize>,
    ) -> Result<(Matrix, Matrix, Matrix), String> {
        let dx = self.gpu.matrix(dy.rows, dy.cols)?;
        let dg = self.gpu.matrix(1, dy.cols)?;
        let db = self.gpu.matrix(1, dy.cols)?;
        self.gpu.launch(
            "layer_norm_dx",
            dy.rows,
            &mut [
                Ptr(dy.ptr()),
                Ptr(n.xhat.ptr()),
                Ptr(n.inv.ptr()),
                Ptr(g.map(|i| self.weights[i].ptr()).unwrap_or(0)),
                Ptr(dx.ptr()),
                Int(dy.rows as i32),
                Int(dy.cols as i32),
            ],
        )?;
        self.gpu.launch(
            "layer_norm_params",
            blocks(dy.cols),
            &mut [
                Ptr(dy.ptr()),
                Ptr(n.xhat.ptr()),
                Ptr(dg.ptr()),
                Ptr(db.ptr()),
                Int(dy.rows as i32),
                Int(dy.cols as i32),
            ],
        )?;
        Ok((dx, dg, db))
    }
    fn axpy(&self, dst: &Matrix, src: &Matrix, scale: f32) -> Result<(), String> {
        self.gpu.launch(
            "axpy",
            blocks(dst.len()),
            &mut [Ptr(dst.ptr()), Ptr(src.ptr()), shape(dst), Float(scale)],
        )
    }
    fn scale(&self, x: &Matrix, scale: f32) -> Result<(), String> {
        self.gpu.launch(
            "scale",
            blocks(x.len()),
            &mut [Ptr(x.ptr()), shape(x), Float(scale)],
        )
    }
    fn sum(&self, x: &Matrix) -> Result<Matrix, String> {
        let out = self.gpu.matrix(1, x.cols)?;
        self.gpu.launch(
            "column_sum",
            blocks(x.cols),
            &mut [
                Ptr(x.ptr()),
                Ptr(out.ptr()),
                Int(x.rows as i32),
                Int(x.cols as i32),
            ],
        )?;
        Ok(out)
    }
    /// Assemble the same features as CPU; replace embedding entries using current device weights.
    pub fn forward(
        &self,
        net: &PolicyValueNet,
        observations: &[&Observation],
    ) -> Result<(Mat, Vec<f32>, Cache), String> {
        if net.cfg != self.config || net.card_fingerprint != self.card_fingerprint {
            return Err("CUDA model configuration or card table does not match".into());
        }
        if observations.is_empty() {
            return Err("empty CUDA inference batch".into());
        }
        let xs: Vec<_> = observations.iter().map(|o| o.x.as_slice()).collect();
        let cards: Vec<_> = observations.iter().map(|o| o.cards.as_slice()).collect();
        let masks: Vec<_> = observations.iter().map(|o| o.mask.as_slice()).collect();
        let (input, used) = net.assemble(&xs, &cards, &masks)?;
        let input = self.gpu.from_f32(input.rows, input.cols, &input.data)?;
        let card_buffer = self.gpu.upload(&used)?;
        let slots = net.n_card_slots();
        let dim = net.cfg.embedding_dim;
        self.gpu.launch(
            "embed",
            blocks(input.rows * slots * dim),
            &mut [
                Ptr(input.ptr()),
                Ptr(card_buffer.ptr),
                Ptr(self.weights[net::E].ptr()),
                Int(input.rows as i32),
                Int(input.cols as i32),
                Int(net.n_input_features() as i32),
                Int(slots as i32),
                Int(dim as i32),
            ],
        )?;
        let z = self.linear(&input, net::W0, net::B0)?;
        let (stem, stem_norm) = self.norm(&z, Some((net::G0, net::K0)))?;
        let mut h = self.activate(&stem, net.cfg.activation)?;
        let mut layers = Vec::with_capacity(net.cfg.blocks);
        for i in 0..net.cfg.blocks {
            let (normed, norm) = self.norm(&h, Some((net::g(i), net::k(i))))?;
            let pre = self.linear(&normed, net::w_up(i), net::b_up(i))?;
            let activated = self.activate(&pre, net.cfg.activation)?;
            let down = self.linear(&activated, net::w_down(i), net::b_down(i))?;
            self.axpy(&h, &down, net.cfg.residual_scale)?;
            layers.push(Block {
                normed,
                pre,
                activated,
                norm,
            });
        }
        let head_norm = if net.cfg.head_norm {
            let (y, n) = self.norm(&h, None)?;
            h = y;
            Some(n)
        } else {
            None
        };
        let logits = self.linear(&h, net.p.wp(), net.p.bp())?;
        let value_pre = self.linear(&h, net.p.wv_hidden(), net.p.bv_hidden())?;
        let value_hidden = self.activate(&value_pre, Activation::Gelu)?;
        let values = self.linear(&value_hidden, net.p.wv(), net.p.bv())?;
        self.gpu.launch(
            "value_tanh",
            blocks(values.len()),
            &mut [Ptr(values.ptr()), shape(&values)],
        )?;
        let logits = Mat::from_vec(logits.rows, logits.cols, logits.buffer.read()?);
        let values = values.buffer.read()?;
        Ok((
            logits,
            values,
            Cache {
                input,
                cards: used,
                stem,
                stem_norm,
                blocks: layers,
                head: h,
                head_norm,
                value_pre,
                value_hidden,
            },
        ))
    }
    pub fn infer(
        &self,
        net: &PolicyValueNet,
        observations: &[&Observation],
    ) -> Result<(Mat, Vec<f32>), String> {
        let (logits, values, _) = self.forward(net, observations)?;
        Ok((logits, values))
    }
    fn backward(
        &self,
        net: &PolicyValueNet,
        c: &Cache,
        dlogits: &Mat,
        dvalue: &[f32],
    ) -> Result<Vec<Matrix>, String> {
        let mut grad = self
            .weights
            .iter()
            .map(|m| self.gpu.matrix(m.rows, m.cols))
            .collect::<Result<Vec<_>, _>>()?;
        let dl = self
            .gpu
            .from_f32(dlogits.rows, dlogits.cols, &dlogits.data)?;
        let dv = self.gpu.from_f32(dvalue.len(), 1, dvalue)?;
        grad[net.p.wp()] = self.gpu.gemm(&c.head, true, &dl, false)?;
        grad[net.p.bp()] = self.sum(&dl)?;
        grad[net.p.wv()] = self.gpu.gemm(&c.value_hidden, true, &dv, false)?;
        grad[net.p.bv()] = self.sum(&dv)?;
        let mut dh = self.gpu.gemm(&dl, false, &self.weights[net.p.wp()], true)?;
        let dvh = self.gpu.gemm(&dv, false, &self.weights[net.p.wv()], true)?;
        self.act_grad(&dvh, &c.value_pre, Activation::Gelu)?;
        grad[net.p.wv_hidden()] = self.gpu.gemm(&c.head, true, &dvh, false)?;
        grad[net.p.bv_hidden()] = self.sum(&dvh)?;
        let add = self
            .gpu
            .gemm(&dvh, false, &self.weights[net.p.wv_hidden()], true)?;
        self.axpy(&dh, &add, 1.)?;
        if let Some(n) = &c.head_norm {
            dh = self.norm_grad(&dh, n, None)?.0;
        }
        for i in (0..net.cfg.blocks).rev() {
            let b = &c.blocks[i];
            let dr = dh.copy()?;
            self.scale(&dr, net.cfg.residual_scale)?;
            grad[net::w_down(i)] = self.gpu.gemm(&b.activated, true, &dr, false)?;
            grad[net::b_down(i)] = self.sum(&dr)?;
            let du = self
                .gpu
                .gemm(&dr, false, &self.weights[net::w_down(i)], true)?;
            self.act_grad(&du, &b.pre, net.cfg.activation)?;
            grad[net::w_up(i)] = self.gpu.gemm(&b.normed, true, &du, false)?;
            grad[net::b_up(i)] = self.sum(&du)?;
            let dn = self
                .gpu
                .gemm(&du, false, &self.weights[net::w_up(i)], true)?;
            let (dx, dg, db) = self.norm_grad(&dn, &b.norm, Some(net::g(i)))?;
            grad[net::g(i)] = dg;
            grad[net::k(i)] = db;
            self.axpy(&dh, &dx, 1.)?;
        }
        self.act_grad(&dh, &c.stem, net.cfg.activation)?;
        let (dz, dg, db) = self.norm_grad(&dh, &c.stem_norm, Some(net::G0))?;
        grad[net::G0] = dg;
        grad[net::K0] = db;
        grad[net::W0] = self.gpu.gemm(&c.input, true, &dz, false)?;
        grad[net::B0] = self.sum(&dz)?;
        let dx = self.gpu.gemm(&dz, false, &self.weights[net::W0], true)?;
        let vocabulary = self.weights[net::E].rows;
        let mut offsets = vec![0i32; vocabulary + 1];
        for &id in &c.cards {
            if id > 0 {
                offsets[id as usize + 1] += 1;
            }
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut positions = vec![0i32; (*offsets.last().unwrap() as usize).max(1)];
        let mut cursors = offsets.clone();
        for (position, &id) in c.cards.iter().enumerate() {
            if id > 0 {
                let i = id as usize;
                positions[cursors[i] as usize] = position as i32;
                cursors[i] += 1;
            }
        }
        let offsets = self.gpu.upload(&offsets)?;
        let positions = self.gpu.upload(&positions)?;
        let e = &grad[net::E];
        self.gpu.launch(
            "embed_grad",
            blocks(e.len()),
            &mut [
                Ptr(dx.ptr()),
                Ptr(offsets.ptr),
                Ptr(positions.ptr),
                Ptr(e.ptr()),
                Int(vocabulary as i32),
                Int(dx.cols as i32),
                Int(net.n_input_features() as i32),
                Int(net.n_card_slots() as i32),
                Int(net.cfg.embedding_dim as i32),
            ],
        )?;
        Ok(grad)
    }
    pub fn gradients(
        &self,
        net: &PolicyValueNet,
        rows: &[&Sample],
        opts: &PpoOptions,
    ) -> Result<(Vec<Matrix>, LossMetrics), String> {
        opts.validate()?;
        let observations: Vec<_> = rows.iter().map(|s| &s.transition.obs).collect();
        let (logits, values, cache) = self.forward(net, &observations)?;
        let (dl, dv, metrics) = ppo::output_gradients(rows, &logits, &values, opts)?;
        let grads = self.backward(net, &cache, &dl, &dv)?;
        let norm = 1. / rows.len() as f64;
        for g in &grads {
            self.scale(g, norm as f32)?;
        }
        let mut mean = LossMetrics::default();
        mean.add_scaled(&metrics, norm);
        if !mean.total_loss(opts).is_finite() || !mean.approx_kl.is_finite() {
            return Err("nonfinite CUDA PPO loss".into());
        }
        Ok((grads, mean))
    }
    fn squares(&self, matrices: &[&Matrix]) -> Result<f64, String> {
        let count: usize = matrices.iter().map(|m| blocks(m.len())).sum();
        let parts = self.gpu.buffer(count * 8)?;
        let mut offset = 0;
        for m in matrices {
            self.gpu.launch(
                "sum_squares",
                blocks(m.len()),
                &mut [Ptr(m.ptr()), Ptr(parts.ptr + (offset * 8) as u64), shape(m)],
            )?;
            offset += blocks(m.len());
        }
        // One synchronization/download for all tensors. Preserve the per-tensor sum order.
        let values = parts.read::<f64>()?;
        offset = 0;
        let mut sum = 0.;
        for m in matrices {
            let end = offset + blocks(m.len());
            sum += values[offset..end].iter().sum::<f64>();
            offset = end;
        }
        Ok(sum)
    }
    pub fn apply(
        &mut self,
        net: &PolicyValueNet,
        grad: &[Matrix],
        lr: f64,
        opts: &PpoOptions,
    ) -> Result<f64, String> {
        opts.validate()?;
        if grad.len() != self.weights.len()
            || grad
                .iter()
                .zip(&self.weights)
                .any(|(g, w)| g.rows != w.rows || g.cols != w.cols || !g.belongs_to(&self.gpu))
        {
            return Err("CUDA gradient shape or device context mismatch".into());
        }
        if !lr.is_finite() || lr <= 0. {
            return Err("learning rate must be finite and positive".into());
        }
        let norm = self.squares(&grad.iter().collect::<Vec<_>>())?.sqrt();
        if !norm.is_finite() {
            return Err("nonfinite CUDA gradient norm".into());
        }
        let scale = if norm > opts.max_grad_norm {
            opts.max_grad_norm / norm
        } else {
            1.
        };
        self.optimizer_step = self
            .optimizer_step
            .checked_add(1)
            .ok_or("optimizer step overflow")?;
        let bias1 = 1. - 0.9f64.powf(self.optimizer_step as f64);
        let bias2 = 1. - 0.999f64.powf(self.optimizer_step as f64);
        for (i, g) in grad.iter().enumerate() {
            self.gpu.launch(
                "adam",
                blocks(g.len()),
                &mut [
                    Ptr(self.weights[i].ptr()),
                    Ptr(self.m[i].ptr()),
                    Ptr(self.v[i].ptr()),
                    Ptr(g.ptr()),
                    shape(g),
                    Double(scale),
                    Double(bias1),
                    Double(bias2),
                    Double(lr),
                    Double(if net.p.is_weight(i) {
                        opts.weight_decay
                    } else {
                        0.
                    }),
                ],
            )?;
        }
        // Validate device state before accepting the update. No full gradient download.
        if !self
            .squares(
                &self
                    .weights
                    .iter()
                    .chain(&self.m)
                    .chain(&self.v)
                    .collect::<Vec<_>>(),
            )?
            .is_finite()
        {
            return Err("nonfinite CUDA optimizer state".into());
        }
        Ok(norm)
    }
    pub fn download_gradients(
        &self,
        net: &PolicyValueNet,
        grad: &[Matrix],
    ) -> Result<Params, String> {
        let mut out = net.p.zeros_like();
        for (dst, src) in out.tensors.iter_mut().zip(grad) {
            dst.data = src.buffer.read()?;
        }
        Ok(out)
    }
    pub fn download(&self, trainer: &mut Trainer) -> Result<(), String> {
        for (dst, src) in [
            (&mut trainer.net.p, &self.weights),
            (&mut trainer.m, &self.m),
            (&mut trainer.v, &self.v),
        ] {
            for (d, s) in dst.tensors.iter_mut().zip(src) {
                d.data = s.buffer.read()?;
            }
        }
        trainer.optimizer_step = self.optimizer_step;
        Ok(())
    }
    pub fn synchronize(&self) -> Result<(), String> {
        self.gpu.sync()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        net: &PolicyValueNet,
        samples: &[Sample],
        epochs: usize,
        batch: usize,
        lr: f64,
        options: &PpoOptions,
        rng: &mut impl Rng,
    ) -> Result<ppo::UpdateMetrics, String> {
        options.validate()?;
        if samples.is_empty() || epochs == 0 || batch == 0 || !lr.is_finite() || lr <= 0. {
            return Err("PPO needs samples, positive epochs, batch size and learning rate".into());
        }
        let mut order: Vec<_> = (0..samples.len()).collect();
        let mut summary = ppo::UpdateMetrics::default();
        let mut seen = 0;
        'epochs: for epoch in 0..epochs {
            order.shuffle(rng);
            for chunk in order.chunks(batch) {
                let rows: Vec<_> = chunk.iter().map(|&i| &samples[i]).collect();
                let start = Instant::now();
                let (grad, metrics) = self.gradients(net, &rows, options)?;
                self.synchronize()?;
                summary.gradient_seconds += start.elapsed().as_secs_f64();
                summary.evaluated_samples += rows.len();
                summary.max_approx_kl = summary.max_approx_kl.max(metrics.approx_kl);
                if options.target_kl > 0. && metrics.approx_kl > options.target_kl {
                    summary.early_stop = true;
                    break 'epochs;
                }
                let start = Instant::now();
                summary.mean_grad_norm += self.apply(net, &grad, lr, options)?;
                summary.optimizer_seconds += start.elapsed().as_secs_f64();
                summary.loss.add_scaled(&metrics, rows.len() as f64);
                seen += rows.len();
                if epoch == 0 {
                    summary.first_epoch_samples += rows.len();
                }
                summary.updates += 1;
            }
            summary.epochs_completed += 1;
        }
        let sums = summary.loss.clone();
        summary.loss = LossMetrics::default();
        summary.loss.add_scaled(&sums, 1. / seen.max(1) as f64);
        summary.mean_grad_norm /= summary.updates.max(1) as f64;
        summary.trained_samples = seen;
        summary.effective_epochs = seen as f64 / samples.len() as f64;
        Ok(summary)
    }
}
