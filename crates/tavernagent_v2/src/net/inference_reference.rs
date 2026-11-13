// Allocating inference retained as a numerical test oracle.
use super::*;
impl PolicyValueNet {
    /// FP32 logits and value, using the same operations as the training forward pass.
    fn reference_infer_f32(
        &self,
        x: &[f32],
        cards: &[i32],
        mask: &[bool],
    ) -> Result<(Vec<f32>, f32), String> {
        let (x_in, used) = self.assemble(&[x], &[cards], &[mask])?;
        let x_in = self.entity.forward(self, &x_in, &used).0;
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
        {
            logits = self
                .entity
                .policy(
                    self,
                    &Mat::from_vec(1, h.len(), h.clone()),
                    &x_in,
                    &Mat::from_vec(1, logits.len(), logits),
                )
                .0
                .data;
        }
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
}

#[test]
fn fused_activations_preserve_values_and_derivatives() {
    let xs = (-4000..=4000).map(|i| i as f32 / 100.0).chain([
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ]);
    for x in xs {
        for act in [Activation::Gelu, Activation::Silu, Activation::Relu] {
            let (value, derivative) = act.apply_and_grad(x);
            assert_eq!(
                value.to_bits(),
                act.apply(x).to_bits(),
                "{act:?} value at {x}"
            );
            assert_eq!(
                derivative.to_bits(),
                act.grad(x).to_bits(),
                "{act:?} derivative at {x}"
            );
        }
    }
}

#[test]
fn inference_workspace_matches_allocating_reference_across_models_and_observations() {
    let configs = [
        NetConfig {
            architecture: Architecture::EntityActions,
            hidden: 16,
            blocks: 2,
            embedding_dim: 4,
            head_norm: true,
            critic_distribution: true,
            critic_health: true,
            ..Default::default()
        },
        NetConfig {
            architecture: Architecture::EntityActions,
            hidden: 320,
            ..Default::default()
        },
        NetConfig {
            architecture: Architecture::EntityActions,
            hidden: 24,
            blocks: 1,
            embedding_dim: 8,
            activation: Activation::Silu,
            ..Default::default()
        },
        NetConfig {
            architecture: Architecture::EntityActions,
            hidden: 16,
            blocks: 2,
            embedding_dim: 4,
            activation: Activation::Relu,
            ..Default::default()
        },
    ];
    let mut nets: Vec<_> = configs
        .into_iter()
        .map(|cfg| PolicyValueNet::new(41, cfg).unwrap())
        .collect();
    // Revisit shapes after growing/shrinking buffers, including changed parameters
    // and an invalid request, to detect stale observation/weight caches.
    for pass in 0..3 {
        for net in &mut nets {
            net.p.tensors[W0].data[0] += 0.01;
            assert!(net.infer_f32(&[], &[], &[]).is_err());
            let mut x = vec![0.0; N_FEATURES];
            let mut cards = vec![0; N_CARD_SLOTS];
            let mask: Vec<_> = (0..N_ACTIONS).map(|i| (i + pass) % 3 != 0).collect();
            if pass != 1 {
                for (i, v) in x.iter_mut().enumerate() {
                    *v = ((i * 7 + pass) % 17) as f32 / 20.0;
                }
                let card = net
                    .allowed
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find(|(_, yes)| **yes)
                    .unwrap()
                    .0 as i32;
                for (i, v) in cards.iter_mut().enumerate() {
                    if (i + pass) % 3 != 0 {
                        *v = card;
                    }
                }
            }
            let expected = net.reference_infer_f32(&x, &cards, &mask).unwrap();
            let actual = net.infer_f32(&x, &cards, &mask).unwrap();
            assert_eq!(actual, expected, "{:?}, pass {pass}", net.cfg);
            let (batch_logits, batch_values, _) = net.forward(&[&x], &[&cards], &[&mask]).unwrap();
            for (a, b) in actual.0.iter().zip(&batch_logits.data) {
                assert!((a - b).abs() < 1e-5, "infer vs training: {a} != {b}");
            }
            assert!((actual.1 - batch_values[0]).abs() < 1e-5);
        }
    }
}
