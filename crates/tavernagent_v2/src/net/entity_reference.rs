// Expanded action heads retained only as a numerical test oracle.
use super::*;
struct ReferenceActionCache {
    groups: Vec<MlpCache>,
}
impl Layout {
    fn reference_policy(
        &self,
        net: &PolicyValueNet,
        head: &Mat,
        encoded: &Mat,
        globals: &Mat,
    ) -> (Mat, ReferenceActionCache) {
        let mut logits = Mat::zeros(head.rows, N_ACTIONS);
        for r in 0..head.rows {
            logits.row_mut(r)[..a::A_BUY].copy_from_slice(globals.row(r));
        }
        let mut groups = Vec::with_capacity(5);
        for (kind, &count) in ACTION_COUNTS.iter().enumerate() {
            let mut input = Mat::zeros(head.rows * count, Self::action_dim(head.cols, kind));
            for r in 0..head.rows {
                for j in 0..count {
                    let (first, second) = self.action_slots(kind, j);
                    let row = input.row_mut(r * count + j);
                    row[..head.cols].copy_from_slice(head.row(r));
                    row[head.cols..head.cols + WIDTH]
                        .copy_from_slice(&encoded.row(r)[first * WIDTH..(first + 1) * WIDTH]);
                    if let Some(second) = second {
                        row[head.cols + WIDTH..head.cols + WIDTH * 2]
                            .copy_from_slice(&encoded.row(r)[second * WIDTH..(second + 1) * WIDTH]);
                        row[head.cols + WIDTH * 2 + j % crate::actions::N_POS] = 1.0;
                    }
                }
            }
            let (out, cache) = mlp(net, net.p.extra() + ENCODER_TENSORS + kind * 4, input);
            for r in 0..head.rows {
                for j in 0..count {
                    logits.row_mut(r)[ACTION_STARTS[kind] + j] = out.data[r * count + j];
                }
            }
            groups.push(cache);
        }
        (logits, ReferenceActionCache { groups })
    }

    fn reference_policy_backward(
        &self,
        net: &PolicyValueNet,
        cache: &ReferenceActionCache,
        dlogits: &Mat,
        grads: &mut Params,
        dh: &mut Mat,
        dx: &mut Mat,
    ) -> Mat {
        let mut globals = Mat::zeros(dlogits.rows, a::A_BUY);
        for r in 0..dlogits.rows {
            globals
                .row_mut(r)
                .copy_from_slice(&dlogits.row(r)[..a::A_BUY]);
        }
        for (kind, &count) in ACTION_COUNTS.iter().enumerate() {
            let mut grad = Mat::zeros(dlogits.rows * count, 1);
            for r in 0..dlogits.rows {
                for j in 0..count {
                    grad.data[r * count + j] = dlogits.row(r)[ACTION_STARTS[kind] + j];
                }
            }
            let di = mlp_backward(
                net,
                net.p.extra() + ENCODER_TENSORS + kind * 4,
                &cache.groups[kind],
                &grad,
                grads,
            );
            let hidden = dh.cols;
            for r in 0..dlogits.rows {
                for j in 0..count {
                    let row = di.row(r * count + j);
                    for (d, x) in dh.row_mut(r).iter_mut().zip(row) {
                        *d += x;
                    }
                    let (first, second) = self.action_slots(kind, j);
                    for (d, x) in dx.row_mut(r)[first * WIDTH..(first + 1) * WIDTH]
                        .iter_mut()
                        .zip(&row[hidden..hidden + WIDTH])
                    {
                        *d += x;
                    }
                    if let Some(second) = second {
                        for (d, x) in dx.row_mut(r)[second * WIDTH..(second + 1) * WIDTH]
                            .iter_mut()
                            .zip(&row[hidden + WIDTH..hidden + WIDTH * 2])
                        {
                            *d += x;
                        }
                    }
                }
            }
        }
        globals
    }
}

#[test]
fn factored_heads_match_expanded_logits_probabilities_and_gradients() {
    fn close(label: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        let max_abs = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let scale = b.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            max_abs <= 3e-6 + 3e-5 * scale,
            "{label}: max abs {max_abs}, scale {scale}"
        );
    }
    fn inputs(rows: usize, cols: usize, salt: usize) -> Mat {
        Mat::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| {
                    if (i + salt).is_multiple_of(7) {
                        0.0
                    } else {
                        ((i * 13 + salt) % 97) as f32 / 48.0 - 1.0
                    }
                })
                .collect(),
        )
    }
    for activation in [Activation::Gelu, Activation::Silu, Activation::Relu] {
        let net = PolicyValueNet::new(
            19,
            NetConfig {
                architecture: Architecture::EntityActions,
                hidden: 320,
                activation,
                ..Default::default()
            },
        )
        .unwrap();
        let layout = &net.entity;
        for batch in [1, 3, 32] {
            let head = inputs(batch, 320, 3);
            let encoded = inputs(batch, layout.slots * WIDTH + GLOBAL_WIDTH + N_ACTIONS, 5);
            let globals = inputs(batch, a::A_BUY, 7);
            let (actual, cache) = layout.policy(&net, &head, &encoded, &globals);
            let (expected, reference) = layout.reference_policy(&net, &head, &encoded, &globals);
            close("logits", &actual.data, &expected.data);
            for r in 0..batch {
                let mask: Vec<_> = (0..N_ACTIONS).map(|i| (i + r) % 3 != 0).collect();
                close(
                    "probabilities",
                    &masked_softmax(actual.row(r), &mask),
                    &masked_softmax(expected.row(r), &mask),
                );
            }
            let dout = inputs(batch, N_ACTIONS, 11);
            let mut grads = net.p.zeros_like();
            let mut reference_grads = net.p.zeros_like();
            let mut dh = head.zeros_like();
            let mut rh = head.zeros_like();
            let mut dx = encoded.zeros_like();
            let mut rx = encoded.zeros_like();
            let dg = layout.policy_backward(&net, &cache, &dout, &mut grads, &mut dh, &mut dx);
            let rg = layout.reference_policy_backward(
                &net,
                &reference,
                &dout,
                &mut reference_grads,
                &mut rh,
                &mut rx,
            );
            close("global logits gradient", &dg.data, &rg.data);
            close("trunk gradient", &dh.data, &rh.data);
            close("entity gradient", &dx.data, &rx.data);
            for (name, (actual, expected)) in net
                .p
                .names()
                .iter()
                .zip(grads.tensors.iter().zip(&reference_grads.tensors))
            {
                close(name, &actual.data, &expected.data);
            }
        }
    }
}
